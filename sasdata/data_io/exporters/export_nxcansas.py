"""
Export SasData to NXcanSAS (HDF5) files.

The files follow the NXcanSAS definition (https://manual.nexusformat.org/classes/applications/NXcanSAS.html) as
written by SasView: each dataset is written to its own SASentry, with its data in a SASdata group. Uncertainties are
linked to their data with the ``uncertainties`` attribute and resolutions to Q with the ``resolutions`` attribute, so
the files can be read by both :class:`sasdata.data_io.importers.import_hdf5.HDF5Importer` and SasView.

Quantities are written in their own units, which are not converted. SasView does not convert the units of HDF5 files
either, so data should be in 1/Å and 1/cm to be read correctly by SasView.
"""

import re
from pathlib import Path

import h5py
import numpy as np

from sasdata.data import SasMeasurement
from sasdata.data_io.exporter import Exporter
from sasdata.dataset_types import one_dim, three_dim, two_dim
from sasdata.quantities.quantity import Quantity

# canSAS and NeXus classes of the groups written by Metadata.as_h5, keyed by a pattern matching the group name.
GROUP_CLASSES: dict[str, tuple[str, str]] = {
    r"sasentry\d*": ("SASentry", "NXentry"),
    r"sasdata\d*": ("SASdata", "NXdata"),
    r"sassample": ("SASsample", "NXsample"),
    r"sasinstrument": ("SASinstrument", "NXinstrument"),
    r"sassource": ("SASsource", "NXsource"),
    r"sascollimation\d*": ("SAScollimation", "NXcollimator"),
    r"sasaperture\d*": ("SASaperture", "NXaperture"),
    r"sasdetector\d*": ("SASdetector", "NXdetector"),
    r"sasprocess\d*": ("SASprocess", "NXprocess"),
    r"sasnote\d*": ("SASnote", "NXnote"),
}

# The abscissae of each dataset type, in the order expected by the I_axes attribute.
AXES = {
    one_dim.name: ["Q"],
    two_dim.name: ["Qx", "Qy"],
    three_dim.name: ["Qx", "Qy", "Qz"],
}

# The resolutions of each axis, in the order they are listed in the resolutions attribute.
RESOLUTIONS = {
    "Q": ["dQ", "dQw", "dQl"],
    "Qx": ["dQx"],
    "Qy": ["dQy"],
    "Qz": ["dQz"],
}


class NXcanSASExporter(Exporter):
    """Exporter for NXcanSAS HDF5 files. Supports 1D, 2D and 3D data, with any number of datasets per file."""

    name = "NXcanSAS"
    aliases = ("NXcanSAS HDF5", "HDF5", "h5", "nexus")
    extensions = (".h5", ".hdf5", ".hdf", ".nxs")
    supported_dataset_types = (one_dim, two_dim, three_dim)
    multiple_datasets = True

    def write(self, data: dict[str, SasMeasurement], path: Path):
        with h5py.File(path, "w") as f:
            f.attrs["NX_class"] = "NXroot"
            f.attrs["creator"] = "sasdata"
            for index, (key, datum) in enumerate(data.items()):
                entry = f.create_group(f"sasentry{index + 1:02d}")
                self.write_entry(entry, key, datum)
            f.attrs["default"] = "sasentry01"

    def write_entry(self, entry: h5py.Group, key: str, datum: SasMeasurement):
        """Write one dataset, and its metadata, to a SASentry group."""
        entry.attrs["version"] = "1.1"
        entry.attrs["name"] = datum.name
        if not key.startswith("sasentry"):
            # The importer keys each entry by this attribute, or by the group name if it is absent.
            entry.attrs["sasview_key"] = key
        entry.attrs["default"] = "sasdata01"
        datum.metadata.as_h5(entry)
        if "definition" not in entry:
            entry.create_dataset("definition", data=["NXcanSAS"])
        if not any(name.startswith("run") for name in entry):
            # A run is required by the NXcanSAS definition.
            entry.create_dataset("run", data=[""])
        self.write_data(entry.create_group("sasdata01"), datum)
        add_classes(entry)

    @staticmethod
    def write_data(group: h5py.Group, datum: SasMeasurement):
        """Write the quantities of a dataset to a SASdata group."""
        contents = {name: quantity for name, quantity in datum._data_contents.items()}
        axes = AXES[datum.dataset_type.name]
        shape = grid_shape(contents, axes)

        group.attrs["signal"] = "I"
        group.attrs["I_axes"] = ",".join(axes)
        group.attrs["Q_indices"] = np.arange(len(axes), dtype=np.int32)
        for axis_index, axis in enumerate(axes):
            group.attrs[f"{axis}_indices"] = np.int32(axis_index)

        for name, quantity in contents.items():
            values = reshape(quantity.value, shape)
            dataset = write_quantity(group, name, values, quantity)
            if quantity.has_error:
                error_name = f"{name}dev"
                write_quantity(group, error_name, reshape(quantity.standard_error.value, shape), quantity)
                dataset.attrs["uncertainties"] = error_name
            resolutions = [r for r in RESOLUTIONS.get(name, []) if r in contents]
            if len(resolutions) == 1:
                dataset.attrs["resolutions"] = resolutions[0]
            elif resolutions:
                # Multiple resolutions (e.g. slit smearing) are stored as an array of names.
                dataset.attrs["resolutions"] = np.array(resolutions, dtype=h5py.string_dtype())

        if "mask" in contents:
            group.attrs["mask"] = "mask"


def write_quantity(group: h5py.Group, name: str, values, quantity: Quantity) -> h5py.Dataset:
    """Write values to a dataset with the units of the quantity."""
    dataset = group.create_dataset(name, data=np.asarray(values))
    dataset.attrs["units"] = quantity.units.ascii_symbol
    return dataset


def add_classes(group: h5py.Group):
    """Add the canSAS_class and NX_class attributes to every recognised group below (and including) ``group``."""
    name = group.name.split("/")[-1]
    for pattern, (cansas_class, nx_class) in GROUP_CLASSES.items():
        if re.fullmatch(pattern, name):
            group.attrs["canSAS_class"] = cansas_class
            group.attrs["NX_class"] = nx_class
            break
    for child in group.values():
        if isinstance(child, h5py.Group):
            add_classes(child)


def grid_shape(contents: dict[str, Quantity], axes: list[str]) -> tuple[int, ...] | None:
    """
    For 2D data stored as flat lists of points, find the detector grid the points lie on, so the data can be written
    as 2D arrays (which SasView requires to recognise 2D NXcanSAS data). Returns None if the data is already gridded,
    is not 2D, or the points do not lie on a regular grid.
    """
    if axes != ["Qx", "Qy"]:
        return None
    qx = np.asarray(contents["Qx"].value)
    qy = np.asarray(contents["Qy"].value)
    if qx.ndim != 1 or qy.shape != qx.shape or qx.size < 4:
        return None
    for fast, slow in ((qx, qy), (qy, qx)):
        # The number of points before the slow axis changes value gives the length of the fast axis.
        changes = np.flatnonzero(~np.isclose(slow, slow[0]))
        if changes.size == 0:
            continue
        n_fast = int(changes[0])
        if n_fast < 2 or qx.size % n_fast:
            continue
        shape = (qx.size // n_fast, n_fast)
        fast_grid, slow_grid = fast.reshape(shape), slow.reshape(shape)
        if np.allclose(fast_grid, fast_grid[0], equal_nan=True) and \
                np.allclose(slow_grid, slow_grid[:, :1], equal_nan=True):
            return shape
    return None


def reshape(values, shape: tuple[int, ...] | None):
    """Reshape values to the grid shape, if there is one and the values fit it."""
    values = np.asarray(values)
    if shape is not None and values.size == int(np.prod(shape)):
        return values.reshape(shape)
    return values
