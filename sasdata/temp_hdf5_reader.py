import os
import h5py


import logging

import numpy as np


from h5py._hl.dataset import Dataset as HDF5Dataset
from h5py._hl.group import Group as HDF5Group


from sasdata.data import SasData
from sasdata.data_backing import Dataset as SASDataDataset, Group as SASDataGroup
from sasdata.metadata import Instrument, Collimation, Aperture, Source, BeamSize, Detector
from sasdata.quantities.accessors import AccessorTarget

from sasdata.quantities.quantity import NamedQuantity, Quantity
from sasdata.quantities import units
from sasdata.quantities.unit_parser import parse

# test_file = "./example_data/1d_data/33837rear_1D_1.75_16.5_NXcanSAS_v3.h5"
# test_file = "./example_data/1d_data/33837rear_1D_1.75_16.5_NXcanSAS.h5"
test_file = "./example_data/2d_data/BAM_2D.h5"
# test_file = "./example_data/2d_data/14250_2D_NoDetInfo_NXcanSAS_v3.h5"
# test_file = "./example_data/2d_data/33837rear_2D_1.75_16.5_NXcanSAS_v3.h5"

logger = logging.getLogger(__name__)


def recurse_hdf5(hdf5_entry):
    if isinstance(hdf5_entry, HDF5Dataset):
        #
        # print(hdf5_entry.dtype)
        # print(type(hdf5_entry.dtype))

        attributes = {name: hdf5_entry.attrs[name] for name in hdf5_entry.attrs}

        if isinstance(hdf5_entry.dtype, np.dtypes.BytesDType):
            data = hdf5_entry[()][0].decode("utf-8")

            return SASDataDataset[str](
                name=hdf5_entry.name, data=data, attributes=attributes
            )

        else:
            data = np.array(hdf5_entry, dtype=hdf5_entry.dtype)

            return SASDataDataset[np.ndarray](
                name=hdf5_entry.name, data=data, attributes=attributes
            )

    elif isinstance(hdf5_entry, HDF5Group):
        return SASDataGroup(
            name=hdf5_entry.name,
            children={key: recurse_hdf5(hdf5_entry[key]) for key in hdf5_entry.keys()},
        )

    else:
        raise TypeError(
            f"Unknown type found during HDF5 parsing: {type(hdf5_entry)} ({hdf5_entry})"
        )


GET_UNITS_FROM_ELSEWHERE = units.meters


def connected_data(node: SASDataGroup, name_prefix="") -> list[NamedQuantity]:
    """In the context of NeXus files, load a group of data entries that are organised together
    match up the units and errors with their values"""
    # Gather together data with its error terms

    uncertainty_map = {}
    uncertainties = set()
    entries = {}

    for name in node.children:
        child = node.children[name]

        if "units" in child.attributes:
            units = parse(child.attributes["units"])
        else:
            units = GET_UNITS_FROM_ELSEWHERE

        quantity = NamedQuantity(
            name=name_prefix + child.name, value=child.data, units=units
        )

        # Turns out people can't be trusted to use the same keys here
        if "uncertainty" in child.attributes or "uncertainties" in child.attributes:
            try:
                uncertainty_name = child.attributes["uncertainty"]
            except:
                uncertainty_name = child.attributes["uncertainties"]
            uncertainty_map[name] = uncertainty_name
            uncertainties.add(uncertainty_name)

        entries[name] = quantity

    output = []

    for name, entry in entries.items():
        if name not in uncertainties:
            if name in uncertainty_map:
                uncertainty = entries[uncertainty_map[name]]
                new_entry = entry.with_standard_error(uncertainty)
                output.append(new_entry)
            else:
                output.append(entry)

    return output


def parse_apertures(node) -> list[Aperture]:
    result = []
    aps = [a for a in node if "aperture" in a]
    for ap in aps:
        distance = None
        size = None
        if "distance" in node[ap]:
            distance = node[ap]["distance"]
        if "size" in node[ap]:
            x = y = z = None
            if "x" in node[ap]:
                x = node[ap]["size"]["x"]
            if "y" in node[ap]:
                y = node[ap]["size"]["y"]
            if "z" in node[ap]:
                z = node[ap]["size"]["z"]
            if x is not None or y is not None or z is not None:
                size = (x, y, z)
        result.append(Aperture(distance=distance, size=size, size_name=size_name, name=name, apType=apType))
    return result

def parse_beam_size(node) -> BeamSize:
    name = None
    x = None
    y = None
    z = None
    if "name" in node.attrs:
        name = node.atrs["keys"]
    if "x" in node:
        x = node["x"]
    if "y" in node:
        y = node["y"]
    if "z" in node:
        z = node["z"]
    return BeamSize(name=name, x=x, y=y, z=z)


def parse_source(node) -> Source:
    beam_shape = None
    beam_size = None
    wavelength = None
    wavelength_min = None
    wavelength_max = None
    wavelength_spread = None
    if "beam_shape" in node:
        beam_shape = node["beam_shape"]
    if "wavelength" in node:
        wavelength = node["wavelength"]
    if "wavelength_min" in node:
        wavelength = node["wavelength_min"]
    if "wavelength_max" in node:
        wavelength = node["wavelength_max"]
    if "wavelength_spread" in node:
        wavelength = node["wavelength_spread"]
    if "beam_size" in node:
        beam_size = parse_beam_size(node["beam_size"])
    return Source(
        radiation=node["radiation"].asstr()[0],
        beam_shape=beam_shape,
        beam_size=beam_size,
        wavelength=wavelength,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_spread=wavelength_spread,
    )

def parse_detector(node) -> Detector:
    name = None
    if "name" in node:
        name = node["name"].asstr()[0]
    distance = None
    if "SDD" in node:
        magnitude = node["SDD"].astype(float)[0]
        unit = node["SDD"].attrs["units"]
        distance = Quantity(magnitude, units.symbol_lookup[unit])
    offset = None
    orientation = None
    beam_center = None
    pixel_size = None
    slit_length = None

    return Detector(name=name, distance=distance, offset=offset, orientation=orientation, beam_center=beam_center, pixel_size=pixel_size, slit_length=slit_length)



def parse_collimation(node) -> Collimation:
    if "length" in node:
        length = node["length"]
    else:
        length = None
    return Collimation(length=length, apertures=parse_apertures(node))


def parse_instrument(raw, node) -> Instrument:
    collimations = [
        parse_collimation(node[x])
        for x in node
        if "collimation" in x
    ]
    return Instrument(
        raw,
        collimations=collimations,
        detector=[parse_detector(node[d]) for d in node if "detector" in d],
        source=parse_source(node["sassource"]),
    )


def load_data(filename) -> list[SasData]:
    with h5py.File(filename, "r") as f:
        loaded_data: list[SasData] = []

        for root_key in f.keys():
            entry = f[root_key]

            data_contents = []
            raw_metadata = {}

            entry_keys = [key for key in entry.keys()]

            if "sasdata" not in entry_keys and "data" not in entry_keys:
                logger.warning("No sasdata or data key")

            for key in entry_keys:
                component = entry[key]
                lower_key = key.lower()
                if lower_key == "sasdata" or lower_key == "data":
                    datum = recurse_hdf5(component)
                    # TODO: Use named identifier
                    data_contents = connected_data(datum, "FILE_ID_HERE")

                else:
                    raw_metadata[key] = recurse_hdf5(component)

            instrument = None
            if "sasinstrument" in f["sasentry01"]:
                instrument = parse_instrument(
                    AccessorTarget(SASDataGroup("root", raw_metadata)).with_path_prefix(
                        "sasinstrument|instrument"
                    ),
                    f["sasentry01"]["sasinstrument"],
                )

            loaded_data.append(
                SasData(
                    name=root_key,
                    data_contents=data_contents,
                    raw_metadata=SASDataGroup("root", raw_metadata),
                    instrument=instrument,
                    verbose=False,
                )
            )

        return loaded_data


if __name__ == "__main__":
    data = load_data(test_file)

    for dataset in data:
        print(dataset.summary(include_raw=False))
