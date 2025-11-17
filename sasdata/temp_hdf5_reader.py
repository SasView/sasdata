import logging
from collections.abc import Callable

import h5py
import numpy as np
from h5py._hl.dataset import Dataset as HDF5Dataset
from h5py._hl.group import Group as HDF5Group

from sasdata.data import SasData
from sasdata.data_backing import Dataset as SASDataDataset
from sasdata.data_backing import Group as SASDataGroup
from sasdata.dataset_types import one_dim, two_dim, three_dim
from sasdata.metadata import (
    Aperture,
    BeamSize,
    Collimation,
    Detector,
    Instrument,
    Metadata,
    MetaNode,
    Process,
    Rot3,
    Sample,
    Source,
    Vec3,
)
from sasdata.quantities import units
from sasdata.quantities.quantity import NamedQuantity, Quantity
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


def connected_data(node: SASDataGroup, name_prefix="") -> dict[str, Quantity]:
    """In the context of NeXus files, load a group of data entries that are organised together
    match up the units and errors with their values"""
    # Gather together data with its error terms

    uncertainty_map = {}
    uncertainties = set()
    entries = {}

    for name in node.children:
        child = node.children[name]

        if "units" in child.attributes and child.attributes["units"]:
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

    output : dict[str, Quantity] = {}

    for name, entry in entries.items():
        if name not in uncertainties:
            if name in uncertainty_map:
                uncertainty = entries[uncertainty_map[name]]
                new_entry = entry.with_standard_error(uncertainty)
                output[name] = new_entry
            else:
                output[name] = entry

    return output

### Begin metadata parsing code

def parse_quantity(node : HDF5Group) -> Quantity[float]:
    """Pull a single quantity with length units out of an HDF5 node"""
    if node.shape == (): # scalar dataset
        magnitude =  node.astype(float)[()]
    else: # vector dataset
        magnitude =  node.astype(float)[0]
    unit = node.attrs["units"]
    return Quantity(magnitude, parse(unit))

def parse_string(node : HDF5Group) -> str:
    """Access string data from a node"""
    if node.shape == (): # scalar dataset
        return node.asstr()[()]
    else: # vector dataset
        return node.asstr()[0]

def opt_parse[T](node: HDF5Group, key: str, subparser: Callable[[HDF5Group], T]) -> T | None:
    """Parse a subnode if it is present"""
    if key in node:
        return subparser(node[key])
    return None

def attr_parse(node: HDF5Group, key: str) -> str | None:
    """Parse an attribute if it is present"""
    if key in node.attrs:
        return node.attrs[key]
    return None


def parse_apterture(node : HDF5Group) -> Aperture:
    distance = opt_parse(node, "distance", parse_quantity)
    name = attr_parse(node, "name")
    size = opt_parse(node, "size", parse_vec3)
    size_name = None
    type_ = attr_parse(node, "type")
    if size:
        size_name = attr_parse(node["size"], "name")
    else:
        size_name = None
    return Aperture(distance=distance, size=size, size_name=size_name, name=name, type_=type_)

def parse_beam_size(node : HDF5Group) -> BeamSize:
    name = attr_parse(node, "name")
    size = parse_vec3(node)
    return BeamSize(name=name, size=size)

def parse_source(node : HDF5Group) -> Source:
    radiation = opt_parse(node, "radiation", parse_string)
    beam_shape = opt_parse(node, "beam_shape", parse_string)
    beam_size = opt_parse(node, "beam_size", parse_beam_size)
    wavelength = opt_parse(node, "wavelength", parse_quantity)
    if wavelength is None:
        wavelength = opt_parse(node, "incident_wavelength", parse_quantity)
    wavelength_min = opt_parse(node, "wavelength_min", parse_quantity)
    wavelength_max = opt_parse(node, "wavelength_max", parse_quantity)
    wavelength_spread = opt_parse(node, "wavelength_spread", parse_quantity)
    if wavelength_spread is None:
        wavelength_spread = opt_parse(node, "incident_wavelength_spread", parse_quantity)
    return Source(
        radiation=radiation,
        beam_shape=beam_shape,
        beam_size=beam_size,
        wavelength=wavelength,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_spread=wavelength_spread,
    )

def parse_vec3(node : HDF5Group) -> Vec3:
    """Parse a measured 3-vector"""
    x = opt_parse(node, "x", parse_quantity)
    y = opt_parse(node, "y", parse_quantity)
    z = opt_parse(node, "z", parse_quantity)
    return Vec3(x=x, y=y, z=z)

def parse_rot3(node : HDF5Group) -> Rot3:
    """Parse a measured rotation"""
    roll = opt_parse(node, "roll", parse_quantity)
    pitch = opt_parse(node, "pitch", parse_quantity)
    yaw = opt_parse(node, "yaw", parse_quantity)
    return Rot3(roll=roll, pitch=pitch, yaw=yaw)

def parse_detector(node : HDF5Group) -> Detector:
    name = opt_parse(node, "name", parse_string)
    distance = opt_parse(node, "SDD", parse_quantity)
    offset = opt_parse(node, "offset", parse_vec3)
    orientation = opt_parse(node, "orientation", parse_rot3)
    beam_center = opt_parse(node, "beam_center", parse_vec3)
    pixel_size = opt_parse(node, "pixel_size", parse_vec3)
    slit_length = opt_parse(node, "slit_length", parse_quantity)

    return Detector(name=name,
                    distance=distance,
                    offset=offset,
                    orientation=orientation,
                    beam_center=beam_center,
                    pixel_size=pixel_size,
                    slit_length=slit_length)



def parse_collimation(node : HDF5Group) -> Collimation:
    length = opt_parse(node, "length", parse_quantity)
    return Collimation(length=length, apertures=[parse_apterture(node[ap])
                                                 for ap in node if "aperture" in ap])


def parse_instrument(node : HDF5Group) -> Instrument:
    sassourcekey = next(k for k in node.keys() if k.lower() == "sassource")
    return Instrument(
        collimations= [parse_collimation(node[x]) for x in node if "collimation" in x],
        detector=[parse_detector(node[d]) for d in node if "detector" in d],
        source = parse_source(node[sassourcekey]),
    )

def parse_sample(node : HDF5Group) -> Sample:
    name = attr_parse(node, "name")
    sample_id = opt_parse(node, "ID", parse_string)
    thickness = opt_parse(node, "thickness", parse_quantity)
    try: # vector
        transmission = opt_parse(node, "transmission", lambda n: float(n[0].astype(str)))
    except ValueError: # scalar
        transmission = opt_parse(node, "transmission", lambda n: float(n[()].astype(str)))
    temperature = opt_parse(node, "temperature", parse_quantity)
    position = opt_parse(node, "position", parse_vec3)
    orientation = opt_parse(node, "orientation", parse_rot3)
    details : list[str] = sum([list(node[d].asstr()[()]) for d in node if "details" in d], [])
    return Sample(name=name,
                  sample_id=sample_id,
                  thickness=thickness,
                  transmission=transmission,
                  temperature=temperature,
                  position=position,
                  orientation=orientation,
                  details=details)

def parse_term(node : HDF5Group) -> tuple[str, str | Quantity[float]] | None:
    name = attr_parse(node, "name")
    unit = attr_parse(node, "unit")
    value = attr_parse(node, "value")
    if name is None or value is None:
        return None
    if unit and unit.strip():
        return (name, Quantity(float(value), units.symbol_lookup[unit]))
    return (name, value)


def parse_process(node : HDF5Group) -> Process:
    name = opt_parse(node, "name", parse_string)
    date = opt_parse(node, "date", parse_string)
    description = opt_parse(node, "description", parse_string)
    term_values = [parse_term(node[n]) for n in node if "term" in n]
    terms = {tup[0]: tup[1] for tup in term_values if tup is not None}
    notes = [parse_string(node[n]) for n in node if "note" in n]
    return Process(name=name, date=date, description=description, terms=terms, notes=notes)

def load_raw(node: HDF5Group | HDF5Dataset) -> MetaNode:
    name = node.name.split("/")[-1]
    match node:
        case HDF5Group():
            attrib = {a: node.attrs[a] for a in node.attrs}
            contents = [load_raw(node[v]) for v in node]
            return MetaNode(name=name, attrs=attrib, contents=contents)
        case HDF5Dataset(dtype=dt):
            attrib = {a: node.attrs[a] for a in node.attrs}
            if (str(dt).startswith("|S")):
                if "units" in attrib:
                    contents = Quantity(float(node.asstr()[0]), parse(attrib["units"]))
                else:
                    contents = node.asstr()[0]
            else:
                if "units" in attrib and attrib["units"]:
                    contents = Quantity(node[()], parse(attrib["units"]))
                else:
                    contents = node[()]
            return MetaNode(name=name, attrs=attrib, contents=contents)
        case _:
            raise RuntimeError(f"Cannot load raw data of type {type(node)}")

def parse_metadata(node : HDF5Group) -> Metadata:
    instrument = opt_parse(node, "SASinstrument", parse_instrument)
    sample = opt_parse(node, "SASsample", parse_sample)
    process = [parse_process(node[p]) for p in node if "SASprocess" in p]
    title = opt_parse(node, "title", parse_string)
    run = [parse_string(node[r]) for r in node if "run" in r]
    definition = opt_parse(node, "definition", parse_string)
    raw =  load_raw(node)
    return Metadata(process=process,
                    instrument=instrument,
                    sample=sample,
                    title=title,
                    run=run,
                    raw=raw,
                    definition=definition)

### End Metadata parsing code


def load_data(filename: str) -> dict[str, SasData]:
    with h5py.File(filename, "r") as f:
        loaded_data: dict[str, SasData] = {}

        for root_key in f.keys():
            entry = f[root_key]

            data_contents : dict[str, Quantity] = {}

            entry_keys = entry

            if not [k for k in entry if k.lower().startswith("sasdata") or k.lower().startswith("data")]:
                logger.warning("No sasdata or data key")
                logger.warning(f"Known keys: {[k for k in entry_keys]}")

            for key in entry_keys:
                component = entry[key]
                lower_key = key.lower()
                if lower_key.startswith("sasdata") or lower_key.startswith("data"):
                    datum = recurse_hdf5(component)
                    data_contents = connected_data(datum, str(filename))

            metadata = parse_metadata(f[root_key])

            if "Qz" in data_contents:
                dataset_type = three_dim
            elif "Qy" in data_contents:
                dataset_type = two_dim
            else:
                dataset_type = one_dim

            entry_key = entry.attrs["sasview_key"] if "sasview_key" in entry.attrs else root_key

            loaded_data[entry_key] = SasData(
                    name=root_key,
                    dataset_type=dataset_type,
                    data_contents=data_contents,
                    metadata=metadata,
                    verbose=False,
                )

        return loaded_data


if __name__ == "__main__":
    data = load_data(test_file)

    for dataset in data.values():
        print(dataset.summary())