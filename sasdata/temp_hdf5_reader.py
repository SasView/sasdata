import os
import h5py


import logging

import numpy as np


from h5py._hl.dataset import Dataset as HDF5Dataset
from h5py._hl.group import Group as HDF5Group


from sasdata.raw_form import RawData
from sasdata.raw_form import Dataset as SASDataDataset, Group as SASDataGroup

from quantities.quantity import NamedQuantity
from quantities import units

test_file = "./example_data/1d_data/33837rear_1D_1.75_16.5_NXcanSAS_v3.h5"
# test_file = "./example_data/1d_data/33837rear_1D_1.75_16.5_NXcanSAS.h5"

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
                name=hdf5_entry.name,
                data=data,
                attributes=attributes)

        else:
            data = np.array(hdf5_entry, dtype=hdf5_entry.dtype)

            return SASDataDataset[np.ndarray](
                name=hdf5_entry.name,
                data=data,
                attributes=attributes)

    elif isinstance(hdf5_entry, HDF5Group):
        return SASDataGroup(
            name=hdf5_entry.name,
            children={key: recurse_hdf5(hdf5_entry[key]) for key in hdf5_entry.keys()})

    else:
        raise TypeError(f"Unknown type found during HDF5 parsing: {type(hdf5_entry)} ({hdf5_entry})")

def parse_units_placeholder(string: str) -> units.Unit:
    #TODO: Remove when not needed
    return units.meters

def connected_data(node: SASDataGroup, name_prefix="") -> list[NamedQuantity]:
    """ In the context of NeXus files, load a group of data entries that are organised together
    match up the units and errors with their values"""
    # Gather together data with its error terms

    uncertainty_map = {}
    uncertainties = set()
    entries = {}

    for name in node.children:

        child = node.children[name]
        # TODO: Actual unit parser here
        units = parse_units_placeholder(child.attributes["units"])

        quantity = NamedQuantity(name=name_prefix+child.name,
                                 value=child.data,
                                 units=units)

        if "uncertainty" in child.attributes:
            uncertainty_name = child.attributes["uncertainty"]
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


def load_data(filename) -> list[RawData]:
    with h5py.File(filename, 'r') as f:

        loaded_data: list[RawData] = []

        for root_key in f.keys():

            entry = f[root_key]

            data_contents = []
            raw_metadata = {}

            entry_keys = [key for key in entry.keys()]

            if "sasdata" not in entry_keys:
                logger.warning("No sasdata key")

            for key in entry_keys:
                component = entry[key]
                if key.lower() == "sasdata":
                    datum = recurse_hdf5(component)
                    # TODO: Use named identifier
                    data_contents = connected_data(datum, "FILE_ID_HERE")

                else:
                    raw_metadata[key] = recurse_hdf5(component)


            loaded_data.append(
                RawData(
                    name=root_key,
                    data_contents=data_contents,
                    raw_metadata=raw_metadata))

        return loaded_data




data = load_data(test_file)

for dataset in data:
    print(dataset.summary())