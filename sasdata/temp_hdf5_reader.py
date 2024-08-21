import os
import h5py


import logging

import numpy as np


from h5py._hl.dataset import Dataset as HDF5Dataset
from h5py._hl.group import Group as HDF5Group


from sasdata.data import DataSet
from sasdata.raw_form import RawData
from sasdata.raw_form import Dataset as SASDataDataset, Group as SASDataGroup

test_file = "./example_data/1d_data/33837rear_1D_1.75_16.5_NXcanSAS_v3.h5"
test_file = "./example_data/1d_data/33837rear_1D_1.75_16.5_NXcanSAS.h5"

logger = logging.getLogger(__name__)

def hdf5_attr(entry):
    return entry

def recurse_hdf5(hdf5_entry):
    if isinstance(hdf5_entry, HDF5Dataset):
        #
        # print(hdf5_entry.dtype)
        # print(type(hdf5_entry.dtype))

        if isinstance(hdf5_entry.dtype, np.dtypes.BytesDType):
            data = hdf5_entry[()][0].decode("utf-8")
        else:
            data = np.array(hdf5_entry, dtype=hdf5_entry.dtype)

        attributes = {name: hdf5_attr(hdf5_entry.attrs[name]) for name in hdf5_entry.attrs}

        return SASDataDataset(
            name=hdf5_entry.name,
            data=data,
            attributes=attributes)

    elif isinstance(hdf5_entry, HDF5Group):
        return SASDataGroup(
            name=hdf5_entry.name,
            children={key: recurse_hdf5(hdf5_entry[key]) for key in hdf5_entry.keys()})

    else:
        raise TypeError(f"Unknown type found during HDF5 parsing: {type(hdf5_entry)} ({hdf5_entry})")

def load_data(filename) -> list[RawData]:
    with h5py.File(filename, 'r') as f:

        loaded_data: list[RawData] = []

        for root_key in f.keys():

            print(root_key)

            entry = f[root_key]

            data_contents = []
            raw_metadata = {}

            entry_keys = [key for key in entry.keys()]

            if "sasdata" not in entry_keys:
                logger.warning("")

            for key in entry_keys:
                component = entry[key]
                if key.lower() == "sasdata":
                    print("found sasdata, skipping for now")

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
    print(dataset)