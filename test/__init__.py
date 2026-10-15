import os
import pathlib

from sasdata.data import SasData
from sasdata.data_io.importers.import_ascii import load_data_default_params
from sasdata.data_io.importers.import_hdf5 import load_data as hdf_load_data
from sasdata.data_io.importers.import_xml import load_data as xml_load_data
from test.test_files import data, json, mumag, reference, sesans_data


def local_load(path: str) -> SasData:
    """Get local file path"""
    base = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(f"{base}.h5"):
        return hdf_load_data(f"{base}.h5").values()
    if os.path.exists(f"{base}.xml"):
        return xml_load_data(f"{base}.xml").values()
    if os.path.exists(f"{base}.txt"):
        return load_data_default_params(f"{base}.txt")


def local_reference_finder(path: str):
    module_path = pathlib.Path(reference.__file__).resolve().parent
    return f"{os.path.join(module_path, path)}"


def local_data_finder(path: str):
    module_path = pathlib.Path(data.__file__).resolve().parent
    return f"{os.path.join(module_path, path)}"


def local_json_finder(path: str):
    module_path = pathlib.Path(json.__file__).resolve().parent
    return f"{os.path.join(module_path, path)}"


def local_sesans_finder(path: str):
    module_path = pathlib.Path(sesans_data.__file__).resolve().parent
    return f"{os.path.join(module_path, path)}"


def local_mumag_finder(path: str):
    module_path = pathlib.Path(mumag.__file__).resolve().parent
    return f"{os.path.join(module_path, path)}"
