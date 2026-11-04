import os
import pathlib

from sasdata.data import SasData
from sasdata.data_io.importer import Importer
from test.test_files import data, json, mumag, reference, sesans_data


def local_load(path: str) -> list[SasData]:
    """Load a local file, letting the Importer infer the file type from its contents"""
    base = os.path.join(os.path.dirname(__file__), path)
    for extension in ("h5", "xml", "txt"):
        if os.path.exists(f"{base}.{extension}"):
            return list(Importer.load(f"{base}.{extension}").values())


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
