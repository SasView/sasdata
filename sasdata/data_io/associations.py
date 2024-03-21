from typing import TYPE_CHECKING, ClassVar
from importlib import import_module
import sasdata.data_io.importers as importers

if TYPE_CHECKING:
    from sasdata.data_io.data_import import Registry
    from sasdata.data_io.importer import Importer

MODULES = {}


def import_associations(obj: Registry):
    obj.modules = find_importers()


def find_module(package_name: str, cls_name: str) -> dict[str, ClassVar]:
    return {cls_name: import_module(cls_name, package_name)}


def find_importers() -> dict[str, Importer]:
    for module in importers.__all__:
        try:
            loaded_mod = find_module('Importer', f'sasdata.data_io.importers.{module}')
            MODULES.update(loaded_mod)
        except ImportError:
            # Modules without an Importer object should throw this error
            pass
    return MODULES
