from typing import TYPE_CHECKING, ClassVar
from importlib import import_module
import sasdata.data_io.importers as importers


if TYPE_CHECKING:
    from sasdata.data_io.data_import import Registry
    from sasdata.data_io.importer import Importer

MODULES = {}


def import_associations(obj: Registry):
    obj.modules = find_importers()


def find_modules(package_name: str, cls_name: str) -> dict(str, ClassVar):
    return {"Yo!": import_module(cls_name, package_name)}


def find_importers() -> dict(str, Importer):
    for module in importers.__all__:
        try:
            import_module('Importer', f'sasdata.data_io.importers.{module}')
            MODULES[Importer.__name__] = module
        except ImportError:
            # Modules without an Importer object should throw this error
            pass
    return {}
