import logging
import time
from collections import defaultdict
from types import ModuleType
from typing import Union
from abc import abstractmethod

from sasdata.data_io.registry import ExtensionRegistry, FILE_LIKE

logger = logging.getLogger(__name__)

# TODO: The following tasks are generic to the entire project. Much of this initial commit will be rewritten.

# TODO: Importers and Exporters should be separated from one another
#  sasdata.data_io.importers - holds former 'Reader' classes
#  sasdata.data_io.exporters - holds similar objects but with only the 'write' methods

# TODO: Replace all read/write nomenclature with import/export and use generic language for in Registry classes

# TODO: Find all plugins in a directory and associate using ext class attribute
#  This should (hopefully) simplify the plugin finders

# TODO: Data objects should hold [File|String|Binary]IO object
#  Similarly, Import.import_data() should accept these objects in addition to what is currently allowed

# TODO: Replace all Data1D and Data2D references with Data once Data object exists


class Registry(ExtensionRegistry):
    """
    Registry class to track and handle modules that can perform a specific set of actions.
    """
    cls_name = ''
    fnc = ''

    def __init__(self, files: Union):
        """
        Creates an instance of the Registry class, using the given function name
        :param files: A function name to use as a basis for registering extensions
        """
        self.as_super = super(Registry, self)
        self.as_super.__init__()

        # Writers
        self.writers = defaultdict(list)

        # List of wildcards
        self.wildcards = ['All (*.*)|*.*']

        # Creation time, for testing
        self._created = time.time()

        # The function modules will need to have
        # Defaults to the read method if no value is supplied
        self.cls_name = self.cls_name if self.cls_name else 'Reader'
        self.fnc = self.fnc if self.fnc else 'read'

    def find_plugins(self, dir: str):
        """
        Find readers in a given directory. This method
        can be used to inspect user plug-in directories to
        find new readers/writers.
        :param dir: directory to search into
        :return: number of readers found
        """
        pass

    def associate_file_type(self, ext: str, module: ModuleType) -> bool:
        """
        Look into a module to find whether it contains a
        Reader class. If so, APPEND it to readers and (potentially)
        to the list of writers for the given extension
        :param ext: file extension [string]
        :param module: module object
        """
        pass

    def associate_plugin(self, file_extension: str, module: ModuleType) -> None:
        """
        Append a module to the list of plugins
        :param file_extension: file extension [string]
        :param reader: reader object
        """
        if self._is_plugin_module(module):
            self.associate_file_type(file_extension, module)

    @abstractmethod
    def _is_plugin_module(self, module: ModuleType) -> bool:
        """
        Look in a module and return whether it contains the expected class or not.
        :param module: module object
        :returns: Is the class in the module?
        """
        raise NotImplementedError(f"The _identify_plugin method is required for the class {self.__class__.__name__}.")

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Enable direct calling of the class to allow transient instances."""
        raise NotImplementedError(f"The __call__ method is required in the class {self.__class__.__name__}.")
