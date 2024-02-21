"""
    File handler to support different file extensions.
    Uses reflectometer registry utility.
    The default readers are found in the 'readers' sub-module
    and registered by default at initialization time.
    To add a new default reader, one must register it in
    the register_readers method found in readers/__init__.py.
    A utility method (find_plugins) is available to inspect
    a directory (for instance, a user plug-in directory) and
    look for new readers/writers.
"""

import logging
from types import ModuleType
from typing import Optional, Union, List
from itertools import zip_longest

from sasdata.dataloader.data_info import Data1D, Data2D
from sasdata.data_io.io_base import Registry
from sasdata.data_util.util import unique_preserve_order

logger = logging.getLogger(__name__)


class Export(Registry):

    def associate_exporter(self, ext: str, module: ModuleType) -> bool:
        """
        Append a reader object to readers
        :param ext: file extension [string]
        :param module: reader object
        """
        return self.associate_file_reader(ext, module)

    def export_data(self, path: str, data, format: Optional[str] = None):
        """
        Call the writer for the file type of path.
        Raises ValueError if no writer is available.
        Raises KeyError if format is not available.
        May raise a writer-defined exception if writer fails.
        """
        if format is None:
            writers = self.lookup_writers(path)
        else:
            writers = self.writers[format]
        for writing_function in writers:
            try:
                return writing_function(path, data)
            except Exception as exc:
                msg = f"Saving file {path} using the {type(writing_function).__name__} writer failed.\n {str(exc)}"
                logger.exception(msg)  # give other loaders a chance to succeed

    def lookup_writers(self, path):
        """
        :return: the loader associated with the file type of path.
        :Raises ValueError: if file type is not known.
        """
        # Find matching extensions
        extlist = [ext for ext in self.extensions() if path.endswith(ext)]

        # Sort matching extensions by decreasing order of length
        extlist.sort(key=len)

        # Combine loaders for matching extensions into one big list
        writers = [writer for ext in extlist for writer in self.writers[ext]]
        # Remove duplicates if they exist
        writers = unique_preserve_order(writers)
        # Raise an error if there are no matching extensions
        if len(writers) == 0:
            raise ValueError("Unknown file type for " + path)
        # All done
        return writers

    def _get_registry_creation_time(self) -> float:
        """
        Internal method used to test the uniqueness
        of the registry object
        """
        return self.__registry._created

    def find_plugins(self, directory: str) -> int:
        """
        Find plugins in a given directory
        :param directory: directory to look into to find new readers/writers
        """
        return self.__registry.find_plugins(directory)

    def get_wildcards(self):
        """
        Return the list of wildcards
        """
        return self.__registry.wildcards

    def __call__(self, file_path_list: List[str], data_list: List[Union[Data1D, Data2D]],
                 ext: Union[list[str], str]) -> list[bool]:
        """Allow direct calls to the export system for transient file export systems.
        :param file_path_list: A list file path type object. Each item can either be a local file path or a URI.
        :return: A list of loaded Data1D/2D objects.
        """
        success = []
        for f_path, d_path, ext in zip_longest(file_path_list, data_list, ext, fillvalue=ext[0]):
            success.append(self.export_data(f_path, d_path, ext))
        return success
