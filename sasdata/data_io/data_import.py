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
from pathlib import Path
from itertools import zip_longest

from sasdata.dataloader.data_info import Data1D, Data2D
from sasdata.data_io.io_base import Registry

logger = logging.getLogger(__name__)


class Import(Registry):

    def associate_importer(self, ext: str, module: ModuleType) -> bool:
        """
        Append a reader object to readers
        :param ext: file extension [string]
        :param module: reader object
        """
        return self.associate_file_reader(ext, module)

    def import_data(self, file_path_list: Union[List[Union[str, Path]], str, Path],
             ext: Optional[Union[List[str], str]] = None,
             debug: Optional[bool] = False,
             use_defaults: Optional[bool] = True):
        """
        Call the loader for the file type of path.

        :param file_path_list: A list of pathlib.Path objects and/or string representations of file paths
        :param ext: A list of explicit extensions, to force the use of a particular reader for a particular file.
                    **Usage** If any ext is passed, the length of the ext list should be the same as the length of
                    the file path list. A single extention, as a string or a list of length 1, will apply  ext to all
                    files in the file path list. Any other case will result in an error.
        :param debug: when True, print the traceback for each loader that fails
        :param use_defaults:
            Flag to use the default readers as a backup if the
            main reader fails or no reader exists

        Defaults to the ascii (multi-column), cansas XML, and cansas NeXuS
        readers if no reader was registered for the file's extension.
        """
        # Coerce file path list and ext to lists
        file_path_list = [file_path_list] if isinstance(file_path_list, (str, Path)) else file_path_list
        ext = [ext] if isinstance(ext, str) else ext
        # Ensure ext has at least 1 value in it to ensure zip_longest has a value for the fillvalue
        if not ext:
            ext = [None]
        if len(ext) > 1 and len(ext) != len(file_path_list):
            raise IndexError(f"The file extensions, {ext}, and file paths, {file_path_list} are not the same length. ")
        output = []
        # Use zip_longest for times where no ext or a single ext is passed
        # Note: load() returns a list, so list comprehension would create a list of lists, without multiple loops
        for file_path, ext_n in zip_longest(file_path_list, ext, fillvalue=ext[0]):
            output.extend(self.as_super.load(file_path, ext=ext_n))
        return output

    def __call__(self, file_path_list: List[str]) -> List[Union[Data1D, Data2D]]:
        """Allow direct calls to the loader system for transient file loader systems.
        :param file_path_list: A list of string representations of file paths. Each item can either be a local file path
            or a URI.
        :return: A list of loaded Data1D/2D objects.
        """
        return self.import_data(file_path_list)
