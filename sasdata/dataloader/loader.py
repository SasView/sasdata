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
#####################################################################
# This software was developed by the University of Tennessee as part of the
# Distributed Data Analysis of Neutron Scattering Experiments (DANSE)
# project funded by the US National Science Foundation.
# See the license text in license.txt
# copyright 2008, University of Tennessee
######################################################################

import logging
import os
import sys
import time
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

from sasdata.data_util.registry import ExtensionRegistry
from sasdata.data_util.util import unique_preserve_order
from sasdata.dataloader.data_info import Data1D, Data2D

# Default readers are defined in the readers sub-module
from . import readers

logger = logging.getLogger(__name__)


class Registry(ExtensionRegistry):
    """
    Registry class for file format extensions.
    Readers and writers are supported.
    """
    def __init__(self):
        self.as_super = super(Registry, self)
        self.as_super.__init__()

        # Writers
        self.writers = defaultdict(list)

        # List of wildcards
        self.wildcards = ['All (*.*)|*.*']

        # Creation time, for testing
        self._created = time.time()

        # Register default readers
        readers.read_associations(self)

    def load(self, file_path_list: list[str | Path] | str | Path,
             ext: list[str] | str | None = None,
             debug: bool | None = False,
             use_defaults: bool | None = True):
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
        file_path_list = [file_path_list] if isinstance(file_path_list, (str | Path)) else file_path_list
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

    def find_plugins(self, dir: str):
        """
        Find readers in a given directory. This method
        can be used to inspect user plug-in directories to
        find new readers/writers.
        :param dir: directory to search into
        :return: number of readers found
        """
        readers_found = 0
        temp_path = os.path.abspath(dir)
        if not os.path.isdir(temp_path):
            temp_path = os.path.join(os.getcwd(), dir)
        if not os.path.isdir(temp_path):
            temp_path = os.path.join(os.path.dirname(__file__), dir)
        if not os.path.isdir(temp_path):
            temp_path = os.path.join(os.path.dirname(sys.path[0]), dir)

        dir = temp_path
        # Check whether the directory exists
        if not os.path.isdir(dir):
            msg = f"DataLoader could nt locate plugin folder. {dir} does not exist"
            logger.warning(msg)
            return readers_found

        for item in os.listdir(dir):
            full_path = os.path.join(dir, item)
            if os.path.isfile(full_path):

                # Process python files
                if item.endswith('.py'):
                    toks = os.path.splitext(os.path.basename(item))
                    try:
                        sys.path.insert(0, os.path.abspath(dir))
                        module = __import__(toks[0], globals(), locals())
                        if self._identify_plugin(module):
                            readers_found += 1
                    except Exception as exc:
                        msg = f"Loader: Error importing {item}\n  {str(exc)}"
                        logger.error(msg)

                # Process zip files
                elif item.endswith('.zip'):
                    try:
                        # Find the modules in the zip file
                        zfile = ZipFile(item)
                        nlist = zfile.namelist()

                        sys.path.insert(0, item)
                        for mfile in nlist:
                            try:
                                # Change OS path to python path
                                fullname = mfile.replace('/', '.')
                                fullname = os.path.splitext(fullname)[0]
                                module = __import__(fullname, globals(), locals(), [""])
                                if self._identify_plugin(module):
                                    readers_found += 1
                            except Exception as exc:
                                msg = f"Loader: Error importing {mfile}\n  {str(exc)}"
                                logger.error(msg)

                    except Exception as exc:
                        msg = f"Loader: Error importing  {item}\n  {str(exc)}"
                        logger.error(msg)

        return readers_found

    def associate_file_type(self, ext: str, module: ModuleType) -> bool:
        """
        Look into a module to find whether it contains a
        Reader class. If so, APPEND it to readers and (potentially)
        to the list of writers for the given extension
        :param ext: file extension [string]
        :param module: module object
        """
        reader_found = False

        if hasattr(module, "Reader"):
            try:
                # Find supported extensions
                loader = module.Reader()
                if ext not in self.readers:
                    self.readers[ext] = []
                # Append the new reader to the list
                self.readers[ext].append(loader.read)

                reader_found = True

                # Keep track of wildcards
                type_name = module.__name__
                if hasattr(loader, 'type_name'):
                    type_name = loader.type_name

                wcard = f"{type_name} files (*{ext.lower()})|*{ext.lower()}"
                if wcard not in self.wildcards:
                    self.wildcards.append(wcard)

                # Check whether writing is supported
                if hasattr(loader, 'write'):
                    if ext not in self.writers:
                        self.writers[ext] = []
                    # Append the new writer to the list
                    self.writers[ext].append(loader.write)

            except Exception as exc:
                msg = f"Loader: Error accessing  Reader in {module.__name__}\n {str(exc)}"
                logger.error(msg)
        return reader_found

    def associate_file_reader(self, file_extension, reader):
        """
        Append a reader object to readers
        :param file_extension: file extension [string]
        :param reader: reader object
        """
        reader_found = False

        try:
            # Find supported extensions
            if file_extension not in self.readers:
                self.readers[file_extension] = []
            # Append the new reader to the list
            self.readers[file_extension].append(reader.read)

            reader_found = True

            # Keep track of wildcards
            if hasattr(reader, 'type_name'):
                type_name = reader.type_name

                wcard = f"{type_name} files (*{file_extension.lower()})|*{file_extension.lower()}"
                if wcard not in self.wildcards:
                    self.wildcards.append(wcard)

        except Exception as exc:
            msg = f"Loader: Error accessing Reader in {reader.__name__}\n  {str(exc)}"
            logger.error(msg)
        return reader_found

    def _identify_plugin(self, module: ModuleType):
        """
        Look into a module to find whether it contains a
        Reader class. If so, add it to readers and (potentially)
        to the list of writers.
        :param module: module object
        :returns: True if successful
        """
        reader_found = False

        if hasattr(module, "Reader"):
            try:
                # Find supported extensions
                reader = module.Reader()
                for ext in reader.ext:
                    if ext not in self.readers:
                        self.readers[ext] = []
                    # When finding a reader at run time,
                    # treat this reader as the new default
                    self.readers[ext].insert(0, reader.read)

                    reader_found = True

                    # Keep track of wildcards
                    file_description = reader.type_name if hasattr(reader, 'type_name') else module.__name__
                    wcard = f"{file_description} files (*{ext.lower()})|*{ext.lower()}"
                    if wcard not in self.wildcards:
                        self.wildcards.append(wcard)

                # Check whether writing is supported
                if hasattr(reader, 'write'):
                    for ext in reader.ext:
                        if ext not in self.writers:
                            self.writers[ext] = []
                        self.writers[ext].insert(0, reader.write)

            except Exception as exc:
                msg = f"Loader: Error accessing Reader in {module.__name__}\n {str(exc)}"
                logger.error(msg)
        return reader_found

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

    def save(self, path: str, data, format: str | None = None):
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


class Loader:
    """
    Utility class to use the Registry as a singleton.
    """
    __registry = Registry()

    def associate_file_type(self, ext: str, module: ModuleType) -> bool:
        """
        Look into a module to find whether it contains a
        Reader class. If so, append it to readers and (potentially)
        to the list of writers for the given extension
        :param ext: file extension [string]
        :param module: module object
        """
        return self.__registry.associate_file_type(ext, module)

    def associate_file_reader(self, ext: str, loader) -> bool:
        """
        Append a reader object to readers
        :param ext: file extension [string]
        :param module: reader object
        """
        return self.__registry.associate_file_reader(ext, loader)

    def load(self, file_path_list: list[str | Path] | str | Path,
             format: list[str] | str | None = None
             ) -> list[Data1D | Data2D]:
        """
        Load a file or series of files
        :param file_path_list: String representations of any number of file paths. This can either be a list or a string
        :param format: specified format to use (optional)
        :return: a list of DataInfo objects and/or loading exceptions.
        """
        return self.__registry.load(file_path_list, format)

    def save(self, file: str, data, format: str) -> bool:
        """
        Save a DataInfo object to file
        :param file: file name (path)
        :param data: DataInfo object
        :param format: format to write the data in
        """
        return self.__registry.save(file, data, format)

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

    def __call__(self, file_path_list: list[str]) -> list[Data1D | Data2D]:
        """Allow direct calls to the loader system for transient file loader systems.
        :param file_path_list: A list of string representations of file paths. Each item can either be a local file path
            or a URI.
        :return: A list of loaded Data1D/2D objects.
        """
        return self.load(file_path_list)