import os
import logging
import sys
import time
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
from types import ModuleType
from typing import Union

from sasdata.data_util.registry import ExtensionRegistry

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
    Registry class for file format extensions.
    Readers and writers are supported.
    """
    def __init__(self):
        self.as_super = super(Registry, self)
        self.as_super.__init__()

        # List of wildcards
        self.wildcards = ['All (*.*)|*.*']

        # Creation time, for testing
        self._created = time.time()

        # List of modules this registry object can interact with
        self.plugins = defaultdict(list)

    def find_plugins(self, path: Union[Path, str], ) -> int:
        """Find plugin modules in the given directory. This method can be used to inspect user plugin directories.
        :param path: directory to search into
        :return: number of readers found
        """
        readers_found = 0
        temp_path = os.path.abspath(path)
        if not os.path.isdir(temp_path):
            temp_path = os.path.join(os.getcwd(), path)
        if not os.path.isdir(temp_path):
            temp_path = os.path.join(os.path.dirname(__file__), path)
        if not os.path.isdir(temp_path):
            temp_path = os.path.join(os.path.dirname(sys.path[0]), path)

        path = temp_path
        # Check whether the directory exists
        if not os.path.isdir(path):
            msg = f"DataLoader could nt locate plugin folder. {path} does not exist"
            logger.warning(msg)
            return readers_found

        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isfile(full_path):

                # Process python files
                if item.endswith('.py'):
                    toks = os.path.splitext(os.path.basename(item))
                    try:
                        sys.path.insert(0, os.path.abspath(path))
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
                if ext not in self.modules:
                    self.modules[ext] = []
                # Append the new reader to the list
                self.modules[ext].append(loader.read)

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
            if file_extension not in self.modules:
                self.modules[file_extension] = []
            # Append the new reader to the list
            self.modules[file_extension].append(reader.read)

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

    def _identify_plugin(self, module: ModuleType, attr: list[str]) -> bool:
        """
        Look into a module to find whether it contains a
        Reader class. If so, add it to readers and (potentially)
        to the list of writers.
        :param module: module object
        :param attr: A list of
        :returns: True if successful
        """
        reader_found = False

        if hasattr(module, "Reader"):
            try:
                # Find supported extensions
                reader = module.Reader()
                for ext in reader.ext:
                    if ext not in self.modules:
                        self.modules[ext] = []
                    # When finding a reader at run time,
                    # treat this reader as the new default
                    self.modules[ext].insert(0, reader.read)

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
