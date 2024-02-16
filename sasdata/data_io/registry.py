"""
File extension registry.

This provides routines for opening files based on extension, and registers the built-in file extensions.
"""
import os
from typing import Any, Optional, List, Union
from collections import defaultdict
from pathlib import Path
import logging
from io import BytesIO, StringIO, FileIO

from sasdata.data_util.loader_exceptions import NoKnownLoaderException
from sasdata.data_util.util import unique_preserve_order
from sasdata.data_io.open import CustomFileOpen

# Class-specific types
PATH_LIKE = str, Path, os.path
FILE_LIKE = PATH_LIKE, BytesIO, StringIO, FileIO
OPEN_LIKE = Union[list[FILE_LIKE], FILE_LIKE]

DEPRECATION_MESSAGE = ("\rSupport for module {}, associated with the file extension {} has been removed. An attempt to "
                       "load the file {} was made. Should it be successful, the results cannot be guaranteed.")

logger = logging.getLogger(__name__)


class ExtensionRegistry:
    """Associate a module with a file extension. Note that there may be multiple modules for the same extension.

    Example: ::

        registry = ExtensionRegistry()

        # Add an association by setting an element
        registry['.zip'] = unzip

        # Multiple extensions for one module
        registry['.tgz'] = untar
        registry['.tar.gz'] = untar

        # Generic extensions to use after trying more specific extensions;
        # these will be checked after the more specific extensions fail.
        registry['.gz'] = gunzip

        # Multiple modules for one extension
        registry['.cx'] = cx1
        registry['.cx'] = cx2
        registry['.cx'] = cx3

        # Show registered extensions
        print(registry.extensions())

        # Register a format name for explicit control from caller
        registry['CX Number 3'] = cx3
        print registry.formats()

        # Retrieve modules for a file name
        registry.lookup('hello.cx') -> [cx3,cx2,cx1]

        # Run module on a filename
        registry.call_fcn('hello.cx') ->
            for module in registry.lookup('hello.cx'):
                try:
                    return module('hello.cx')
                except:
                    pass

        # Use a specific format ignoring extension
        registry.call_fcn('hello.cx', format='CX Number 3') -> return cx3('hello.cx')
    """

    # A dictionary of previously allowed extensions and formats that are no longer supported mapped to
    #   a string outlining the reason for deprecation.
    _deprecated_extensions = {}

    # A list of modules that should be used as backup modules, should all registered modules fail
    _fall_back_modules = []

    # File open mode for the registry object. Defaults to binary reading
    _mode = 'rb'

    def __init__(self):
        self.modules = defaultdict(list)

    def __setitem__(self, fmt: str, module):
        self.modules[fmt].insert(0, module)

    def __getitem__(self, ext: str) -> List:
        return self.modules[ext]

    def __contains__(self, ext: str) -> bool:
        return ext in self.modules

    def formats(self) -> List[str]:
        """
        Return a sorted list of the registered formats.
        """
        names = [a for a in self.modules.keys() if not a.startswith('.')]
        names.sort()
        return names

    def extensions(self) -> List[str]:
        """
        Return a sorted list of registered extensions.
        """
        exts = [a for a in self.modules.keys() if a.startswith('.')]
        exts.sort()
        return exts

    def lookup(self, path: PATH_LIKE) -> List[callable]:
        """
        Return all modules associated with the file type of path.

        :param path: Data file path
        :return: List of available modules based on the file extension (maybe empty)
        """
        # Find matching lower-case extensions
        path = Path(path)
        path_lower = str(path.absolute()).lower()
        # Allow multi extension files e.g.: `.tar.gz` files should match both `.tar.gz` and `.gz`
        # Combine readers for matching extensions into one big list
        modules = [module for ext in self.extensions() for module in self.modules[ext] if path_lower.endswith(ext)]
        # include generic readers in list of available readers to ensure error handling works properly
        modules.extend(self._fall_back_modules)
        # Ensure the list of readers only includes unique values and the order is maintained
        return unique_preserve_order(modules)

    def call_fcn(self, path: str, ext: Optional[str] = None) -> Any:
        """
        Call the registry for a single file.

        :param path: Data file path
        :param ext: Data file extension (optional)
        :return: Information relevant to the registry. This should be defined in subclasses.
        """
        if ext is None:
            # Empty string extensions are valid extensions and should *not* fall into this code branch.
            modules = self.lookup(path)
            ext = Path(path).suffix
        else:
            modules = self.modules.get(ext.lower(), [])
        # No registered modules found for this particular file extension
        if not modules:
            raise NoKnownLoaderException("No loaders match format %r" % ext)

        # A mapping of module name to errors thrown
        errors = {}
        with CustomFileOpen(path, self._mode) as file_handler:
            for module in modules:
                try:
                    results = module(path, file_handler)
                    # Check if the file read support is deprecated
                    if ext.lower() in self._deprecated_extensions:
                        logger.warning(DEPRECATION_MESSAGE.format(ext, path))
                        errors[module.__name__] = DEPRECATION_MESSAGE.format(module._name__, ext, path)
                    return results
                except Exception as e:
                    logger.warning(f"The module {module.__name__} was unable to process the contents of {path}.")

                    errors[module.__name__] = e
            # If we get here it is because all no modules were successful
            raise NoKnownLoaderException(f"Sasdata is unable to manage contents 0f {path}.")
