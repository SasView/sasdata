"""
This is the base file importer class all importers should inherit from.
All generic functionality required for file import is built into this class.
"""

import codecs
import logging
import os.path
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Optional, Any

from sasdata.data_util.registry import CustomFileOpen

logger = logging.getLogger(__name__)


def decode(s):
    # Attempt to decode files using common encodings
    # *NB* windows-1252, aka cp1252, overlaps with most ASCII-style encodings
    for codec in ['utf-8', 'windows-1252']:
        try:
            return codecs.decode(s, codec) if isinstance(s, bytes) else s
        except (ValueError, UnicodeError):
            # If the specific codec fails, try the next one.
            pass
        except Exception as e:
            logger.warning(e)
    # Give warning if unable to decode the item using the codecs
    logger.warning(f"Unable to decode {s}")


# Data 1D fields for iterative purposes
FIELDS_1D = 'x', 'y', 'dx', 'dy', 'dxl', 'dxw'
# Data 2D fields for iterative purposes
FIELDS_2D = 'data', 'qx_data', 'qy_data', 'q_data', 'err_data', 'dqx_data', 'dqy_data', 'mask'


class Importer:
    # The following are class-level objects that should not be modified at the instance level
    # String to describe the type of data this reader can load
    type_name = ""
    # Dictionary mapping allowed extensions to their respective descriptions
    ext = {}
    # Bypass extension check and try to load anyway
    allow_all = True

    def __init__(self):
        # List of Data1D and Data2D objects to be sent back to data_loader
        self.imported_data = []
        # For non-transient importers, keep a handle on all data loaded since inception
        self._all_data = {}
        # Path object using the file path sent to reader
        self.filepath = None
        # Starting file position to begin reading data from
        self.f_pos = 0
        # File extension of the data file passed to the reader
        self.extension = None
        # Open file handle
        self.f_open = None

    def read(self, filepath: Union[str, Path], file_handler: Optional[CustomFileOpen] = None,
             f_pos: Optional[int] = 0) -> List[Any]:
        """
        Basic file reader

        :param filepath: The string representation of the path to a file to be loaded. This can be a URI or a local file
        :param file_handler: A CustomFileOpen instance used to handle file operations
        :param f_pos: The initial file position to start reading from
        :return: A list of Data1D and Data2D objects
        """
        self.filepath = Path(filepath)
        self.f_pos = f_pos
        if not file_handler:
            # Allow direct calls to the readers without generating a file_handler, but higher-level calls should
            #   already have file_handler defined
            with CustomFileOpen(filepath, 'rb') as file_handler:
                return self._read(file_handler)
        return self._read(file_handler)

    def _read(self, file_handler: CustomFileOpen) -> list[Any]:
        """
        Private method to handle file loading

        :param file_handler: A CustomFileOpen instance used to handle file operations
        :return: A list of Data1D and Data2D objects
        """
        self.f_open = file_handler.fd
        # Move to the desired initial file position in case of successive reads on the same handle
        self.f_open.seek(self.f_pos)

        basename, extension = self.filepath.stem, self.filepath.suffix
        self.extension = extension.lower()
        if self.extension in self.ext or self.allow_all:
            try:
                # All raised exceptions are handled by ExtensionRegistry.load(). No exception handling here.
                self.get_file_contents()
            finally:
                # TODO: Do something here to ensure data quality
                pass
        else:
            # TODO: Throw meaningful error (unknown file type!!!!)
            pass

        # Return a list of parsed entries that data_loader can manage
        final_data = self.imported_data.copy()
        self.reset_state()
        return final_data

    def reset_state(self):
        """
        Resets the class state to a base case when loading a new data file so previous
        data files do not appear a second time
        """
        self.imported_data = []

    def next_line(self) -> str:
        """
        Returns the next line in the file as a string.
        """
        return decode(self.f_open.readline())

    def next_lines(self) -> str:
        """
        Returns the next line in the file as a string.
        """
        for _ in self.f_open:
            yield self.next_line()

    def readall(self) -> str:
        """
        Returns the entire file as a string.
        """
        self.f_open.seek(self.f_pos)
        return decode(self.f_open.read())

    @abstractmethod
    def get_file_contents(self):
        """
        Reader specific class to access the contents of the file
        All reader classes that inherit from FileReader must implement
        """
        pass