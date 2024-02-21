"""
A custom context manager to handle file io.

This creates a simple file handle that ensures files are opened and closed properly.
"""
from urllib.request import urlopen
from pathlib import Path
from os import path
from io import BytesIO, FileIO, StringIO
from typing import Union, Optional

PATHLIKE = Union[str, Path, path]
FILELIKE = Union[BytesIO, FileIO, StringIO]


class CustomFileOpen:
    """A custom context manager to handle files, regardless of where the file is located.

        ...

        Attributes
        ----------
        file : Path
            A pathlib.Path object that points to the file location.
        filename : str
            The name of the file, with no path information associated.
        mode : str
            The file io mode, see https://docs.python.org/3/library/functions.html#open (default `rb`)
        fd : Union[BytesIO, FileIO, StringIO]
            An open file handle
        errors : list[Exception]
            A list of exceptions caught during file handling that should be handled post file-handling.

        Methods
        -------
        __enter__()
            Called when creating a context manager from this class. Should never be called directly.
            Opens the file, if necessary, and returns the instance with the given name from the manager.
            Usage:
                ``` f_open = CustomFileOpen(file, mode, filename)
                    with f_open as fd:
                        contents = fd.fd.read()
                ```
        __exit__()
            Called when the context manager exits. Should never be called directly.
            Closes any active file handles.
        """

    def __init__(self, file: Union[PATHLIKE, FILELIKE], mode: Optional[str] = 'rb',
                 full_path: Optional[PATHLIKE] = None):
        """Create an instance of the file handler.

        Parameters
        ----------
            file (Union[PATHLIKE, BytesIO, FileIO, StringIO]): A string representation of a file path, a Python path
                object, or a file io object.
            mode (Optional[str]): The file open mode (if needed). (default `rb`)
            full_path (Optional[str]): A complete filepath to the file. (default: None)
        """
        if hasattr(file, 'read'):
            # io style objects
            self.fd = file
            self.file = Path(full_path)
        else:
            self.file = Path(file)
            self.fd = None
        self.filename = self.file.name
        self.mode = mode
        self.errors = []

    def __enter__(self):
        """A context method that either fetches a file from a URL, opens a local file, or keeps the existing file open.

        Returns
        -------
            self : The CustomFileOpen instance.
        """
        if self.fd is not None:
            # Likely an IO-like object was passed to the manager so the file is already opened.
            pass
        elif '://' in self.file:
            # Use urllib.request package to access remote files
            with urlopen(self.filename) as req:
                content = req.read()
                self.fd = BytesIO(content)
                self.fd.name = self.filename
        else:
            # Use native open to access local files
            self.fd = open(self.file, self.mode)
        # Return the instance to allow access to the filename, and any open file handles.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        """Close all open file handles when exiting the context manager.

        Parameters
        ----------
            exc_type (Optional[Exception]): A string representation of a file path, a Python path
                object, or a file io object.
            exc_val (Optional[str]): The
            exc_tb (Optional[str]): A complete filepath to the file. (default: None)

        Returns
        -------
            bool : Should the context manager suppress errors thrown during execution?
        """
        if self.fd is not None:
            self.fd.close()
            self.fd = None
        return self._check_error(exc_type, exc_val, exc_tb)

    def _check_error(self, exc_type, exc_val, exc_tb) -> bool:
        """Check if the

        Parameters
        ----------
            exc_type (Optional[Exception]): A string representation of a file path, a Python path
                object, or a file io object.
            exc_val (Optional[str]): The
            exc_tb (Optional[str]): A complete filepath to the file. (default: None)

        Returns
        -------
            bool : Should the error be suppressed?
        """
        suppress = False
        # TODO: This suppresses *ALL* exceptions. Check for specific exception types.
        if exc_type:
            exception = Exception(exc_type, exc_val, exc_tb)
            self.errors.append(exception)
            return suppress
        return suppress
