"""Exceptions specific to loading data."""

from typing import Optional, List


class NoKnownLoaderException(Exception):
    """Exception for files with no associated reader based on the file extension of the loaded file. This exception
    should only be thrown by loader.py.
    """
    def __init__(self, e: Optional[str] = None):
        super().__init__(e)


class FileLoadException(Exception):
    """Base class all sasdata exceptions should be built off"""
    def __init__(self, loader: str, path: str, e: Optional[str] = None):
        """Initialize the base loader exception
        :param loader: The name of the reader that threw the exception.
        :param path: The file path that was being loaded.
        :param e: The error message, as a string, that describes the issue the loader faced.
        """
        self.path = path if path else ""
        self.loader = loader if loader else ""
        self.message = f"The file loader {loader} encountered an issue while loading the file {path}.\n"
        self.message += e


class DefaultReaderException(FileLoadException):
    """Exception for files with no associated reader. This should be thrown by default readers only to tell Loader to
    try the next reader.
    """
    def __init__(self, e=None, reader=None, path=None):
        super().__init__(reader, path, e)


class FileContentsException(FileLoadException):
    """Exception for files with an associated reader, but with no loadable data. This is useful for catching loader or
    file format issues.
    """
    def __init__(self, e=None, reader=None, path=None):
        super().__init__(reader, path, e)


class DataReaderException(FileLoadException):
    """Exception for files that were able to mostly load, but had minor issues along the way. Any exceptions of this
    type should be put into the data_info.errors
    """
    def __init__(self, e=None, reader=None, path=None, unread: Optional[List[str]] = None):
        """
        :param reader: The name of the reader that threw the exception.
        :param e: The error message, as a string, that describes the issue the loader faced.
        :param unread: A list of items that the loader system was unable handle for some reason.
        """
        super().__init__(reader, path, e)
        # Append each unread item to the error message after ensuring unread is a list.
        for item in list(unread):
            self.message += f"{item}\n"
