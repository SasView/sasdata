"""
Module to associate default readers to file extensions.
The module reads an xml file to get the readers for each file extension.
The readers are tried in order they appear when reading a file.
"""
############################################################################
#This software was developed by the University of Tennessee as part of the
#Distributed Data Analysis of Neutron Scattering Experiments (DANSE)
#project funded by the US National Science Foundation.
#If you use DANSE applications to do scientific research that leads to
#publication, we ask that you acknowledge the use of the software with the
#following sentence:
#This work benefited from DANSE software developed under NSF award DMR-0520547.
#copyright 2009, University of Tennessee
#############################################################################
import logging

from . import abs_reader
from . import anton_paar_saxs_reader
from . import ascii_reader
from . import cansas_reader
from . import cansas_reader_HDF5
from . import csv_reader
from . import danse_reader
from . import red2d_reader
from . import sesans_reader

logger = logging.getLogger(__name__)

FILE_ASSOCIATIONS = {
    ".xml": cansas_reader,
    ".ses": sesans_reader,
    ".sesans": sesans_reader,
    ".h5": cansas_reader_HDF5,
    ".hdf": cansas_reader_HDF5,
    ".hdf5": cansas_reader_HDF5,
    ".nxs": cansas_reader_HDF5,
    ".txt": ascii_reader,
    ".csv": csv_reader,
    ".dat": red2d_reader,
    ".abs": abs_reader,
    ".cor": abs_reader,
    ".sans": danse_reader,
    ".pdh": anton_paar_saxs_reader
}

GENERIC_READERS = [
    ascii_reader,
    cansas_reader,
    cansas_reader_HDF5
]


def read_associations(loader, settings=None):
    """
    Use the specified settings dictionary to associate readers to file extension.
    :param loader: Loader object
    :param settings: A dictionary in the format {str(file extension): data_loader_class} that is used to associate a
    file extension to a data loader class
    """
    # Default to a known list of extensions
    if not settings:
        settings = FILE_ASSOCIATIONS
    # For each FileType entry, get the associated reader and extension
    for ext, reader in settings.items():
        # Associate the extension with a particular reader
        if not loader.associate_file_type(ext.lower(), reader):
            msg = f"read_associations: skipping association for {ext.lower()}"
            logger.warning(msg)


def get_fallback_readers(settings=None, use_defaults=True):
    # type: ([FileReader], bool) -> [callable]
    """
    Returns a list of fallback readers that the data loader system will use in an attempt to load a data file.
    A list of loaders can be passed as an argument which will be appended to (if use_defaults is True) or override the
    list of default readers.
    :param settings: A list of modules to use as default readers. If None is passed, a default list will be used.
    :param use_defaults: Boolean to say if the default fallback readers should be added to the list of readers returned.
    :return: Final list of fallback loader modules the dataloader system will try if necessary
    """
    default_readers = GENERIC_READERS.copy() if use_defaults else []
    read_methods = []
    if isinstance(settings, (list, set, tuple)):
        default_readers.extend(settings)
    for module in default_readers:
        loader = module.Reader()
        # Append the new reader to the list
        read_methods.append(loader.read)
    # This might return duplicates -> Registry handles this issue upstream
    return read_methods
