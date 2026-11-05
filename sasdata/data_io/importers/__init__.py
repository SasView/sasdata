"""
Importers for each of the file formats supported by sasdata.

Each module defines a child of :class:`sasdata.data_io.importer.Importer`. Importing this package registers all of
them, so :meth:`Importer.find_importer` can select the correct one from the contents of a file.
"""

from sasdata.data_io.importers.import_ascii import AsciiImporter
from sasdata.data_io.importers.import_hdf5 import HDF5Importer
from sasdata.data_io.importers.import_sesans import SesansImporter
from sasdata.data_io.importers.import_xml import XmlImporter

__all__ = [
    'AsciiImporter', 'HDF5Importer', 'SesansImporter', 'XmlImporter',
]
