"""
Unit tests for the new recursive cansas reader
"""

import os
import unittest
import pytest
import logging
import warnings
from io import StringIO

from lxml import etree
from lxml.etree import XMLSyntaxError
from xml.dom import minidom

from sasdata.dataloader.filereader import decode
from sasdata.dataloader.loader import Loader
from sasdata.dataloader.data_info import Data1D, Data2D
from sasdata.dataloader.readers.xml_reader import XMLreader
from sasdata.dataloader.readers.cansas_reader import Reader
from sasdata.dataloader.readers.cansas_constants import CansasConstants
from sasdata.temp_hdf5_reader import load_data

test_file_names = [
    # "simpleexamplefile",
    "nxcansas_1Dand2D_multisasentry",
    "nxcansas_1Dand2D_multisasdata",
    "MAR07232_rest",
    "x25000_no_di",
]


def local_load(path: str):
    """Get local file path"""
    return os.path.join(os.path.dirname(__file__), path)


@pytest.mark.current
@pytest.mark.parametrize("f", test_file_names)
def test_load_file(f):
    data = load_data(local_load(f"data/{f}.h5"))

    with open(local_load(f"reference/{f}.txt")) as infile:
        expected = "".join(infile.readlines())
    assert data[0].summary() == expected
