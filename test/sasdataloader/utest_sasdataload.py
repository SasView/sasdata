"""
Unit tests for the new recursive cansas reader
"""

import numpy as np
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
from sasdata.quantities.quantity import Quantity
import sasdata.quantities.unit_parser as unit_parser
from sasdata.temp_hdf5_reader import load_data as hdf_load_data
from sasdata.temp_xml_reader import load_data as xml_load_data

test_hdf_file_names = [
    # "simpleexamplefile",
    "nxcansas_1Dand2D_multisasentry",
    "nxcansas_1Dand2D_multisasdata",
    "MAR07232_rest",
    "x25000_no_di",
]

test_xml_file_names = [
    "ISIS_1_0",
    "ISIS_1_1",
    "ISIS_1_1_doubletrans",
    "ISIS_1_1_notrans",
    "TestExtensions",
    "cansas1d",
    "cansas1d_badunits",
    "cansas1d_notitle",
    "cansas1d_slit",
    "cansas1d_units",
    "cansas_test",
    "cansas_test_modified",
    "cansas_xml_multisasentry_multisasdata",
    "valid_cansas_xml",
]


def local_load(path: str):
    """Get local file path"""
    return os.path.join(os.path.dirname(__file__), path)


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_hdf_file_names)
def test_hdf_load_file(f):
    data = hdf_load_data(local_load(f"data/{f}.h5"))

    with open(local_load(f"reference/{f}.txt")) as infile:
        expected = "".join(infile.readlines())
    keys = sorted([d for d in data])
    assert "".join(data[k].summary() for k in keys) == expected


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_xml_file_names)
def test_xml_load_file(f):
    data = xml_load_data(local_load(f"data/{f}.xml"))

    with open(local_load(f"reference/{f}.txt")) as infile:
        expected = "".join(infile.readlines())
    keys = sorted([d for d in data])
    assert "".join(data[k].summary() for k in keys) == expected

@pytest.mark.sasdata
def test_filter_data():
    data = xml_load_data(local_load("data/cansas1d_notitle.xml"))
    for k, v in data.items():
        assert v.metadata.raw.filter("transmission") == ["0.327"]
        assert v.metadata.raw.filter("wavelength") == [Quantity(6.0, unit_parser.parse("A"))]
        assert v.metadata.raw.filter("SDD") == [Quantity(4.15, unit_parser.parse("m"))]
    data = hdf_load_data(local_load("data/nxcansas_1Dand2D_multisasentry.h5"))
    for k, v in data.items():
        print([y
               for x in v.metadata.raw.contents if x.name.startswith("sasinstrument")
               for y in x.contents if y.name.startswith("sasdetector")
               ])
        assert v.metadata.raw.filter("radiation") == ["Spallation Neutron Source"]
        assert v.metadata.raw.filter("SDD") == [Quantity(np.array([2845.26], dtype=np.float32), unit_parser.parse("mm")),
                                                Quantity(np.array([4385.28], dtype=np.float32), unit_parser.parse("mm"))]
