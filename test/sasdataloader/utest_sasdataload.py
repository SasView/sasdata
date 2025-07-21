"""
Unit tests for the new recursive cansas reader
"""

import numpy as np
import os
import pytest
from dataclasses import dataclass
from typing import Literal


from sasdata.quantities.quantity import Quantity
import sasdata.quantities.units as units
from sasdata.temp_hdf5_reader import load_data as hdf_load_data
from sasdata.temp_xml_reader import load_data as xml_load_data
from sasdata.temp_ascii_reader import AsciiReaderParams
from sasdata.temp_ascii_reader import load_data as ascii_load_data


@dataclass
class TestCase:
    filename: str
    ascii_reader_params: AsciiReaderParams | None
    # Key is the index of the row.
    expected_values: dict[int, dict[str, float]]
    loader: Literal["ascii", "xml", "hdf5", "sesans"]


test_cases = [
    TestCase(
        filename="ascii_test_1.txt",
        expected_values={
            0: {"Q": 0.002618, "I": 0.02198, "dI": 0.002704},
            -1: {"Q": 0.0497, "I": 8.346, "dI": 0.191},
        },
        loader="ascii",
    )
]


@pytest.mark.parametrize("test_case", test_cases)
def test_load_file(test_case: TestCase):
    match test_case.loader:
        case "ascii":
            if test_case.ascii_reader_params is not None:
                loaded_data = ascii_load_data(test_case.ascii_reader_params)[0]
        # TODO: Support other loaders
        case _:
            raise ValueError("Invalid loader")
    for index, values in test_case.expected_values.items():
        for column, expected_value in values.items():
            assert loaded_data._data_contents[column] == pytest.approx(expected_value)


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

    with open(local_load(f"reference/{f}.txt"), encoding="utf-8") as infile:
        expected = "".join(infile.readlines())
    keys = sorted([d for d in data])
    assert "".join(data[k].summary() for k in keys) == expected


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_xml_file_names)
def test_xml_load_file(f):
    data = xml_load_data(local_load(f"data/{f}.xml"))

    with open(local_load(f"reference/{f}.txt"), encoding="utf-8") as infile:
        expected = "".join(infile.readlines())
    keys = sorted([d for d in data])
    assert "".join(data[k].summary() for k in keys) == expected


@pytest.mark.sasdata
def test_filter_data():
    data = xml_load_data(local_load("data/cansas1d_notitle.xml"))
    for k, v in data.items():
        assert v.metadata.raw.filter("transmission") == ["0.327"]
        assert v.metadata.raw.filter("wavelength")[0] == Quantity(6.0, units.angstroms)
        assert v.metadata.raw.filter("SDD")[0] == Quantity(4.15, units.meters)
    data = hdf_load_data(local_load("data/nxcansas_1Dand2D_multisasentry.h5"))
    for k, v in data.items():
        assert v.metadata.raw.filter("radiation") == ["Spallation Neutron Source"]
        assert v.metadata.raw.filter("SDD") == [
            Quantity(np.array([2845.26], dtype=np.float32), units.millimeters),
            Quantity(np.array([4385.28], dtype=np.float32), units.millimeters),
        ]
