"""
Unit tests for the new recursive cansas reader
"""

import io
import json
import os

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Literal
from abc import ABC

import sasdata.quantities.units as units
from sasdata.data import SasData, SasDataEncoder
from sasdata.quantities.quantity import Quantity
from sasdata.temp_hdf5_reader import load_data as hdf_load_data
from sasdata.temp_xml_reader import load_data as xml_load_data
from sasdata.temp_ascii_reader import AsciiReaderParams, load_data_default_params
from sasdata.temp_ascii_reader import load_data as ascii_load_data


@dataclass
class BaseTestCase(ABC):
    expected_values: dict[int, dict[str, float]]


@dataclass
class AsciiTestCase(BaseTestCase):
    # If this is a string/list of strings then the other params will be guessed.
    reader_params: AsciiReaderParams | str | list[str]


@dataclass
class XmlTestCase(BaseTestCase):
    filename: str


@dataclass
class Hdf5TestCase(BaseTestCase):
    filename: str


@dataclass
class SesansTestCase(BaseTestCase):
    filename: str


@dataclass
class TestCase:
    filename: str | list[str]
    # Key is the index of the row.
    expected_values: dict[int, dict[str, float]]
    loader: Literal["ascii", "xml", "hdf5", "sesans"]
    ascii_reader_params: AsciiReaderParams | None = None


test_cases = [
    pytest.param(
        TestCase(
            filename="ascii_test_1.txt",
            expected_values={
                0: {"Q": 0.002618, "I": 0.02198, "dI": 0.002704},
                -1: {"Q": 0.0497, "I": 8.346, "dI": 0.191},
            },
            loader="ascii",
        ),
        marks=pytest.mark.xfail(
            reason="The ASCII reader cannot make the right guesses for this file."
        ),
    ),
    TestCase(
        filename="test_3_columns.txt",
        expected_values={
            0: {"Q": 0, "I": 2.83954, "dI": 0.6},
            -1: {"Q": 1.22449, "I": 7.47487, "dI": 1.05918},
        },
        loader="ascii",
    ),
    pytest.param(
        TestCase(
            filename="detector_rectangular.DAT",
            expected_values={
                0: {
                    "Qx": -0.009160664,
                    "Qy": -0.1683881,
                    "I": 16806.79,
                    "dI": 0.01366757,
                },
                -1: {"Qx": 0.2908819, "Qy": 0.1634992, "I": 8147.779, "dI": 0.05458562},
            },
            loader="ascii",
        ),
        marks=pytest.mark.xfail(
            reason="Guesses for 2D ASCII files are currently wrong, so the data loaded won't be correct."
        ),
    ),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_load_file(test_case: TestCase):
    full_filename = local_load(os.path.join("data", test_case.filename))
    match test_case.loader:
        case "ascii":
            if test_case.ascii_reader_params is None:
                loaded_data = load_data_default_params(full_filename)[0]
            else:
                loaded_data = ascii_load_data(test_case.ascii_reader_params)[0]

        # TODO: Support other loaders
        case _:
            raise ValueError("Invalid loader")
    for index, values in test_case.expected_values.items():
        for column, expected_value in values.items():
            # TODO: Handle other uncertainities.
            if column == "dI":
                assert loaded_data._data_contents["I"]._variance[
                    index
                ] == pytest.approx(expected_value**2)
            else:
                assert loaded_data._data_contents[column].value[index] == pytest.approx(
                    expected_value
                )


test_hdf_file_names = [
    "simpleexamplefile",
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
    # "cansas_xml_multisasentry_multisasdata",
    "valid_cansas_xml",
]


def local_load(path: str):
    """Get local file path"""
    base = os.path.join(os.path.dirname(__file__), path)
    if (os.path.exists(f"{base}.h5")):
        return f"{base}.h5"
    if (os.path.exists(f"{base}.xml")):
        return f"{base}.xml"
    return f"{base}"

def example_data_load(path: str):
    try:
        return xml_load_data(local_load(f"data/{path}.xml"))
    except OSError:
        return hdf_load_data(local_load(f"data/{path}.h5"))


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_hdf_file_names)
def test_hdf_load_file(f):
    data = hdf_load_data(local_load(f"data/{f}"))

    with open(local_load(f"reference/{f}.txt"), encoding="utf-8") as infile:
        expected = "".join(infile.readlines())
    keys = sorted([d for d in data])
    assert "".join(data[k].summary() for k in keys) == expected


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_xml_file_names)
def test_xml_load_file(f):
    data = xml_load_data(local_load(f"data/{f}"))

    with open(local_load(f"reference/{f}.txt"), encoding="utf-8") as infile:
        expected = "".join(infile.readlines())
    keys = sorted([d for d in data])
    assert "".join(data[k].summary() for k in keys) == expected


@pytest.mark.sasdata
def test_filter_data():
    data = xml_load_data(local_load("data/cansas1d_notitle"))
    for k, v in data.items():
        assert v.metadata.raw.filter("transmission") == ["0.327"]
        assert v.metadata.raw.filter("wavelength")[0] == Quantity(6.0, units.angstroms)
        assert v.metadata.raw.filter("SDD")[0] == Quantity(4.15, units.meters)
    data = hdf_load_data(local_load("data/nxcansas_1Dand2D_multisasentry"))
    for k, v in data.items():
        assert v.metadata.raw.filter("radiation") == ["Spallation Neutron Source"]
        assert v.metadata.raw.filter("SDD") == [
            Quantity(np.array([2845.26], dtype=np.float32), units.millimeters),
            Quantity(np.array([4385.28], dtype=np.float32), units.millimeters),
        ]


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_hdf_file_names + test_xml_file_names)
def test_json_serialise(f):
    data = example_data_load(f)

    with open(local_load(f"json/{f}.json"), encoding="utf-8") as infile:
        expected = json.loads("".join(infile.readlines()))
    assert json.loads(SasDataEncoder().encode(data)) == expected


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_hdf_file_names + test_xml_file_names)
def test_json_deserialise(f):
    expected = example_data_load(f)

    with open(local_load(f"json/{f}.json"), encoding="utf-8") as infile:
        raw = json.loads("".join(infile.readlines()))
        parsed = {}
        for k in raw:
            parsed[k] = SasData.from_json(raw[k])

    for k in expected:
        expect = expected[k]
        pars = parsed[k]
        assert pars.name == expect.name
        # assert pars._data_contents == expect._data_contents
        assert pars.dataset_type == expect.dataset_type
        assert pars.mask == expect.mask
        assert pars.model_requirements == expect.model_requirements


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_xml_file_names + test_hdf_file_names)
def test_h5_round_trip_serialise(f):
    expected = example_data_load(f)

    bio = io.BytesIO()
    SasData.save_h5(expected, bio)
    bio.seek(0)

    result = hdf_load_data(bio)
    bio.close()

    for name, entry in result.items():
        assert expected[name].metadata.title == entry.metadata.title
        assert expected[name].metadata.run == entry.metadata.run
        assert expected[name].metadata.definition == entry.metadata.definition
        assert expected[name].metadata.process == entry.metadata.process
        assert expected[name].metadata.instrument == entry.metadata.instrument
        assert expected[name].metadata.sample == entry.metadata.sample
        assert expected[name].ordinate.units == entry.ordinate.units
        assert np.all(expected[name].ordinate.value == entry.ordinate.value)
        assert expected[name].abscissae.units == entry.abscissae.units
        assert np.all(expected[name].abscissae.value == entry.abscissae.value)
