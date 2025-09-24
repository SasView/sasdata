"""
Unit tests for the new recursive cansas reader
"""

import io
import json
import os

import numpy as np
import pytest

import sasdata.quantities.units as units
from sasdata.data import SasData, SasDataEncoder
from sasdata.quantities.quantity import Quantity
from sasdata.temp_hdf5_reader import load_data as hdf_load_data
from sasdata.temp_xml_reader import load_data as xml_load_data

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
@pytest.mark.parametrize("f", test_hdf_file_names)
def test_json_serialise(f):
    data = hdf_load_data(local_load(f"data/{f}"))

    with open(local_load(f"json/{f}.json"), encoding="utf-8") as infile:
        expected = json.loads("".join(infile.readlines()))
    assert json.loads(SasDataEncoder().encode(data)) == expected


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_hdf_file_names)
def test_json_deserialise(f):
    expected = hdf_load_data(local_load(f"data/{f}"))

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
    try:
        result = xml_load_data(local_load(f"data/{f}.xml"))
    except OSError:
        result = hdf_load_data(local_load(f"data/{f}.h5"))


    for name, expected in result.items():
        bio = io.BytesIO()
        expected.save_h5(bio)
        bio.seek(0)

        for result in hdf_load_data(bio).values():
            assert expected.metadata.title == result.metadata.title
            assert expected.metadata.run == result.metadata.run
            assert expected.metadata.definition == result.metadata.definition
            assert expected.metadata.process == result.metadata.process
            assert expected.metadata.instrument == result.metadata.instrument
            assert expected.metadata.sample == result.metadata.sample
