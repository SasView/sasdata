"""
Unit tests for the new recursive cansas reader
"""

import io
import json
import os

import h5py
import numpy as np
import pytest

import sasdata.quantities.units as units
from sasdata.data import SasData, SasDataEncoder
from sasdata.quantities.quantity import Quantity
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
    # "cansas_xml_multisasentry_multisasdata",
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


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_hdf_file_names)
def test_json_serialise(f):
    data = hdf_load_data(local_load(f"data/{f}.h5"))

    with open(local_load(f"json/{f}.json"), encoding="utf-8") as infile:
        expected = json.loads("".join(infile.readlines()))
    assert json.loads(SasDataEncoder().encode(data["sasentry01"])) == expected


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_hdf_file_names)
def test_json_deserialise(f):
    expected = hdf_load_data(local_load(f"data/{f}.h5"))["sasentry01"]

    with open(local_load(f"json/{f}.json"), encoding="utf-8") as infile:
        parsed = SasData.from_json(json.loads("".join(infile.readlines())))
    assert parsed.name == expected.name
    assert parsed._data_contents == expected._data_contents
    assert parsed.dataset_type == expected.dataset_type
    assert parsed.mask == expected.mask
    assert parsed.model_requirements == expected.model_requirements


def safe_assert(e, r):
    match (e, r):
        case (Quantity(), _):
            assert e.value == r[()]
            assert e.units.symbol == r.attrs["units"]
        case (str(), bytes()):
            assert e == r.decode("utf-8")
        case (_, h5py.Dataset()):
            safe_assert(e, r[()])
        case _:
            assert e == r


@pytest.mark.sasdata3
@pytest.mark.parametrize("f", test_xml_file_names)
def test_h5_serialise(f):
    for name, expected in xml_load_data(local_load(f"data/{f}.xml")).items():
        bio = io.BytesIO()
        expected.save_h5(bio)
        bio.seek(0)

        result = h5py.File(bio)

        if expected.name:
            assert expected.name == result.attrs["name"]

        if expected.metadata:
            assert_metadata(expected.metadata, result["metadata"])


def assert_metadata(e, r):
    if e.title:
        safe_assert(e.title, r["title"][()])
    if e.definition:
        safe_assert(e.definition, r["definition"][()])
    if e.sample:
        assert_sample(e.sample, r["sassample"])


def assert_sample(e, r):
    if e.name:
        safe_assert(e.name, r.attrs["name"])
    if e.sample_id:
        safe_assert(e.sample_id, r["ID"])
    if e.thickness:
        safe_assert(e.thickness, r["thickness"])
    if e.temperature:
        safe_assert(e.temperature, r["temperature"])
    if e.transmission:
        safe_assert(e.transmission, r["transmission"])
    if e.position:
        assert_vec(e.position, r["position"])
    if e.orientation:
        assert_rot(e.orientation, r["orientation"])


def assert_vec(e, r):
    if e.x:
        safe_assert(e.x, r["x"])
    if e.y:
        safe_assert(e.y, r["y"])
    if e.z:
        safe_assert(e.z, r["z"])


def assert_rot(e, r):
    if e.roll:
        safe_assert(e.roll, r["roll"])
    if e.pitch:
        safe_assert(e.pitch, r["pitch"])
    if e.yaw:
        safe_assert(e.yaw, r["yaw"])
