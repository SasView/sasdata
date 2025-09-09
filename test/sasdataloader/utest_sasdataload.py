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
    if e.instrument:
        assert_instrument(e.instrument, r["sasinstrument"])
    if e.process:
        for ei, ri in zip(e.process, sorted([x for x in r if "sasprocess" in x])):
            assert_process(ei, ri)


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


def assert_process(e, r):
    if e.name:
        safe_assert(e.name, r["name"])
    if e.date:
        safe_assert(e.date, r["date"])
    if e.description:
        safe_assert(e.description, r["description"])
    for ei, ri in zip(e.notes, [x for x in r if "note" in x]):
        safe_assert(ei, ri)


def assert_instrument(e, r):
    if e.source:
        assert_source(e.source, r["sassource"])
    if e.detector:
        for ei, ri in zip(e.detector, [x for x in r if "detector" in r]):
            assert_detector(ei, ri)
    if e.collimations:
        for ei, ri in zip(e.collimations, [x for x in r if "collimation" in r]):
            assert_collimation(ei, ri)


def assert_source(e, r):
    if e.radiation:
        safe_assert(e.radiation, r["radiation"])
    if e.beam_shape:
        safe_assert(e.beam_shape, r["beam_shape"])
    if e.beam_size:
        assert_beam_size(e.beam_size, r["beam_size"])
    if e.wavelength:
        safe_assert(e.wavelength, r["wavelength"])
    if e.wavelength_min:
        safe_assert(e.wavelength_min, r["wavelength_min"])
    if e.wavelength_max:
        safe_assert(e.wavelength_max, r["wavelength_max"])
    if e.wavelength_spread:
        safe_assert(e.wavelength_spread, r["wavelength_spread"])


def assert_beam_size(e, r):
    if e.name:
        safe_assert(e.name, r.attrs["name"])
    assert_vec(e.size, r)


def assert_detector(e, r):
    if e.name:
        safe_assert(e.name, r["name"])
    if e.distance:
        safe_assert(e.distance, r["SDD"])
    if e.offset:
        assert_vec(e.offset, r["offset"])
    if e.orientation:
        assert_rot(e.orientation, r["orientation"])
    if e.beam_center:
        assert_vec(e.beam_center, r["beam_center"])
    if e.pixel_size:
        assert_vec(e.pixel_size, r["pixel_size"])
    if e.slit_length:
        safe_assert(e.slit_length, r["slit_length"])


def assert_collimation(e, r):
    if e.length:
        safe_assert(e.length, r["length"])
    for ei, ri in zip(e.apertures, [x for x in r if "aperture" in x]):
        assert_aperture(ei, ri)


def assert_aperture(e, r):
    if e.distance:
        safe_assert(e.distance, r["distance"])
    if e.name:
        safe_assert(e.name, r.attrs["name"])
    if e.size:
        assert_vec(e.size, r["size"])
        if e.size_name:
            safe_assert(e.size_name, r["size"].attrs["name"])
    if e.type_:
        safe_assert(e.type_, r.attrs["type"])
