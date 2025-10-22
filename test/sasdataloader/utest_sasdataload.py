"""
Unit tests for the new recursive cansas reader
"""

import io
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

import sasdata.quantities.units as units
from sasdata.data import SasData, SasDataEncoder
from sasdata.dataset_types import one_dim
from sasdata.guess import guess_columns
from sasdata.quantities.quantity import Quantity
from sasdata.quantities.units import per_angstrom
from sasdata.temp_ascii_reader import (
    AsciiMetadataCategory,
    AsciiReaderMetadata,
    AsciiReaderParams,
    load_data_default_params,
)
from sasdata.temp_ascii_reader import load_data as ascii_load_data
from sasdata.temp_hdf5_reader import load_data as hdf_load_data
from sasdata.temp_xml_reader import load_data as xml_load_data

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
    if os.path.exists(f"{base}.h5"):
        return f"{base}.h5"
    if os.path.exists(f"{base}.xml"):
        return f"{base}.xml"
    return f"{base}"


def local_reference_load(path: str):
    return local_load(f"{os.path.join('reference', path)}")


def local_data_load(path: str):
    return local_load(f"{os.path.join('data', path)}")


def example_data_load(path: str):
    try:
        return xml_load_data(local_load(f"data/{path}.xml"))
    except OSError:
        return hdf_load_data(local_load(f"data/{path}.h5"))


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
@pytest.mark.parametrize("f", test_xml_file_names)
def test_json_serialise(f):
    data = example_data_load(f)

    with open(local_load(f"json/{f}.json"), encoding="utf-8") as infile:
        expected = json.loads("".join(infile.readlines()))
    assert json.loads(SasDataEncoder().encode(data)) == expected


@pytest.mark.sasdata
@pytest.mark.parametrize("f", test_xml_file_names)
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


@dataclass(kw_only=True)
class BaseTestCase:
    expected_values: dict[int, dict[str, float]]
    expected_metadata: dict[str, Any] = field(default_factory=dict)
    metadata_file: None | str = None
    round_trip: bool = False


@dataclass(kw_only=True)
class AsciiTestCase(BaseTestCase):
    # If this is a string of strings then the other params will be guessed.
    reader_params: AsciiReaderParams | str


@dataclass(kw_only=True)
class BulkAsciiTestCase(AsciiTestCase):
    reader_params: AsciiReaderParams
    expected_values: dict[str, dict[int, dict[str, float]]]
    expected_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(kw_only=True)
class XmlTestCase(BaseTestCase):
    filename: str
    entry: str = "sasentry01"
    round_trip: bool = True


@dataclass(kw_only=True)
class Hdf5TestCase(BaseTestCase):
    filename: str
    entry: str = "sasentry01"
    round_trip: bool = True


@dataclass(kw_only=True)
class SesansTestCase(BaseTestCase):
    filename: str


test_cases = [
    pytest.param(
        AsciiTestCase(
            reader_params=local_data_load("ascii_test_1.txt"),
            expected_values={
                0: {"Q": 0.002618, "I": 0.02198, "dI": 0.002704},
                -1: {"Q": 0.0497, "I": 8.346, "dI": 0.191},
            },
        ),
        marks=pytest.mark.xfail(reason="The ASCII reader cannot make the right guesses for this file."),
    ),
    AsciiTestCase(
        reader_params=local_data_load("test_3_columns.txt"),
        expected_values={
            0: {"Q": 0, "I": 2.83954, "dI": 0.6},
            -1: {"Q": 1.22449, "I": 7.47487, "dI": 1.05918},
        },
    ),
    pytest.param(
        AsciiTestCase(
            reader_params=local_data_load("detector_rectangular.DAT"),
            expected_values={
                0: {
                    "Qx": -0.009160664,
                    "Qy": -0.1683881,
                    "I": 16806.79,
                    "dI": 0.01366757,
                },
                -1: {
                    "Qx": 0.2908819,
                    "Qy": 0.1634992,
                    "I": 8147.779,
                    "dI": 0.05458562,
                },
            },
        ),
        marks=pytest.mark.xfail(
            reason="Guesses for 2D ASCII files are currently wrong, so the data loaded won't be correct."
        ),
    ),
    BulkAsciiTestCase(
        reader_params=AsciiReaderParams(
            filenames=[
                local_data_load(filename)
                for filename in [
                    "1_33_1640_22.874115.csv",
                    "2_42_1640_23.456895.csv",
                    "3_61_1640_23.748285.csv",
                    "4_103_1640_24.039675.csv",
                    "5_312_1640_24.331065.csv",
                    "6_1270_1640_24.331065.csv",
                ]
            ],
            columns=[(column, per_angstrom) for column in guess_columns(3, one_dim)],
            separator_dict={"Comma": True},
            metadata=AsciiReaderMetadata(
                master_metadata={
                    "magnetic": AsciiMetadataCategory(
                        values={
                            "counting_index": 0,
                            "applied_magnetic_field": 1,
                            "saturation_magnetization": 2,
                            "demagnetizing_field": 3,
                        }
                    )
                }
            ),
        ),
        expected_values={},
        expected_metadata={
            "1_33_1640_22.874115.csv": {
                "counting_index": ["1"],
                "applied_magnetic_field": ["33"],
                "saturation_magnetization": ["1640"],
                "demagnetizing_field": ["22"],
            },
            "6_1270_1640_24.331065.csv": {
                "counting_index": ["6"],
                "applied_magnetic_field": ["1270"],
                "saturation_magnetization": ["1640"],
                "demagnetizing_field": ["24"],
            },
        },
    ),
    XmlTestCase(
        filename=local_data_load("ISIS_1_0.xml"),
        entry="79680main_1D_2.2_10.0",
        expected_values={
            0: {"Q": 0.009, "I": 85.3333, "dI": 0.852491, "dQ": 0},
            -2: {"Q": 0.281, "I": 0.408902, "dQ": 0},
            -1: {"Q": 0.283, "I": 0, "dI": 0, "dQ": 0},
        },
        expected_metadata={
            # TODO: Add more.
            "radiation": "neutron"
        },
    ),
    Hdf5TestCase(
        filename=local_data_load("simpleexamplefile.h5"),
        metadata_file=local_reference_load("simpleexamplefile.txt"),
        expected_values={
            0: {"Q": 0.5488135039273248, "I": 0.6778165367962301},
            -1: {"Q": 0.004695476192547066, "I": 0.4344166255581208},
        },
    ),
    Hdf5TestCase(
        filename=local_data_load("MAR07232_rest.h5"),
        metadata_file=local_reference_load("MAR07232_rest.txt"),
        expected_values={},
    ),
    Hdf5TestCase(
        filename=local_data_load("x25000_no_di.h5"),
        metadata_file=local_reference_load("x25000_no_di.txt"),
        expected_values={},
    ),
    Hdf5TestCase(
        filename=local_data_load("nxcansas_1Dand2D_multisasentry.h5"),
        metadata_file=local_reference_load("nxcansas_1Dand2D_multisasentry.txt"),
        expected_values={},
    ),
    Hdf5TestCase(
        filename=local_data_load("nxcansas_1Dand2D_multisasdata.h5"),
        metadata_file=local_reference_load("nxcansas_1Dand2D_multisasdata.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("ISIS_1_0.xml"),
        entry="79680main_1D_2.2_10.0",
        metadata_file=local_reference_load("ISIS_1_0.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("ISIS_1_1.xml"),
        entry="79680main_1D_2.2_10.0",
        metadata_file=local_reference_load("ISIS_1_1.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("ISIS_1_1_doubletrans.xml"),
        entry="79680main_1D_2.2_10.0",
        metadata_file=local_reference_load("ISIS_1_1_doubletrans.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("ISIS_1_1_notrans.xml"),
        entry="79680main_1D_2.2_10.0",
        metadata_file=local_reference_load("ISIS_1_1_notrans.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("TestExtensions.xml"),
        entry="TK49 c10_SANS",
        metadata_file=local_reference_load("TestExtensions.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("cansas1d.xml"),
        entry="Test title",
        metadata_file=local_reference_load("cansas1d.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("cansas1d_badunits.xml"),
        entry="Test title",
        metadata_file=local_reference_load("cansas1d_badunits.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("cansas1d_notitle.xml"),
        entry="SasData01",
        metadata_file=local_reference_load("cansas1d_notitle.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("cansas1d_slit.xml"),
        entry="Test title",
        metadata_file=local_reference_load("cansas1d_slit.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("cansas1d_units.xml"),
        entry="Test title",
        metadata_file=local_reference_load("cansas1d_units.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("cansas_test.xml"),
        entry="ILL-D11 example1: 2A 5mM 0%D2O",
        metadata_file=local_reference_load("cansas_test.txt"),
        expected_values={},
    ),
    XmlTestCase(
        filename=local_data_load("cansas_test_modified.xml"),
        entry="ILL-D11 example1: 2A 5mM 0%D2O",
        metadata_file=local_reference_load("cansas_test_modified.txt"),
        expected_values={},
    ),
    # XmlTestCase(filename=local_data_load("cansas_xml_multisasentry_multisasdata.xml"), metadata_file=local_reference_load("cansas_xml_multisasentry_multisasdata.txt"), expected_values={}),
    XmlTestCase(
        filename=local_data_load("valid_cansas_xml.xml"),
        entry="80514main_1D_2.2_10.0",
        metadata_file=local_reference_load("valid_cansas_xml.txt"),
        expected_values={},
    ),
]


def join_actual_expected(
    actual: list[SasData], expected: dict[str, dict[int, dict[str, float]]]
) -> list[tuple[SasData, dict[int, dict[str, float]]]]:
    return_value = []
    for actual_datum in actual:
        matching_expected_datum = expected.get(actual_datum.name)
        if matching_expected_datum is None:
            continue
        return_value.append((actual_datum, matching_expected_datum))
    return return_value


def is_uncertainty(column: str) -> bool:
    for uncertainty_str in ["I", "Q", "Qx", "Qy"]:
        if column == "d" + uncertainty_str:
            return True
    return False


@pytest.mark.dataload
@pytest.mark.parametrize("test_case", test_cases)
def test_load_file(test_case: BaseTestCase):
    match test_case:
        case BulkAsciiTestCase():
            loaded_data = ascii_load_data(test_case.reader_params)
        case AsciiTestCase():
            if isinstance(test_case.reader_params, str):
                loaded_data = load_data_default_params(test_case.reader_params)[0]
            elif isinstance(test_case.reader_params, AsciiReaderParams):
                loaded_data = ascii_load_data(test_case.reader_params)[0]
            else:
                raise TypeError("Invalid type for reader_params.")
        case Hdf5TestCase():
            combined_data = hdf_load_data(test_case.filename)
            loaded_data = combined_data[test_case.entry]
        # TODO: Support SESANS
        case XmlTestCase():
            # Not bulk, so just assume we get one dataset.
            combined_data = xml_load_data(test_case.filename)
            loaded_data = combined_data[test_case.entry]
        case _:
            raise ValueError("Invalid loader")
    if isinstance(test_case, BulkAsciiTestCase):
        loaded_expected_pairs = join_actual_expected(loaded_data, test_case.expected_values)
        metadata_filenames = test_case.expected_metadata.keys()
    else:
        loaded_expected_pairs = [(loaded_data, test_case.expected_values)]
        metadata_filenames = [loaded_data.name]
    for loaded, expected in loaded_expected_pairs:
        for index, values in expected.items():
            for column, expected_value in values.items():
                if is_uncertainty(column):
                    assert loaded._data_contents[column[1::]]._variance[index] == pytest.approx(expected_value**2)
                else:
                    assert loaded._data_contents[column].value[index] == pytest.approx(expected_value)

    for filename in metadata_filenames:
        current_metadata_dict = test_case.expected_metadata.get(filename)
        current_datum = (
            next(filter(lambda d: d.name == filename, loaded_data)) if isinstance(loaded_data, list) else loaded_data
        )
        if current_metadata_dict is None:
            continue
        for metadata_key, value in current_metadata_dict.items():
            assert current_datum.metadata.raw.filter(metadata_key) == value

    if test_case.metadata_file is not None:
        with open(test_case.metadata_file, encoding="utf-8") as infile:
            expected = "".join(infile.readlines())
        keys = sorted([d for d in combined_data])
        assert "".join(combined_data[k].summary() for k in keys) == expected

    if test_case.round_trip:
        bio = io.BytesIO()
        SasData.save_h5(combined_data, bio)
        bio.seek(0)

        result = hdf_load_data(bio)
        bio.close()

        for name, entry in result.items():
            assert combined_data[name].metadata.title == entry.metadata.title
            assert combined_data[name].metadata.run == entry.metadata.run
            assert combined_data[name].metadata.definition == entry.metadata.definition
            assert combined_data[name].metadata.process == entry.metadata.process
            assert combined_data[name].metadata.instrument == entry.metadata.instrument
            assert combined_data[name].metadata.sample == entry.metadata.sample
            assert combined_data[name].ordinate.units == entry.ordinate.units
            assert np.all(combined_data[name].ordinate.value == entry.ordinate.value)
            assert combined_data[name].abscissae.units == entry.abscissae.units
            assert np.all(combined_data[name].abscissae.value == entry.abscissae.value)
