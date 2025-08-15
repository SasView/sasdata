"""
Unit tests for the new recursive cansas reader
"""

import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from sasdata.data import SasData
from sasdata.dataset_types import one_dim
from sasdata.guess import guess_columns
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


def local_load(path: str):
    """Get local file path"""
    return os.path.join(os.path.dirname(__file__), path)


def local_data_load(path: str):
    return local_load(f"{os.path.join('data', path)}")


@dataclass(kw_only=True)
class BaseTestCase:
    expected_values: dict[int, dict[str, float]]
    expected_metadata: dict[str, Any] = field(default_factory=dict)


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


@dataclass(kw_only=True)
class Hdf5TestCase(BaseTestCase):
    filename: str


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
    pytest.param(
        Hdf5TestCase(
            filename=local_data_load("simpleexamplefile.h5"),
            expected_values={
                0: {"Q": 0.5488135039273248, "I": 0.6778165367962301},
                -1: {"Q": 0.004695476192547066, "I": 0.4344166255581208},
            },
        ),
        marks=pytest.mark.xfail(
            reason="Failing because of some Regex issue. The test looks correct, so this may be an issue with the HDF5 reader itself."
        ),
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
            loaded_data = hdf_load_data(test_case.filename)
        # TODO: Support SESANS
        case XmlTestCase():
            # Not bulk, so just assume we get one dataset.
            loaded_data = next(iter(xml_load_data(test_case.filename).items()))[1]
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
                uncertainty_handled = False
                for uncertainty_str in ["I", "Q", "Qx", "Qy"]:
                    if column == "d" + uncertainty_str:
                        uncertainty_handled = True
                        assert loaded._data_contents[uncertainty_str]._variance[index] == pytest.approx(
                            expected_value**2
                        )
                if not uncertainty_handled:
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
