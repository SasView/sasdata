from typing import Literal
import pytest
import os

from sasdata.temp_ascii_reader import (
    load_data,
    guess_params_from_filename,
    load_data_default_params,
    AsciiReaderParams,
)
from sasdata.ascii_reader_metadata import AsciiReaderMetadata, AsciiMetadataCategory
from sasdata.dataset_types import one_dim
from sasdata.quantities.units import per_angstrom, per_centimeter
from sasdata.guess import guess_columns

# TODO: These are using the private _data_contents temporarily. Later, there will be a public way of accessing these,
# and that should be used instead.

# TODO: Look into parameterizing this, although its not trivial due to the setup, and tests being a bit different.


def find(filename: str, locations: Literal["sasdataloader", "mumag"]) -> str:
    # This match statement is here in case we want to pull data out of other locations.
    match locations:
        case "sasdataloader":
            return os.path.join(
                os.path.dirname(__file__), "sasdataloader", "data", filename
            )
        case "mumag":
            return os.path.join(
                os.path.dirname(__file__),
                "mumag",
                "Nanoperm_perpendicular_Honecker_et_al",
                filename,
            )


def test_ascii_1():
    filename = find("ascii_test_1.txt", "sasdataloader")
    params = guess_params_from_filename(filename, one_dim)
    # Need to change the columns as they won't be right.
    # TODO: <ignore> unitless
    params.columns = [
        ("Q", per_angstrom),
        ("I", per_centimeter),
        ("dI", per_centimeter),
        ("<ignore>", None),
        ("<ignore>", None),
        ("<ignore>", None),
    ]
    loaded_data = load_data(params)[0]
    # Check the first, and last rows to see if they are correct.
    for name, datum in loaded_data._data_contents.items():
        match name:
            case "Q":
                assert datum.value[0] == pytest.approx(0.002618)
                assert datum.value[-1] == pytest.approx(0.0497)
            case "I":
                assert datum.value[0] == pytest.approx(0.02198)
                assert datum.value[-1] == pytest.approx(8.346)
            case "dI":
                assert datum.value[0] == pytest.approx(0.002704)
                assert datum.value[-1] == pytest.approx(0.191)


def test_ascii_2():
    filename = find("test_3_columns.txt", "sasdataloader")
    loaded_data = load_data_default_params(filename)[0]

    for name, datum in loaded_data._data_contents.items():
        match name:
            case "Q":
                assert datum.value[0] == pytest.approx(0)
                assert datum.value[-1] == pytest.approx(1.22449)
            case "I":
                assert datum.value[0] == pytest.approx(2.83954)
                assert datum.value[-1] == pytest.approx(7.47487)
            case "dI":
                assert datum.value[0] == pytest.approx(0.6)
                assert datum.value[-1] == pytest.approx(1.05918)


def test_ascii_2d():
    filename = find("detector_rectangular.DAT", "sasdataloader")
    # Make sure that the dataset type is guessed as 2D data.
    loaded_data = load_data_default_params(filename)[0]

    for name, datum in loaded_data._data_contents.items():
        match name:
            case "Qx":
                assert datum.value[0] == pytest.approx(-0.009160664)
                assert datum.value[-1] == pytest.approx(0.2908819)
            case "Qy":
                assert datum.value[0] == pytest.approx(-0.1683881)
                assert datum.value[-1] == pytest.approx(0.1634992)
            case "I":
                assert datum.value[0] == pytest.approx(16806.79)
                assert datum.value[-1] == pytest.approx(8147.779)
            case "dI":
                assert datum.value[0] == pytest.approx(0.01366757)
                assert datum.value[-1] == pytest.approx(0.05458562)


def test_mumag_metadata():
    filenames = [
        "1_33_1640_22.874115.csv",
        "1_33_1640_22.874115.csv",
        "2_42_1640_23.456895.csv",
        "3_61_1640_23.748285.csv",
        "4_103_1640_24.039675.csv",
        "5_312_1640_24.331065.csv",
        "6_1270_1640_24.331065.csv",
    ]
    param_filenames = []
    for filename in filenames:
        param_filenames.append(find(filename, "mumag"))

    metadata = AsciiReaderMetadata(
        master_metadata={
            "magnetic": AsciiMetadataCategory(
                values={
                    "counting_index": 0,
                    "applied_magnetic_field": 1,
                    "saturation_magnetization": 2,
                    "demagnetizing_field": 3,
                }
            ),
        },
    )
    params = AsciiReaderParams(
        filenames=param_filenames,
        columns=[(column, per_angstrom) for column in guess_columns(3, one_dim)],
        separator_dict={"Comma": True},
        metadata=metadata,
    )
    data = load_data(params)
    for datum in data:
        match datum.name:
            case "1_33_1640_22.874115.csv":
                assert datum.metadata.raw.filter("counting_index") == ["1"]
                assert datum.metadata.raw.filter("applied_magnetic_field") == ["33"]
                assert datum.metadata.raw.filter("saturation_magnetization") == ["1640"]
                assert datum.metadata.raw.filter("demagnetizing_field") == ["22.874115"]
            case "6_1270_1640_24.331065.csv":
                assert datum.metadata.raw.filter("counting_index") == ["6"]
                assert datum.metadata.raw.filter("applied_magnetic_field") == ["1270"]
                assert datum.metadata.raw.filter("saturation_magnetization") == ["1640"]
                assert datum.metadata.raw.filter("demagnetizing_field") == ["24.331065"]
