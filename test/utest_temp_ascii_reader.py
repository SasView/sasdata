from typing import Literal
import pytest
import os

from sasdata.guess import guess_column_count, guess_columns
from sasdata.temp_ascii_reader import (
    load_data,
    AsciiReaderParams,
    guess_params_from_filename,
)
from sasdata.dataset_types import one_dim
from sasdata.quantities.units import per_angstrom, per_centimeter

# TODO: These are using the private _data_contents temporarily. Later, there will be a public way of accessing these,
# and that should be used instead.

# TODO: Look into parameterizing this, although its not trivial due to the setup, and tests being a bit different.


def find(filename: str, locations: Literal["sasdataloader"]) -> str:
    # This match statement is here in case we want to pull data out of other locations.
    match locations:
        case "sasdataloader":
            return os.path.join(
                os.path.dirname(__file__), "sasdataloader", "data", filename
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
    params = guess_params_from_filename(filename, one_dim)
    loaded_data = load_data(params)[0]

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
