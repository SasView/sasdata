from typing import Literal

import pytest

from sasdata.ascii_reader_metadata import AsciiMetadataCategory, AsciiReaderMetadata
from sasdata.data_io.importers.import_ascii import (
    AsciiImporter,
    AsciiReaderParams,
    guess_params_from_filename,
)
from sasdata.dataset_types import one_dim
from sasdata.guess import guess_columns
from sasdata.quantities.units import per_angstrom, per_centimeter
from test import local_data_finder, local_mumag_finder

# TODO: These are using the private _data_contents temporarily. Later, there will be a public way of accessing these,
# and that should be used instead.

# TODO: Look into parameterizing this, although its not trivial due to the setup, and tests being a bit different.


def find(filename: str, locations: Literal["sasdataloader", "mumag"]) -> str:
    # This match statement is here in case we want to pull data out of other locations.
    match locations:
        case "sasdataloader":
            return local_data_finder(filename)
        case "mumag":
            return local_mumag_finder(filename)


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
    loaded_data = list(AsciiImporter(params=params).load_data().values())[0]
    assert len(loaded_data._data_contents) > 0
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
    loaded_data = AsciiImporter().load_data(filename)["test_3_columns.txt"]
    assert len(loaded_data._data_contents) > 0
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


@pytest.mark.xfail(
    reason="Guesses for 2D ASCII files are currently wrong, so the data loaded won't be correct."
)
def test_ascii_2d():
    filename = find("detector_rectangular.DAT", "sasdataloader")
    # Make sure that the dataset type is guessed as 2D data.
    loaded_data = list(AsciiImporter().load_data(filename).values())[0]
    assert len(loaded_data._data_contents) > 0

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
        "Nanoperm_perpendicular_Honecker_et_al/1_33_1640_22.874115.csv",
        "Nanoperm_perpendicular_Honecker_et_al/2_42_1640_23.456895.csv",
        "Nanoperm_perpendicular_Honecker_et_al/3_61_1640_23.748285.csv",
        "Nanoperm_perpendicular_Honecker_et_al/4_103_1640_24.039675.csv",
        "Nanoperm_perpendicular_Honecker_et_al/5_312_1640_24.331065.csv",
        "Nanoperm_perpendicular_Honecker_et_al/6_1270_1640_24.331065.csv",
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
        # The MuMag files are tab separated.
        separator_dict={"Comma": True, "Tab": True},
        metadata=metadata,
    )
    data = AsciiImporter(params=params).load_data()
    assert len(data) == len(filenames)
    for datum in data:
        # Each file has 60 rows of Q, I, dI data.
        assert len(datum._data_contents["Q"].value) == 60
        assert len(datum._data_contents["I"].value) == 60
        match datum.name:
            case "1_33_1640_22.874115.csv":
                assert datum._data_contents["Q"].value[0] == pytest.approx(0.02732)
                assert datum.metadata.raw.filter("counting_index") == ["1"]
                assert datum.metadata.raw.filter("applied_magnetic_field") == ["33"]
                assert datum.metadata.raw.filter("saturation_magnetization") == ["1640"]
                assert datum.metadata.raw.filter("demagnetizing_field") == ["22"]
            case "6_1270_1640_24.331065.csv":
                assert datum.metadata.raw.filter("counting_index") == ["6"]
                assert datum.metadata.raw.filter("applied_magnetic_field") == ["1270"]
                assert datum.metadata.raw.filter("saturation_magnetization") == ["1640"]
                assert datum.metadata.raw.filter("demagnetizing_field") == ["24"]


# Tests for excluded lines, and for lines that are skipped when loading (issue #261).

ALL_SEPARATORS = {"Comma": True, "Whitespace": True, "Tab": True}
Q_I_DI_COLUMNS = [("Q", per_angstrom), ("I", per_centimeter), ("dI", per_centimeter)]


def load_text(tmp_path, text: str, columns=Q_I_DI_COLUMNS, **kwargs) -> dict:
    """Write text to a temporary ASCII file, load it with the given columns (Q, I, dI by default), and return the
    loaded quantities."""
    filename = tmp_path / "data.txt"
    filename.write_text(text)
    params = AsciiReaderParams(
        filenames=[str(filename)],
        columns=columns,
        separator_dict=ALL_SEPARATORS,
        **kwargs,
    )
    return load_data(params)[0]._data_contents


# Line numbers in the comments are the (0-based) indices used for excluded_lines.
HEADER_AND_DATA = (
    "Q I dI\n"  # 0
    "0.01 100 1\n"  # 1
    "0.02 90 2\n"  # 2
    "0.03 80 3\n"  # 3
    "0.04 70 4\n"  # 4
)


def test_no_excluded_lines(tmp_path):
    data = load_text(tmp_path, HEADER_AND_DATA, starting_line=1)
    assert data["Q"].value == pytest.approx([0.01, 0.02, 0.03, 0.04])
    assert data["I"].value == pytest.approx([100, 90, 80, 70])


def test_excluded_line_is_removed(tmp_path):
    data = load_text(tmp_path, HEADER_AND_DATA, starting_line=1, excluded_lines={2})
    # The excluded row must be removed, not replaced by zeros.
    assert data["Q"].value == pytest.approx([0.01, 0.03, 0.04])
    assert data["I"].value == pytest.approx([100, 80, 70])


def test_multiple_excluded_lines(tmp_path):
    data = load_text(tmp_path, HEADER_AND_DATA, starting_line=1, excluded_lines={1, 4})
    assert data["Q"].value == pytest.approx([0.02, 0.03])
    assert data["I"].value == pytest.approx([90, 80])


def test_all_data_lines_excluded(tmp_path):
    data = load_text(tmp_path, HEADER_AND_DATA, starting_line=1, excluded_lines={1, 2, 3, 4})
    assert len(data["Q"].value) == 0
    assert len(data["I"].value) == 0


def test_excluded_lines_before_starting_line_are_ignored(tmp_path):
    # Excluding a header line (which is already skipped by the starting line) must not affect the data.
    data = load_text(tmp_path, HEADER_AND_DATA, starting_line=1, excluded_lines={0})
    assert data["Q"].value == pytest.approx([0.01, 0.02, 0.03, 0.04])


def test_excluded_line_numbers_count_from_start_of_file(tmp_path):
    # Line numbers in excluded_lines refer to the whole file, not to the position after the starting line.
    text = "# comment\n# comment\nQ I dI\n0.01 100 1\n0.02 90 2\n0.03 80 3\n"
    data = load_text(tmp_path, text, starting_line=3, excluded_lines={4})
    assert data["Q"].value == pytest.approx([0.01, 0.03])


def test_uncertainties_stay_aligned_with_excluded_lines(tmp_path):
    # dI is merged into I as its uncertainty, so it must lose the same rows as I.
    data = load_text(tmp_path, HEADER_AND_DATA, starting_line=1, excluded_lines={2})
    _, standard_error = data["I"].in_units_of_with_standard_error(per_centimeter)
    assert standard_error == pytest.approx([1, 3, 4])


def test_unreadable_lines_are_skipped_not_zeroed(tmp_path):
    text = (
        "Q I dI\n"
        "0.01 100 1\n"
        "\n"  # blank line
        "not a data line\n"
        "0.02 90 2\n"
        "0.03 80\n"  # too few columns
        "0.04 70 4\n"
        "\n"  # trailing blank line
    )
    data = load_text(tmp_path, text, starting_line=1)
    assert data["Q"].value == pytest.approx([0.01, 0.02, 0.04])
    assert data["I"].value == pytest.approx([100, 90, 70])
    assert 0 not in data["Q"].value


def test_extra_columns_are_ignored(tmp_path):
    text = "0.01 100 1 0.001 5\n0.02 90 2 0.002 5\n"
    data = load_text(tmp_path, text)
    assert data["Q"].value == pytest.approx([0.01, 0.02])
    assert data["I"].value == pytest.approx([100, 90])


# Tests for ignored columns (issue #263).

IGNORE = ("<ignore>", None)


def test_ignored_column_in_the_middle(tmp_path):
    columns = [("Q", per_angstrom), IGNORE, ("I", per_centimeter), ("dI", per_centimeter)]
    data = load_text(tmp_path, "0.01 999 100 1\n0.02 999 90 2\n", columns=columns)
    assert data["Q"].value == pytest.approx([0.01, 0.02])
    assert data["I"].value == pytest.approx([100, 90])
    _, standard_error = data["I"].in_units_of_with_standard_error(per_centimeter)
    assert standard_error == pytest.approx([1, 2])


def test_column_without_unit_is_ignored(tmp_path):
    # A column with no unit isn't loaded, but still takes up its position in the line.
    columns = [("Q", per_angstrom), ("dQ", None), ("I", per_centimeter), ("dI", per_centimeter)]
    data = load_text(tmp_path, "0.01 999 100 1\n0.02 999 90 2\n", columns=columns)
    assert "dQ" not in data
    assert data["Q"].value == pytest.approx([0.01, 0.02])
    assert data["I"].value == pytest.approx([100, 90])
