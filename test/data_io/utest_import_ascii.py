from typing import Literal

import pytest

from sasdata.ascii_reader_metadata import AsciiMetadataCategory, AsciiReaderMetadata
from sasdata.data_io.importers.import_ascii import (
    AsciiReaderParams,
    guess_params_from_filename,
    load_data,
    load_data_default_params,
)
from sasdata.dataset_types import one_dim, two_dim
from sasdata.guess import guess_column_count, guess_columns, guess_dataset_type
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
    loaded_data = load_data(params)[0]
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
    loaded_data = load_data_default_params(filename)[0]
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
    loaded_data = load_data_default_params(filename)[0]
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
    data = load_data(params)
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


THREE_COLUMN_DATA = "0.01 100 1\n0.02 90 2\n0.03 80 3\n"


def test_guess_params_honours_requested_dataset_type(tmp_path):
    # guess_dataset_type() classifies every .dat file as 2D. An explicit request
    # for 1D must win, otherwise the 1D columns are checked against the 2D type.
    filename = tmp_path / "data.dat"
    filename.write_text(THREE_COLUMN_DATA)
    params = guess_params_from_filename(str(filename), one_dim)
    assert params.dataset_type is one_dim
    data = load_data(params)[0]
    assert data.dataset_type is one_dim
    assert data["Q"].value == pytest.approx([0.01, 0.02, 0.03])
    assert data["I"].value == pytest.approx([100, 90, 80])


# Qx wraps around at the end of each Qy row, as in 2D detector files.
GRID_DATA = "-0.1 -0.1 5\n0.0 -0.1 6\n0.1 -0.1 7\n-0.1 0.0 8\n0.0 0.0 9\n0.1 0.0 10\n"


def test_guess_params_with_guessed_dataset_type(tmp_path):
    # Passing guess_dataset_type() uses the guessed type (as load_data_default_params does).
    filename = tmp_path / "data.dat"
    filename.write_text(GRID_DATA)
    params = guess_params_from_filename(str(filename), guess_dataset_type(str(filename)))
    assert params.dataset_type is two_dim


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


def test_dat_file_with_monotonic_q_is_guessed_as_1d(tmp_path):
    filename = tmp_path / "data.dat"
    filename.write_text(THREE_COLUMN_DATA)
    assert guess_dataset_type(str(filename)) is one_dim


def test_non_dat_file_is_always_guessed_as_1d(tmp_path):
    filename = tmp_path / "data.txt"
    filename.write_text(GRID_DATA)
    assert guess_dataset_type(str(filename)) is one_dim


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("detector_square.dat", two_dim),
        ("exp18_14_igor_2dqxqy.dat", two_dim),
        ("nans_in_2d_data.DAT", two_dim),  # has a blank line after every data row
        ("nans_in_1d_data.dat", one_dim),
    ],
)
def test_dat_dataset_type_guess_on_real_files(filename, expected):
    assert guess_dataset_type(find(filename, "sasdataloader")) is expected


# Layout of an Anton Paar SAXSess export: a title, a line of integers and two
# lines of instrument parameters before the Q, I, dI rows. Classic Mac line endings.
SAXSESS_DATA = "\r".join(
    [
        "",
        "SAXS BOX",
        "       4        30         0         0         0         0         0         0",
        "  0.000000E+00   0.261200E+03   0.000000E+00   0.100000E+01   0.154200E+00",
        "  0.100000E+01   0.155999E+01   0.000000E+00   0.000000E+00   0.000000E+00",
        "  0.754396E-01   0.975440E+00   0.938981E+02",
        "  0.150879E+00   0.975745E+00   0.991879E+02",
        "  0.226319E+00   0.975473E+00   0.767819E+02",
        "  0.301758E+00   0.974517E+00   0.131319E+03",
        "",
    ]
)


def test_saxsess_header_lines_are_not_read_as_data(tmp_path):
    filename = tmp_path / "beam_profile.DAT"
    filename.write_bytes(SAXSESS_DATA.encode())
    data = load_data_default_params(str(filename))[0]
    assert data.dataset_type is one_dim
    assert data["Q"].value == pytest.approx([0.0754396, 0.150879, 0.226319, 0.301758])
    assert data["I"].value == pytest.approx([0.975440, 0.975745, 0.975473, 0.974517])


def test_blank_lines_between_rows_do_not_change_the_column_count():
    rows = [["0.01", "100", "1"], [""], ["0.02", "90", "2"], [""], ["0.03", "80", "3"], [""]]
    assert guess_column_count(rows, 0) == 3
