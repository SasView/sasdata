import re

import numpy as np
from scipy.stats import mode

from sasdata.dataset_types import DatasetType, one_dim, two_dim


def _is_blank(line: list[str]) -> bool:
    return len(line) == 0 or line == [""]


def guess_column_count(split_csv: list[list[str]], starting_pos: int) -> int:
    """Guess the amount of columns present in the data.

    Blank lines are not counted, so files with an empty line between data rows
    are not guessed as having a single column.
    """
    candidate_lines = [line for line in split_csv[starting_pos::] if not _is_blank(line)]
    if not candidate_lines:
        return 0
    return int(mode([len(line) for line in candidate_lines]).mode)


def guess_columns(col_count: int, dataset_type: DatasetType) -> list[str]:
    """Based on the amount of columns specified in col_count, try to find a set
    of columns that best matchs the dataset_type.

    """
    # Ideally we want an exact match but if the ordering is bigger than the col
    # count then we can accept that as well.
    for order_list in dataset_type.expected_orders:
        if (
            len(order_list) >= col_count
            or order_list == dataset_type.expected_orders[-1]
        ):
            return_value = order_list[:]
            # If we have any extra columns than expected, then we'll just ignore them.
            excess = col_count - len(order_list)
            for _ in range(excess):
                return_value.append("<ignore>")
            return return_value

    return dataset_type.expected_orders[-1]


symbols_to_ignore = [".", "-", "+", "e", "E"]


def _is_numeric_row(row: list[str]) -> bool:
    for column in row:
        amended_column = column
        for symbol in symbols_to_ignore:
            amended_column = amended_column.replace(symbol, "")
        if not amended_column.isdigit():
            return False
    return True


def guess_starting_position(split_csv: list[list[str]]) -> int:
    """Find the first line of the data block.

    The data block is made of numeric lines with the most common column count.
    Numeric header lines with a different column count (for example the
    parameter lines of an Anton Paar SAXSess file) are skipped. If no line is
    numeric, then just return 0 as the starting position.

    """
    numeric_rows = [(i, len(row)) for i, row in enumerate(split_csv) if not _is_blank(row) and _is_numeric_row(row)]
    if not numeric_rows:
        return 0
    data_column_count = int(mode([count for _, count in numeric_rows]).mode)
    return next(i for i, count in numeric_rows if count == data_column_count)


def _split_all_separators(line: str) -> list[str]:
    # Same splitting as the ASCII importer with every separator enabled.
    return re.split(r"\s+|,|\t", line.strip())


def guess_dataset_type(filename: str, split_csv: list[list[str]] | None = None) -> DatasetType:
    """Guess whether an ASCII file holds 1D or 2D data.

    Only ``.dat`` files can be 2D. They are guessed from their contents: 2D data
    has at least three columns, and its first column (Qx) repeats or wraps
    around, while the Q column of 1D data is strictly monotonic.

    :param filename: The file to guess for.
    :param split_csv: The file's lines, already split. Read from ``filename`` if not given.
    """
    if not filename.lower().endswith(".dat"):
        return one_dim
    if split_csv is None:
        with open(filename) as file:
            split_csv = [_split_all_separators(line) for line in file.readlines()]
    start = guess_starting_position(split_csv)
    column_count = guess_column_count(split_csv, start)
    if column_count < 3:
        return one_dim
    first_column = []
    for row in split_csv[start:]:
        if len(row) < column_count:
            continue
        try:
            first_column.append(float(row[0]))
        except ValueError:
            continue
    first_column = np.array(first_column)
    first_column = first_column[np.isfinite(first_column)]
    if len(first_column) < 2:
        return two_dim
    steps = np.diff(first_column)
    if np.all(steps > 0) or np.all(steps < 0):
        return one_dim
    return two_dim
