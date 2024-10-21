#!/usr/bin/env python

from sasdata.data import SasData
from sasdata.quantities.units import NamedUnit
from sasdata.quantities.quantity import NamedQuantity
from enum import Enum
import numpy as np
import re

class AsciiSeparator(Enum):
    Comma = 0,
    Whitespace = 1,
    Tab = 2

# TODO: This has mostly been copied from the ASCII dialog but really the widget should use the implementation here.
def split_line(separator_dict: dict[str, bool], line: str) -> list[str]:
    """Split a line in a CSV file based on which seperators the user has
    selected on the widget.

    """
    expr = ''
    for seperator, isenabled in separator_dict:
        if isenabled:
            if expr != r'':
                expr += r'|'
            match seperator:
                case 'Comma':
                    seperator_text = r','
                case 'Whitespace':
                    seperator_text = r'\s+'
                case 'Tab':
                    seperator_text = r'\t'
            expr += seperator_text

    return re.split(expr, line)

# TODO: Implement error handling.
def load_quantities(filename: str, starting_line: int, columns: list[tuple[str, NamedUnit]], separators: list[AsciiSeparator], excluded_lines: list[int], separator_dict: dict[str, bool]) -> list[NamedQuantity]:
    with open(filename) as ascii_file:
        lines = ascii_file.readlines()
        arrays: list[np.ndarray] = []
        for _ in columns:
            arrays.append(np.zeros(len(lines)))
        for i, current_line in enumerate(lines):
            if i < starting_line or current_line in excluded_lines:
                continue
            line_split = split_line(separator_dict)
            for j, token in enumerate(line_split):
                # TODO: Data might not be floats. Maybe don't hard code this.
                arrays[i][j] = float(token)
    quantities = [NamedQuantity(name, arrays[i], unit) for i, (name, unit) in enumerate(columns)]
    return quantities

def load_data(filename: str, starting_line: int, columns: list[tuple[str, NamedUnit]], separators: list[AsciiSeparator], excluded_lines: list[int], separator_dict: dict[str, bool]) -> SasData:
    quantities = load_quantities(filename, starting_line, columns, separators, excluded_lines, separator_dict)
    # Name is placeholder; this might come from the metadata.
    return SasData(filename, quantities, None)
