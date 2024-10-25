#!/usr/bin/env python

from sasdata.data import SasData
from sasdata.quantities.units import NamedUnit
from sasdata.quantities.quantity import NamedQuantity
from sasdata.quantities.accessors import AccessorTarget, Group
from sasdata.metadata import Metadata
from enum import Enum
from dataclasses import dataclass
import numpy as np
import re

class AsciiSeparator(Enum):
    Comma = 0,
    Whitespace = 1,
    Tab = 2

@dataclass
class AsciiReaderParams:
    filename: str
    starting_line: int
    columns: list[tuple[str, NamedUnit]]
    # sepearators: list[AsciiSeparator]
    excluded_lines: set[int]
    separator_dict: dict[str, bool]
    # The value of the metadatum will need to be parsed based on what it actually is.
    raw_metadata: dict[str, str]


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
def load_quantities(params: AsciiReaderParams) -> list[NamedQuantity]:
    with open(params.filename) as ascii_file:
        lines = ascii_file.readlines()
        arrays: list[np.ndarray] = []
        for _ in params.columns:
            arrays.append(np.zeros(len(lines)))
        for i, current_line in enumerate(lines):
            if i < params.starting_line or current_line in params.excluded_lines:
                continue
            line_split = split_line(params.separator_dict, current_line)
            for j, token in enumerate(line_split):
                # Sometimes in the split, there might be an extra column that doesn't need to be there (e.g. an empty
                # string.) This won't convert to a float so we need to ignore it.
                if j >= len(params.columns):
                    continue
                # TODO: Data might not be floats. Maybe don't hard code this.
                arrays[j][i] = float(token)
    quantities = [NamedQuantity(name, arrays[i], unit) for i, (name, unit) in enumerate(params.columns)]
    return quantities

def load_metadata(params: AsciiReaderParams) -> Group:
    root_group = Group('root', {})
    # TODO: Actually fill this metadata in based on params.
    return root_group

def load_data(params: AsciiReaderParams) -> SasData:
    quantities = load_quantities(params)
    # Name is placeholder; this might come from the metadata.
    return SasData(params.filename, quantities, load_metadata(params)))
