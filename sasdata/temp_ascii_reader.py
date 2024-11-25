#!/usr/bin/env python

from sasdata.data import SasData
from sasdata.quantities.units import NamedUnit
from sasdata.quantities.quantity import NamedQuantity
from sasdata.quantities.accessors import AccessorTarget, Group
from sasdata.metadata import Metadata
from sasdata.data_backing import Dataset, Group
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
            arrays.append(np.zeros(len(lines) - params.starting_line))
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
                arrays[j][i - params.starting_line] = float(token)
    quantities = [NamedQuantity(name, arrays[i], unit) for i, (name, unit) in enumerate(params.columns)]
    return quantities

# TODO: idk if metadata dict is gonna stay flat like this. May need to change later.
def metadata_dict_to_data_backing(metadata_dict: dict[str, dict[str, str]]) -> dict[str, Dataset | Group]:
    root_children = {}
    for top_level_key, top_level_item in metadata_dict.items():
        children = {}
        for metadatum_name, metadatum in top_level_item.items():
            children[metadatum_name] = Dataset(metadatum_name, metadatum, {})
        # This is a special set which needs to live at the root of the group.
        # TODO: the 'other' name will probably need to change.
        if top_level_key == 'other':
            root_children = root_children | children
        else:
            group = Group(top_level_key, children)
            root_children[top_level_key] = group
    return Group('root', root_children)

# TODO: There may be a better place for this.
# pairings = [('I', 'Idev')] # TODO: fill later.
pairings = {'I': 'dI'}

def merge_uncertainties(quantities: list[NamedQuantity[list]]) -> list[NamedQuantity]:
    new_quantities = []
    error_quantity_names = pairings.values()
    for quantity in quantities:
        if quantity in error_quantity_names:
            continue
        pairing = pairings.get(quantity.name, '')
        error_quantity = None
        for other_quantity in quantities:
            if other_quantity.name == pairing:
                error_quantity = other_quantity
        if not error_quantity is None:
            to_add = quantity.with_standard_error(error_quantity)
        else:
            to_add = quantity
        new_quantities.append(to_add)
    return new_quantities

def load_data(params: AsciiReaderParams) -> SasData:
    quantities = load_quantities(params)
    # Name is placeholder; this might come from the metadata.
    metadata = metadata_dict_to_data_backing(params.raw_metadata)
    return SasData(params.filename, merge_uncertainties(quantities), metadata)
