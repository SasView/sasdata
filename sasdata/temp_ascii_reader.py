#!/usr/bin/env python

from sasdata.ascii_reader_metadata import AsciiMetadataCategory, AsciiReaderMetadata, pairings, bidirectional_pairings
from sasdata.data import SasData
from sasdata.quantities.units import NamedUnit
from sasdata.quantities.quantity import NamedQuantity
from sasdata.quantities.accessors import AccessorTarget, Group
from sasdata.metadata import Metadata
from sasdata.data_backing import Dataset, Group
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import re
from os import path

class AsciiSeparator(Enum):
    Comma = 0,
    Whitespace = 1,
    Tab = 2

def initialise_metadata(filenames: list[str]) -> AsciiReaderMetadata:
    metadata = AsciiReaderMetadata()
    for filename in filenames:
        basename = path.basename(filename)
        metadata.filename_separator[basename] = '_'
        metadata.filename_specific_metadata[basename] = {}
    return metadata

# TODO: Turn them all of for now so the caller can turn one of them on. But is this the desired behaviour?
def initialise_separator_dict() -> dict[str, bool]:
    return {'Whitespace': False,
            'Comma': False,
            'Tab': False}

@dataclass
class AsciiReaderParams:
    filenames: list[str] # These will be the FULL file path. Will need to convert to basenames for some functions.
    columns: list[tuple[str, NamedUnit]]
    metadata: AsciiReaderMetadata = field(init=False)
    starting_line: int = 0
    excluded_lines: set[int] = field(default_factory=set)
    separator_dict: dict[str, bool] = field(default_factory=initialise_separator_dict)

    def __post__init__(self):
        self.metadata = initialise_metadata(self.filenames)

def split_line(separator_dict: dict[str, bool], line: str) -> list[str]:
    """Split a line in a CSV file based on which seperators the user has
    selected on the widget.

    """
    expr = ''
    for seperator, isenabled in separator_dict.items():
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

    return re.split(expr, line.strip())

# TODO: Implement error handling.
def load_quantities(params: AsciiReaderParams, filename: str) -> list[NamedQuantity]:
    with open(filename) as ascii_file:
        lines = ascii_file.readlines()
        arrays: list[np.ndarray] = []
        for _ in params.columns:
            arrays.append(np.zeros(len(lines) - params.starting_line))
        for i, current_line in enumerate(lines):
            if i < params.starting_line or current_line in params.excluded_lines:
                continue
            line_split = split_line(params.separator_dict, current_line)
            try:
                for j, token in enumerate(line_split):
                    # Sometimes in the split, there might be an extra column that doesn't need to be there (e.g. an empty
                    # string.) This won't convert to a float so we need to ignore it.
                    if j >= len(params.columns):
                        continue
                    # TODO: Data might not be floats. Maybe don't hard code this.
                    arrays[j][i - params.starting_line] = float(token)
            except ValueError:
                # If any of the lines contain non-numerical data, then this line can't be read in as a quantity so it
                # should be ignored entirely.
                print(f'Line {i + 1} skipped.')
                continue
    file_quantities = [NamedQuantity(name, arrays[i], unit) for i, (name, unit) in enumerate(params.columns)]
    return file_quantities

def metadata_to_data_backing(metadata: dict[str, AsciiMetadataCategory[str]]) -> Group:
    root_children = {}
    for top_level_key, top_level_item in metadata.items():
        children = {}
        for metadatum_name, metadatum in top_level_item.values.items():
            children[metadatum_name] = Dataset(metadatum_name, metadatum, {})
        # This is a special set which needs to live at the root of the group.
        # TODO: the 'other' name will probably need to change.
        if top_level_key == 'other':
            root_children = root_children | children
        else:
            group = Group(top_level_key, children)
            root_children[top_level_key] = group
    return Group('root', root_children)

def merge_uncertainties(quantities: list[NamedQuantity[list]]) -> list[NamedQuantity]:
    new_quantities = []
    error_quantity_names = pairings.values()
    for quantity in quantities:
        if quantity.name in error_quantity_names:
            continue
        pairing = bidirectional_pairings.get(quantity.name, '')
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

def load_data(params: AsciiReaderParams) -> list[SasData]:
    loaded_data: list[SasData] = []
    for filename in params.filenames:
        quantities = load_quantities(params, filename)
        metadata = metadata_to_data_backing(params.metadata.all_file_metadata(path.basename(filename)))
        loaded_data.append(SasData(filename, merge_uncertainties(quantities), metadata))
    return loaded_data
