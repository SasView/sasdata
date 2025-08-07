import re
from dataclasses import dataclass, field, replace
from enum import Enum
from os import path

import numpy as np

from sasdata.ascii_reader_metadata import (
    AsciiMetadataCategory,
    AsciiReaderMetadata,
    bidirectional_pairings,
    pairings,
)
from sasdata.data import SasData
from sasdata.dataset_types import DatasetType, one_dim, unit_kinds
from sasdata.default_units import get_default_unit
from sasdata.guess import (
    guess_column_count,
    guess_columns,
    guess_dataset_type,
    guess_starting_position,
)
from sasdata.metadata import Metadata, MetaNode
from sasdata.quantities.quantity import Quantity
from sasdata.quantities.units import NamedUnit

from sasdata.data import SasData
from sasdata.quantities.units import NamedUnit
from enum import Enum

class AsciiSeparator(Enum):
    Comma = 0,
    Whitespace = 1,
    Tab = 2

def load_data(filename: str, starting_line: int, columns: list[tuple[str, NamedUnit]], separators: list[AsciiSeparator]) -> list[SasData]:
    raise NotImplementedError()
