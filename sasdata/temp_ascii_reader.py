#!/usr/bin/env python

from enum import Enum

from sasdata.data import SasData
from sasdata.quantities.units import NamedUnit


class AsciiSeparator(Enum):
    Comma = (0,)
    Whitespace = (1,)
    Tab = 2


def load_data(filename: str, starting_line: int, columns: list[tuple[str, NamedUnit]], separators: list[AsciiSeparator]) -> list[SasData]:
    raise NotImplementedError()
