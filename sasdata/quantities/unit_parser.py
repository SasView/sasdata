from sasdata.quantities.units import Dimensions, NamedUnit
from re import findall

def split_unit_str(unit_str: str) -> list[str]:
    return findall(r'[A-Za-z]+|[-\d]+', unit_str)

def parse_unit(unit_str: str) -> NamedUnit:
    # TODO: Not implemented. This is just to enable testing.
    return NamedUnit(1, Dimensions())
