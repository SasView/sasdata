from sasdata.quantities.units import Dimensions, NamedUnit, Unit, symbol_lookup
from re import findall

def split_unit_str(unit_str: str) -> list[str]:
    return findall(r'[A-Za-z]+|[-\d]+', unit_str)

def parse_single_unit(unit_str: str) -> tuple[Unit | None, str]:
    """Attempts to find a single unit for unit_str. Return this unit, and the remaining string in a tuple. If a unit
    cannot be parsed, the unit will be None, and the remaining string will be the entire unit_str"""
    current_unit = ''
    string_pos = 0
    for char in unit_str:
        potential_unit_str = current_unit + char
        potential_symbol = symbol_lookup.get(potential_unit_str, None)
        if potential_symbol is None:
            break
        string_pos += 1
        current_unit= potential_unit_str
    if current_unit == '':
        return (None, unit_str)
    remaining_str = unit_str[string_pos::]
    return (symbol_lookup[current_unit], remaining_str)

# Its probably useful to work out the unit first, and then later work out if a named unit exists for it. Hence why there
# are two functions.

def parse_unit(unit_str: str) -> Unit:
    # TODO: Not implemented. This is just to enable testing.
    return Unit(1, Dimensions())

def parse_named_unit(unit_str: str) -> NamedUnit:
    # TODO: Not implemented.
    return NamedUnit(1, Dimensions())
