from sasdata.quantities.units import Dimensions, NamedUnit, Unit, symbol_lookup, unit_groups, UnitGroup
from re import findall, fullmatch

# TODO: This shouldn't be in this file but I don't want to edit Lucas' code before he is finished.

all_units_groups = [group.units for group in unit_groups.values()]
all_units: list[NamedUnit] = []
for group in all_units_groups:
    all_units.extend(group)

def multiply_dimensions(dimensions_1: Dimensions, dimensions_2: Dimensions) -> Dimensions:
    """Multiply each dimension in dimensions_1 with the same dimension in dimensions_2"""
    return Dimensions(
        length=dimensions_1.length * dimensions_2.length,
        time=dimensions_1.time * dimensions_2.time,
        mass=dimensions_1.mass * dimensions_2.mass,
        current=dimensions_1.current * dimensions_2.current,
        temperature=dimensions_1.temperature * dimensions_2.temperature,
        moles_hint=dimensions_1.moles_hint * dimensions_2.moles_hint,
        angle_hint=dimensions_1.angle_hint * dimensions_2.angle_hint
    )

def combine_units(unit_1: Unit, unit_2: Unit):
    """Combine unit_1, and unit_2 into one unit."""
    return Unit(unit_1.scale * unit_2.scale, unit_1.dimensions * unit_2.dimensions)

def split_unit_str(unit_str: str) -> list[str]:
    """Separate the letters from the numbers in unit_str"""
    return findall(r'[A-Za-z]+|[-\d]+|/', unit_str)

def validate_unit_str(unit_str: str) -> bool:
    """Validate whether unit_str is valid. This doesn't mean that the unit specified in unit_str exists but rather it
    only consists of letters, and numbers as a unit string should."""
    return not fullmatch(r'[A-Za-z1-9\-\+/]+', unit_str) is None

def parse_single_unit(unit_str: str, unit_group: UnitGroup | None = None, longest_unit: bool = True) -> tuple[Unit | None, str]:
    """Attempts to find a single unit for unit_str. Return this unit, and the remaining string in a tuple. If a unit
    cannot be parsed, the unit will be None, and the remaining string will be the entire unit_str.

    The shortest_unit parameter specifies how to resolve ambiguities. If it is true, then it will parse the longest unit
    available. Otherwise, it will stop parsing as soon as it has found any unit.

    If unit_group is set, it will only try to parse units within that group. This is useful for resolving ambiguities.
    """
    current_unit = ''
    string_pos = 0
    if unit_group is None:
        lookup_dict = symbol_lookup
    else:
        lookup_dict = dict([(name, unit) for name, unit in symbol_lookup.items() if unit in unit_group.units])
    for next_char in unit_str:
        potential_unit_str = current_unit + next_char
        potential_symbols = [symbol for symbol in lookup_dict.keys() if symbol.startswith(potential_unit_str)]
        if len(potential_symbols) == 0:
            break
        string_pos += 1
        current_unit= potential_unit_str
        if not longest_unit and current_unit in lookup_dict.keys():
            break
    if current_unit == '':
        return (None, unit_str)
    remaining_str = unit_str[string_pos::]
    return (lookup_dict[current_unit], remaining_str)

def parse_unit_strs(unit_str: str, current_units: list[Unit] | None=None, longest_unit: bool = True) -> list[Unit]:
    """Recursively parse units from unit_str until no more characters are present."""
    if current_units is None:
        current_units = []
    if unit_str == '':
        return current_units
    parsed_unit, remaining_str = parse_single_unit(unit_str, longest_unit=longest_unit)
    if parsed_unit is not None:
        current_units += [parsed_unit]
        return parse_unit_strs(remaining_str, current_units, longest_unit)
    else:
        raise ValueError(f'Could not interpret {remaining_str}')

def unit_power(to_modify: Unit, power: int):
    """Raise to_modify to power"""
    # FIXME: This is horrible but I'm not sure how to fix this without changing the Dimension class itself.
    dimension_multiplier = Dimensions(power, power, power, power, power, power, power)
    scale_multiplier = 1 if power > 0 else -1
    return Unit(to_modify.scale ** scale_multiplier, multiply_dimensions(to_modify.dimensions, dimension_multiplier))


# Its probably useful to work out the unit first, and then later work out if a named unit exists for it. Hence why there
# are two functions.

def parse_unit_stack(unit_str: str, longest_unit: bool = True) -> list[Unit]:
    """Split unit_str into a stack of parsed units."""
    unit_stack: list[Unit] = []
    split_str = split_unit_str(unit_str)
    inverse_next_unit = False
    for token in split_str:
        try:
            if token == '/':
                inverse_next_unit = True
                continue
            power = int(token)
            to_modify = unit_stack[-1]
            modified = unit_power(to_modify, power)
            unit_stack[-1] = modified
        except ValueError:
            new_units = parse_unit_strs(token, None, longest_unit)
            if inverse_next_unit:
                # TODO: Assume the power is going to be -1. This might not be true.
                power = -1
                new_units[0] = unit_power(new_units[0], power)
            unit_stack += new_units
        # This error will happen if it tries to read a modifier but there are no units on the stack. We will just have
        # to ignore it. Strings being parsed shouldn't really have it anyway (e.g. -1m).
        except IndexError:
            pass
    return unit_stack

def parse_unit(unit_str: str, longest_unit: bool = True) -> Unit:
    """Parse unit_str into a unit."""
    try:
        if not validate_unit_str(unit_str):
            raise ValueError('unit_str contains forbidden characters.')
        parsed_unit = Unit(1, Dimensions())
        unit_stack = parse_unit_stack(unit_str, longest_unit)
        for unit in unit_stack:
            parsed_unit = combine_units(parsed_unit, unit)
        return parsed_unit
    except KeyError:
        raise ValueError('Unit string contains an unrecognised pattern.')

def parse_unit_from_group(unit_str: str, from_group: UnitGroup) -> Unit | None:
    """Tries to use the given unit group to resolve ambiguities. Parse a unit twice with different options, and returns
    whatever conforms to the unit group."""
    longest_parsed_unit = parse_unit(unit_str, True)
    shortest_parsed_unit = parse_unit(unit_str, False)
    if longest_parsed_unit in from_group.units:
        return longest_parsed_unit
    elif shortest_parsed_unit in from_group.units:
        return shortest_parsed_unit
    else:
        return None

def parse_named_unit(unit: str | Unit) -> NamedUnit:
    """Parses unit into a named unit. Parses unit into a Unit if it is not already, and then finds an equivaelent named
    unit. Please note that this might not be the expected unit from the string itself. E.g. 'kgm/2' will become
    newtons."""
    if isinstance(unit, str):
        generic_unit = parse_unit(unit)
    elif isinstance(unit, Unit):
        generic_unit = unit
    else:
        raise ValueError('Unit must be a string, or Unit')
    for named_unit in all_units:
        if named_unit == generic_unit:
            return named_unit
    raise ValueError('A named unit does not exist for this unit.')

def parse_named_unit_from_group(unit_str: str, from_group: UnitGroup) -> NamedUnit:
    """Parses unit_str into a named unit. The named unit found must be part of from_group. If two units are found, the
    unit that is present in from_group is returned. This is useful in cases of ambiguities."""
    parsed_unit = parse_unit_from_group(unit_str, from_group)
    if parsed_unit is None:
        raise ValueError('That unit cannot be parsed from the specified group.')
    return parse_named_unit(parsed_unit)

if __name__ == "__main__":
    to_parse = input('Enter a unit to parse: ')
    try:
        generic_unit = parse_unit(to_parse)
        print(f'Generic Unit: {generic_unit}')
        named_unit = parse_named_unit(generic_unit)
        print(f'Named Unit: {named_unit}')
    except ValueError:
        print('There is no named unit available.')
