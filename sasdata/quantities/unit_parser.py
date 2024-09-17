from sasdata.quantities.units import Dimensions, NamedUnit, Unit, symbol_lookup, unit_groups
from re import findall

# TODO: This shouldn't be in this file but I don't want to edit Lucas' code before he is finished.

all_units_groups = [group.units for group in unit_groups.values()]
all_units: list[NamedUnit] = []
for group in all_units_groups:
    all_units.extend(group)

def multiply_dimensions(dimensions_1: Dimensions, dimensions_2: Dimensions) -> Dimensions:
    return Dimensions(
        length=dimensions_1.length * dimensions_2.length,
        time=dimensions_1.time * dimensions_2.time,
        mass=dimensions_1.mass * dimensions_2.mass,
        current=dimensions_1.current * dimensions_2.current,
        temperature=dimensions_1.temperature * dimensions_2.temperature,
        moles_hint=dimensions_1.moles_hint * dimensions_2.moles_hint,
        angle_hint=dimensions_1.angle_hint * dimensions_2.angle_hint
    )

def sum_dimensions(dimensions: Dimensions):
    return sum([
        dimensions.length,
        dimensions.time,
        dimensions.mass,
        dimensions.current,
        dimensions.temperature,
        dimensions.moles_hint,
        dimensions.angle_hint
    ])

def combine_units(unit_1: Unit, unit_2: Unit):
    return Unit(unit_1.scale * unit_2.scale, unit_1.dimensions * unit_2.dimensions)

def split_unit_str(unit_str: str) -> list[str]:
    return findall(r'[A-Za-z]+|[-\d]+', unit_str)

def parse_single_unit(unit_str: str) -> tuple[Unit | None, str]:
    """Attempts to find a single unit for unit_str. Return this unit, and the remaining string in a tuple. If a unit
    cannot be parsed, the unit will be None, and the remaining string will be the entire unit_str"""
    current_unit = ''
    string_pos = 0
    for char in unit_str:
        potential_unit_str = current_unit + char
        potential_symbols = [symbol for symbol in symbol_lookup.keys() if symbol.startswith(potential_unit_str)]
        if len(potential_symbols) == 0:
            break
        string_pos += 1
        current_unit= potential_unit_str
    if current_unit == '':
        return (None, unit_str)
    remaining_str = unit_str[string_pos::]
    return (symbol_lookup[current_unit], remaining_str)

def parse_unit_strs(unit_str: str, current_units: list[Unit] | None=None) -> list[Unit]:
    if current_units is None:
        current_units = []
    if unit_str == '':
        return current_units
    parsed_unit, remaining_str = parse_single_unit(unit_str)
    if not parsed_unit is None:
        current_units += [parsed_unit]
    return parse_unit_strs(remaining_str, current_units)


# Its probably useful to work out the unit first, and then later work out if a named unit exists for it. Hence why there
# are two functions.

def parse_unit_stack(unit_str: str) -> list[Unit]:
    # TODO: This doesn't work for 1/ (or any fraction) yet.
    unit_stack: list[Unit] = []
    split_str = split_unit_str(unit_str)
    for token in split_str:
        try:
            dimension_modifier = int(token)
            to_modify = unit_stack[-1]
            # FIXME: This is horrible but I'm not sure how to fix this without changing the Dimension class itself.
            multiplier = Dimensions(dimension_modifier, dimension_modifier, dimension_modifier, dimension_modifier, dimension_modifier, dimension_modifier, dimension_modifier)
            to_modify = Unit(to_modify.scale ** dimension_modifier, multiply_dimensions(to_modify.dimensions, multiplier))
            unit_stack[-1] = to_modify
        except ValueError:
            new_units = parse_unit_strs(token)
            unit_stack += new_units
        # This error will happen if it tries to read a modifier but there are no units on the stack. We will just have
        # to ignore it. Strings being parsed shouldn't really have it anyway (e.g. -1m).
        except IndexError:
            pass
    return unit_stack

def parse_unit(unit_str: str) -> Unit:
    parsed_unit = Unit(1, Dimensions())
    unit_stack = parse_unit_stack(unit_str)
    for unit in unit_stack:
        parsed_unit = combine_units(parsed_unit, unit)
    return parsed_unit

def parse_named_unit(unit_str: str) -> NamedUnit:
    # TODO: Not actually sure if this includes all units.
    generic_unit = parse_unit(unit_str)
    for named_unit in all_units:
        if named_unit == generic_unit:
            return named_unit
    raise ValueError('A named unit does not exist for this unit.')

if __name__ == "__main__":
    print(parse_named_unit('kmh-1'))
