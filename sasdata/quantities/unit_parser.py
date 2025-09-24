from re import findall, fullmatch

from sasdata.quantities.units import Dimensions, NamedUnit, Unit, UnitGroup, symbol_lookup, unit_groups

# TODO: This shouldn't be in this file but I don't want to edit Lucas' code before he is finished.

all_units_groups = [group.units for group in unit_groups.values()]
unit_groups_by_dimension_hash = {hash(group.units[0].dimensions): group for group in unit_groups.values()}
all_units: list[NamedUnit] = []
for group in all_units_groups:
    all_units.extend(group)


def split_unit_str(unit_str: str) -> list[str]:
    """Separate the letters from the numbers in unit_str"""
    return findall(r"[A-Za-zΩ%Å]+|[-\d]+|/", unit_str)


def validate_unit_str(unit_str: str) -> bool:
    """Validate whether unit_str is valid. This doesn't mean that the unit specified in unit_str exists but rather it
    only consists of letters, and numbers as a unit string should."""
    return fullmatch(r"[A-Za-zΩµ%Å^1-9\-\+/\ \._]+", unit_str) is not None


def parse_single_unit(
    unit_str: str, unit_group: UnitGroup | None = None, longest_unit: bool = True
) -> tuple[Unit | None, str]:
    """Attempts to find a single unit for unit_str. Return this unit, and the remaining string in a tuple. If a unit
    cannot be parsed, the unit will be None, and the remaining string will be the entire unit_str.

    The shortest_unit parameter specifies how to resolve ambiguities. If it is true, then it will parse the longest unit
    available. Otherwise, it will stop parsing as soon as it has found any unit.

    If unit_group is set, it will only try to parse units within that group. This is useful for resolving ambiguities.
    """
    current_unit = ""
    string_pos = 0
    if unit_group is None:
        lookup_dict = symbol_lookup
    else:
        lookup_dict = dict([(name, unit) for name, unit in symbol_lookup.items() if unit in unit_group.units])
    for next_char in unit_str:
        potential_unit_str = current_unit + next_char
        potential_symbols = [
            symbol
            for symbol, unit in lookup_dict.items()
            if symbol.startswith(potential_unit_str) or unit.startswith(potential_unit_str)
        ]
        if len(potential_symbols) == 0:
            break
        string_pos += 1
        current_unit = potential_unit_str
        if not longest_unit and current_unit in lookup_dict:
            break
    if current_unit == "":
        return None, unit_str
    matching_types = [unit for symbol, unit in lookup_dict.items() if symbol == current_unit or unit == current_unit]
    if not matching_types:
        raise KeyError(f"No known type matching {current_unit}")
    final_unit = matching_types[0]
    remaining_str = unit_str[string_pos::]
    return final_unit, remaining_str


def parse_unit_strs(unit_str: str, current_units: list[Unit] | None = None, longest_unit: bool = True) -> list[Unit]:
    """Recursively parse units from unit_str until no more characters are present."""
    if current_units is None:
        current_units = []
    if unit_str == "":
        return current_units
    parsed_unit, remaining_str = parse_single_unit(unit_str, longest_unit=longest_unit)
    if parsed_unit is not None:
        current_units += [parsed_unit]
        return parse_unit_strs(remaining_str, current_units, longest_unit)
    else:
        raise ValueError(f"Could not interpret {remaining_str}")


# Its probably useful to work out the unit first, and then later work out if a named unit exists for it. Hence why there
# are two functions.


def parse_unit_stack(unit_str: str, longest_unit: bool = True) -> list[Unit]:
    """Split unit_str into a stack of parsed units."""
    unit_stack: list[Unit] = []
    split_str = split_unit_str(unit_str)
    inverse_next_unit = False
    for token in split_str:
        try:
            if token == "/":
                inverse_next_unit = True
                continue
            power = int(token)
            to_modify = unit_stack[-1]
            modified = to_modify**power
            # modified = unit_power(to_modify, power)
            unit_stack[-1] = modified
        except ValueError:
            new_units = parse_unit_strs(token, None, longest_unit)
            if inverse_next_unit:
                # TODO: Assume the power is going to be -1. This might not be true.
                power = -1
                new_units[0] = new_units[0] ** power
                # new_units[0] = unit_power(new_units[0], power)
            unit_stack += new_units
        # This error will happen if it tries to read a modifier but there are no units on the stack. We will just have
        # to ignore it. Strings being parsed shouldn't really have it anyway (e.g. -1m).
        except IndexError:
            pass
    return unit_stack


def known_mistake(unit_str: str) -> Unit | None:
    """Take known broken units from historical files
    and give them a reasonible parse"""
    import sasdata.quantities.units as units

    mistakes = {"µm": units.micrometers, "per_centimeter": units.per_centimeter, "per_angstrom": units.per_angstrom}
    if unit_str in mistakes:
        return mistakes[unit_str]
    return None


def parse_unit(unit_str: str, longest_unit: bool = True) -> Unit:
    """Parse unit_str into a unit."""
    if result := known_mistake(unit_str):
        return result
    try:
        if not validate_unit_str(unit_str):
            raise ValueError("unit_str contains forbidden characters.")
        parsed_unit = Unit(1, Dimensions())
        unit_stack = parse_unit_stack(unit_str, longest_unit)
        for unit in unit_stack:
            # parsed_unit = combine_units(parsed_unit, unit)
            parsed_unit *= unit
        return parsed_unit
    except KeyError as ex:
        raise ValueError(f"Unit string contains an unrecognised pattern: {unit_str}") from ex


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


def parse_named_unit(unit_string: str, rtol: float = 1e-14) -> NamedUnit:
    """Parses unit into a named unit. Parses unit into a Unit if it is not already, and then finds an equivaelent named
    unit. Please note that this might not be the expected unit from the string itself. E.g. 'kgm/2' will become
    newtons.

    :param unit_string: string describing the units, e.g. km/s
    :param rtol: relative tolerance for matching scale factors
    """
    unit = parse_unit(unit_string)
    named_unit = find_named_unit(unit)
    if named_unit is None:
        raise ValueError(f"We don't have a for this unit: '{unit}'")
    else:
        return named_unit


def find_named_unit(unit: Unit, rtol: float = 1e-14) -> NamedUnit | None:
    """Find a named unit matching the one provided"""
    dimension_hash = hash(unit.dimensions)
    if dimension_hash in unit_groups_by_dimension_hash:
        unit_group = unit_groups_by_dimension_hash[hash(unit.dimensions)]

        for named_unit in unit_group.units:
            if abs(named_unit.scale - unit.scale) < rtol * named_unit.scale:
                return named_unit

    return None


def parse_named_unit_from_group(unit_str: str, from_group: UnitGroup) -> NamedUnit:
    """Parses unit_str into a named unit. The named unit found must be part of from_group. If two units are found, the
    unit that is present in from_group is returned. This is useful in cases of ambiguities."""
    parsed_unit = parse_unit_from_group(unit_str, from_group)
    if parsed_unit is None:
        raise ValueError("That unit cannot be parsed from the specified group.")
    return find_named_unit(parsed_unit)


def parse(string: str, name_lookup: bool = True, longest_unit: bool = True, lookup_rtol: float = 1e-14):
    unit = parse_unit(string, longest_unit=longest_unit)
    if name_lookup:
        named = find_named_unit(unit, rtol=lookup_rtol)
        if named is not None:
            return named

    return unit


if __name__ == "__main__":
    to_parse = input("Enter a unit to parse: ")
    try:
        generic_unit = parse_unit(to_parse)
        print(f"Generic Unit: {generic_unit}")
        named_unit = find_named_unit(generic_unit)
        print(f"Named Unit: {named_unit}")
    except ValueError:
        print("There is no named unit available.")
