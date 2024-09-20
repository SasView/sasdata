from sasdata.quantities.unit_parser import parse_named_unit, parse_named_unit_from_group, parse_unit
from sasdata.quantities.units import meters, speed, meters_per_second, per_angstrom, kilometers_per_square_hour
from pytest import raises


def test_parse():
    parsed_metres = parse_named_unit('m')
    assert parsed_metres == meters
    # Have to specify a group because this is ambigious with inverse of milliseconds.
    parsed_metres_per_second = parse_named_unit_from_group('ms-1', speed)
    assert parsed_metres_per_second == meters_per_second
    parsed_inverse_angstroms = parse_named_unit('A-1')
    assert parsed_inverse_angstroms == per_angstrom
    parsed_kilometers_per_square_hour = parse_named_unit('kmh-2')
    assert parsed_kilometers_per_square_hour == kilometers_per_square_hour

def test_parse_errors():
    # Fails because the unit is not in that specific group.
    with raises(ValueError, match='That unit cannot be parsed from the specified group.'):
        parse_named_unit_from_group('km', speed)
    # Fails because part of the unit matches but there is an unknown unit '@'
    with raises(ValueError, match='unit_str contains forbidden characters.'):
        parse_unit('km@-1')
    # Fails because 'da' is not a unit.
    with raises(ValueError, match='Unit string contains an unrecognised pattern.'):
        parse_unit('mmda2')
