from sasdata.quantities.unit_parser import parse_named_unit, parse_named_unit_from_group
from sasdata.quantities.units import meters, speed, meters_per_second, per_angstrom, kilometers_per_square_hour


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
