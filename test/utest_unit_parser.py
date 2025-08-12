import pytest

from sasdata.quantities import units
from sasdata.quantities.unit_parser import parse_named_unit, parse_named_unit_from_group, parse_unit
from sasdata.quantities.units import Unit

named_units_for_testing = [
    ("m", units.meters),
    ("A-1", units.per_angstrom),
    ("1/A", units.per_angstrom),
    ("1/angstroms", units.per_angstrom),
    ("micrometer", units.micrometers),
    ("micron", units.micrometers),
    ("kmh-2", units.kilometers_per_square_hour),
    ("km/h2", units.kilometers_per_square_hour),
    ("kgm/s2", units.newtons),
    ("m m", units.square_meters),
    ("mm", units.millimeters),
    ("A^-1", units.per_angstrom),
    ("V/Amps", units.ohms),
    ("Ω", units.ohms),
    ("Å", units.angstroms),
    ("%", units.percent),
    ("per_centimeter", units.per_centimeter),
]

latex_units_for_testing = [
    (r"\Omega", units.ohms),  # Test omega is Ω
    (r"\AA", units.angstroms),  # Test angstrom is Å
    (r"\%", units.percent),  # Test percent is NOT a comment
    (r"{\mu}A", units.microamperes),  # Test µ with an ASCII unit
    (r"{\mu}\Omega", units.microohms),  # Test µ with LaTeX unit
    (r"mm", units.millimeters),  # Test that most units just use ASCII in LaTeX
]


unnamed_units_for_testing = [("m13", units.meters**13), ("kW/sr", units.kilowatts / units.stradians)]


@pytest.mark.parametrize("string, expected_units", named_units_for_testing)
def test_name_parse(string: str, expected_units: Unit):
    """Test basic parsing"""
    assert parse_named_unit(string) == expected_units


@pytest.mark.parametrize("string, expected_units", named_units_for_testing + unnamed_units_for_testing)
def test_equivalent(string: str, expected_units: Unit):
    """Check dimensions of parsed units"""
    assert parse_unit(string).equivalent(expected_units)


@pytest.mark.parametrize("string, expected_units", named_units_for_testing + unnamed_units_for_testing)
def test_scale_same(string: str, expected_units: Unit):
    """Test basic parsing"""
    assert parse_unit(string).scale == pytest.approx(expected_units.scale, rel=1e-14)


@pytest.mark.parametrize("latex_string, units", latex_units_for_testing)
def test_latex_parse(latex_string: str, units: Unit):
    """Test that proper LaTeX formats for units are being generated"""
    assert units.latex_symbol == latex_string


def test_parse_from_group():
    """Test group based disambiguation"""
    parsed_metres_per_second = parse_named_unit_from_group("ms-1", units.speed)
    assert parsed_metres_per_second == units.meters_per_second


def test_parse_errors():
    # Fails because the unit is not in that specific group.
    with pytest.raises(ValueError, match="That unit cannot be parsed from the specified group."):
        parse_named_unit_from_group("km", units.speed)
    # Fails because part of the unit matches but there is an unknown unit '@'
    with pytest.raises(ValueError, match="unit_str contains forbidden characters."):
        parse_unit("km@-1")
    # Fails because 'da' is not a unit.
    with pytest.raises(ValueError, match="Unit string contains an unrecognised pattern."):
        parse_unit("mmda2")
