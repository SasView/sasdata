import math

import pytest

import sasdata.quantities.units as units
from sasdata.quantities.units import UnknownUnit

EQUAL_TERMS = {
    "Pressure": [units.pascals, units.newtons / units.meters**2, units.micronewtons * units.millimeters**-2],
    "Resistance": [units.ohms, units.volts / units.amperes, 1e-3 / units.millisiemens],
    "Angular frequency": [(units.rotations / units.minutes), (units.radians * units.hertz) * 2 * math.pi / 60.0],
    "Unknown Units": [UnknownUnit("Pizzas"), UnknownUnit(["Pizzas"])],
    "Unknown Fractional Units": [
        UnknownUnit("Slices", denominator=["Pizzas"]),
        UnknownUnit(["Slices"], denominator=["Pizzas"]),
    ],
    "Unknown Multiplication": [
        UnknownUnit("Pizzas") * UnknownUnit("People"),
        UnknownUnit(["Pizzas", "People"]),
    ],
    "Unknown Multiplication with Units": [
        UnknownUnit("Pizzas") * units.meters,
        units.meters * UnknownUnit(["Pizzas"]),
    ],
    "Unknown Power": [
        UnknownUnit(["Slices"], denominator=["Pizza"]) * UnknownUnit(["Slices"], denominator=["Pizza"]),
        UnknownUnit(["Slices"], denominator=["Pizza"]) ** 2,
    ],
    "Unknown Fractional Power": [
        UnknownUnit(["Pizza", "Pizza", "Pizza"]),
        UnknownUnit(["Pizza", "Pizza"]) ** 1.5,
    ],
    "Unknown Division": [
        UnknownUnit("Slices") / UnknownUnit("Pizza"),
        UnknownUnit(["Slices"], denominator=["Pizza"]),
        (1 / UnknownUnit("Pizza")) * UnknownUnit("Slices"),
        1 / (UnknownUnit("Pizza") / UnknownUnit("Slices")),
    ],
    "Unknown Complicated Math": [
        (UnknownUnit("Slices") / UnknownUnit("Person"))
        / (UnknownUnit("Slices") / UnknownUnit("Pizzas"))
        * UnknownUnit("Person"),
        UnknownUnit("Pizzas"),
    ],
}


@pytest.fixture(params=EQUAL_TERMS)
def equal_term(request):
    return EQUAL_TERMS[request.param]


def test_unit_equality(equal_term):
    for i, unit_1 in enumerate(equal_term):
        for unit_2 in equal_term[i + 1 :]:
            assert unit_1.equivalent(unit_2), "Units should be equivalent"
            assert unit_1 == unit_2, "Units should be equal"


EQUIVALENT_TERMS = {
    "Angular frequency": [units.rotations / units.minutes, units.degrees * units.hertz],
}


@pytest.fixture(params=EQUIVALENT_TERMS)
def equivalent_term(request):
    return EQUIVALENT_TERMS[request.param]


def test_unit_equivalent(equivalent_term):
    units = equivalent_term
    for i, unit_1 in enumerate(units):
        for unit_2 in units[i + 1 :]:
            assert unit_1.equivalent(unit_2), "Units should be equivalent"
            assert unit_1 != unit_2, "Units not should be equal"


DISSIMILAR_TERMS = {
    "Frequency and Angular frequency": [(units.rotations / units.minutes), (units.hertz)],
    "Different Unknown Units": [UnknownUnit("Pizzas"), UnknownUnit(["Donuts"])],
    "Unknown Multiplication with Units": [
        UnknownUnit("Pizzas") * units.meters,
        units.seconds * UnknownUnit(["Pizzas"]),
    ],
}


@pytest.fixture(params=DISSIMILAR_TERMS)
def dissimilar_term(request):
    return DISSIMILAR_TERMS[request.param]


def test_unit_dissimilar(dissimilar_term):
    units = dissimilar_term
    for i, unit_1 in enumerate(units):
        for unit_2 in units[i + 1 :]:
            assert not unit_1.equivalent(unit_2), "Units should not be equivalent"


def test_unit_names():
    pizza = UnknownUnit(["Pizza"])
    slice = UnknownUnit(["Slice"])
    pineapple = UnknownUnit(["Pineapple"])
    pie = UnknownUnit(["Pie"])
    empty = UnknownUnit([])

    assert str(empty) == ""

    assert str(pizza) == "Pizza"
    assert str(pizza * pineapple) == "Pineapple Pizza"
    assert str(pizza * pizza) == "Pizza^2"

    assert str(1 / pizza) == "1 / Pizza"
    assert str(slice / pizza) == "Slice / Pizza"
    assert str((slice / pizza) ** 2) == "Slice^2 / Pizza^2"

    assert str(pie**0.5) == "Pie^0.5"  # A valid unit, because pie are square
