import math
import pytest

import sasdata.quantities.units as units
from sasdata.quantities.units import ArbitraryUnit


EQUAL_TERMS = {
    "Pressure": [units.pascals, units.newtons / units.meters**2, units.micronewtons * units.millimeters**-2],
    "Resistance": [units.ohms, units.volts / units.amperes, 1e-3 / units.millisiemens],
    "Angular frequency": [(units.rotations / units.minutes), (units.radians * units.hertz) * 2 * math.pi / 60.0],
    "Arbitrary Units": [ArbitraryUnit("Pizzas"), ArbitraryUnit(["Pizzas"])],
    "Arbitrary Fractional Units": [
        ArbitraryUnit("Slices", denominator=["Pizzas"]),
        ArbitraryUnit(["Slices"], denominator=["Pizzas"]),
    ],
    "Arbitrary Multiplication": [
        ArbitraryUnit("Pizzas") * ArbitraryUnit("People"),
        ArbitraryUnit(["Pizzas", "People"]),
    ],
    "Arbitrary Multiplication with Units": [
        ArbitraryUnit("Pizzas") * units.meters,
        units.meters * ArbitraryUnit(["Pizzas"]),
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
            print(unit_1, unit_2)
            assert unit_1.equivalent(unit_2), "Units should be equivalent"
            assert unit_1 != unit_2, "Units not should be equal"


DISSIMILAR_TERMS = {
    "Frequency and Angular frequency": [(units.rotations / units.minutes), (units.hertz)],
    "Different Arbitrary Units": [ArbitraryUnit("Pizzas"), ArbitraryUnit(["Donuts"])],
    "Arbitrary Multiplication with Units": [
        ArbitraryUnit("Pizzas") * units.meters,
        units.seconds * ArbitraryUnit(["Pizzas"]),
    ],
}


@pytest.fixture(params=DISSIMILAR_TERMS)
def dissimilar_term(request):
    return DISSIMILAR_TERMS[request.param]


def test_unit_dissimilar(dissimilar_term):
    units = dissimilar_term
    for i, unit_1 in enumerate(units):
        for unit_2 in units[i + 1 :]:
            print(unit_1, unit_2)
            assert not unit_1.equivalent(unit_2), "Units should not be equivalent"
