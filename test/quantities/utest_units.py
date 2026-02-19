import math

import pytest

import sasdata.quantities.units as units
from sasdata.quantities.units import Unit, ArbitraryUnit


class EqualUnits:
    def __init__(self, test_name: str, *units):
        self.test_name = "Equality: " + test_name
        self.units: list[Unit] = list(units)

    def run_test(self):
        for i, unit_1 in enumerate(self.units):
            for unit_2 in self.units[i + 1 :]:
                print(unit_1, unit_2)
                assert unit_1.equivalent(unit_2), "Units should be equivalent"
                assert unit_1 == unit_2, "Units should be equal"


def test_unit_equality(equal_term):
    for i, unit_1 in enumerate(equal_term):
        for unit_2 in equal_term[i + 1 :]:
            assert unit_1.equivalent(unit_2), "Units should be equivalent"
            assert unit_1 == unit_2, "Units should be equal"


EQUIVALENT_TERMS = {
    "Angular frequency": [units.rotations / units.minutes, units.degrees * units.hertz],
}

pizza_a = ArbitraryUnit("Pizzas")
pizza_b = ArbitraryUnit(["Pizzas"])
print(f"A: {pizza_a}\tB: {pizza_b}")


@pytest.fixture(params=EQUIVALENT_TERMS)
def equivalent_term(request):
    return EQUIVALENT_TERMS[request.param]

    EqualUnits("Arbitrary Units",
               ArbitraryUnit("Pizzas"),
               ArbitraryUnit(["Pizzas"])),

    EqualUnits("Arbitrary Units",
               ArbitraryUnit("Slices", denominator=["Pizzas"]),
               ArbitraryUnit(["Slices"], denominator=["Pizzas"])),

    DissimilarUnits("Different Arbitrary Units",
               ArbitraryUnit("Pizzas"),
               ArbitraryUnit(["Doughnuts"])),


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
        for unit_2 in units[i + 1:]:
            assert not unit_1.equivalent(unit_2), "Units should not be equivalent"


def test_unit_operations():
    pizza = UnknownUnit(["Pizza"])
    slice = UnknownUnit("Slice")
    pineapple = UnknownUnit("Pineapple")
    pie = UnknownUnit("Pie")
    empty = UnknownUnit([])

    with pytest.raises(RuntimeError):
        UnknownUnit("a/b")
    with pytest.raises(RuntimeError):
        UnknownUnit(["a^b"])
    with pytest.raises(RuntimeError):
        UnknownUnit({"a b": 1})
    with pytest.raises(RuntimeError):
        UnknownUnit("a", {"a*b": 1})
    with pytest.raises(RuntimeError):
        UnknownUnit("a", ["a^b"])
    with pytest.raises(RuntimeError):
        UnknownUnit("a", "a/b")

    assert str(empty) == ""

    assert str(pizza) == "Pizza"
    assert str(pizza * pineapple) == "Pineapple Pizza"
    assert str(pizza * pizza) == "Pizza^2"

    assert str(1 / pizza) == "Pizza^-1"
    assert str(1 / pizza / pineapple) == "Pineapple^-1 Pizza^-1"
    assert str(slice / pizza) == "Slice Pizza^-1"
    assert str(slice / pizza / pineapple) == "Slice Pineapple^-1 Pizza^-1"
    assert str((slice / pizza) ** 2) == "Slice^2 Pizza^-2"

    assert str(pie**0.5) == "Pie^0.5"  # A valid unit, because pie are square
