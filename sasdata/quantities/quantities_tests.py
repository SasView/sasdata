import numpy as np

from sasdata.quantities.quantity import BaseQuantity, UnitError
import sasdata.quantities.units as units
import sasdata.quantities.si as si
import pytest
def test_in_units_of_calculation():
    """ Just a couple of unit conversions """
    assert BaseQuantity(1, units.meters).in_units_of(units.kilometers) == 1e-3
    assert BaseQuantity(10, units.minutes).in_units_of(units.seconds) == 600
    assert BaseQuantity(7, units.kilonewtons).in_units_of(units.kg_force) == pytest.approx(7000 / 9.81, abs=1)
    assert BaseQuantity(0, units.meters).in_units_of(units.exameters) == 0


def test_unit_compounding_pow():
    """ Test units compound correctly when __pow__ is used"""
    assert (BaseQuantity(1, units.millimeters) ** 2).in_units_of(units.square_meters) == 1e-6
    assert (BaseQuantity(1, units.minutes) ** 3).in_units_of(units.seconds ** 3) == 60 ** 3

def test_unit_compounding_mul():
    """ Test units compound correctly when __mul__ is used"""
    assert (BaseQuantity(4, units.minutes) * BaseQuantity(0.25, units.hertz)).in_units_of(units.none) == 60
    assert (BaseQuantity(250, units.volts) * BaseQuantity(8, units.amperes)).in_units_of(units.kilowatts) == 2

def test_unit_compounding_div():
    """ Test units compound correctly when __truediv__ is used"""
    assert (BaseQuantity(10, units.kilometers) / BaseQuantity(2, units.minutes)
            ).in_units_of(units.meters_per_second) == pytest.approx(250/3, abs=1e-6)

    assert (BaseQuantity(1, units.nanowebers) / (BaseQuantity(1, units.millimeters) ** 2)).in_units_of(units.millitesla) == 1

def test_value_mul():
    """ Test value part of quantities multiply correctly"""
    assert (BaseQuantity(1j, units.seconds) * BaseQuantity(1j, units.watts)).in_units_of(units.joules) == -1

def test_scalar_mul():
    assert (BaseQuantity(1, units.seconds) * 10).in_units_of(units.seconds) == 10
    assert (10 * BaseQuantity(1, units.seconds)).in_units_of(units.seconds) == 10
    assert (1000 * BaseQuantity(1, units.milliseconds)).in_units_of(units.seconds) == 1

def test_scalar_div():

    assert (BaseQuantity(1, units.seconds) / 10).in_units_of(units.seconds) == 0.1
    assert (10 / BaseQuantity(1, units.seconds)).in_units_of(units.hertz) == 10
    assert (0.001 / BaseQuantity(1, units.milliseconds)).in_units_of(units.hertz) == 1

def test_good_add_sub():
    """ Test that adding and subtracting units works """
    assert (BaseQuantity(1, units.seconds) + BaseQuantity(1, units.milliseconds)).in_units_of(units.seconds) == 1.001
    assert (BaseQuantity(1, units.seconds) - BaseQuantity(1, units.milliseconds)).in_units_of(units.seconds) == 0.999

    assert (BaseQuantity(1, units.inches) + BaseQuantity(1, units.feet)).in_units_of(units.inches) == 13


@pytest.mark.parametrize("unit_1", si.all_si)
@pytest.mark.parametrize("unit_2", si.all_si)
def test_mixed_quantity_add_sub(unit_1, unit_2):
    if unit_1.equivalent(unit_2):
        assert (BaseQuantity(0, unit_1) + BaseQuantity(0, unit_2)).in_units_of(unit_1) == 0

    else:
        with pytest.raises(UnitError):
            BaseQuantity(1, unit_1) + BaseQuantity(1, unit_2)

def assert_unit_ratio(u1: units.Unit, u2: units.Unit, value: float, abs=1e-9):
    """ Helper function for testing units that are multiples of each other """

    assert u1.equivalent(u2), "Units should be compatible for this test"
    assert (BaseQuantity(1, u1) / BaseQuantity(1, u2)).in_units_of(units.none) == pytest.approx(value, abs=abs)


def test_american_units():
    assert_unit_ratio(units.feet, units.inches, 12)
    assert_unit_ratio(units.yards, units.inches, 36)
    assert_unit_ratio(units.miles, units.inches, 63360)
    assert_unit_ratio(units.pounds_force_per_square_inch, units.pounds_force / (units.inches**2), 1, abs=1e-5)

def test_percent():
    assert BaseQuantity(5, units.percent).in_units_of(units.none) == pytest.approx(0.05, 1e-10)

@pytest.mark.parametrize("unit_1", si.all_si)
@pytest.mark.parametrize("unit_2", si.all_si)
def test_conversion_errors(unit_1, unit_2):
    """ Test conversion errors are thrown when units are not compatible """

    if unit_1 == unit_2:
        assert BaseQuantity(1, unit_1).in_units_of(unit_2) == 1

    else:
        with pytest.raises(UnitError):
            BaseQuantity(1, units.seconds).in_units_of(units.meters)

