import numpy as np

from sasdata.quantities.quantity import Quantity, UnitError
import sasdata.quantities.units as units
import pytest
def test_in_units_of_calculation():
    """ Just a couple of unit conversions """
    assert Quantity(1, units.meters).in_units_of(units.kilometers) == 1e-3
    assert Quantity(10, units.minutes).in_units_of(units.seconds) == 600
    assert Quantity(7, units.kilonewtons).in_units_of(units.kg_force) == pytest.approx(7000/9.81, abs=1)
    assert Quantity(0, units.meters).in_units_of(units.exameters) == 0


def test_unit_compounding_pow():
    assert (Quantity(1, units.millimeters)**2).in_units_of(units.square_meters) == 1e-6
    assert (Quantity(1, units.minutes)**3).in_units_of(units.seconds**3) == 60**3

def test_unit_compounding_mul():
    assert (Quantity(4, units.minutes) * Quantity(0.25, units.hertz)).in_units_of(units.none) == 60
    assert (Quantity(250, units.volts) * Quantity(8, units.amperes)).in_units_of(units.kilowatts) == 2

def test_unit_compounding_div():
    assert (Quantity(10, units.kilometers) / Quantity(2, units.minutes)
            ).in_units_of(units.meters_per_second) == pytest.approx(250/3, abs=1e-6)

    assert (Quantity(1, units.nanowebers) / (Quantity(1, units.millimeters)**2)).in_units_of(units.millitesla) == 1

def test_value_mul():
    assert (Quantity(1j, units.seconds) * Quantity(1j, units.watts)).in_units_of(units.joules) == -1


def test_conversion_errors():



    with pytest.raises(UnitError):
        Quantity(1, units.seconds).in_units_of(units.meters)