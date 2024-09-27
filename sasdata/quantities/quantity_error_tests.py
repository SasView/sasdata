from sasdata.quantities import units
from sasdata.quantities.quantity import NamedQuantity
import pytest
import numpy as np

@pytest.mark.parametrize("x_err, y_err, x_units, y_units",
                         [(1, 1, units.meters, units.meters),
                          (1, 1, units.centimeters, units.centimeters),
                          (1, 2, units.meters, units.millimeters)])
def test_addition_propagation(x_err, y_err, x_units, y_units):
    """ Test that errors in addition of independent variables works with different units in the mix"""

    expected_err = np.sqrt((x_err*x_units.scale)**2 + (y_err*y_units.scale)**2)

    x = NamedQuantity("x", 0, x_units, standard_error=x_err)
    y = NamedQuantity("y", 0, y_units, standard_error=y_err)

    _, err = (x + y).in_si_with_standard_error()

    assert err == pytest.approx(expected_err, abs=1e-8)