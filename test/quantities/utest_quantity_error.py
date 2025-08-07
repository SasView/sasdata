import numpy as np
import pytest

from sasdata.quantities import units
from sasdata.quantities.quantity import NamedQuantity


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

@pytest.mark.parametrize("x_val, y_val, x_units, y_units",
                         [(1, 1, units.meters, units.meters),
                          (1, 1, units.centimeters, units.centimeters),
                          (2, 2, units.meters, units.meters),
                          (1, 2, units.centimeters, units.centimeters),
                          (1, 2, units.meters, units.millimeters),
                          (3, 4, units.milliseconds, units.microseconds),
                          (0, 1, units.meters, units.meters)])
def test_asymmetry_propagation(x_val, y_val, x_units, y_units):

    x = NamedQuantity("x", x_val, x_units, standard_error=np.sqrt(x_val))
    y = NamedQuantity("y", y_val, y_units, standard_error=np.sqrt(y_val))

    x_si, x_err = x.in_si_with_standard_error()
    y_si, y_err = y.in_si_with_standard_error()

    numerator = x-y
    denominator = x+y
    a = numerator/denominator

    # Check numerator and denominator
    expected_error = np.sqrt(x_err ** 2 + y_err ** 2)

    value, error = numerator.in_si_with_standard_error()
    assert error == pytest.approx(expected_error, rel=1e-6)

    value, error = denominator.in_si_with_standard_error()
    assert error == pytest.approx(expected_error, rel=1e-6)

    # check whole thing
    value, error = a.in_si_with_standard_error()
    expected_error = (2 / (x_si + y_si)**2) * np.sqrt((x_err*y_si)**2 + (y_err*x_si)**2)
    assert error == pytest.approx(expected_error, rel=1e-6)

@pytest.mark.parametrize("x_val, y_val, x_units, y_units",
                         [(1, 1, units.meters, units.meters),
                          (1, 1, units.centimeters, units.centimeters),
                          (2, 2, units.meters, units.meters),
                          (1, 2, units.centimeters, units.centimeters),
                          (1, 2, units.meters, units.millimeters),
                          (3, 4, units.milliseconds, units.microseconds),
                          (0, 1, units.meters, units.meters)])
def test_power_propagation(x_val, y_val, x_units, y_units):

    x = NamedQuantity("x", x_val, x_units, standard_error=np.sqrt(x_val))
    y = NamedQuantity("y", y_val, y_units, standard_error=np.sqrt(y_val))

    x_si, x_err = x.in_si_with_standard_error()
    y_si, y_err = y.in_si_with_standard_error()

    x_var = x_err ** 2
    y_var = y_err ** 2

    z = (x*y)**3

    # check whole thing
    value, error = z.in_si_with_standard_error()
    expected_variance = 9*((x_si*y_si)**4)*(x_var*y_si*y_si + x_si*x_si*y_var)
    assert error == pytest.approx(np.sqrt(expected_variance), rel=1e-6)

@pytest.mark.parametrize("k", [0.1, 0.5, 1, 2, 10])
@pytest.mark.parametrize("x_val, y_val, x_units, y_units",
                         [(1, 1, units.meters, units.meters),
                          (1, 1, units.centimeters, units.centimeters),
                          (2, 2, units.meters, units.meters),
                          (1, 2, units.centimeters, units.centimeters),
                          (1, 2, units.meters, units.millimeters),
                          (3, 4, units.milliseconds, units.microseconds),
                          (0, 1, units.meters, units.meters),
                          (0, 0, units.meters, units.meters)])
def test_complex_power_propagation(x_val, y_val, x_units, y_units, k):

    x = NamedQuantity("x", x_val, x_units, standard_error=np.sqrt(k*x_val))
    y = NamedQuantity("y", y_val, y_units, standard_error=np.sqrt(k*y_val))

    x_si, x_err = x.in_si_with_standard_error()
    y_si, y_err = y.in_si_with_standard_error()

    x_var = x_err ** 2
    y_var = y_err ** 2

    z = (x+y)**3 + x**3 + y**3

    value, error = z.in_si_with_standard_error()
    expected_variance = \
        9*x_var*(x_si**2 + (x_si+y_si)**2)**2 + \
        9*y_var*(y_si**2 + (x_si+y_si)**2)**2

    assert error == pytest.approx(np.sqrt(expected_variance), rel=1e-6)

@pytest.mark.parametrize("k_x", [0.1, 0.5, 1, 2, 10])
@pytest.mark.parametrize("k_y", [0.1, 0.5, 1, 2, 10])
@pytest.mark.parametrize("x_val, y_val, x_units, y_units",
                         [(1, 1, units.meters, units.meters),
                          (1, 1, units.centimeters, units.centimeters),
                          (2, 2, units.meters, units.meters),
                          (1, 2, units.centimeters, units.centimeters),
                          (1, 2, units.meters, units.millimeters),
                          (3, 4, units.milliseconds, units.microseconds),
                          (0, 1, units.meters, units.meters),
                          (0, 0, units.meters, units.meters)])
def test_complex_propagation(x_val, y_val, x_units, y_units, k_x, k_y):

    x = NamedQuantity("x", x_val, x_units, standard_error=np.sqrt(k_x*x_val))
    y = NamedQuantity("y", y_val, y_units, standard_error=np.sqrt(k_y*y_val))

    cx = NamedQuantity("cx", 1.7, x_units)
    cy = NamedQuantity("cy", 1.2, y_units)
    c0 = 4*NamedQuantity("c0", value=7, units=units.none)

    cx_si = cx.in_si()
    cy_si = cy.in_si()

    c0_si = c0.in_si()

    x_si, x_err = x.in_si_with_standard_error()
    y_si, y_err = y.in_si_with_standard_error()

    x_var = x_err ** 2
    y_var = y_err ** 2

    z = (((x-cx)**4 + (y-cy)**4)**(1/4)) + c0*(-x-y)

    value, error = z.in_si_with_standard_error()

    denom_factor = ((x_si - cx_si)**4 + (y_si - cy_si)**4)**(-3/4)
    x_num = (cx_si - x_si)**3
    y_num = (cy_si - y_si)**3

    expected_variance = x_var*(c0_si + x_num*denom_factor)**2 + y_var*(c0_si + y_num*denom_factor)**2

    assert error == pytest.approx(np.sqrt(expected_variance), rel=1e-8)

