from collections.abc import Callable

import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

from sasdata.quantities import units
from sasdata.quantities.plotting import quantity_plot
from sasdata.quantities.quantity import NamedQuantity, Quantity
from sasdata.transforms.rebinning import InterpolationOptions, calculate_interpolation_matrix_1d

test_functions = [
    lambda x: x**2,
    lambda x: 2*x,
    lambda x: x**3
]

test_interpolation_orders = [
    InterpolationOptions.LINEAR,
    InterpolationOptions.CUBIC
]


@pytest.mark.parametrize("fun", test_functions)
@pytest.mark.parametrize("order", test_interpolation_orders)
def test_interpolate_matrix_inside(fun: Callable[[Quantity[ArrayLike]], Quantity[ArrayLike]], order: InterpolationOptions, show_plots: bool):
    original_points = NamedQuantity("x_base", np.linspace(-10,10, 31), units.meters)
    test_points = NamedQuantity("x_test", np.linspace(-5, 5, 11), units.meters)


    mapping, _ = calculate_interpolation_matrix_1d(original_points, test_points, order=order)

    y_original = fun(original_points)
    y_test = y_original @ mapping
    y_expected = fun(test_points)

    test_units = y_expected.units

    y_values_test = y_test.in_units_of(test_units)
    y_values_expected = y_expected.in_units_of(test_units)

    if show_plots:
        print(y_values_test)
        print(y_values_expected)

        quantity_plot(original_points, y_original)
        quantity_plot(test_points, y_test)
        quantity_plot(test_points, y_expected)
        plt.show()

    assert len(y_values_test) == len(y_values_expected)

    for t, e in zip(y_values_test, y_values_expected):
        assert t == pytest.approx(e, abs=2)


@pytest.mark.parametrize("fun", test_functions)
@pytest.mark.parametrize("order", test_interpolation_orders)
def test_interpolate_different_units(fun: Callable[[Quantity[ArrayLike]], Quantity[ArrayLike]], order: InterpolationOptions, show_plots: bool):
    original_points = NamedQuantity("x_base", np.linspace(-10,10, 107), units.meters)
    test_points = NamedQuantity("x_test", np.linspace(-5000, 5000, 11), units.millimeters)

    mapping, _ = calculate_interpolation_matrix_1d(original_points, test_points, order=order)

    y_original = fun(original_points)
    y_test = y_original @ mapping
    y_expected = fun(test_points)

    test_units = y_expected.units

    y_values_test = y_test.in_units_of(test_units)
    y_values_expected = y_expected.in_units_of(test_units)

    if show_plots:
        print(y_values_test)
        print(y_test.in_si())
        print(y_values_expected)

        plt.plot(original_points.in_si(), y_original.in_si())
        plt.plot(test_points.in_si(), y_test.in_si(), "x")
        plt.plot(test_points.in_si(), y_expected.in_si(), "o")
        plt.show()

    assert len(y_values_test) == len(y_values_expected)

    for t, e in zip(y_values_test, y_values_expected):
        assert t == pytest.approx(e, rel=5e-2)

@pytest.mark.parametrize("order", test_interpolation_orders)
def test_linear(order: InterpolationOptions):
    """ Test linear interpolation between two points"""
    x_and_y = NamedQuantity("x_base", np.linspace(-10, 10, 2), units.meters)
    new_x = NamedQuantity("x_test", np.linspace(-5000, 5000, 101), units.millimeters)

    mapping, _ = calculate_interpolation_matrix_1d(x_and_y, new_x, order=order)

    linear_points = x_and_y @ mapping

    for t, e in zip(new_x.in_si(), linear_points.in_si()):
        assert t == pytest.approx(e, rel=1e-3)
