import numpy as np
import pytest
from abscissa import Abscissa, GridAbscissa, MeshgridAbscissa, ScatterAbscissa
from exceptions import InterpretationError
from quantities.quantity import Quantity
from quantities.units import none


def test_deterimine_1d_grid():
    """ Test that 1D ordered data is grid type"""
    q = Quantity(np.arange(10), units=none)

    determined = Abscissa.determine([q], q)

    assert isinstance(determined, GridAbscissa)


def test_deterimine_1d_scatter():
    """ Test that 1D unordered data is scatter type """
    a = Quantity(np.array([1, 2, 3, 4, 5, 0, 9, 8, 7, 6]), units=none)
    d = Quantity(np.arange(10), units=none)


    determined = Abscissa.determine([a], d)

    assert isinstance(determined, ScatterAbscissa)

def test_2D_scatter():
    """ Test the nD scatter case with 2D data """
    q = Quantity(np.arange(10), units=none)

    determined = Abscissa.determine([q, q], q)

    assert isinstance(determined, ScatterAbscissa)

def test_2D_meshgrid():
    """ Test the meshgrid case with 2x5"""
    q = Quantity(np.arange(10).reshape(2, 5), units=none)

    determined = Abscissa.determine([q, q], q)

    assert isinstance(determined, MeshgridAbscissa)


def test_2D_grid():
    """ Test the nD grid case with 2x5 """
    a1 = Quantity(np.arange(2), units=none)
    a2 = Quantity(np.arange(5), units=none)

    d = Quantity(np.arange(10).reshape(2, 5), units=none)

    determined = Abscissa.determine([a1, a2], d)

    assert isinstance(determined, GridAbscissa)

def test_2D_grid_axis_error():
    """ Test the nD grid case with bad axes """

    a1 = Quantity(np.arange(2), units=none)
    a2 = Quantity(np.arange(5), units=none)

    d = Quantity(np.arange(10).reshape(5, 2, 1), units=none)

    with pytest.raises(InterpretationError):

        Abscissa.determine([a1, a2], d)


def test_2D_meshgrid_error_mismatched_dimensionality():
    """ Test the nD meshgrid case with bad axes """

    q = Quantity(np.arange(10).reshape(5, 2), units=none)

    with pytest.raises(InterpretationError):

        deterimined = Abscissa.determine([q, q, q], q) # three axes, each 2D
        print(type(deterimined))