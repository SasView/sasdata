import unittest

import numpy as np
from numpy.testing import assert_allclose

from sasdata.data_util import interpolations

RTOL = 1e-12

class Data1DTests(unittest.TestCase):
    """
    This testing class for plottable_1D is incomplete.
    Creating class to test _perform_operation and _interpolation_operation only. CMW 1-21-2024
    """

    def test_linear(self):
        """
        Test whether interpolation is performed correctly.
        """
        # check interpolation
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 5, 6, 8]
        x_interp = [1.2, 3.5, 4.5]
        y_interp = [1.6, 5.5, 7.]
        result = interpolations.linear(x_interp, x, y)
        assert_allclose(result[0], y_interp, RTOL)
        self.assertIsNone(result[1])

        # check sorting
        x = [1, 3, 2, 4, 5]
        y = [1, 5, 4, 7, 8]
        x_interp = [1.2, 3.5, 4.5]
        y_interp = [1.6, 6, 7.5]
        result = interpolations.linear(x_interp, x, y)
        assert_allclose(result[0], y_interp, RTOL)

        # check error propagation
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 4, 5, 6])
        dy = np.array([0.1, 0.2, 0.3, 0.4])
        x_interp = np.array([1.2, 3.5])
        y_interp = np.array([1.6, 5.5])
        i2 = np.searchsorted(x, x_interp)
        i1 = i2-1
        dy_interp = np.sqrt(dy[i1]**2*((x_interp-x[i2])/(x[i1]-x[i2]))**2+dy[i2]**2*((x_interp-x[i1])/(x[i2]-x[i1]))**2)
        result = interpolations.linear(x_interp, x, y, dy=dy)
        assert_allclose(result[0], y_interp, RTOL)
        assert_allclose(result[1], dy_interp, RTOL)

    def test_linear_scales(self):
        """
        Test whether interpolation is performed correctly with different scales.
        """
        # check linear
        x = [1., 2, 3, 4, 5]
        y = [1., 4, 5, 6, 8]
        x_interp = [1.2, 3.5, 4.5]
        y_interp = [1.6, 5.5, 7.]
        result = interpolations.linear_scales(x_interp, x, y, scale='linear')
        assert_allclose(result[0], y_interp, RTOL)
        self.assertIsNone(result[1])

        # check log
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 4, 5, 6, 8])
        dy = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        x_interp = [1.2, 3.5, 4.5]

        result = interpolations.linear_scales(x_interp, x, y, dy=dy, scale='log')
        assert_allclose(result[0], np.array([1.44, 5.5131300913615755, 6.983904974860978]), RTOL)
        assert_allclose(result[1], np.array([
            0.10779966010303317,
            0.24972135075650462,
            0.31845097763629354,
        ]))

        # check log
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 4, 5, 6, 8])
        x_interp = [1.2, 3.5, 4.5]

        result = interpolations.linear_scales(x_interp, x, y, dy=None, scale='log')
        assert_allclose(result[0], np.array([1.44, 5.5131300913615755, 6.983904974860978]), RTOL)
        self.assertIsNone(result[1])

