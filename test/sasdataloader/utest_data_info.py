import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from sasdata.dataloader.data_info import Data1D
from sasdata.data_util.uncertainty import Uncertainty

RTOL = 1e-12


class Data1DTests(unittest.TestCase):
    """
    This testing class for plottable_1D is incomplete.
    Creating class to test _perform_operation and _interpolation_operation only. CMW 1-21-2024
    """

    def test_interpolation_operation(self):
        """
        Test whether the operation check and interpolation is performed correctly.
        """

        # test x2 range within x1 range
        # INTERPOLATION OPERATION TEST 1
        data1 = Data1D(x=[1, 2, 3, 4], y=[2, 3, 4, 5])
        data2 = Data1D(x=[2, 3], y=[0.2, 0.5])
        data1._interpolation_operation(data2)
        assert_allclose(np.array([2., 3.]), data1._operation.x, RTOL)
        assert_allclose(np.array([2., 3.]), data2._operation.x, RTOL)
        assert_allclose(np.array([3., 4.]), data1._operation.y, RTOL)
        assert_allclose(np.array([0.2, 0.5]), data2._operation.y, RTOL)

        # test x1 range within x2 range
        data1 = Data1D(x=[2, 3], y=[0.2, 0.5])
        data2 = Data1D(x=[1, 2, 3, 4], y=[2, 3, 4, 5])
        data1._interpolation_operation(data2)
        assert_allclose(np.array([2., 3.]), data1._operation.x, RTOL)
        assert_allclose(np.array([2., 3.]), data2._operation.x, RTOL)
        assert_allclose(np.array([0.2, 0.5]), data1._operation.y, RTOL)
        assert_allclose(np.array([3., 4.]), data2._operation.y, RTOL)

        # test overlap of x2 at high x1
        data1 = Data1D(x=[1, 2, 3, 4], y=[2, 3, 4, 5])
        data2 = Data1D(x=[3, 4, 5], y=[0.2, 0.5, 0.7])
        data1._interpolation_operation(data2)
        assert_allclose(np.array([3., 4.]), data1._operation.x, RTOL)
        assert_allclose(np.array([3., 4.]), data2._operation.x, RTOL)
        assert_allclose(np.array([4., 5.]), data1._operation.y, RTOL)
        assert_allclose(np.array([0.2, 0.5]), data2._operation.y, RTOL)

        # test overlap of x2 at low x1
        data1 = Data1D(x=[1, 2, 3, 4], y=[2, 3, 4, 5])
        data2 = Data1D(x=[0.2, 1, 2], y=[0.2, 0.5, 0.7])
        data1._interpolation_operation(data2)
        assert_allclose(np.array([1., 2.]), data1._operation.x, RTOL)
        assert_allclose(np.array([1., 2.]), data2._operation.x, RTOL)
        assert_allclose(np.array([2., 3.]), data1._operation.y, RTOL)
        assert_allclose(np.array([0.5, 0.7]), data2._operation.y, RTOL)

        # test equal x1 and x 2
        data1 = Data1D(x=[1, 2, 3, 4], y=[2, 3, 4, 5])
        data2 = Data1D(x=[1, 2, 3, 4], y=[0.2, 0.3, 0.4, 0.5])
        data1._interpolation_operation(data2)
        assert_allclose(np.array([1., 2., 3., 4.]), data1._operation.x, RTOL)
        assert_allclose(np.array([1., 2., 3., 4.]), data2._operation.x, RTOL)
        assert_allclose(np.array([2., 3., 4., 5.]), data1._operation.y, RTOL)
        assert_allclose(np.array([0.2, 0.3, 0.4, 0.5]), data2._operation.y, RTOL)

        # check once that these are all 0 or None if not supplied in original datasets
        assert_equal(data1._operation.dy, 0)
        assert_equal(data2._operation.dy, 0)
        self.assertIsNone(data1._operation.dx)
        self.assertIsNone(data1._operation.dxl)
        self.assertIsNone(data1._operation.dxw)
        self.assertIsNone(data1._operation.lam)
        self.assertIsNone(data1._operation.dlam)

        self.assertIsNone(data2._operation.dx)
        self.assertIsNone(data2._operation.dxl)
        self.assertIsNone(data2._operation.dxw)
        self.assertIsNone(data2._operation.lam)
        self.assertIsNone(data2._operation.dlam)

        # test tolerance
        data1 = Data1D(x=[1, 2, 3, 4, 5], y=[2, 3, 4, 5, 6])
        data2 = Data1D(x=[1, 2.19999, 3, 4.2, 5.6, 6], y=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        data1._interpolation_operation(data2, tolerance=0.1)
        assert_allclose(np.array([1., 2., 3., 4.]), data1._operation.x, RTOL)
        assert_allclose(np.array([1., 2., 3., 4.]), data2._operation.x, RTOL)
        assert_allclose(np.array([2, 3, 4., 5.]), data1._operation.y, RTOL)
        assert_allclose(np.array([0.2, 0.3, 0.4, 0.5]), data2._operation.y, RTOL)

        # test interpolation
        data1 = Data1D(x=[1, 2, 3, 4, 5], y=[2, 3, 4, 5, 6])
        data2 = Data1D(x=[2, 2.5, 3.5, 5], y=[0.4, 0.5, 0.6, 0.7])
        data1._interpolation_operation(data2)
        assert_allclose(np.array([2., 3., 4., 5.]), data1._operation.x, RTOL)
        assert_allclose(np.array([2., 3., 4., 5.]), data2._operation.x, RTOL)
        assert_allclose(np.array([3., 4., 5., 6.]), data1._operation.y, RTOL)
        assert_allclose(np.array([0.4, 0.5519189701334538, 0.6356450684803129, 0.7]), data2._operation.y, RTOL)

        # check these are copied over appropriately with no interpolation
        # test overlap of x2 at low x1
        data1 = Data1D(x=[1, 2, 3, 4],
                       y=[2, 3, 4, 5],
                       dy=[0.02, 0.03, 0.04, 0.05],
                       dx=[0.01, 0.02, 0.03, 0.04],
                       lam=[10, 11, 12, 13],
                       dlam=[0.1, 0.11, 0.12, 0.13])
        data1.dxl = np.array([0.1, 0.2, 0.3, 0.4])
        data1.dxw = np.array([0.4, 0.3, 0.2, 0.4])
        data2 = Data1D(x=[0.2, 1, 2],
                       y=[0.2, 0.5, 0.7],
                       dy=[0.002, 0.005, 0.007],
                       dx=[0.002, 0.01, 0.02],
                       lam=[13, 12, 11],
                       dlam=[0.13, 0.12, 0.11])
        data2.dxl = np.array([0.5, 0.6, 0.7])
        data2.dxw = np.array([0.7, 0.6, 0.5])
        data1._interpolation_operation(data2)

        assert_allclose(np.array([0.02, 0.03]), data1._operation.dy, RTOL)
        assert_allclose(np.array([0.01, 0.02]), data1._operation.dx, RTOL)
        assert_allclose(np.array([10, 11]), data1._operation.lam, RTOL)
        assert_allclose(np.array([0.1, 0.11]), data1._operation.dlam, RTOL)
        assert_allclose(np.array([0.1, 0.2]), data1._operation.dxl, RTOL)
        assert_allclose(np.array([0.4, 0.3]), data1._operation.dxw, RTOL)

        assert_allclose(np.array([0.005, 0.007]), data2._operation.dy, RTOL)
        assert_allclose(np.array([0.01, 0.02]), data2._operation.dx, RTOL)
        assert_allclose(np.array([12, 11.]), data2._operation.lam, RTOL)
        assert_allclose(np.array([0.12, 0.11]), data2._operation.dlam, RTOL)
        assert_allclose(np.array([0.6, 0.7]), data2._operation.dxl, RTOL)
        assert_allclose(np.array([0.6, 0.5]), data2._operation.dxw, RTOL)

        # check these are copied over appropriately with interpolation
        # test overlap of x2 at low x1
        data1 = Data1D(x=[1, 1.5, 2, 3],
                       y=[2, 3, 4, 5],
                       dy=[0.02, 0.03, 0.04, 0.05],
                       dx=[0.01, 0.02, 0.03, 0.04],
                       lam=[10, 11, 12, 13],
                       dlam=[0.1, 0.11, 0.12, 0.13])
        data1.dxl = np.array([0.1, 0.2, 0.3, 0.4])
        data1.dxw = np.array([0.4, 0.3, 0.2, 0.4])
        data2 = Data1D(x=[0.2, 1, 2],
                       y=[0.2, 0.5, 0.7],
                       dy=[0.002, 0.005, 0.007],
                       dx=[0.002, 0.01, 0.02],
                       lam=[13, 12, 11],
                       dlam=[0.13, 0.12, 0.11])
        data2.dxl = np.array([0.5, 0.6, 0.7])
        data2.dxw = np.array([0.7, 0.6, 0.5])
        data1._interpolation_operation(data2)

        assert_allclose(np.array([0.02, 0.03, 0.04]), data1._operation.dy, RTOL)
        assert_allclose(np.array([0.01, 0.02, 0.03]), data1._operation.dx, RTOL)
        assert_allclose(np.array([10, 11, 12]), data1._operation.lam, RTOL)
        assert_allclose(np.array([0.1, 0.11, 0.12]), data1._operation.dlam, RTOL)
        assert_allclose(np.array([0.1, 0.2, 0.3]), data1._operation.dxl, RTOL)
        assert_allclose(np.array([0.4, 0.3, 0.2]), data1._operation.dxw, RTOL)

        assert_allclose(np.array([0.005, 0.0043663206993972085, 0.007]), data2._operation.dy)
        self.assertIsNone(data2._operation.dx)
        self.assertIsNone(data2._operation.lam)
        self.assertIsNone(data2._operation.dlam)
        self.assertIsNone(data2._operation.dxl)
        self.assertIsNone(data2._operation.dxw)

    def test_perform_operation(self):
        """
        Test that the operation is performed correctly for two datasets.
        """
        def operation(a, b):
            return a - b

        data1 = Data1D(x=[1, 2, 3, 4],
                       y=[2, 3, 4, 5],
                       dy=[0.02, 0.03, 0.04, 0.05],
                       dx=[0.01, 0.02, 0.03, 0.04],
                       lam=[10, 11, 12, 13],
                       dlam=[0.1, 0.11, 0.12, 0.13])
        data1.dxl = np.array([0.1, 0.2, 0.3, 0.4])
        data1.dxw = np.array([0.4, 0.3, 0.2, 0.4])
        data2 = Data1D(x=[0.2, 1, 2],
                       y=[0.2, 0.5, 0.7],
                       dy=[0.002, 0.005, 0.007],
                       dx=[0.002, 0.01, 0.03],
                       lam=[13, 12, 11],
                       dlam=[0.13, 0.12, 0.11])
        data2.dxl = np.array([0.5, 0.6, 0.7])
        data2.dxw = np.array([0.7, 0.6, 0.5])
        result = data1._perform_operation(data2, operation)

        assert_allclose(np.array([1., 2.]), result.x, RTOL)
        assert_allclose(np.array([1.5, 2.3]), result.y, RTOL)
        # determine target values using Uncertainty (not a check for correctness of Uncertainty)
        u1 = Uncertainty(np.array([3, 4]), np.array([0.02**2, 0.03**2]))
        u2 = Uncertainty(np.array([0.5, 0.7]), np.array([0.005**2, 0.007**2]))
        u3 = u1-u2
        assert_allclose(np.sqrt(np.abs(u3.variance)), result.dy, RTOL)
        assert_equal(result.lam, data1._operation.lam)
        assert_equal(result.dlam, data1._operation.dlam)
        assert_allclose(np.sqrt(np.abs((data1._operation.dx**2+data2._operation.dx**2)/2)), result.dx, RTOL)
        assert_allclose(np.sqrt(np.abs((data1._operation.dxl**2+data2._operation.dxl**2)/2)), result.dxl, RTOL)
        assert_allclose(np.sqrt(np.abs((data1._operation.dxw**2+data2._operation.dxw**2)/2)), result.dxw, RTOL)


if __name__ == '__main__':
    unittest.main()


