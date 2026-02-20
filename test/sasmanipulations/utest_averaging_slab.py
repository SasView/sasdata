"""
Unit tests for SlabX and SlabY averagers (moved out of utest_averaging_analytical.py).
"""
import unittest

import numpy as np

from sasdata.data_util.averaging import SlabX, SlabY
from sasdata.dataloader import data_info
from test.sasmanipulations.helper import (
    MatrixToData2D,
    expected_slabx_area,
    expected_slaby_area,
    integrate_1d_output,
    make_dd_from_func,
)

# TODO - also check the errors are being calculated correctly

class SlabXTests(unittest.TestCase):
    """
    This class contains all the unit tests for the SlabX class from
    manipulations.py
    """

    def test_slabx_init(self):
        """
        Test that SlabX's __init__ method does what it's supposed to.
        """
        qx_min = 1
        qx_max = 2
        qy_min = 3
        qy_max = 4
        nbins = 100
        fold = True

        slab_object = SlabX(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max), nbins=nbins, fold=fold)

        self.assertEqual(slab_object.qx_min, qx_min)
        self.assertEqual(slab_object.qx_max, qx_max)
        self.assertEqual(slab_object.qy_min, qy_min)
        self.assertEqual(slab_object.qy_max, qy_max)
        self.assertEqual(slab_object.nbins, nbins)
        self.assertEqual(slab_object.fold, fold)

    def test_slabx_multiple_detectors(self):
        """
        Test that SlabX raises an error when there are multiple detectors
        """
        averager_data = MatrixToData2D(np.ones([100, 100]))
        detector1 = data_info.Detector()
        detector2 = data_info.Detector()
        averager_data.data.detector.append(detector1)
        averager_data.data.detector.append(detector2)

        slab_object = SlabX()
        self.assertRaises(ValueError, slab_object, averager_data.data)

    def test_slabx_no_points_to_average(self):
        """
        Test SlabX raises ValueError when the ROI contains no data
        """
        test_data = np.ones([100, 100])
        averager_data = MatrixToData2D(data2d=test_data)

        # Region of interest well outside region with data
        qx_min = 2 * averager_data.qmax
        qx_max = 3 * averager_data.qmax
        qy_min = 2 * averager_data.qmax
        qy_max = 3 * averager_data.qmax

        slab_object = SlabX(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max))
        self.assertRaises(ValueError, slab_object, averager_data.data)

    def test_slabx_averaging_without_fold(self):
        """
        Test that SlabX can average correctly when x is the major axis
        """
        def func(x, y):
            return x**2 * y
        averager_data, matrix_size = make_dd_from_func(func, matrix_size=201)


        # Set up region of interest to average over - the limits are arbitrary.
        qx_min = -0.5 * averager_data.qmax  # = -0.5
        qx_max = averager_data.qmax  # = 1
        qy_min = -0.5 * averager_data.qmax  # = -0.5
        qy_max = averager_data.qmax  # = 1
        nbins = int((qx_max - qx_min) / 2 * matrix_size)
        # Explicitly not using fold in this test
        fold = False

        slab_object = SlabX(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max), nbins=nbins, fold=fold)
        data1d = slab_object(averager_data.data)

        expected_area = expected_slabx_area(qx_min, qx_max, qy_min, qy_max)
        actual_area = integrate_1d_output(data1d, method="simpson")

        self.assertAlmostEqual(actual_area, expected_area, 2)

    def test_slabx_averaging_with_fold(self):
        """
        Test that SlabX can average correctly when x is the major axis
        """
        def func(x, y):
            return x**2 * y
        averager_data, matrix_size = make_dd_from_func(func, matrix_size=201)

        # Set up region of interest to average over - the limits are arbitrary.
        qx_min = -0.5 * averager_data.qmax  # = -0.5
        qx_max = averager_data.qmax  # = 1
        qy_min = -0.5 * averager_data.qmax  # = -0.5
        qy_max = averager_data.qmax  # = 1
        nbins = int((qx_max - qx_min) / 2 * matrix_size)
        # Explicitly using fold in this test
        fold = True

        slab_object = SlabX(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max), nbins=nbins, fold=fold)
        data1d = slab_object(averager_data.data)

        # Negative values of x are not graphed when fold = True
        qx_min_fold = 0
        expected_area = expected_slabx_area(qx_min_fold, qx_max, qy_min, qy_max)
        actual_area = integrate_1d_output(data1d, method="simpson")

        self.assertAlmostEqual(actual_area, expected_area, 2)

class SlabYTests(unittest.TestCase):
    """
    This class contains all the unit tests for the SlabY class from
    manipulations.py
    """

    def test_slaby_init(self):
        """
        Test that SlabY's __init__ method does what it's supposed to.
        """
        qx_min = 1
        qx_max = 2
        qy_min = 3
        qy_max = 4
        nbins = 100
        fold = True

        slab_object = SlabY(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max), nbins=nbins, fold=fold)

        self.assertEqual(slab_object.qx_min, qx_min)
        self.assertEqual(slab_object.qx_max, qx_max)
        self.assertEqual(slab_object.qy_min, qy_min)
        self.assertEqual(slab_object.qy_max, qy_max)
        self.assertEqual(slab_object.nbins, nbins)
        self.assertEqual(slab_object.fold, fold)

    def test_slaby_multiple_detectors(self):
        """
        Test that SlabY raises an error when there are multiple detectors
        """
        averager_data = MatrixToData2D(np.ones([100, 100]))
        detector1 = data_info.Detector()
        detector2 = data_info.Detector()
        averager_data.data.detector.append(detector1)
        averager_data.data.detector.append(detector2)
        slab_object = SlabY()
        self.assertRaises(ValueError, slab_object, averager_data.data)

    def test_slaby_no_points_to_average(self):
        """
        Test SlabY raises ValueError when the ROI contains no data
        """
        test_data = np.ones([100, 100])
        averager_data = MatrixToData2D(data2d=test_data)

        # Region of interest well outside region with data
        qx_min = 2 * averager_data.qmax
        qx_max = 3 * averager_data.qmax
        qy_min = 2 * averager_data.qmax
        qy_max = 3 * averager_data.qmax

        slab_object = SlabY(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max))
        self.assertRaises(ValueError, slab_object, averager_data.data)

    def test_slaby_averaging_without_fold(self):
        """
        Test that SlabY can average correctly when y is the major axis
        """
        def func(x, y):
            return x * y**2
        averager_data, matrix_size = make_dd_from_func(func, matrix_size=201)


        # Set up region of interest to average over - the limits are arbitrary.
        qx_min = -0.5 * averager_data.qmax  # = -0.5
        qx_max = averager_data.qmax  # = 1
        qy_min = -0.5 * averager_data.qmax  # = -0.5
        qy_max = averager_data.qmax  # = 1
        nbins = int((qy_max - qy_min) / 2 * matrix_size)
        # Explicitly not using fold in this test
        fold = False

        slab_object = SlabY(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max), nbins=nbins, fold=fold)
        data1d = slab_object(averager_data.data)

        expected_area = expected_slaby_area(qx_min, qx_max, qy_min, qy_max)
        actual_area = integrate_1d_output(data1d, method="simpson")

        self.assertAlmostEqual(actual_area, expected_area, 2)

    def test_slaby_averaging_with_fold(self):
        """
        Test that SlabY can average correctly when y is the major axis
        """
        def func(x, y):
            return x * y**2

        averager_data, matrix_size = make_dd_from_func(func, matrix_size=201)

        # Set up region of interest to average over - the limits are arbitrary.
        qx_min = -0.5 * averager_data.qmax  # = -0.5
        qx_max = averager_data.qmax  # = 1
        qy_min = -0.5 * averager_data.qmax  # = -0.5
        qy_max = averager_data.qmax  # = 1
        nbins = int((qy_max - qy_min) / 2 * matrix_size)
        # Explicitly using fold in this test
        fold = True

        slab_object = SlabY(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max), nbins=nbins, fold=fold)
        data1d = slab_object(averager_data.data)

        # Negative values of y are not graphed when fold = True, so don't
        # include them in the area calculation.
        qy_min_fold = 0
        expected_area = expected_slaby_area(qx_min, qx_max, qy_min_fold, qy_max)
        actual_area = integrate_1d_output(data1d, method="simpson")

        self.assertAlmostEqual(actual_area, expected_area, 2)

if __name__ == '__main__':
    unittest.main()
