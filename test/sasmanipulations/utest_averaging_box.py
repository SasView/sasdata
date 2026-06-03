"""
Unit tests for SlabX and SlabY averagers (moved out of utest_averaging_analytical.py).
"""
import unittest

import numpy as np

from sasdata.data_util.averaging import Boxavg, Boxsum
from sasdata.dataloader import data_info
from test.sasmanipulations.helper import (
    MatrixToData2D,
    expected_boxavg_and_err,
    expected_boxsum_and_err,
    make_uniform_dd,
)

# TODO - also check the errors are being calculated correctly

class BoxsumTests(unittest.TestCase):
    """
    This class contains all the unit tests for the Boxsum class from
    manipulations.py
    """

    def test_boxsum_init(self):
        """
        Test that Boxsum's __init__ method does what it's supposed to.
        """
        qx_min = 1
        qx_max = 2
        qy_min = 3
        qy_max = 4

        box_object = Boxsum(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max))

        self.assertEqual(box_object.qx_min, qx_min)
        self.assertEqual(box_object.qx_max, qx_max)
        self.assertEqual(box_object.qy_min, qy_min)
        self.assertEqual(box_object.qy_max, qy_max)

    def test_boxsum_multiple_detectors(self):
        """
        Test Boxsum raises an error when there are multiple detectors.
        """
        dd = make_uniform_dd((100, 100), value=1.0)
        detector1 = data_info.Detector()
        detector2 = data_info.Detector()
        dd.data.detector.append(detector1)
        dd.data.detector.append(detector2)

        box_object = Boxsum()
        self.assertRaises(ValueError, box_object, dd.data)

    def test_boxsum_total(self):
        """
        Test that Boxsum can find the sum of all of a data set
        """
        # Creating a 100x100 matrix for a distribution which is flat in y
        # and linear in x.
        test_data = np.tile(np.arange(100), (100, 1))
        dd = MatrixToData2D(data2d=test_data)

        box_object = Boxsum(qx_range=(-1 * dd.qmax, dd.qmax), qy_range=(-1 * dd.qmax, dd.qmax))
        result, error, npoints = box_object(dd.data)
        correct_sum, correct_error = expected_boxsum_and_err(test_data)

        self.assertAlmostEqual(result, correct_sum, 6)
        self.assertAlmostEqual(error, correct_error, 6)

    def test_boxsum_subset_total(self):
        """
        Test that Boxsum can find the sum of a portion of a data set
        """
        # Creating a 100x100 matrix for a distribution which is flat in y
        # and linear in x.
        test_data = np.tile(np.arange(100), (100, 1))
        dd = MatrixToData2D(data2d=test_data)

        # region corresponds to central 50x50 in original test
        box_object = Boxsum(qx_range=(-0.5 * dd.qmax, 0.5 * dd.qmax), qy_range=(-0.5 * dd.qmax, 0.5 * dd.qmax))
        result, error, npoints = box_object(dd.data)
        inner_portion = test_data[25:75, 25:75]
        correct_sum, correct_error = expected_boxsum_and_err(inner_portion)

        self.assertAlmostEqual(result, correct_sum, 6)
        self.assertAlmostEqual(error, correct_error, 6)

    def test_boxsum_zero_sum(self):
        """
        Test that Boxsum returns 0 when there are no points within the ROI
        """
        test_data = np.ones([100, 100])
        test_data[25:75, 25:75] = 0
        dd = MatrixToData2D(data2d=test_data)

        box_object = Boxsum(qx_range=(-0.5 * dd.qmax, 0.5 * dd.qmax), qy_range=(-0.5 * dd.qmax, 0.5 * dd.qmax))
        result, error, npoints = box_object(dd.data)

        self.assertAlmostEqual(result, 0, 6)
        self.assertAlmostEqual(error, 0, 6)


class BoxavgTests(unittest.TestCase):
    """
    This class contains all the unit tests for the Boxavg class from
    manipulations.py
    """

    def test_boxavg_init(self):
        """
        Test that Boxavg's __init__ method does what it's supposed to.
        """
        qx_min = 1
        qx_max = 2
        qy_min = 3
        qy_max = 4

        box_object = Boxavg(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max))

        self.assertEqual(box_object.qx_min, qx_min)
        self.assertEqual(box_object.qx_max, qx_max)
        self.assertEqual(box_object.qy_min, qy_min)
        self.assertEqual(box_object.qy_max, qy_max)

    def test_boxavg_multiple_detectors(self):
        """
        Test Boxavg raises an error when there are multiple detectors.
        """
        dd = make_uniform_dd((100, 100), value=1.0)
        detector1 = data_info.Detector()
        detector2 = data_info.Detector()
        dd.data.detector.append(detector1)
        dd.data.detector.append(detector2)

        box_object = Boxavg()
        self.assertRaises(ValueError, box_object, dd.data)

    def test_boxavg_total(self):
        """
        Test that Boxavg can find the average of all of a data set
        """
        # Creating a 100x100 matrix for a distribution which is flat in y
        # and linear in x.
        test_data = np.tile(np.arange(100), (100, 1))
        dd = MatrixToData2D(data2d=test_data)

        box_object = Boxavg(qx_range=(-1 * dd.qmax, dd.qmax), qy_range=(-1 * dd.qmax, dd.qmax))
        result, error = box_object(dd.data)
        correct_avg, correct_error = expected_boxavg_and_err(test_data)

        self.assertAlmostEqual(result, correct_avg, 6)
        self.assertAlmostEqual(error, correct_error, 6)

    def test_boxavg_subset_total(self):
        """
        Test that Boxavg can find the average of a portion of a data set
        """
        # Creating a 100x100 matrix for a distribution which is flat in y
        # and linear in x.
        test_data = np.tile(np.arange(100), (100, 1))
        dd = MatrixToData2D(data2d=test_data)

        box_object = Boxavg(qx_range=(-0.5 * dd.qmax, 0.5 * dd.qmax), qy_range=(-0.5 * dd.qmax, 0.5 * dd.qmax))
        result, error = box_object(dd.data)
        inner_portion = test_data[25:75, 25:75]
        correct_avg, correct_error = expected_boxavg_and_err(inner_portion)

        self.assertAlmostEqual(result, correct_avg, 6)
        self.assertAlmostEqual(error, correct_error, 6)

    def test_boxavg_zero_average(self):
        """
        Test that Boxavg returns 0 when there are no points within the ROI
        """
        test_data = np.ones([100, 100])
        # Make a hole in the middle with zeros
        test_data[25:75, 25:75] = np.zeros([50, 50])
        dd = MatrixToData2D(data2d=test_data)

        box_object = Boxavg(qx_range=(-0.5 * dd.qmax, 0.5 * dd.qmax), qy_range=(-0.5 * dd.qmax, 0.5 * dd.qmax))
        result, error = box_object(dd.data)

        self.assertAlmostEqual(result, 0, 6)
        self.assertAlmostEqual(error, 0, 6)

if __name__ == '__main__':
    unittest.main()
