"""
This file contains unit tests for the various averagers found in
sasdata/data_util/manipulations.py - These tests are based on analytical
formulae rather than imported data files.
"""

import unittest

import numpy as np

from sasdata.data_util.binning import DirectionalAverage
from test.sasmanipulations.helper import MatrixToData2D

# TODO - also check the errors are being calculated correctly

class DirectionalAverageValidationTests(unittest.TestCase):
    """
    This class tests DirectionalAverage's data validation checks.
    """

    def test_missing_coordinate_data(self):
        """
        Ensure a ValueError is raised if no axis data is supplied.
        """
        self.assertRaises(ValueError, DirectionalAverage,
                          major_axis=None, minor_axis=None)

    def test_inappropriate_limits_arrays(self):
        """
        Ensure a ValueError is raised if the wrong number of limits is suppied.
        """
        self.assertRaises(ValueError, DirectionalAverage, major_axis=[],
                          minor_axis=[], lims=([], []))

    def test_nbins_not_int(self):
        """
        Ensure a TypeError is raised if the parameter nbins is not a number 
        that can be converted to integer.
        """
        self.assertRaises(TypeError, DirectionalAverage, major_axis=np.array([0, 1]),
                          minor_axis=np.array([0, 1]), nbins=np.array([]))

    def test_axes_unequal_lengths(self):
        """
        Ensure ValueError is raised if the major and minor axes don't match.
        """
        self.assertRaises(ValueError, DirectionalAverage, major_axis=[0, 1, 2],
                          minor_axis=[3, 4])

    def test_no_limits_on_an_axis(self):
        """
        Ensure correct behaviour when there are no limits provided.
        The min. and max. values from major/minor_axis are taken as the limits.
        """
        dir_avg = DirectionalAverage(major_axis=np.array([1, 2, 3]),
                                     minor_axis=np.array([4, 5, 6]))
        self.assertEqual(dir_avg.major_lims, (1, 3))
        self.assertEqual(dir_avg.minor_lims, (4, 6))

class DirectionalAverageFunctionalityTests(unittest.TestCase):
    """
    Placeholder
    """

    def setUp(self):
        """
        Setup for the DirectionalAverageFunctionalityTests tests.
        """

        # 21 bins, with spacing 0.1
        self.qx_data = np.linspace(-1, 1, 21)
        self.qy_data = self.qx_data
        x, y = np.meshgrid(self.qx_data, self.qy_data)
        # quadratic in x, linear in y
        data = x * x * y
        self.data2d = MatrixToData2D(data)

        # ROI is the first quadrant only. Same limits for both axes.
        self.lims = (0.0, 1.0)
        self.in_roi = (self.lims[0] <= self.qx_data) & \
                      (self.qx_data <= self.lims[1])
        self.nbins = int(np.sum(self.in_roi))
        # Note that the bin width is less than the spacing of the datapoints,
        # because we're insisting that there be as many bins as datapoints.
        self.bin_width = (self.lims[1] - self.lims[0]) / self.nbins

        self.directional_average = \
            DirectionalAverage(major_axis=self.data2d.data.qx_data,
                               minor_axis=self.data2d.data.qy_data,
                               lims=(self.lims,self.lims), nbins=self.nbins)

    def test_bin_width(self):
        """
        Test that the bin width is calculated correctly.
        """
        self.assertAlmostEqual(np.average(self.directional_average.bin_widths), self.bin_width)

    def test_get_bin_interval(self):
        """
        Test that the get_bin_interval method works correctly.
        """
        for b in range(self.nbins):
            bin_start, bin_end = self.directional_average.get_bin_interval(b)
            expected_bin_start = self.lims[0] + b * self.bin_width
            expected_bin_end = self.lims[0] + (b + 1) * self.bin_width
            self.assertAlmostEqual(bin_start, expected_bin_start, 10)
            self.assertAlmostEqual(bin_end, expected_bin_end, 10)

    def test_get_bin_index(self):
        """
        Test that the get_bin_index method works correctly.
        """
        # use values at the edges of bins, and values in the middles
        values = np.linspace(self.lims[0], self.lims[1], self.nbins * 2)
        expected_indices = np.repeat(np.arange(self.nbins), 2)
        for n, v in enumerate(values):
            self.assertAlmostEqual(self.directional_average.get_bin_index(v),
                                   expected_indices[n], 10)

    def test_binary_weights(self):
        """
        Test weights are calculated correctly when the bins & ROI are aligned.
        When aligned perfectly, the weights should be ones and zeros only.

        Variations on this test will be needed once fractional weighting is
        possible. These should have ROIs which do not line up perfectly with
        the bins.
        """

        # I think this test needs mocks, it'd be very complex otherwise.
        # I'm struggling to come up with a test for this one.
        pass

    def test_directional_averaging(self):
        """
        Test that a directinal average is computed correctly.

        Variations on this test will be needed once fractional weighting is
        possible. These should have ROIs which do not line up perfectly with
        the bins.
        """
        x_axis_values, intensity, errors = \
            self.directional_average(data=self.data2d.data.data,
                                     err_data=self.data2d.data.err_data)

        expected_x = self.qx_data[self.in_roi]
        expected_intensity = np.mean(self.qy_data[self.in_roi]) * expected_x**2

        np.testing.assert_array_almost_equal(x_axis_values, expected_x, 10)
        np.testing.assert_array_almost_equal(intensity, expected_intensity, 10)

    def test_no_points_in_roi(self):
        """
        Test that ValueError is raised if there were on points in the ROI.
        """
        # move the region of interest to outside the range of the data
        self.directional_average.major_lims = (2, 3)
        self.directional_average.minor_lims = (2, 3)
        self.assertRaises(ValueError, self.directional_average,
                          self.data2d.data.data, self.data2d.data.err_data)

if __name__ == '__main__':
    unittest.main()
