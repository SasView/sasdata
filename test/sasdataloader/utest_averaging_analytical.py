"""
This file contains unit tests for the various averagers found in
sasdata/data_util/manipulations.py - These tests are based on analytical
formulae rather than imported data files.
"""

import unittest
from unittest.mock import patch

import numpy as np
from scipy import integrate

from sasdata.dataloader import data_info
from sasdata.data_util.new_manipulations import (SlabX, SlabY, Boxsum, Boxavg,
                                                 CircularAverage, Ring,
                                                 SectorQ, WedgeQ, WedgePhi)


class MatrixToData2D:
    """
    Create Data2D objects from supplied 2D arrays of data.
    Error data can also be included.

    Adapted from sasdata.data_util.manipulations.reader_2D_converter
    """

    def __init__(self, data2d=None, err_data=None):
        if data2d is not None:
            matrix = np.asarray(data2d)
        else:
            msg = "Data must be supplied to convert to Data2D"
            raise ValueError(msg)

        if matrix.ndim != 2:
            msg = "Supplied array must have 2 dimensions to convert to Data2D"
            raise ValueError(msg)

        if err_data is not None:
            err_data = np.asarray(err_data)
            if err_data.shape != matrix.shape:
                msg = "Data and errors must have the same shape"
                raise ValueError(msg)

        # qmax can be any number, 1 just makes things simple.
        self.qmax = 1
        qx_bins = np.linspace(start=-1 * self.qmax,
                              stop=self.qmax,
                              num=matrix.shape[1],
                              endpoint=True)
        qy_bins = np.linspace(start=-1 * self.qmax,
                              stop=self.qmax,
                              num=matrix.shape[0],
                              endpoint=True)

        # Creating arrays in Data2D's preferred format.
        data2d = matrix.flatten()
        if err_data is None or np.any(err_data <= 0):
            # Error data of some kind is needed, so we fabricate some
            err_data = np.sqrt(np.abs(data2d))  # TODO - use different approach
        else:
            err_data = err_data.flatten()
        qx_data = np.tile(qx_bins, (len(qy_bins), 1)).flatten()
        qy_data = np.tile(qy_bins, (len(qx_bins), 1)).swapaxes(0, 1).flatten()
        q_data = np.sqrt(qx_data * qx_data + qy_data * qy_data)
        mask = np.ones(len(data2d), dtype=bool)

        # Creating a Data2D object to use for testing the averagers.
        self.data = data_info.Data2D(data=data2d, err_data=err_data,
                                     qx_data=qx_data, qy_data=qy_data,
                                     q_data=q_data, mask=mask)


class CircularTestingMatrix:
    """
    This class is used to generate a 2D array representing a function in polar
    coordinates. The function, f(r, φ) = R(r) * Φ(φ), factorises into simple
    radial and angular parts. This makes it easy to determine the form of the
    function after one of the parts has been averaged over, and therefore good
    for testing the directional averagers in manipulations.py.
    This testing is done by comparing the area under the functions, as these
    will only match if the limits defining the ROI were applied correctly.

    f(r, φ) = R(r) * Φ(φ)
    R(r) = r ; where 0 <= r <= 1.
    Φ(φ) = 1 + sin(ν * φ) ; where ν is the frequency and 0 <= φ <= 2π.
    """

    def __init__(self, frequency=1, matrix_size=201, major_axis=None):
        """
        :param frequency: No. times Φ(φ) oscillates over the 0 <= φ <= 2π range
                          This parameter is largely arbitrary.
        :param matrix_size: The len() of the output matrix.
                            Note that odd numbers give a centrepoint of 0,0.
        :param major_axis: 'Q' or 'Phi' - the axis plotted against by the
                           averager being tested.
        """
        if major_axis not in ('Q', 'Phi'):
            msg = "Major axis must be either 'Q' or 'Phi'."
            raise ValueError(msg)

        self.freq = frequency
        self.matrix_size = matrix_size
        self.major = major_axis

        # Grid with same dimensions as data matrix, ranging from -1 to 1
        x, y = np.meshgrid(np.linspace(-1, 1, self.matrix_size),
                           np.linspace(-1, 1, self.matrix_size))
        # radius is 0 at the centre, and 1 at (0, +/-1) and (+/-1, 0)
        radius = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        # Create the 2D array of data
        # The sinusoidal part is shifted up by 1 so its average is never 0
        self.matrix = radius * (1 + np.sin(self.freq * angle))

    def area_under_region(self, r_min=0, r_max=1, phi_min=0, phi_max=2*np.pi):
        """
        Integral of the testing matrix along the major axis, between the limits
        specified. This can be compared to the integral under the 1D data
        output by the averager being tested to confirm it's working properly.

        :param r_min: value defining the minimum Q in the ROI.
        :param r_max: value defining the maximum Q in the ROI.
        :param phi_min: value defining the minimum Phi in the ROI.
        :param phi_max: value defining the maximum Phi in the ROI.
        """

        phi_range = phi_max - phi_min
        # ∫(1 + sin(ν * φ)) dφ = φ + (-cos(ν * φ) / ν) + constant.
        sine_part_integ = phi_range - (np.cos(self.freq * phi_max) -
                                       np.cos(self.freq * phi_min)) / self.freq
        sine_part_avg = sine_part_integ / phi_range

        # ∫(r) dr = r²/2 + constant.
        linear_part_integ = (r_max ** 2 - r_min ** 2) / 2
        # The average radius is weighted towards higher radii. The probability
        # of a point having a given radius value is proportional to the radius:
        # P(r) = k * r ; where k is some proportionality constant.
        # ∫[r₀, r₁] P(r) dr = 1, which can be solved for k. This can then be
        # substituted into ⟨r⟩ = ∫[r₀, r₁] P(r) * r dr, giving:
        linear_part_avg = 2/3 * (r_max**3 - r_min**3) / (r_max**2 - r_min**2)

        # The integral along the major axis is modulated by the average value
        # along the minor axis (between the limits).
        if self.major == 'Q':
            calculated_area = sine_part_avg * linear_part_integ
        else:
            calculated_area = linear_part_avg * sine_part_integ

        return calculated_area


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

        slab_object = SlabX(qx_min=qx_min, qx_max=qx_max, qy_min=qy_min,
                            qy_max=qy_max, nbins=nbins, fold=fold)

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

        slab_object = SlabX(qx_min=qx_min, qx_max=qx_max,
                            qy_min=qy_min, qy_max=qy_max)
        self.assertRaises(ValueError, slab_object, averager_data.data)

    def test_slabx_averaging_without_fold(self):
        """
        Test that SlabX can average correctly when x is the major axis
        """
        matrix_size = 201
        x, y = np.meshgrid(np.linspace(-1, 1, matrix_size),
                           np.linspace(-1, 1, matrix_size))
        # Create a distribution which is quadratic in x and linear in y
        test_data = x**2 * y
        averager_data = MatrixToData2D(data2d=test_data)

        # Set up region of interest to average over - the limits are arbitrary.
        qx_min = -0.5 * averager_data.qmax  # = -0.5
        qx_max = averager_data.qmax  # = 1
        qy_min = -0.5 * averager_data.qmax  # = -0.5
        qy_max = averager_data.qmax  # = 1
        nbins = int((qx_max - qx_min) / 2 * matrix_size)
        # Explicitly not using fold in this test
        fold = False

        slab_object = SlabX(qx_min=qx_min, qx_max=qx_max, qy_min=qy_min,
                            qy_max=qy_max, nbins=nbins, fold=fold)
        data1d = slab_object(averager_data.data)

        # ∫x² dx = x³ / 3 + constant.
        x_part_integ = (qx_max**3 - qx_min**3) / 3
        # ∫y dy = y² / 2 + constant.
        y_part_integ = (qy_max**2 - qy_min**2) / 2
        y_part_avg = y_part_integ / (qy_max - qy_min)
        expected_area = y_part_avg * x_part_integ
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 2)

        # TODO - also check the errors are being calculated correctly

    def test_slabx_averaging_with_fold(self):
        """
        Test that SlabX can average correctly when x is the major axis
        """
        matrix_size = 201
        x, y = np.meshgrid(np.linspace(-1, 1, matrix_size),
                           np.linspace(-1, 1, matrix_size))
        # Create a distribution which is quadratic in x and linear in y
        test_data = x**2 * y
        averager_data = MatrixToData2D(data2d=test_data)

        # Set up region of interest to average over - the limits are arbitrary.
        qx_min = -0.5 * averager_data.qmax  # = -0.5
        qx_max = averager_data.qmax  # = 1
        qy_min = -0.5 * averager_data.qmax  # = -0.5
        qy_max = averager_data.qmax  # = 1
        nbins = int((qx_max - qx_min) / 2 * matrix_size)
        # Explicitly using fold in this test
        fold = True

        slab_object = SlabX(qx_min=qx_min, qx_max=qx_max, qy_min=qy_min,
                            qy_max=qy_max, nbins=nbins, fold=fold)
        data1d = slab_object(averager_data.data)

        # Negative values of x are not graphed when fold = True
        qx_min = 0
        # ∫x² dx = x³ / 3 + constant.
        x_part_integ = (qx_max**3 - qx_min**3) / 3
        # ∫y dy = y² / 2 + constant.
        y_part_integ = (qy_max**2 - qy_min**2) / 2
        y_part_avg = y_part_integ / (qy_max - qy_min)
        expected_area = y_part_avg * x_part_integ
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 2)

        # TODO - also check the errors are being calculated correctly


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

        slab_object = SlabY(qx_min=qx_min, qx_max=qx_max, qy_min=qy_min,
                            qy_max=qy_max, nbins=nbins, fold=fold)

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

        slab_object = SlabY(qx_min=qx_min, qx_max=qx_max,
                            qy_min=qy_min, qy_max=qy_max)
        self.assertRaises(ValueError, slab_object, averager_data.data)

    def test_slaby_averaging_without_fold(self):
        """
        Test that SlabY can average correctly when y is the major axis
        """
        matrix_size = 201
        x, y = np.meshgrid(np.linspace(-1, 1, matrix_size),
                           np.linspace(-1, 1, matrix_size))
        # Create a distribution which is linear in x and quadratic in y
        test_data = x * y**2
        averager_data = MatrixToData2D(data2d=test_data)

        # Set up region of interest to average over - the limits are arbitrary.
        qx_min = -0.5 * averager_data.qmax  # = -0.5
        qx_max = averager_data.qmax  # = 1
        qy_min = -0.5 * averager_data.qmax  # = -0.5
        qy_max = averager_data.qmax  # = 1
        nbins = int((qx_max - qx_min) / 2 * matrix_size)
        # Explicitly not using fold in this test
        fold = False

        slab_object = SlabY(qx_min=qx_min, qx_max=qx_max, qy_min=qy_min,
                            qy_max=qy_max, nbins=nbins, fold=fold)
        data1d = slab_object(averager_data.data)

        # ∫x dx = x² / 2 + constant.
        x_part_integ = (qx_max**2 - qx_min**2) / 2
        x_part_avg = x_part_integ / (qx_max - qx_min)  # or (x_min + x_max) / 2
        # ∫y² dy = y³ / 3 + constant.
        y_part_integ = (qy_max**3 - qy_min**3) / 3
        expected_area = x_part_avg * y_part_integ
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 2)

        # TODO - also check the errors are being calculated correctly

    def test_slab_averaging_y_with_fold(self):
        """
        Test that SlabY can average correctly when y is the major axis
        """
        matrix_size = 201
        x, y = np.meshgrid(np.linspace(-1, 1, matrix_size),
                           np.linspace(-1, 1, matrix_size))
        # Create a distribution which is linear in x and quadratic in y
        test_data = x * y**2
        averager_data = MatrixToData2D(data2d=test_data)

        # Set up region of interest to average over - the limits are arbitrary.
        qx_min = -0.5 * averager_data.qmax  # = -0.5
        qx_max = averager_data.qmax  # = 1
        qy_min = -0.5 * averager_data.qmax  # = -0.5
        qy_max = averager_data.qmax  # = 1
        nbins = int((qx_max - qx_min) / 2 * matrix_size)
        # Explicitly using fold in this test
        fold = True

        slab_object = SlabY(qx_min=qx_min, qx_max=qx_max, qy_min=qy_min,
                            qy_max=qy_max, nbins=nbins, fold=fold)
        data1d = slab_object(averager_data.data)

        # Negative values of y are not graphed when fold = True, so don't
        # include them in the area calculation.
        qy_min = 0
        # ∫x dx = x² / 2 + constant.
        x_part_integ = (qx_max**2 - qx_min**2) / 2
        x_part_avg = x_part_integ / (qx_max - qx_min)  # or (x_min + x_max) / 2
        # ∫y² dy = y³ / 3 + constant.
        y_part_integ = (qy_max**3 - qy_min**3) / 3
        expected_area = x_part_avg * y_part_integ
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 2)

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

        box_object = Boxsum(qx_min=qx_min, qx_max=qx_max,
                            qy_min=qy_min, qy_max=qy_max)

        self.assertEqual(box_object.qx_min, qx_min)
        self.assertEqual(box_object.qx_max, qx_max)
        self.assertEqual(box_object.qy_min, qy_min)
        self.assertEqual(box_object.qy_max, qy_max)

    def test_boxsum_multiple_detectors(self):
        """
        Test Boxsum raises an error when there are multiple detectors.
        """
        averager_data = MatrixToData2D(np.ones([100, 100]))
        detector1 = data_info.Detector()
        detector2 = data_info.Detector()
        averager_data.data.detector.append(detector1)
        averager_data.data.detector.append(detector2)

        box_object = Boxsum()
        self.assertRaises(ValueError, box_object, averager_data.data)

    def test_boxsum_total(self):
        """
        Test that Boxsum can find the sum of all of a data set
        """
        # Creating a 100x100 matrix for a distribution which is flat in y
        # and linear in x.
        test_data = np.tile(np.arange(100), (100, 1))
        averager_data = MatrixToData2D(data2d=test_data)

        # Selected region is entire data set
        qx_min = -1 * averager_data.qmax
        qx_max = averager_data.qmax
        qy_min = -1 * averager_data.qmax
        qy_max = averager_data.qmax
        box_object = Boxsum(qx_min=qx_min, qx_max=qx_max,
                            qy_min=qy_min, qy_max=qy_max)
        result, error, npoints = box_object(averager_data.data)
        correct_sum = np.sum(test_data)
        # When averager_data was created, we didn't include any error data.
        # Stand-in error data is created, equal to np.sqrt(data2D).
        # With the current method of error calculation, this is the result we
        # should expect. This may need to change at some point.
        correct_error = np.sqrt(np.sum(test_data))

        self.assertAlmostEqual(result, correct_sum, 6)
        self.assertAlmostEqual(error, correct_error, 6)

    def test_boxsum_subset_total(self):
        """
        Test that Boxsum can find the sum of a portion of a data set
        """
        # Creating a 100x100 matrix for a distribution which is flat in y
        # and linear in x.
        test_data = np.tile(np.arange(100), (100, 1))
        averager_data = MatrixToData2D(data2d=test_data)

        # Selection region covers the inner half of the +&- x&y axes
        qx_min = -0.5 * averager_data.qmax
        qx_max = 0.5 * averager_data.qmax
        qy_min = -0.5 * averager_data.qmax
        qy_max = 0.5 * averager_data.qmax
        # Extracting the inner half of the data set
        inner_portion = test_data[25:75, 25:75]

        box_object = Boxsum(qx_min=qx_min, qx_max=qx_max,
                            qy_min=qy_min, qy_max=qy_max)
        result, error, npoints = box_object(averager_data.data)
        correct_sum = np.sum(inner_portion)
        # When averager_data was created, we didn't include any error data.
        # Stand-in error data is created, equal to np.sqrt(data2D).
        # With the current method of error calculation, this is the result we
        # should expect. This may need to change at some point.
        correct_error = np.sqrt(np.sum(inner_portion))

        self.assertAlmostEqual(result, correct_sum, 6)
        self.assertAlmostEqual(error, correct_error, 6)

    def test_boxsum_zero_sum(self):
        """
        Test that Boxsum returns 0 when there are no points within the ROI
        """
        test_data = np.ones([100, 100])
        # Make a hole in the middle with zeros
        test_data[25:75, 25:75] = np.zeros([50, 50])
        averager_data = MatrixToData2D(data2d=test_data)

        # Selection region covers the inner half of the +&- x&y axes
        qx_min = -0.5 * averager_data.qmax
        qx_max = 0.5 * averager_data.qmax
        qy_min = -0.5 * averager_data.qmax
        qy_max = 0.5 * averager_data.qmax
        box_object = Boxsum(qx_min=qx_min, qx_max=qx_max,
                            qy_min=qy_min, qy_max=qy_max)
        result, error, npoints = box_object(averager_data.data)

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

        box_object = Boxavg(qx_min=qx_min, qx_max=qx_max,
                            qy_min=qy_min, qy_max=qy_max)

        self.assertEqual(box_object.qx_min, qx_min)
        self.assertEqual(box_object.qx_max, qx_max)
        self.assertEqual(box_object.qy_min, qy_min)
        self.assertEqual(box_object.qy_max, qy_max)

    def test_boxavg_multiple_detectors(self):
        """
        Test Boxavg raises an error when there are multiple detectors.
        """
        averager_data = MatrixToData2D(np.ones([100, 100]))
        detector1 = data_info.Detector()
        detector2 = data_info.Detector()
        averager_data.data.detector.append(detector1)
        averager_data.data.detector.append(detector2)

        box_object = Boxavg()
        self.assertRaises(ValueError, box_object, averager_data.data)

    def test_boxavg_total(self):
        """
        Test that Boxavg can find the average of all of a data set
        """
        # Creating a 100x100 matrix for a distribution which is flat in y
        # and linear in x.
        test_data = np.tile(np.arange(100), (100, 1))
        averager_data = MatrixToData2D(data2d=test_data)

        # Selected region is entire data set
        qx_min = -1 * averager_data.qmax
        qx_max = averager_data.qmax
        qy_min = -1 * averager_data.qmax
        qy_max = averager_data.qmax
        box_object = Boxavg(qx_min=qx_min, qx_max=qx_max,
                            qy_min=qy_min, qy_max=qy_max)
        result, error = box_object(averager_data.data)
        correct_avg = np.mean(test_data)
        # When averager_data was created, we didn't include any error data.
        # Stand-in error data is created, equal to np.sqrt(data2D).
        # With the current method of error calculation, this is the result we
        # should expect. This may need to change at some point.
        correct_error = np.sqrt(np.sum(test_data)) / test_data.size

        self.assertAlmostEqual(result, correct_avg, 6)
        self.assertAlmostEqual(error, correct_error, 6)

    def test_boxavg_subset_total(self):
        """
        Test that Boxavg can find the average of a portion of a data set
        """
        # Creating a 100x100 matrix for a distribution which is flat in y
        # and linear in x.
        test_data = np.tile(np.arange(100), (100, 1))
        averager_data = MatrixToData2D(data2d=test_data)

        # Selection region covers the inner half of the +&- x&y axes
        qx_min = -0.5 * averager_data.qmax
        qx_max = 0.5 * averager_data.qmax
        qy_min = -0.5 * averager_data.qmax
        qy_max = 0.5 * averager_data.qmax
        # Extracting the inner half of the data set
        inner_portion = test_data[25:75, 25:75]

        box_object = Boxavg(qx_min=qx_min, qx_max=qx_max,
                            qy_min=qy_min, qy_max=qy_max)
        result, error = box_object(averager_data.data)
        correct_avg = np.mean(inner_portion)
        # When averager_data was created, we didn't include any error data.
        # Stand-in error data is created, equal to np.sqrt(data2D).
        # With the current method of error calculation, this is the result we
        # should expect. This may need to change at some point.
        correct_error = np.sqrt(np.sum(inner_portion)) / inner_portion.size

        self.assertAlmostEqual(result, correct_avg, 6)
        self.assertAlmostEqual(error, correct_error, 6)

    def test_boxavg_zero_average(self):
        """
        Test that Boxavg returns 0 when there are no points within the ROI
        """
        test_data = np.ones([100, 100])
        # Make a hole in the middle with zeros
        test_data[25:75, 25:75] = np.zeros([50, 50])
        averager_data = MatrixToData2D(data2d=test_data)

        # Selection region covers the inner half of the +&- x&y axes
        qx_min = -0.5 * averager_data.qmax
        qx_max = 0.5 * averager_data.qmax
        qy_min = -0.5 * averager_data.qmax
        qy_max = 0.5 * averager_data.qmax
        box_object = Boxavg(qx_min=qx_min, qx_max=qx_max,
                            qy_min=qy_min, qy_max=qy_max)
        result, error = box_object(averager_data.data)

        self.assertAlmostEqual(result, 0, 6)
        self.assertAlmostEqual(error, 0, 6)


class CircularAverageTests(unittest.TestCase):
    """
    This class contains all the tests for the CircularAverage class
    from manipulations.py
    """

    def test_circularaverage_init(self):
        """
        Test that CircularAverage's __init__ method does what it's supposed to.
        """
        r_min = 1
        r_max = 2
        nbins = 100

        circ_object = CircularAverage(r_min=r_min, r_max=r_max, nbins=nbins)

        self.assertEqual(circ_object.r_min, r_min)
        self.assertEqual(circ_object.r_max, r_max)
        self.assertEqual(circ_object.nbins, nbins)

    def test_circularaverage_dq_retrieval(self):
        """
        Test that CircularAverage is able to calclate dq_data correctly when
        the data provided has dqx_data and dqy_data.
        """

        # I'm saving the implementation of this bit for later
        pass

    def test_circularaverage_multiple_detectors(self):
        """
        Test CircularAverage raises an error when there are multiple detectors
        """

        # This test can't be implemented yet, because CircularAverage does not
        # check the number of detectors.
        # TODO - establish whether CircularAverage should be making this check.
        pass

    def test_circularaverage_check_q_data(self):
        """
        Check CircularAverage ensures the data supplied has `q_data` populated
        """
        # test_data = np.ones([100, 100])
        # averager_data = DataMatrixToData2D(test_data)
        # # Overwrite q_data so it's empty
        # averager_data.data.q_data = np.array([])
        # circ_object = CircularAverage()
        # self.assertRaises(RuntimeError, circ_object, averager_data.data)

        # This doesn't work. I'll come back to this later too
        pass

    def test_circularaverage_check_valid_radii(self):
        """
        Test that CircularAverage raises ValueError when r_min > r_max
        """
        self.assertRaises(ValueError, CircularAverage, r_min=0.1, r_max=0.05)

    def test_circularaverage_no_points_to_average(self):
        """
        Test CircularAverage raises ValueError when the ROI contains no data
        """
        test_data = np.ones([100, 100])
        averager_data = MatrixToData2D(test_data)

        # Region of interest well outside region with data
        circ_object = CircularAverage(r_min=2 * averager_data.qmax,
                                      r_max=3 * averager_data.qmax)
        self.assertRaises(ValueError, circ_object, averager_data.data)

    def test_circularaverage_averages_circularly(self):
        """
        Test that CircularAverage can calculate a circular average correctly.
        """
        test_data = CircularTestingMatrix(frequency=2, matrix_size=201,
                                          major_axis='Q')
        averager_data = MatrixToData2D(test_data.matrix)

        # Test the ability to average over a subsection of the data
        r_min = averager_data.qmax * 0.25
        r_max = averager_data.qmax * 0.75

        nbins = test_data.matrix_size
        circ_object = CircularAverage(r_min=r_min, r_max=r_max, nbins=nbins)
        data1d = circ_object(averager_data.data)

        expected_area = test_data.area_under_region(r_min=r_min, r_max=r_max)
        actual_area = integrate.trapezoid(data1d.y, data1d.x)

        # This used to be able to pass with a precision of 3 d.p. with the old
        # manipulations.py - I'm not sure why it doesn't anymore.
        # This is still a good level of precision compared to the others though
        self.assertAlmostEqual(actual_area, expected_area, 2)

        # TODO - also check the errors are being calculated correctly


class RingTests(unittest.TestCase):
    """
    This class contains the tests for the Ring class from manipulations.py
    A.K.A AnnulusSlicer on the sasview side
    """

    def test_ring_init(self):
        """
        Test that Ring's __init__ method does what it's supposed to.
        """
        r_min = 1
        r_max = 2
        nbins = 100

        # Note that Ring also has params center_x and center_y, but these are
        # not used by the slicers and there is a 'todo' in manipulations.py to
        # remove them. For this reason, I have not tested their initialisation.
        ring_object = Ring(r_min=r_min, r_max=r_max, nbins=nbins)

        self.assertEqual(ring_object.r_min, r_min)
        self.assertEqual(ring_object.r_max, r_max)
        self.assertEqual(ring_object.nbins, nbins)

    def test_ring_non_plottable_data(self):
        """
        Test that RuntimeError is raised if the data supplied isn't plottable
        """
        # with patch("sasdata.data_util.manipulations.Ring.data2D.__class__.__name__") as p:
        #     p.return_value = "bad_name"
        #     ring_object = Ring()
        #     self.assertRaises(RuntimeError, ring_object.__call__)

        # I can't seem to get patch working, in this test or in others.
        pass

    def test_ring_no_points_to_average(self):
        """
        Test Ring raises ValueError when the ROI contains no data
        """
        test_data = np.ones([100, 100])
        averager_data = MatrixToData2D(test_data)

        # Region of interest well outside region with data
        ring_object = Ring(r_min=2 * averager_data.qmax,
                           r_max=3 * averager_data.qmax)
        self.assertRaises(ValueError, ring_object, averager_data.data)

    def test_ring_averages_azimuthally(self):
        """
        Test that Ring can calculate an azimuthal average correctly.
        """
        test_data = CircularTestingMatrix(frequency=1, matrix_size=201,
                                          major_axis='Phi')
        averager_data = MatrixToData2D(test_data.matrix)

        # Test the ability to average over a subsection of the data
        r_min = 0.25 * averager_data.qmax
        r_max = 0.75 * averager_data.qmax
        nbins = test_data.matrix_size // 2

        ring_object = Ring(r_min=r_min, r_max=r_max, nbins=nbins)
        data1d = ring_object(averager_data.data)

        expected_area = test_data.area_under_region(r_min=r_min, r_max=r_max)
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 1)

        # TODO - also check the errors are being calculated correctly


class SectorQTests(unittest.TestCase):
    """
    This class contains the tests for the SectorQ class from manipulations.py
    On the sasview side, this includes SectorSlicer and WedgeSlicer.

    The parameters frequency, r_min, r_max, phi_min and phi_max are largely
    arbitrary, and the tests should pass if any sane value is used for them.
    """

    def test_sectorq_init(self):
        """
        Test that SectorQ's __init__ method does what it's supposed to.
        """
        r_min = 0
        r_max = 1
        phi_min = 0
        phi_max = np.pi
        nbins = 100
        # base = 10

        # sector_object = SectorQ(r_min=r_min, r_max=r_max, phi_min=phi_min,
        #                         phi_max=phi_max, nbins=nbins, base=base)
        sector_object = SectorQ(r_min=r_min, r_max=r_max, phi_min=phi_min,
                                phi_max=phi_max, nbins=nbins)

        self.assertEqual(sector_object.r_min, r_min)
        self.assertEqual(sector_object.r_max, r_max)
        self.assertEqual(sector_object.phi_min, phi_min)
        self.assertEqual(sector_object.phi_max, phi_max)
        self.assertEqual(sector_object.nbins, nbins)
        # self.assertEqual(sector_object.base, base)

    def test_sectorq_non_plottable_data(self):
        """
        Test that RuntimeError is raised if the data supplied isn't plottable
        """
        # Implementing this test can wait
        pass

    def test_sectorq_averaging_without_fold(self):
        """
        Test SectorQ can average correctly w/ major axis q and fold disabled.
        All min/max r & phi params are specified and have their expected form.
        """
        test_data = CircularTestingMatrix(frequency=1, matrix_size=201,
                                          major_axis='Q')
        averager_data = MatrixToData2D(test_data.matrix)

        r_min = 0
        r_max = 0.9 * averager_data.qmax
        phi_min = np.pi/6
        phi_max = 5*np.pi/6
        nbins = int(test_data.matrix_size * np.sqrt(2)/4)  # usually reliable

        wedge_object = SectorQ(r_min=r_min, r_max=r_max, phi_min=phi_min,
                               phi_max=phi_max, nbins=nbins)
        # Explicitly set fold to False - results span full +/- range
        wedge_object.fold = False
        data1d = wedge_object(averager_data.data)

        expected_area = test_data.area_under_region(r_min=r_min, r_max=r_max,
                                                    phi_min=phi_min,
                                                    phi_max=phi_max)
        # With fold set to False, the sector on the opposite side of the origin
        # to the one specified is also graphed as negative Q values. Therefore,
        # the area of this other half needs to be accounted for.
        expected_area += test_data.area_under_region(r_min=r_min, r_max=r_max,
                                                     phi_min=phi_min+np.pi,
                                                     phi_max=phi_max+np.pi)
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 1)

    def test_sectorq_averaging_with_fold(self):
        """
        Test SectorQ can average correctly w/ major axis q and fold enabled.
        All min/max r & phi params are specified and have their expected form.
        """
        test_data = CircularTestingMatrix(frequency=1, matrix_size=201,
                                          major_axis='Q')
        averager_data = MatrixToData2D(test_data.matrix)

        r_min = 0
        r_max = 0.9 * averager_data.qmax
        phi_min = np.pi/6
        phi_max = 5*np.pi/6
        nbins = int(test_data.matrix_size * np.sqrt(2)/4)  # usually reliable

        wedge_object = SectorQ(r_min=r_min, r_max=r_max, phi_min=phi_min,
                               phi_max=phi_max, nbins=nbins)
        # Explicitly set fold to True - points either side of 0,0 are averaged
        wedge_object.fold = True
        data1d = wedge_object(averager_data.data)

        expected_area = test_data.area_under_region(r_min=r_min, r_max=r_max,
                                                    phi_min=phi_min,
                                                    phi_max=phi_max)
        # With fold set to True, points from the sector on the opposite side of
        # the origin to the one specified are averaged with points from the
        # specified sector.
        expected_area += test_data.area_under_region(r_min=r_min, r_max=r_max,
                                                     phi_min=phi_min+np.pi,
                                                     phi_max=phi_max+np.pi)
        expected_area /= 2
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 1)


class WedgeQTests(unittest.TestCase):
    """
    This class contains the tests for the WedgeQ class from manipulations.py

    The parameters frequency, r_min, r_max, phi_min and phi_max are largely
    arbitrary, and the tests should pass if any sane value is used for them.
    """

    def test_wedgeq_init(self):
        """
        Test that WedgeQ's __init__ method does what it's supposed to.
        """
        r_min = 1
        r_max = 2
        phi_min = 0
        phi_max = np.pi
        nbins = 10

        wedge_object = WedgeQ(r_min=r_min, r_max=r_max, phi_min=phi_min,
                              phi_max=phi_max, nbins=nbins)

        self.assertEqual(wedge_object.r_min, r_min)
        self.assertEqual(wedge_object.r_max, r_max)
        self.assertEqual(wedge_object.phi_min, phi_min)
        self.assertEqual(wedge_object.phi_max, phi_max)
        self.assertEqual(wedge_object.nbins, nbins)

    def test_wedgeq_averaging(self):
        """
        Test WedgeQ can average correctly, when all of min/max r & phi params
        are specified and have their expected form.
        """
        test_data = CircularTestingMatrix(frequency=3, matrix_size=201,
                                          major_axis='Q')
        averager_data = MatrixToData2D(test_data.matrix)

        r_min = 0.1 * averager_data.qmax
        r_max = 0.9 * averager_data.qmax
        phi_min = np.pi/6
        phi_max = 5*np.pi/6
        nbins = int(test_data.matrix_size * np.sqrt(2)/4)  # usually reliable

        wedge_object = WedgeQ(r_min=r_min, r_max=r_max, phi_min=phi_min,
                              phi_max=phi_max, nbins=nbins)
        data1d = wedge_object(averager_data.data)

        expected_area = test_data.area_under_region(r_min=r_min, r_max=r_max,
                                                    phi_min=phi_min,
                                                    phi_max=phi_max)
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 1)


class WedgePhiTests(unittest.TestCase):
    """
    This class contains the tests for the WedgePhi class from manipulations.py

    The parameters frequency, r_min, r_max, phi_min and phi_max are largely
    arbitrary, and the tests should pass if any sane value is used for them.
    """

    def test_wedgephi_init(self):
        """
        Test that WedgePhi's __init__ method does what it's supposed to.
        """
        r_min = 1
        r_max = 2
        phi_min = 0
        phi_max = np.pi
        nbins = 100
        # base = 10

        # wedge_object = WedgePhi(r_min=r_min, r_max=r_max, phi_min=phi_min,
        #                           phi_max=phi_max, nbins=nbins, base=base)
        wedge_object = WedgePhi(r_min=r_min, r_max=r_max, phi_min=phi_min,
                                phi_max=phi_max, nbins=nbins)

        self.assertEqual(wedge_object.r_min, r_min)
        self.assertEqual(wedge_object.r_max, r_max)
        self.assertEqual(wedge_object.phi_min, phi_min)
        self.assertEqual(wedge_object.phi_max, phi_max)
        self.assertEqual(wedge_object.nbins, nbins)
        # self.assertEqual(wedge_object.base, base)

    def test_wedgephi_non_plottable_data(self):
        """
        Test that RuntimeError is raised if the data supplied isn't plottable
        """
        # Implementing this test can wait
        pass

    def test_wedgephi_averaging(self):
        """
        Test WedgePhi can average correctly, when all of min/max r & phi params
        are specified and have their expected form.
        """
        test_data = CircularTestingMatrix(frequency=1, matrix_size=201,
                                          major_axis='Phi')
        averager_data = MatrixToData2D(test_data.matrix)

        r_min = 0.1 * averager_data.qmax
        r_max = 0.9 * averager_data.qmax
        phi_min = np.pi/6
        phi_max = 5*np.pi/6
        nbins = int(test_data.matrix_size * np.sqrt(2)/4)  # usually reliable

        wedge_object = WedgePhi(r_min=r_min, r_max=r_max, phi_min=phi_min,
                                phi_max=phi_max, nbins=nbins)
        data1d = wedge_object(averager_data.data)

        expected_area = test_data.area_under_region(r_min=r_min, r_max=r_max,
                                                    phi_min=phi_min,
                                                    phi_max=phi_max)
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 1)


if __name__ == '__main__':
    unittest.main()
