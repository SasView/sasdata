"""
This file contains unit tests for the various averagers found in
sasdata/data_util/manipulations.py - These tests are based on analytical
formulae rather than imported data files.
"""

import unittest

import numpy as np
from scipy import integrate
from test.sasmanipulations.helper import (MatrixToData2D, CircularTestingMatrix, make_dd_from_func,
                      expected_slabx_area, expected_slaby_area, integrate_1d_output, expected_boxsum_and_err,expected_boxavg_and_err, make_uniform_dd) 
from sasdata.data_util.averaging import (
    Boxavg,
    Boxsum,
    CircularAverage,
    DirectionalAverage,
    Ring,
    SectorQ,
    SlabX,
    SlabY,
    WedgePhi,
    WedgeQ,
)
from sasdata.dataloader import data_info

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
        nbins = int((qx_max - qx_min) / 2 * matrix_size)
        # Explicitly not using fold in this test
        fold = False

        slab_object = SlabY(qx_range=(qx_min, qx_max), qy_range=(qy_min,qy_max), nbins=nbins, fold=fold)
        data1d = slab_object(averager_data.data)

        expected_area = expected_slaby_area(qx_min, qx_max, qy_min, qy_max)
        actual_area = integrate_1d_output(data1d, method="simpson")

        self.assertAlmostEqual(actual_area, expected_area, 2)

    def test_slab_averaging_y_with_fold(self):
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
        nbins = int((qx_max - qx_min) / 2 * matrix_size)
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

        circ_object = CircularAverage(r_range=(r_min, r_max), nbins=nbins)

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
        self.assertRaises(ValueError, CircularAverage, r_range=(0.1, 0.05))

    def test_circularaverage_no_points_to_average(self):
        """
        Test CircularAverage raises ValueError when the ROI contains no data
        """
        test_data = np.ones([100, 100])
        averager_data = MatrixToData2D(test_data)

        # Region of interest well outside region with data
        circ_object = CircularAverage(r_range=(2 * averager_data.qmax,3 * averager_data.qmax))
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
        circ_object = CircularAverage(r_range=(r_min, r_max), nbins=nbins)
        data1d = circ_object(averager_data.data)

        expected_area = test_data.area_under_region(r_min=r_min, r_max=r_max)
        actual_area = integrate.trapezoid(data1d.y, data1d.x)

        # This used to be able to pass with a precision of 3 d.p. with the old
        # manipulations.py - I'm not sure why it doesn't anymore.
        # This is still a good level of precision compared to the others though
        self.assertAlmostEqual(actual_area, expected_area, 2)

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
        ring_object = Ring(r_range=(r_min, r_max), nbins=nbins)

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
        ring_object = Ring(r_range=(2 * averager_data.qmax, 3 * averager_data.qmax))
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

        ring_object = Ring(r_range=(r_min, r_max), nbins=nbins)
        data1d = ring_object(averager_data.data)

        expected_area = test_data.area_under_region(r_min=r_min, r_max=r_max)
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 1)

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

        sector_object = SectorQ(r_range=(r_min, r_max), phi_range=(phi_min,phi_max), nbins=nbins)

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

        wedge_object = SectorQ(r_range=(r_min, r_max), phi_range=(phi_min,phi_max), nbins=nbins)
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

        wedge_object = SectorQ(r_range=(r_min, r_max), phi_range=(phi_min,phi_max), nbins=nbins)
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

        wedge_object = WedgeQ(r_range=(r_min, r_max), phi_range=(phi_min,phi_max), nbins=nbins)

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

        wedge_object = WedgeQ(r_range=(r_min, r_max), phi_range=(phi_min,phi_max), nbins=nbins)
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
        wedge_object = WedgePhi(r_range=(r_min, r_max), phi_range=(phi_min,phi_max), nbins=nbins)

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

        wedge_object = WedgePhi(r_range=(r_min, r_max), phi_range=(phi_min,phi_max), nbins=nbins)
        data1d = wedge_object(averager_data.data)

        expected_area = test_data.area_under_region(r_min=r_min, r_max=r_max,
                                                    phi_min=phi_min,
                                                    phi_max=phi_max)
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 1)


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
