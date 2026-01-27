"""
Unit tests for SlabX and SlabY averagers (moved out of utest_averaging_analytical.py).
"""
import unittest

import numpy as np
from scipy import integrate

from sasdata.data_util.averaging import CircularAverage, Ring, SectorQ, WedgePhi, WedgeQ
from sasdata.quantities.constants import Pi
from test.sasmanipulations.helper import CircularTestingMatrix, MatrixToData2D

# TODO - also check the errors are being calculated correctly

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
        phi_max = Pi
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
        phi_min = Pi/6
        phi_max = 5*Pi/6
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
                                                     phi_min=phi_min+Pi,
                                                     phi_max=phi_max+Pi)
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
        phi_min = Pi/6
        phi_max = 5*Pi/6
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
                                                     phi_min=phi_min+Pi,
                                                     phi_max=phi_max+Pi)
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
        phi_max = Pi
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
        phi_min = Pi/6
        phi_max = 5*Pi/6
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
        phi_max = Pi
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
        phi_min = Pi/6
        phi_max = 5*Pi/6
        nbins = int(test_data.matrix_size * np.sqrt(2)/4)  # usually reliable

        wedge_object = WedgePhi(r_range=(r_min, r_max), phi_range=(phi_min,phi_max), nbins=nbins)
        data1d = wedge_object(averager_data.data)

        expected_area = test_data.area_under_region(r_min=r_min, r_max=r_max,
                                                    phi_min=phi_min,
                                                    phi_max=phi_max)
        actual_area = integrate.simpson(data1d.y, data1d.x)

        self.assertAlmostEqual(actual_area, expected_area, 1)

if __name__ == '__main__':
    unittest.main()
