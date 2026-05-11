import os
import unittest

import numpy as np

from sasdata.data import SasData, sasdata_reader2D_converter
from sasdata.data_util.manipulations import (
    Boxavg,
    Boxsum,
    CircularAverage,
    Ring,
    SectorPhi,
    SectorQ,
    SlabX,
    SlabY,
    position_and_wavelength_to_q,
)
from sasdata.dataset_types import two_dim
from sasdata.metadata import Detector, Instrument, Metadata, Source, Vec3
from sasdata.quantities.constants import Pi, TwoPi
from sasdata.quantities.quantity import Quantity
from sasdata.quantities.units import angstroms, millimeters, none, per_angstrom, per_centimeter
from sasdata.temp_ascii_reader import load_data_default_params as ascii_load_data
from sasdata.temp_hdf5_reader import load_data as hdf_load_data


def find(filename):
    return os.path.join(os.path.dirname(__file__), 'data', filename)


class Averaging(unittest.TestCase):
    """Test averaging manipulations on a flat distribution."""

    def setUp(self):
        """
            Create a flat 2D distribution. All averaging results
            should return the predefined height of the distribution (1.0).
        """
        x_0 = np.ones([100, 100])
        dx_0 = np.ones([100, 100])

        data_contents = {
            "Qx": Quantity(np.arange(100), per_angstrom),
            "Qy": Quantity(np.arange(100), per_angstrom),
            "I": Quantity(x_0, per_centimeter),
            "dI": Quantity(dx_0, per_centimeter)
        }

        wavelength = Quantity(10.0, angstroms)
        source = Source(radiation=None,
                        beam_shape=None,
                        beam_size=None,
                        wavelength=wavelength,
                        wavelength_max=None,
                        wavelength_min=None,
                        wavelength_spread=None)
        detector = Detector(name = None,
                            distance = Quantity(1000.0, millimeters),
                            offset = None,
                            orientation = None,
                            beam_center = Vec3(x=Quantity(0.5 * (len(x_0) - 1), none), y=Quantity(0.5 * (len(x_0) - 1), none), z=None),
                            pixel_size = Vec3(x=Quantity(1.0, millimeters), y=Quantity(1.0, millimeters), z=None),
                            slit_length = None)
        instrument = Instrument(collimations=[],
                                source=source,
                                detector=[detector])
        metadata=Metadata(title=None,
                          run=[],
                          definition=None,
                          process=[],
                          sample=None,
                          instrument=instrument,
                          raw=None)

        self.data = SasData("Test Averaging", data_contents, two_dim, metadata)

        # get_q(dx, dy, det_dist, wavelength) where units are mm,mm,mm,and A
        # respectively.
        self.qmin = position_and_wavelength_to_q(1.0, 1.0, detector.distance.value, source.wavelength.value)
        self.qmax = position_and_wavelength_to_q(49.5, 49.5, detector.distance.value, source.wavelength.value)

        self.qstep = len(x_0)
        x = np.linspace(start=-1 * self.qmax,
                        stop=self.qmax,
                        num=self.qstep,
                        endpoint=True)
        y = np.linspace(start=-1 * self.qmax,
                        stop=self.qmax,
                        num=self.qstep,
                        endpoint=True)
        self.data.x_bins = x
        self.data.y_bins = y
        self.data = sasdata_reader2D_converter(self.data)

    def test_ring_flat_distribution(self):
        """Test ring averaging."""
        r = Ring(r_min=2*self.qmin,
                 r_max=5*self.qmin,
                 center_x=self.data.metadata.instrument.detector[0].beam_center.x,
                 center_y=self.data.metadata.instrument.detector[0].beam_center.y)
        r.nbins_phi = 20

        o = r(self.data)
        for i in range(20):
            self.assertEqual(o._data_contents["I"].value[i], 1.0)

    def test_sectorphi_full(self):
        """Test sector averaging."""
        r = SectorPhi(r_min=self.qmin, r_max=3 * self.qmin,
                      phi_min=0, phi_max=TwoPi)
        r.nbins_phi = 20
        o = r(self.data)
        for i in range(7):
            self.assertEqual(o._data_contents["I"].value[i], 1.0)

    def test_sectorphi_partial(self):
        """Test sector averaging."""
        phi_max = Pi * 1.5
        r = SectorPhi(r_min=self.qmin, r_max=3 * self.qmin,
                      phi_min=0, phi_max=phi_max)
        self.assertEqual(r.phi_max, phi_max)
        r.nbins_phi = 20
        o = r(self.data)
        self.assertEqual(r.phi_max, phi_max)
        for i in range(17):
            self.assertEqual(o._data_contents["I"].value[i], 1.0)


class DataInfoTests(unittest.TestCase):

    def setUp(self):
        filepath = find('MAR07232_rest.h5')
        data_dict = hdf_load_data(filepath)
        self.data = data_dict['sasentry01']

    def test_ring(self):
        """Test ring averaging."""
        if beam_center := self.data.metadata.instrument.detector[0].beam_center:
            center_x = beam_center.x
            center_y = beam_center.y
        else:
            center_x = None
            center_y = None

        r = Ring(r_min=.005, r_max=.01,
                 center_x=center_x,
                 center_y=center_y,
                 nbins=20)
        r.nbins_phi = 20

        o = r(self.data)
        filepath = find('ring_testdata.txt')
        answer_list = ascii_load_data(filepath)
        answer = answer_list[0]

        self.assertEqual(len(answer_list), 1)
        for i in range(r.nbins_phi - 1):
            # Current ascii reader implementation assumes file data is "one_dim"
            self.assertAlmostEqual(o._data_contents["Phi"].value[i], answer._data_contents["Q"].value[i], 4)
            self.assertAlmostEqual(o._data_contents["I"].value[i], answer._data_contents["I"].value[i], 4)
            self.assertAlmostEqual(o._data_contents["I"].variance.value[i], answer._data_contents["I"].variance.value[i], 4)

    def test_circularavg(self):
        """
        Test circular averaging
        The test data was not generated by IGOR.
        """
        r = CircularAverage(r_min=.00, r_max=.025,
                            bin_width=0.0003)
        r.nbins_phi = 20

        o = r(self.data)

        filepath = find('avg_testdata.txt')
        answer = ascii_load_data(filepath)[0]
        for i in range(r.nbins_phi):
            self.assertAlmostEqual(o._data_contents["Q"].value[i], answer._data_contents["Q"].value[i], delta=1e-4)
            self.assertAlmostEqual(o._data_contents["I"].value[i], answer._data_contents["I"].value[i], delta=1e-4)
            self.assertAlmostEqual(o._data_contents["I"].variance.value[i], answer._data_contents["I"].variance.value[i], delta=1e-4)

    def test_box(self):
        """
            Test circular averaging
            The test data was not generated by IGOR.
        """

        r = Boxsum(x_min=.01, x_max=.015, y_min=0.01, y_max=0.015)
        s, ds, npoints = r(self.data)
        self.assertAlmostEqual(s, 34.278990899999997, 4)
        self.assertAlmostEqual(ds, 8.237259999538685, 4)
        self.assertAlmostEqual(npoints, 324.0000, 4)

        r = Boxavg(x_min=.01, x_max=.015, y_min=0.01, y_max=0.015)
        s, ds = r(self.data)
        self.assertAlmostEqual(s, 0.10579935462962962, 4)
        self.assertAlmostEqual(ds, 0.02542364197388483, 4)

    def test_slabX(self):
        """
            Test slab in X
            The test data was not generated by IGOR.
        """

        r = SlabX(x_min=-.01, x_max=.01, y_min=-0.0002,
                  y_max=0.0002, bin_width=0.0004)
        r.fold = False
        o = r(self.data)

        filepath = find('slabx_testdata.txt')
        answer = ascii_load_data(filepath)[0]
        for i in range(len(o._data_contents["Q"].value)):
            self.assertAlmostEqual(o._data_contents["Q"].value[i], answer._data_contents["Q"].value[i], 4)
            self.assertAlmostEqual(o._data_contents["I"].value[i], answer._data_contents["I"].value[i], 4)
            self.assertAlmostEqual(o._data_contents["I"].variance.value[i], answer._data_contents["I"].variance.value[i], 4)

    def test_slabY(self):
        """
            Test slab in Y
            The test data was not generated by IGOR.
        """

        r = SlabY(x_min=.005, x_max=.01, y_min=-
                  0.01, y_max=0.01, bin_width=0.0004)
        r.fold = False
        o = r(self.data)

        filepath = find('slaby_testdata.txt')
        answer = ascii_load_data(filepath)[0]
        for i in range(len(o._data_contents["Q"].value)):
            self.assertAlmostEqual(o._data_contents["Q"].value[i], answer._data_contents["Q"].value[i], 4)
            self.assertAlmostEqual(o._data_contents["I"].value[i], answer._data_contents["I"].value[i], 4)
            self.assertAlmostEqual(o._data_contents["I"].variance.value[i], answer._data_contents["I"].variance.value[i], 4)

    def test_sectorphi_full(self):
        """
            Test sector averaging I(phi)
            When considering the whole azimuthal range (2pi),
            the answer should be the same as ring averaging.
            The test data was not generated by IGOR.
        """

        nbins = 19
        phi_min = Pi / (nbins + 1)
        phi_max = TwoPi - phi_min

        r = SectorPhi(r_min=.005,
                      r_max=.01,
                      phi_min=phi_min,
                      phi_max=phi_max,
                      nbins=nbins)
        o = r(self.data)

        filepath = find('ring_testdata.txt')
        answer = ascii_load_data(filepath)[0]
        for i in range(len(o._data_contents["Q"].value)-1):
            self.assertAlmostEqual(o._data_contents["Q"].value[i], answer._data_contents["Q"].value[i+1], 4)
            self.assertAlmostEqual(o._data_contents["I"].value[i], answer._data_contents["I"].value[i+1], 4)
            self.assertAlmostEqual(o._data_contents["I"].variance.value[i], answer._data_contents["I"].variance.value[i+1], 4)

    def test_sectorphi_quarter(self):
        """
            Test sector averaging I(phi)
            The test data was not generated by IGOR.
        """

        r = SectorPhi(r_min=.005, r_max=.01, phi_min=0, phi_max=Pi / 2.0)
        r.nbins_phi = 20
        o = r(self.data)

        filepath = find('sectorphi_testdata.txt')
        answer = ascii_load_data(filepath)[0]
        for i in range(len(o._data_contents["Q"].value)):
            self.assertAlmostEqual(o._data_contents["Q"].value[i], answer._data_contents["Q"].value[i], 4)
            self.assertAlmostEqual(o._data_contents["I"].value[i], answer._data_contents["I"].value[i], 4)
            self.assertAlmostEqual(o._data_contents["I"].variance.value[i], answer._data_contents["I"].variance.value[i], 4)

    def test_sectorq_full(self):
        """
            Test sector averaging I(q)
            The test data was not generated by IGOR.
        """

        r = SectorQ(r_min=.005, r_max=.01, phi_min=0, phi_max=Pi / 2.0)
        r.nbins_phi = 20
        o = r(self.data)

        filepath = find('sectorq_testdata.txt')
        answer = ascii_load_data(filepath)[0]
        for i in range(len(o._data_contents["Q"].value)):
            self.assertAlmostEqual(o._data_contents["Q"].value[i], answer._data_contents["Q"].value[i], 4)
            self.assertAlmostEqual(o._data_contents["I"].value[i], answer._data_contents["I"].value[i], 4)
            self.assertAlmostEqual(o._data_contents["I"].variance.value[i], answer._data_contents["I"].variance.value[i], 4)

    def test_sectorq_log(self):
        """
            Test sector averaging I(q)
            The test data was not generated by IGOR.
        """

        r = SectorQ(r_min=.005, r_max=.01, phi_min=0, phi_max=Pi / 2.0, base=10)
        r.nbins_phi = 20
        o = r(self.data)

        expected_binning = np.logspace(np.log10(0.005), np.log10(0.01), 20, base=10)
        for i in range(len(o._data_contents["Q"].value)):
            self.assertAlmostEqual(o._data_contents["Q"].value[i], expected_binning[i], 3)

        # TODO: Test for Y values (o.y)
        # print len(self.data.x_bins)
        # print len(self.data.y_bins)
        # print self.data.q_data.shape
        # data_to_bin = np.array(self.data.q_data)
        # data_to_bin = data_to_bin.reshape(len(self.data.x_bins), len(self.data.y_bins))
        # H, xedges, yedges, binnumber = binned_statistic_2d(self.data.x_bins, self.data.y_bins, data_to_bin,
        #     bins=[[0, math.pi / 2.0], expected_binning], statistic='mean')
        # xedges_width = (xedges[1] - xedges[0])
        # xedges_center = xedges[1:] - xedges_width / 2

        # yedges_width = (yedges[1] - yedges[0])
        # yedges_center = yedges[1:] - yedges_width / 2

        # print H.flatten().shape
        # print o.y.shape


if __name__ == '__main__':
    unittest.main()
