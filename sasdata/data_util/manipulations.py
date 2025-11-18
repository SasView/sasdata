"""
Data manipulations for 2D data sets.
Using the meta data information, various types of averaging are performed in Q-space

To test this module use:
```
cd test
PYTHONPATH=../src/ python2  -m sasmanipulations.test.utest_averaging DataInfoTests.test_sectorphi_quarter
```
"""
#####################################################################
# This software was developed by the University of Tennessee as part of the
# Distributed Data Analysis of Neutron Scattering Experiments (DANSE)
# project funded by the US National Science Foundation.
# See the license text in license.txt
# copyright 2008, University of Tennessee
######################################################################


# TODO: copy the meta data from the 2D object to the resulting 1D object
import math
from warnings import warn

import numpy as np

from sasdata.dataloader.data_info import Data1D, Data2D

################################################################################
# Backwards-compatible wrappers that delegate to the new implementations
# in averaging.py. 
# The original manipulations classes used different parameter names.
# The wrappers below translate the old style
# parameters to the new classes to preserve external behaviour.
################################################################################

from sasdata.data_util.averaging import (
    Boxsum as AvgBoxsum,
    Boxavg as AvgBoxavg,
    SlabX as AvgSlabX,
    SlabY as AvgSlabY,
    CircularAverage as AvgCircularAverage,
    Ring as AvgRing,
    SectorQ as AvgSectorQ,
    WedgeQ as AvgWedgeQ,
    WedgePhi as AvgWedgePhi,
    SectorPhi as AvgSectorPhi,
    Ringcut as AvgRingcut,
    Boxcut as AvgBoxcut,
    Sectorcut as AvgSectorcut,
)

warn("sasdata.data_util.manipulations is deprecated. Unless otherwise noted, update your import to "
     "sasdata.data_util.averaging.", DeprecationWarning, stacklevel=2)


def position_and_wavelength_to_q(dx: float, dy: float, detector_distance: float, wavelength: float) -> float:
    """
    :param dx: x-distance from beam center [mm]
    :param dy: y-distance from beam center [mm]
    :param detector_distance: sample to detector distance [mm]
    :param wavelength: neutron wavelength [nm]
    :return: q-value at the given position
    """
    # Distance from beam center in the plane of detector
    plane_dist = math.sqrt(dx * dx + dy * dy)
    # Half of the scattering angle
    theta = 0.5 * math.atan(plane_dist / detector_distance)
    return (4.0 * math.pi / wavelength) * math.sin(theta)


def get_q_compo(dx: float, dy: float, detector_distance: float, wavelength: float, compo: str | None = None) -> float:
    """
    This reduces tiny error at very large q.
    Implementation of this func is not started yet.<--ToDo
    """
    if dy == 0:
        if dx >= 0:
            angle_xy = 0
        else:
            angle_xy = math.pi
    else:
        angle_xy = math.atan(dx / dy)

    if compo == "x":
        out = position_and_wavelength_to_q(dx, dy, detector_distance, wavelength) * math.cos(angle_xy)
    elif compo == "y":
        out = position_and_wavelength_to_q(dx, dy, detector_distance, wavelength) * math.sin(angle_xy)
    else:
        out = position_and_wavelength_to_q(dx, dy, detector_distance, wavelength)
    return out


def flip_phi(phi: float) -> float:
    """
    Force phi to be within the 0 <= to <= 2pi range by adding or subtracting
    2pi as necessary

    :return: phi in >=0 and <=2Pi
    """
    if phi < 0:
        phi_out = phi + (2 * math.pi)
    elif phi > (2 * math.pi):
        phi_out = phi - (2 * math.pi)
    else:
        phi_out = phi
    return phi_out


def get_pixel_fraction_square(x: float, x_min: float, x_max: float) -> float:
    """
    Return the fraction of the length
    from xmin to x.::

           A            B
       +-----------+---------+
       xmin        x         xmax

    :param x: x-value
    :param x_min: minimum x for the length considered
    :param x_max: minimum x for the length considered
    :return: (x-xmin)/(xmax-xmin) when xmin < x < xmax

    """
    if x <= x_min:
        return 0.0
    if x_min < x < x_max:
        return (x - x_min) / (x_max - x_min)
    else:
        return 1.0


def get_intercept(q: float, q_0: float, q_1: float) -> float | None:
    """
    Returns the fraction of the side at which the
    q-value intercept the pixel, None otherwise.
    The values returned is the fraction ON THE SIDE
    OF THE LOWEST Q. ::

            A           B
        +-----------+--------+    <--- pixel size
        0                    1
        Q_0 -------- Q ----- Q_1   <--- equivalent Q range
        if Q_1 > Q_0, A is returned
        if Q_1 < Q_0, B is returned
        if Q is outside the range of [Q_0, Q_1], None is returned

    """
    if q_1 > q_0:
        if q_0 < q <= q_1:
            return (q - q_0) / (q_1 - q_0)
    else:
        if q_1 < q <= q_0:
            return (q - q_1) / (q_0 - q_1)
    return None


def get_pixel_fraction(q_max: float, q_00: float, q_01: float, q_10: float, q_11: float) -> float:
    """
    Returns the fraction of the pixel defined by
    the four corners (q_00, q_01, q_10, q_11) that
    has q < q_max.::

                q_01                q_11
        y=1         +--------------+
                    |              |
                    |              |
                    |              |
        y=0         +--------------+
                q_00                q_10

                    x=0            x=1

    """
    # y side for x = minx
    x_0 = get_intercept(q_max, q_00, q_01)
    # y side for x = maxx
    x_1 = get_intercept(q_max, q_10, q_11)

    # x side for y = miny
    y_0 = get_intercept(q_max, q_00, q_10)
    # x side for y = maxy
    y_1 = get_intercept(q_max, q_01, q_11)

    # surface fraction for a 1x1 pixel
    frac_max = 0

    if x_0 and x_1:
        frac_max = (x_0 + x_1) / 2.0
    elif y_0 and y_1:
        frac_max = (y_0 + y_1) / 2.0
    elif x_0 and y_0:
        if q_00 < q_10:
            frac_max = x_0 * y_0 / 2.0
        else:
            frac_max = 1.0 - x_0 * y_0 / 2.0
    elif x_0 and y_1:
        if q_00 < q_10:
            frac_max = x_0 * y_1 / 2.0
        else:
            frac_max = 1.0 - x_0 * y_1 / 2.0
    elif x_1 and y_0:
        if q_00 > q_10:
            frac_max = x_1 * y_0 / 2.0
        else:
            frac_max = 1.0 - x_1 * y_0 / 2.0
    elif x_1 and y_1:
        if q_00 < q_10:
            frac_max = 1.0 - (1.0 - x_1) * (1.0 - y_1) / 2.0
        else:
            frac_max = (1.0 - x_1) * (1.0 - y_1) / 2.0

    # If we make it here, there is no intercept between
    # this pixel and the constant-q ring. We only need
    # to know if we have to include it or exclude it.
    elif (q_00 + q_01 + q_10 + q_11) / 4.0 < q_max:
        frac_max = 1.0

    return frac_max


def get_dq_data(data2d: Data2D) -> np.array:
    '''
    Get the dq for resolution averaging
    The pinholes and det. pix contribution present
    in both direction of the 2D which must be subtracted when
    converting to 1D: dq_overlap should be calculated ideally at
    q = 0. Note This method works on only pinhole geometry.
    Extrapolate dqx(r) and dqy(phi) at q = 0, and take an average.
    '''
    z_max = max(data2d.q_data)
    z_min = min(data2d.q_data)
    dqx_at_z_max = data2d.dqx_data[np.argmax(data2d.q_data)]
    dqx_at_z_min = data2d.dqx_data[np.argmin(data2d.q_data)]
    dqy_at_z_max = data2d.dqy_data[np.argmax(data2d.q_data)]
    dqy_at_z_min = data2d.dqy_data[np.argmin(data2d.q_data)]
    # Find qdx at q = 0
    dq_overlap_x = (dqx_at_z_min * z_max - dqx_at_z_max * z_min) / (z_max - z_min)
    # when extrapolation goes wrong
    if dq_overlap_x > min(data2d.dqx_data):
        dq_overlap_x = min(data2d.dqx_data)
    dq_overlap_x *= dq_overlap_x
    # Find qdx at q = 0
    dq_overlap_y = (dqy_at_z_min * z_max - dqy_at_z_max * z_min) / (z_max - z_min)
    # when extrapolation goes wrong
    if dq_overlap_y > min(data2d.dqy_data):
        dq_overlap_y = min(data2d.dqy_data)
    # get dq at q=0.
    dq_overlap_y *= dq_overlap_y

    dq_overlap = np.sqrt((dq_overlap_x + dq_overlap_y) / 2.0)
    # Final protection of dq
    if dq_overlap < 0:
        dq_overlap = dqy_at_z_min
    dqx_data = data2d.dqx_data[np.isfinite(data2d.data)]
    dqy_data = data2d.dqy_data[np.isfinite(
        data2d.data)] - dq_overlap
    # def; dqx_data = dq_r dqy_data = dq_phi
    # Convert dq 2D to 1D here
    dq_data = np.sqrt(dqx_data**2 + dqy_data**2)
    return dq_data

################################################################################


def reader2D_converter(data2d: Data2D | None = None) -> Data2D:
    """
    convert old 2d format opened by IhorReader or danse_reader
    to new Data2D format
    This is mainly used by the Readers

    :param data2d: 2d array of Data2D object
    :return: 1d arrays of Data2D object

    """
    warn("reader2D_converter should be imported in the future sasdata.dataloader.data_info.",
         DeprecationWarning, stacklevel=2)
    if data2d.data is None or data2d.x_bins is None or data2d.y_bins is None:
        raise ValueError("Can't convert this data: data=None...")
    new_x = np.tile(data2d.x_bins, (len(data2d.y_bins), 1))
    new_y = np.tile(data2d.y_bins, (len(data2d.x_bins), 1))
    new_y = new_y.swapaxes(0, 1)

    new_data = data2d.data.flatten()
    qx_data = new_x.flatten()
    qy_data = new_y.flatten()
    q_data = np.sqrt(qx_data * qx_data + qy_data * qy_data)
    if data2d.err_data is None or np.any(data2d.err_data <= 0):
        new_err_data = np.sqrt(np.abs(new_data))
    else:
        new_err_data = data2d.err_data.flatten()
    mask = np.ones(len(new_data), dtype=bool)

    output = data2d
    output.data = new_data
    output.err_data = new_err_data
    output.qx_data = qx_data
    output.qy_data = qy_data
    output.q_data = q_data
    output.mask = mask

    return output

################################################################################


class Binning:
    """
    This class just creates a binning object
    either linear or log
    """

    def __init__(self, min_value, max_value, n_bins, base=None):
        """
        :param min_value: the value defining the start of the binning interval.
        :param max_value: the value defining the end of the binning interval.
        :param n_bins: the number of bins.
        :param base: the base used for log, linear binning if None.

        Beware that min_value should always be numerically smaller than
        max_value. Take particular care when binning angles across the
        2pi to 0 discontinuity.
        """
        self.min = min_value
        self.max = max_value
        self.n_bins = n_bins
        self.base = base

    def get_bin_index(self, value):
        """
        :param value: the value in the binning interval whose bin index should
                      be returned. Must be between min_value and max_value.

        The general formula logarithm binning is:
        bin = floor(N * (log(x) - log(min)) / (log(max) - log(min)))
        """
        if self.base:
            temp_x = self.n_bins * (math.log(value, self.base) - math.log(self.min, self.base))
            temp_y = math.log(self.max, self.base) - math.log(self.min, self.base)
        else:
            temp_x = self.n_bins * (value - self.min)
            temp_y = self.max - self.min
        # Bin index calulation
        return int(math.floor(temp_x / temp_y))


################################################################################

class SlabX:
    """
    Wrapper for new SlabX.

    Old signature:
        SlabX(x_min=0, x_max=0, y_min=0, y_max=0, bin_width=0.001, fold=False)
    New signature uses nbins; translate bin_width -> nbins using ceil(range/bin_width)
    """
    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0,
                 y_max=0.0, bin_width=0.001, fold=False):
        # protect against zero-width or negative widths
        width = max(abs(x_max - x_min), 1e-12)
        nbins = int(math.ceil(width / abs(bin_width))) if bin_width != 0 else 1
        self._impl = AvgSlabX(qx_range=(x_min, x_max), qy_range=(y_min, y_max),
                              nbins=nbins, fold=fold)

    def __call__(self, data2D):
        return self._impl(data2D)


class SlabY:
    """
    Wrapper for new SlabY. Same bin_width -> nbins translation as SlabX.
    """
    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0,
                 y_max=0.0, bin_width=0.001, fold=False):
        height = max(abs(y_max - y_min), 1e-12)
        nbins = int(math.ceil(height / abs(bin_width))) if bin_width != 0 else 1
        self._impl = AvgSlabY(qx_range=(x_min, x_max), qy_range=(y_min, y_max),
                              nbins=nbins, fold=fold)

    def __call__(self, data2D):
        return self._impl(data2D)


################################################################################


class Boxsum:
    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0):
        self._impl = AvgBoxsum(qx_range=(x_min, x_max), qy_range=(y_min, y_max))

    def __call__(self, data2D):
        return self._impl(data2D)


class Boxavg:
    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0):
        self._impl = AvgBoxavg(qx_range=(x_min, x_max), qy_range=(y_min, y_max))

    def __call__(self, data2D):
        return self._impl(data2D)

################################################################################


class CircularAverage:
    """
    Wrapper for new CircularAverage.
    Old signature: CircularAverage(r_min=0.0, r_max=0.0, bin_width=0.0005)
    New signature uses r_range and nbins; translate bin_width -> nbins.
    """
    def __init__(self, r_min=0.0, r_max=0.0, bin_width=0.0005):
        width = max(r_max - r_min, 1e-12)
        nbins = int(math.ceil(width / abs(bin_width))) if bin_width != 0 else 1
        self._impl = AvgCircularAverage(r_range=(r_min, r_max), nbins=nbins)

    def __call__(self, data2D):
        return self._impl(data2D)

################################################################################


class Ring:
    """
    Wrapper for new Ring.
    Old signature: Ring(r_min=0, r_max=0, center_x=0, center_y=0, nbins=36)
    New signature: Ring(r_range, nbins)
    center_x/center_y are ignored for compatibility.
    """
    def __init__(self, r_min=0.0, r_max=0.0, center_x=0.0, center_y=0.0, nbins=36):
        self._impl = AvgRing(r_range=(r_min, r_max), center=(center_x, center_y),nbins=nbins)

    def __call__(self, data2D):
        return self._impl(data2D)


class SectorPhi:
    """
    Backwards-compatible name for angular sector averaging.
    Delegates to new SectorPhi/WedgePhi implementation.
    """
    def __init__(self, r_min=0.0, r_max=0.0, phi_min=0.0, phi_max=2 * math.pi, center_x=0.0, center_y=0.0, nbins=20):
        # SectorPhi in new module is essentially WedgePhi; pass through phi_range and nbins
        self._impl = AvgSectorPhi(r_range=(r_min, r_max), phi_range=(phi_min, phi_max),center=(center_x, center_y),
                                  nbins=nbins)

    def __call__(self, data2D):
        return self._impl(data2D)


class SectorQ:
    """
    Wrapper for new SectorQ.
    Old signature: SectorQ(r_min, r_max, phi_min=0, phi_max=2*pi, nbins=20, base=None)
    New signature: SectorQ(r_range, phi_range=(0,2pi), nbins=100, fold=True)
    Keeps the same default folding behaviour (fold True).
    """
    def __init__(self, r_min, r_max, phi_min=0, phi_max=2 * math.pi, center_x=0.0, center_y=0.0, nbins=20, base=None):
        self._impl = AvgSectorQ(r_range=(r_min, r_max), phi_range=(phi_min, phi_max), center=(center_x, center_y),
                                nbins=nbins, fold=True)

    def __call__(self, data2D):
        return self._impl(data2D)

class WedgePhi:
    """
    Wrapper for new WedgePhi (behaviour matches legacy WedgePhi expectations).
    """
    def __init__(self, r_min, r_max, phi_min=0, phi_max=2 * math.pi, center_x=0.0, center_y=0.0, nbins=20):
        self._impl = AvgWedgePhi(r_range=(r_min, r_max), phi_range=(phi_min, phi_max), center=(center_x, center_y),
                                 nbins=nbins)

    def __call__(self, data2D):
        return self._impl(data2D)

class WedgeQ:
    """
    Wrapper for new WedgeQ (behaviour matches legacy WedgeQ expectations).
    """
    def __init__(self, r_min, r_max, phi_min=0, phi_max=2 * math.pi, center_x=0.0, center_y=0.0, nbins=20):
        self._impl = AvgWedgeQ(r_range=(r_min, r_max), phi_range=(phi_min, phi_max), center=(center_x, center_y),
                                nbins=nbins)

    def __call__(self, data2D):
        return self._impl(data2D)

################################################################################


class Ringcut:
    def __init__(self, r_min=0.0, r_max=0.0, center_x=0.0, center_y=0.0):
        # center_x, center_y ignored for compatibility
        self._impl = AvgRingcut(r_range=(r_min, r_max), phi_range=(0.0, 2 * math.pi), center=(center_x, center_y))

    def __call__(self, data2D):
        return self._impl(data2D)

################################################################################


class Boxcut:
    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0):
        self._impl = AvgBoxcut(qx_range=(x_min, x_max), qy_range=(y_min, y_max))

    def __call__(self, data2D):
        return self._impl(data2D)

################################################################################


class Sectorcut:
    def __init__(self, phi_min=0.0, phi_max=math.pi, center_x=0.0, center_y=0.0):
        # The new Sectorcut expects a phi_range; set radial range to full image
        self._impl = AvgSectorcut(phi_range=(phi_min, phi_max), center=(center_x, center_y))

    def __call__(self, data2D):
        return self._impl(data2D)
