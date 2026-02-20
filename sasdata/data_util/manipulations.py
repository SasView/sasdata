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

################################################################################
# Backwards-compatible wrappers that delegate to the new implementations
# in averaging.py.
# The original manipulations classes used different parameter names.
# The wrappers below translate the old style
# parameters to the new classes to preserve external behaviour.
################################################################################
from sasdata.data_util.averaging import (
    Boxavg,
    Boxcut,
    Boxsum,
    Ring,
    Ringcut,
    Sectorcut,
    SectorQ,
    SlabX,
    SlabY,
    WedgePhi,
    WedgeQ,
)
from sasdata.dataloader.data_info import Data1D, Data2D
from sasdata.dataloader.data_info import reader2D_converter as _di_reader2D_converter
from sasdata.quantities.constants import Pi, TwoPi

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
    return (2.0 * TwoPi / wavelength) * math.sin(theta)


def get_q_compo(dx: float, dy: float, detector_distance: float, wavelength: float, compo: str | None = None) -> float:
    """
    This reduces tiny error at very large q.
    Implementation of this func is not started yet.<--ToDo
    """
    if dy == 0:
        if dx >= 0:
            angle_xy = 0
        else:
            angle_xy = Pi
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
        phi_out = phi + (TwoPi)
    elif phi > (TwoPi):
        phi_out = phi % TwoPi
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
    warn("reader2D_converter should be delegated in the future sasdata.dataloader.data_info.",
         DeprecationWarning, stacklevel=2)
    # Delegate to the implementation in data_info

    return _di_reader2D_converter(data2d)

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

class SlabX(SlabX):
    """
    Wrapper for new SlabX.

    Old signature:
        SlabX(x_min=0, x_max=0, y_min=0, y_max=0, bin_width=0.001, fold=False)
    New signature uses nbins; translate bin_width -> nbins using ceil(range/bin_width)
    """
    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0,
                 y_max=0.0, bin_width=0.001, fold=False):
        if fold:
                # Set x_max based on which is further from Qx = 0
                x_max = max(abs(self.x_min),abs(self.x_max))
                # Set x_min based on which is closer to Qx = 0, but will have different limits depending on whether
                # x_min and x_max are on the same side of Qx = 0
                if self.x_min*self.x_max >= 0: # If on same side
                    x_min = min(abs(self.x_min),abs(self.x_max))
                else:
                    x_min = 0.0

        # protect against zero-width or negative widths
        width = max(abs(x_max - x_min), 1e-12)
        nbins = int(math.ceil(width / abs(bin_width))) if bin_width != 0 else 1
        self.nbins=nbins
        super().__init__(qx_range=(x_min, x_max), qy_range=(y_min, y_max),
                              nbins=nbins, fold=fold)


class SlabY(SlabY):
    """
    Wrapper for new SlabY. Same bin_width -> nbins translation as SlabX.
    """
    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0,
                 y_max=0.0, bin_width=0.001, fold=False):
        if fold:
            # Set y_max based on which is further from Qy = 0
            y_max = max(abs(self.y_min),abs(self.y_max))
            # Set y_min based on which is closer to Qy = 0, but will have different limits depending on whether
            # y_min and y_max are on the same side of Qy = 0
            if self.y_min*self.y_max >= 0: # If on same side
                y_min = min(abs(self.y_min),abs(self.y_max))
            else:
                y_min = 0.0

        # protect against zero-width or negative widths
        height = max(abs(y_max - y_min), 1e-12)
        nbins = int(math.ceil(height / abs(bin_width))) if bin_width != 0 else 1
        self.nbins=nbins
        super().__init__(qx_range=(x_min, x_max), qy_range=(y_min, y_max),
                              nbins=nbins, fold=fold)


################################################################################


class Boxsum(Boxsum):
    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0):

        super().__init__(qx_range=(x_min, x_max), qy_range=(y_min, y_max))


class Boxavg(Boxavg):
    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0):

        super().__init__(qx_range=(x_min, x_max), qy_range=(y_min, y_max))

################################################################################

class CircularAverage:
    """
    Perform circular averaging on 2D data

    The data returned is the distribution of counts
    as a function of Q
    """

    def __init__(self, r_min=0.0, r_max=0.0, bin_width=0.0005):
        # Minimum radius included in the average [A-1]
        self.r_min = r_min
        # Maximum radius included in the average [A-1]
        self.r_max = r_max
        # Bin width (step size) [A-1]
        self.bin_width = bin_width

    def __call__(self, data2D, ismask=False):
        """
        Perform circular averaging on the data

        :param data2D: Data2D object
        :return: Data1D object
        """
        # Get data W/ finite values
        data = data2D.data[np.isfinite(data2D.data)]
        q_data = data2D.q_data[np.isfinite(data2D.data)]
        err_data = None
        if data2D.err_data is not None:
            err_data = data2D.err_data[np.isfinite(data2D.data)]
        mask_data = data2D.mask[np.isfinite(data2D.data)]

        dq_data = None
        if data2D.dqx_data is not None and data2D.dqy_data is not None:
            dq_data = get_dq_data(data2D)

        if len(q_data) == 0:
            msg = "Circular averaging: invalid q_data: %g" % data2D.q_data
            raise RuntimeError(msg)

        # Build array of Q intervals
        nbins = int(math.ceil((self.r_max - self.r_min) / self.bin_width))

        x = np.zeros(nbins)
        y = np.zeros(nbins)
        err_y = np.zeros(nbins)
        err_x = np.zeros(nbins)
        y_counts = np.zeros(nbins)

        for npt in range(len(data)):

            if ismask and not mask_data[npt]:
                continue

            frac = 0

            # q-value at the pixel (j,i)
            q_value = q_data[npt]
            data_n = data[npt]

            # No need to calculate the frac when all data are within range
            if self.r_min >= self.r_max:
                raise ValueError("Limit Error: min > max")

            if self.r_min <= q_value and q_value <= self.r_max:
                frac = 1
            if frac == 0:
                continue
            i_q = int(math.floor((q_value - self.r_min) / self.bin_width))

            # Take care of the edge case at phi = 2pi.
            if i_q == nbins:
                i_q = nbins - 1
            y[i_q] += frac * data_n
            # Take dqs from data to get the q_average
            x[i_q] += frac * q_value
            if err_data is None or err_data[npt] == 0.0:
                if data_n < 0:
                    data_n = -data_n
                err_y[i_q] += frac * frac * data_n
            else:
                err_y[i_q] += frac * frac * err_data[npt] * err_data[npt]
            if dq_data is not None:
                # To be consistent with dq calculation in 1d reduction,
                # we need just the averages (not quadratures) because
                # it should not depend on the number of the q points
                # in the qr bins.
                err_x[i_q] += frac * dq_data[npt]
            else:
                err_x = None
            y_counts[i_q] += frac

        # Average the sums
        for n in range(nbins):
            if err_y[n] < 0:
                err_y[n] = -err_y[n]
            err_y[n] = math.sqrt(err_y[n])
            # if err_x is not None:
            #    err_x[n] = math.sqrt(err_x[n])

        err_y = err_y / y_counts
        err_y[err_y == 0] = np.average(err_y)
        y = y / y_counts
        x = x / y_counts
        idx = (np.isfinite(y)) & (np.isfinite(x))

        if err_x is not None:
            d_x = err_x[idx] / y_counts[idx]
        else:
            d_x = None

        if not idx.any():
            msg = "Average Error: No points inside ROI to average..."
            raise ValueError(msg)

        return Data1D(x=x[idx], y=y[idx], dy=err_y[idx], dx=d_x)

################################################################################



class Ring(Ring):
    """
    Wrapper for new Ring.
    """


    @property
    def nbins_phi(self):
        return self.nbins

    @nbins_phi.setter
    def nbins_phi(self, value):
        self.nbins = value

    def __init__(self, r_min=0.0, r_max=0.0, center_x=0.0, center_y=0.0, nbins=36):

        super().__init__(
            r_range=(r_min, r_max),
            center=(center_x, center_y),
            nbins=nbins
            )

class _Sector:
    """
    Defines a sector region on a 2D data set.
    The sector is defined by r_min, r_max, phi_min and phi_max.
    phi_min and phi_max are defined by the right and left lines wrt a central
    line such that phi_max could be less than phi_min if they straddle the
    discontinuity from 2pi to 0.

    Phi is defined between 0 and 2*pi in anti-clockwise
    starting from the negative x-axis.
    """

    def __init__(self, r_min, r_max, phi_min=0, phi_max=2 * math.pi, nbins=20,
                 base=None):
        '''
        :param base: must be a valid base for an algorithm, i.e.,
        a positive number
        '''
        self.r_min = r_min
        self.r_max = r_max
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.nbins = nbins
        self.base = base

        # set up to use the asymmetric sector average - default to symmetric
        self.fold = True

    def _agv(self, data2D, run='phi'):
        """
        Perform sector averaging.

        :param data2D: Data2D object
        :param run:  define the varying parameter ('phi' , or 'sector')

        :return: Data1D object
        """
        if data2D.__class__.__name__ not in ["Data2D", "plottable_2D"]:
            raise RuntimeError("Ring averaging only take plottable_2D objects")

        # Get all the data & info
        data = data2D.data[np.isfinite(data2D.data)]
        q_data = data2D.q_data[np.isfinite(data2D.data)]
        err_data=None
        if data2D.err_data is not None:
            err_data = data2D.err_data[np.isfinite(data2D.data)]
        qx_data = data2D.qx_data[np.isfinite(data2D.data)]
        qy_data = data2D.qy_data[np.isfinite(data2D.data)]
        mask_data = data2D.mask[np.isfinite(data2D.data)]

        dq_data = None
        if data2D.dqx_data is not None and data2D.dqy_data is not None:
            dq_data = get_dq_data(data2D)

        # set space for 1d outputs
        x = np.zeros(self.nbins)
        y = np.zeros(self.nbins)
        y_err = np.zeros(self.nbins)
        x_err = np.zeros(self.nbins)
        y_counts = np.zeros(self.nbins)  # Cycle counts (for the mean)

        # Get the min and max into the region: 0 <= phi < 2Pi
        phi_min = flip_phi(self.phi_min)
        phi_max = flip_phi(self.phi_max)
        # Now calculate the angles for the opposite side sector, here referred
        # to as "minor wing," and ensure these too are within 0 to 2pi
        phi_min_minor = flip_phi(phi_min - math.pi)
        phi_max_minor = flip_phi(phi_max - math.pi)

        #  set up the bins by creating a binning object
        if run.lower() == 'phi':
            # The check here ensures when a range straddles the discontinuity
            # inherent in circular angles (jumping from 2pi to 0) that the
            # Binning class still recieves a continuous interval. phi_min/max
            # are used here instead of self.phi_min/max as they are always in
            # the range 0, 2pi - making their values more predictable.
            # Note that their values must not be altered, as they are used to
            # determine what points (also in the range 0, 2pi) are in the ROI.
            if phi_min > phi_max:
                binning = Binning(phi_min, phi_max + 2 * np.pi, self.nbins, self.base)
            else:
                binning = Binning(phi_min, phi_max, self.nbins, self.base)
        elif self.fold:
            binning = Binning(self.r_min, self.r_max, self.nbins, self.base)
        else:
            binning = Binning(-self.r_max, self.r_max, self.nbins, self.base)

        for n in range(len(data)):
            if not mask_data[n]:
                # ignore points that are masked
                continue

            # q-value at the pixel (j,i)
            q_value = q_data[n]
            data_n = data[n]

            # Is pixel within range?
            is_in = False

            # calculate the phi-value of the pixel (j,i) and convert the range
            # [-pi,pi] returned by the atan2 function to the [0,2pi] range used
            # as the reference frame for these calculations
            phi_value = math.atan2(qy_data[n], qx_data[n]) + math.pi

            # No need to calculate: data outside of the radius
            if self.r_min > q_value or q_value > self.r_max:
                continue

            # For all cases(i.e.,for 'sector' (fold true or false), and 'phi')
            # Find pixels within the main ROI (primary sector (main wing)
            # in the case of sectors)
            if phi_min > phi_max:
                is_in = is_in or (phi_value > phi_min or
                                  phi_value < phi_max)
            else:
                is_in = is_in or (phi_value >= phi_min and
                                  phi_value < phi_max)

            # For sector cuts we need to check if the point is within the
            # "minor wing" before checking if it is in the major wing.
            # There are effectively two ROIs here as each sector on opposite
            # sides of 0,0 need to be checked separately.
            if run.lower() == 'sector' and not is_in:
                if phi_min_minor > phi_max_minor:
                    is_in = (phi_value > phi_min_minor or
                             phi_value < phi_max_minor)
                else:
                    is_in = (phi_value > phi_min_minor and
                             phi_value < phi_max_minor)
                # now, if we want to keep both sides separate we arbitrarily,
                # assign negative q to the qs in the minor wing. As calculated,
                # all qs are postive and in fact all qs in the same ring are
                # the same. This will allow us to plot both sides of 0,0
                # independently.
                if not self.fold:
                    if is_in:
                        q_value *= -1

            # data oustide of the phi range
            if not is_in:
                continue

            # Get the binning index
            if run.lower() == 'phi':
                # If the original range used to instantiate `binning` was
                # shifted by 2pi to accommodate the 2pi to 0 discontinuity,
                # then phi_value needs to be shifted too so that it falls in
                # the continuous range set up for the binning process.
                if phi_min > phi_value:
                    i_bin = binning.get_bin_index(phi_value + 2 * np.pi)
                else:
                    i_bin = binning.get_bin_index(phi_value)
            else:
                i_bin = binning.get_bin_index(q_value)

            # Take care of the edge case at phi = 2pi.
            if i_bin == self.nbins:
                i_bin = self.nbins - 1

            # Get the total y
            y[i_bin] += data_n
            x[i_bin] += q_value
            if err_data is None or err_data[n] == 0.0:
                if data_n < 0:
                    data_n = -data_n
                y_err[i_bin] += data_n
            else:
                y_err[i_bin] += err_data[n]**2

            if dq_data is not None:
                # To be consistent with dq calculation in 1d reduction,
                # we need just the averages (not quadratures) because
                # it should not depend on the number of the q points
                # in the qr bins.
                x_err[i_bin] += dq_data[n]
            else:
                x_err = None
            y_counts[i_bin] += 1

        # Organize the results
        with np.errstate(divide='ignore', invalid='ignore'):
            y = y/y_counts
            y_err = np.sqrt(y_err)/y_counts
            # Calculate x values at the center of the bin depending on the
            # the type of averaging (phi or sector)
            if run.lower() == 'phi':
                # Determining the step size is best done via the binning
                # object's interval, as this is set up so max > min in all
                # cases. One could also use phi_min and phi_max, so long as
                # they have not been changed.
                # In setting up x, phi_min makes a better starting point than
                # self.phi_min, as the resulting array is garenteed to be > 0
                # throughout. This works better with the sasview gui, which
                # will convert the result to the range -pi, pi.
                step = (binning.max - binning.min) / self.nbins
                x = (np.arange(self.nbins) + 0.5) * step + phi_min
            else:
                # set q to the average of the q values within each bin
                x = x/y_counts

                ### Alternate algorithm
                ## We take the center of ring area, not radius.
                ## This is more accurate than taking the radial center of ring.
                #step = (self.r_max - self.r_min) / self.nbins
                #r_inner = self.r_min + step * np.arange(self.nbins)
                #x = math.sqrt((r_inner**2 + (r_inner + step)**2) / 2)

        idx = (np.isfinite(y) & np.isfinite(y_err))
        if x_err is not None:
            d_x = x_err[idx] / y_counts[idx]
        else:
            d_x = None
        if not idx.any():
            msg = "Average Error: No points inside sector of ROI to average..."
            raise ValueError(msg)
        return Data1D(x=x[idx], y=y[idx], dy=y_err[idx], dx=d_x)


class SectorPhi(_Sector):
    """
    Sector average as a function of phi.
    I(phi) is return and the data is averaged over Q.

    A sector is defined by r_min, r_max, phi_min, phi_max.
    The number of bin in phi also has to be defined.
    """

    def __call__(self, data2D):
        """
        Perform sector average and return I(phi).

        :param data2D: Data2D object
        :return: Data1D object
        """
        return self._agv(data2D, 'phi')


class SectorQ(_Sector):
    """
    Sector average as a function of Q for both wings. setting the _Sector.fold
    attribute determines whether or not the two sectors are averaged together
    (folded over) or separate.  In the case of separate (not folded), the
    qs for the "minor wing" are arbitrarily set to a negative value.
    I(Q) is returned and the data is averaged over phi.

    A sector is defined by r_min, r_max, phi_min, phi_max.
    where r_min, r_max, phi_min, phi_max >0.
    The number of bin in Q also has to be defined.
    """

    def __call__(self, data2D):
        """
        Perform sector average and return I(Q).

        :param data2D: Data2D object

        :return: Data1D object
        """
        return self._agv(data2D, 'sector')


class WedgePhi(WedgePhi):
    """
    Wrapper for new WedgePhi (behaviour matches legacy WedgePhi expectations).
    """
    def __init__(self, r_min, r_max, phi_min=0, phi_max=TwoPi, center_x=0.0, center_y=0.0, nbins=10):

        super().__init__(
            r_range=(r_min, r_max),
            phi_range=(phi_min, phi_max),
            center=(center_x, center_y),
            nbins=nbins
            )

class WedgeQ(WedgeQ):
    """
    Wrapper for new WedgeQ (behaviour matches legacy WedgeQ expectations).
    """
    def __init__(self, r_min, r_max, phi_min=0, phi_max=TwoPi, center_x=0.0, center_y=0.0, nbins=10):

        super().__init__(
            r_range=(r_min, r_max),
            phi_range=(phi_min, phi_max),
            center=(center_x, center_y),
            nbins=nbins
            )

################################################################################


class Ringcut(Ringcut):
    def __init__(self, r_min=0.0, r_max=0.0, center_x=0.0, center_y=0.0):
        # center_x, center_y ignored for compatibility

        super().__init__(
            r_range=(r_min, r_max),
            phi_range=(0.0, TwoPi),
            center=(center_x, center_y)
            )

################################################################################


class Boxcut(Boxcut):
    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0):
        super().__init__(qx_range=(x_min, x_max), qy_range=(y_min, y_max))

################################################################################


class Sectorcut(Sectorcut):
    def __init__(self, phi_min=0.0, phi_max=Pi, center_x=0.0, center_y=0.0):
        # The new Sectorcut expects a phi_range; set radial range to full image
        super().__init__(phi_range=(phi_min, phi_max), center=(center_x, center_y))
