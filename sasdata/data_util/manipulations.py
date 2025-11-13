"""
Data manipulations for 2D data sets.
Using the meta data information, various types of averaging
are performed in Q-space

To test this module use:
```
cd test
PYTHONPATH=../src/ python2  -m sasdataloader.test.utest_averaging DataInfoTests.test_sectorphi_quarter
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

import numpy as np

from sasdata.dataloader.data_info import Data1D, Data2D, Data3D
from sasdata.quantities.constants import Pi, TwoPi


def deduce_qz(qx: float, qy: float, wavelength: float) -> float:
    """
    If you know qx, qy, and the wavelength, you can derive qz

    :param qx: qx [inverse length]
    :param dy: qy [inverse length]
    :param wavelength: neutron wavelength [length]

    :return: qz
    """

    k0 = 2*np.pi/wavelength
    twotheta = np.arcsin((qx**2 + qy**2) / k0)
    qz = (1 - np.cos(twotheta)) * k0
    return qz


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
    return (2.0* TwoPi / wavelength) * math.sin(theta)


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
    converting to 1D: dq_overlap should calculated ideally at
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

class _Slab:
    """
    Compute average I(Q) for a region of interest
    """

    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0,
                 y_max=0.0, bin_width=0.001, fold = False):
        # Minimum Qx value [A-1]
        self.x_min = x_min
        # Maximum Qx value [A-1]
        self.x_max = x_max
        # Minimum Qy value [A-1]
        self.y_min = y_min
        # Maximum Qy value [A-1]
        self.y_max = y_max
        # Bin width (step size) [A-1]
        self.bin_width = bin_width
        # If True, I(|Q|) will be return, otherwise,
        # negative q-values are allowed
        self.fold = fold

    def __call__(self, data2D):
        return NotImplemented

    def _avg(self, data2D, maj):
        """
        Compute average I(Q_maj) for a region of interest.
        The major axis is defined as the axis of Q_maj.
        The minor axis is the axis that we average over.

        :param data2D: Data2D object
        :param maj_min: min value on the major axis
        :return: Data1D object
        """
        if len(data2D.detector) > 1:
            msg = "_Slab._avg: invalid number of "
            msg += " detectors: %g" % len(data2D.detector)
            raise RuntimeError(msg)

        # Get data
        data = data2D.data[np.isfinite(data2D.data)]
        err_data =None
        if data2D.err_data is not None:
            err_data = data2D.err_data[np.isfinite(data2D.data)]
        qx_data = data2D.qx_data[np.isfinite(data2D.data)]
        qy_data = data2D.qy_data[np.isfinite(data2D.data)]
        mask_data = data2D.mask[np.isfinite(data2D.data)]

        # Bin width calculation returns negative values when either axis has no points above 0.
        self.bin_width = abs(self.bin_width)

        # Build array of Q intervals
        if maj == 'x':
            if self.fold:
                # Set x_max based on which is further from Qx = 0
                x_max = max(abs(self.x_min),abs(self.x_max))
                # Set x_min based on which is closer to Qx = 0, but will have different limits depending on whether
                # x_min and x_max are on the same side of Qx = 0
                if self.x_min*self.x_max >= 0: # If on same side
                    x_min = min(abs(self.x_min),abs(self.x_max))
                else:
                    x_min = 0
            else:
                x_max = self.x_max
                x_min = self.x_min
            y_max = self.y_max
            y_min = self.y_min
            nbins = int(math.ceil((x_max - x_min) / self.bin_width))
        elif maj == 'y':
            if self.fold:
                # Set y_max based on which is further from Qy = 0
                y_max = max(abs(self.y_min), abs(self.y_max))
                # Set y_min based on which is closer to Qy = 0, but will have different limits depending on whether
                # y_min and y_max are on the same side of Qy = 0
                if self.y_min * self.y_max >= 0:  # If on same side
                    y_min = min(abs(self.y_min), abs(self.y_max))
                else:
                    y_min = 0
            else:
                y_max = self.y_max
                y_min = self.y_min
            x_max = self.x_max
            x_min = self.x_min
            nbins = int(math.ceil((y_max - y_min) / self.bin_width))
        else:
            raise RuntimeError("_Slab._avg: unrecognized axis %s" % str(maj))

        x = np.zeros(nbins)
        y = np.zeros(nbins)
        err_y = np.zeros(nbins)
        y_counts = np.zeros(nbins)

        # Average pixelsize in q space
        for npts in range(len(data)):
            if not mask_data[npts]:
                # ignore points that are masked
                continue
            # default frac
            frac_x = 0
            frac_y = 0
            # get ROI
            if self.fold:
                # If folded, need to satisfy absolute value of Q, but also make sure we're only pulling
                # from data inside the box (an issue when the box is not centered on 0)
                if maj == 'x':
                    if self.x_min <= qx_data[npts] < self.x_max and x_min <= abs(qx_data[npts]) < x_max:
                        frac_x = 1
                    if self.y_min <= qy_data[npts] < self.y_max:
                        frac_y = 1
                elif maj == 'y': # The case where maj != 'x' or 'y' was handled earlier
                    if self.y_min <= qy_data[npts] < self.y_max and y_min <= abs(qy_data[npts]) < y_max:
                        frac_y = 1
                    if self.x_min <= qx_data[npts] < self.x_max:
                        frac_x = 1
            else:
                if self.x_min <= qx_data[npts] < self.x_max:
                    frac_x = 1
                if self.y_min <= qy_data[npts] < self.y_max:
                    frac_y = 1
            frac = frac_x * frac_y

            if frac == 0:
                continue
            # binning: find axis of q
            if maj == 'x':
                q_value = qx_data[npts]
                min_value = x_min
            if maj == 'y':
                q_value = qy_data[npts]
                min_value = y_min
            if self.fold and q_value < 0:
                q_value = -q_value
            # bin
            i_q = int(math.ceil((q_value - min_value) / self.bin_width)) - 1

            # skip outside of max bins
            if i_q < 0 or i_q >= nbins:
                continue

            # TODO: find better definition of x[i_q] based on q_data
            # min_value + (i_q + 1) * self.bin_width / 2.0
            x[i_q] += frac * q_value
            y[i_q] += frac * data[npts]

            if err_data is None or err_data[npts] == 0.0:
                if data[npts] < 0:
                    data[npts] = -data[npts]
                err_y[i_q] += frac * frac * data[npts]
            else:
                err_y[i_q] += frac * frac * err_data[npts] * err_data[npts]
            y_counts[i_q] += frac

        # Average the sums
        for n in range(nbins):
            err_y[n] = math.sqrt(err_y[n])

        err_y = err_y / y_counts
        y = y / y_counts
        x = x / y_counts
        idx = (np.isfinite(y) & np.isfinite(x))

        if not idx.any():
            msg = "Average Error: No points inside ROI to average..."
            raise ValueError(msg)
        return Data1D(x=x[idx], y=y[idx], dy=err_y[idx])


class SlabY(_Slab):
    """
    Compute average I(Qy) for a region of interest
    """

    def __call__(self, data2D):
        """
        Compute average I(Qy) for a region of interest

        :param data2D: Data2D object
        :return: Data1D object
        """
        return self._avg(data2D, 'y')


class SlabX(_Slab):
    """
    Compute average I(Qx) for a region of interest
    """

    def __call__(self, data2D):
        """
        Compute average I(Qx) for a region of interest
        :param data2D: Data2D object
        :return: Data1D object
        """
        return self._avg(data2D, 'x')

################################################################################


class Boxsum:
    """
    Perform the sum of counts in a 2D region of interest.
    """

    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0):
        # Minimum Qx value [A-1]
        self.x_min = x_min
        # Maximum Qx value [A-1]
        self.x_max = x_max
        # Minimum Qy value [A-1]
        self.y_min = y_min
        # Maximum Qy value [A-1]
        self.y_max = y_max

    def __call__(self, data2D):
        """
        Perform the sum in the region of interest

        :param data2D: Data2D object
        :return: number of counts, error on number of counts,
            number of points summed
        """
        y, err_y, y_counts = self._sum(data2D)

        # Average the sums
        counts = 0 if y_counts == 0 else y
        error = 0 if y_counts == 0 else math.sqrt(err_y)

        # Added y_counts to return, SMK & PDB, 04/03/2013
        return counts, error, y_counts

    def _sum(self, data2D):
        """
        Perform the sum in the region of interest

        :param data2D: Data2D object
        :return: number of counts,
            error on number of counts, number of entries summed
        """
        if len(data2D.detector) > 1:
            msg = "Circular averaging: invalid number "
            msg += "of detectors: %g" % len(data2D.detector)
            raise RuntimeError(msg)
        # Get data
        data = data2D.data[np.isfinite(data2D.data)]
        err_data = None
        if data2D.err_data is not None:
            err_data = data2D.err_data[np.isfinite(data2D.data)]
        qx_data = data2D.qx_data[np.isfinite(data2D.data)]
        qy_data = data2D.qy_data[np.isfinite(data2D.data)]
        mask_data = data2D.mask[np.isfinite(data2D.data)]

        y = 0.0
        err_y = 0.0
        y_counts = 0.0

        # Average pixelsize in q space
        for npts in range(len(data)):
            if not mask_data[npts]:
                # ignore points that are masked
                continue
            # default frac
            frac_x = 0
            frac_y = 0

            # get min and max at each points
            qx = qx_data[npts]
            qy = qy_data[npts]

            # get the ROI
            if self.x_min <= qx and self.x_max > qx:
                frac_x = 1
            if self.y_min <= qy and self.y_max > qy:
                frac_y = 1
            # Find the fraction along each directions
            frac = frac_x * frac_y
            if frac == 0:
                continue
            y += frac * data[npts]
            if err_data is None or err_data[npts] == 0.0:
                if data[npts] < 0:
                    data[npts] = -data[npts]
                err_y += frac * frac * data[npts]
            else:
                err_y += frac * frac * err_data[npts] * err_data[npts]
            y_counts += frac
        return y, err_y, y_counts


class Boxavg(Boxsum):
    """
    Perform the average of counts in a 2D region of interest.
    """

    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0):
        super(Boxavg, self).__init__(x_min=x_min, x_max=x_max,
                                     y_min=y_min, y_max=y_max)

    def __call__(self, data2D):
        """
        Perform the sum in the region of interest

        :param data2D: Data2D object
        :return: average counts, error on average counts

        """
        y, err_y, y_counts = self._sum(data2D)

        # Average the sums
        counts = 0 if y_counts == 0 else y / y_counts
        error = 0 if y_counts == 0 else math.sqrt(err_y) / y_counts

        return counts, error

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


class Ring:
    """
    Defines a ring on a 2D data set.
    The ring is defined by r_min, r_max, and
    the position of the center of the ring.

    The data returned is the distribution of counts
    around the ring as a function of phi.

    Phi_min and phi_max should be defined between 0 and 2*pi
    in anti-clockwise starting from the x- axis on the left-hand side
    """
    # Todo: remove center.

    def __init__(self, r_min=0, r_max=0, center_x=0, center_y=0, nbins=36):
        # Minimum radius
        self.r_min = r_min
        # Maximum radius
        self.r_max = r_max
        # Center of the ring in x
        self.center_x = center_x
        # Center of the ring in y
        self.center_y = center_y
        # Number of angular bins
        self.nbins_phi = nbins

    def __call__(self, data2D):
        """
        Apply the ring to the data set.
        Returns the angular distribution for a given q range

        :param data2D: Data2D object

        :return: Data1D object
        """
        if data2D.__class__.__name__ not in ["Data2D", "plottable_2D"]:
            raise RuntimeError("Ring averaging only take plottable_2D objects")

        # Get data
        data = data2D.data[np.isfinite(data2D.data)]
        q_data = data2D.q_data[np.isfinite(data2D.data)]
        err_data = None
        if data2D.err_data is not None:
            err_data = data2D.err_data[np.isfinite(data2D.data)]
        qx_data = data2D.qx_data[np.isfinite(data2D.data)]
        qy_data = data2D.qy_data[np.isfinite(data2D.data)]
        mask_data = data2D.mask[np.isfinite(data2D.data)]

        # Set space for 1d outputs
        phi_bins = np.zeros(self.nbins_phi)
        phi_counts = np.zeros(self.nbins_phi)
        phi_values = np.zeros(self.nbins_phi)
        phi_err = np.zeros(self.nbins_phi)

        # Shift to apply to calculated phi values in order
        # to center first bin at zero
        phi_shift = Pi / self.nbins_phi

        for npt in range(len(data)):
            if not mask_data[npt]:
                # ignore points that are masked
                continue
            frac = 0
            # q-value at the point (npt)
            q_value = q_data[npt]
            data_n = data[npt]

            # phi-value at the point (npt)
            phi_value = math.atan2(qy_data[npt], qx_data[npt]) + Pi

            if self.r_min <= q_value and q_value <= self.r_max:
                frac = 1
            if frac == 0:
                continue
            # binning
            i_phi = int(math.floor((self.nbins_phi) *
                                   (phi_value + phi_shift) / (2 * Pi)))

            # Take care of the edge case at phi = 2pi.
            if i_phi >= self.nbins_phi:
                i_phi = 0
            phi_bins[i_phi] += frac * data[npt]

            if err_data is None or err_data[npt] == 0.0:
                if data_n < 0:
                    data_n = -data_n
                phi_err[i_phi] += frac * frac * math.fabs(data_n)
            else:
                phi_err[i_phi] += frac * frac * err_data[npt] * err_data[npt]
            phi_counts[i_phi] += frac

        for i in range(self.nbins_phi):
            phi_bins[i] = phi_bins[i] / phi_counts[i]
            phi_err[i] = math.sqrt(phi_err[i]) / phi_counts[i]
            phi_values[i] = TwoPi / self.nbins_phi * (1.0 * i)

        idx = (np.isfinite(phi_bins))

        if not idx.any():
            msg = "Average Error: No points inside ROI to average..."
            raise ValueError(msg)

        return Data1D(x=phi_values[idx], y=phi_bins[idx], dy=phi_err[idx])


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

    def __init__(self, r_min, r_max, phi_min=0, phi_max=TwoPi, nbins=20,
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

        # Compute original span in a robust way with the inputs
        # and take modulo 2pi so that a full span is normalised to a full circle
        # and multiples of 2pi are recognized.
        # This avoids zero width when phi_max == phi_min + 2pi.
        span = (self.phi_max - self.phi_min)
        span_mod = span % TwoPi

        # Get the min and max into the region: 0 <= phi < 2Pi
        phi_min = flip_phi(self.phi_min)

        # If the original specified span corresponds to a full circle (or very close),
        # construct a continuous interval for binning from phi_min to phi_min + 2pi.
        if math.isclose(span_mod, 0.0, abs_tol=1e-12) and abs(span) >= TwoPi:
            # Treat as full circle
            phi_max = phi_min + TwoPi
        else:
            # Normal case: map phi_max into [0,2pi] for bins
            phi_max = flip_phi(self.phi_max)



        # Now calculate the angles for the opposite side sector, here referred
        # to as "minor wing," and ensure these too are within 0 to 2pi
        phi_min_minor = flip_phi(phi_min - Pi)
        phi_max_minor = flip_phi(phi_max - Pi)

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
                binning = Binning(phi_min, phi_max + TwoPi, self.nbins, self.base)
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
            phi_value = math.atan2(qy_data[n], qx_data[n]) + Pi

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
                    i_bin = binning.get_bin_index(phi_value + TwoPi)
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

################################################################################


class Ringcut:
    """
    Defines a ring on a 2D data set.
    The ring is defined by r_min, r_max, and
    the position of the center of the ring.

    The data returned is the region inside the ring

    Phi_min and phi_max should be defined between 0 and 2*pi
    in anti-clockwise starting from the x- axis on the left-hand side
    """

    def __init__(self, r_min=0, r_max=0, center_x=0, center_y=0):
        # Minimum radius
        self.r_min = r_min
        # Maximum radius
        self.r_max = r_max
        # Center of the ring in x
        self.center_x = center_x
        # Center of the ring in y
        self.center_y = center_y

    def __call__(self, data2D):
        """
        Apply the ring to the data set.
        Returns the angular distribution for a given q range

        :param data2D: Data2D object

        :return: index array in the range
        """
        if data2D.__class__.__name__ not in ["Data2D", "plottable_2D"]:
            raise RuntimeError("Ring cut only take plottable_2D objects")

        # Get data
        qx_data = data2D.qx_data
        qy_data = data2D.qy_data
        q_data = np.sqrt(qx_data * qx_data + qy_data * qy_data)

        # check whether or not the data point is inside ROI
        out = (self.r_min <= q_data) & (self.r_max >= q_data)
        return out

################################################################################


class Boxcut:
    """
    Find a rectangular 2D region of interest.
    """

    def __init__(self, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0):
        # Minimum Qx value [A-1]
        self.x_min = x_min
        # Maximum Qx value [A-1]
        self.x_max = x_max
        # Minimum Qy value [A-1]
        self.y_min = y_min
        # Maximum Qy value [A-1]
        self.y_max = y_max

    def __call__(self, data2D):
        """
       Find a rectangular 2D region of interest.

       :param data2D: Data2D object
       :return: mask, 1d array (len = len(data))
           with Trues where the data points are inside ROI, otherwise False
        """
        mask = self._find(data2D)

        return mask

    def _find(self, data2D):
        """
        Find a rectangular 2D region of interest.

        :param data2D: Data2D object

        :return: out, 1d array (length = len(data))
           with Trues where the data points are inside ROI, otherwise Falses
        """
        if data2D.__class__.__name__ not in ["Data2D", "plottable_2D"]:
            raise RuntimeError("Boxcut take only plottable_2D objects")
        # Get qx_ and qy_data
        qx_data = data2D.qx_data
        qy_data = data2D.qy_data

        # check whether or not the data point is inside ROI
        outx = (self.x_min <= qx_data) & (self.x_max > qx_data)
        outy = (self.y_min <= qy_data) & (self.y_max > qy_data)

        return outx & outy

################################################################################


class Sectorcut:
    """
    Defines a sector (major + minor) region on a 2D data set.
    The sector is defined by phi_min, phi_max,
    where phi_min and phi_max are defined by the right
    and left lines wrt central line.

    Phi_min and phi_max are given in units of radian
    and (phi_max-phi_min) should not be larger than pi
    """

    def __init__(self, phi_min=0, phi_max=Pi):
        self.phi_min = phi_min
        self.phi_max = phi_max

    def __call__(self, data2D):
        """
        Find a rectangular 2D region of interest.

        :param data2D: Data2D object

        :return: mask, 1d array (len = len(data))

        with Trues where the data points are inside ROI, otherwise False
        """
        mask = self._find(data2D)

        return mask

    def _find(self, data2D):
        """
        Find a rectangular 2D region of interest.

        :param data2D: Data2D object

        :return: out, 1d array (length = len(data))

        with Trues where the data points are inside ROI, otherwise Falses
        """
        if data2D.__class__.__name__ not in ["Data2D", "plottable_2D"]:
            raise RuntimeError("Sectorcut take only plottable_2D objects")
        # Get data
        qx_data = data2D.qx_data
        qy_data = data2D.qy_data

        # get phi from data
        phi_data = np.arctan2(qy_data, qx_data)

        # Get the min and max into the region: -pi <= phi < Pi
        phi_min_major = flip_phi(self.phi_min + Pi) - Pi
        phi_max_major = flip_phi(self.phi_max + Pi) - Pi
        # check for major sector
        if phi_min_major > phi_max_major:
            out_major = (phi_min_major <= phi_data) + \
                (phi_max_major > phi_data)
        else:
            out_major = (phi_min_major <= phi_data) & (
                phi_max_major > phi_data)

        # minor sector
        # Get the min and max into the region: -pi <= phi < Pi
        phi_min_minor = flip_phi(self.phi_min) - Pi
        phi_max_minor = flip_phi(self.phi_max) - Pi

        # check for minor sector
        if phi_min_minor > phi_max_minor:
            out_minor = (phi_min_minor <= phi_data) + \
                (phi_max_minor >= phi_data)
        else:
            out_minor = (phi_min_minor <= phi_data) & \
                (phi_max_minor >= phi_data)
        out = out_major + out_minor

        return out
