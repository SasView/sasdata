from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from sasdata.dataloader.data_info import Data1D, Data2D


def weights_for_interval(array, l_bound, u_bound, interval_type='half-open'):
    """
    If and when fractional binning is implemented (ask Lucas), this function
    will be changed so that instead of outputting zeros and ones, it gives
    fractional values instead. These will depend on how close the array value
    is to being within the interval defined.
    """
    if interval_type == 'half-open':
        in_range = (l_bound <= array) & (array < u_bound)
    elif interval_type == 'closed':
        in_range = (l_bound <= array) & (array <= u_bound)
    else:
        msg = f"Unrecognised interval_type: {interval_type}"
        raise ValueError(msg)

    return np.asarray(in_range, dtype=int)


class Binning:
    """
    TODO - add docstring
    """

    def __init__(self, min_value, max_value, nbins, base=None):
        """
        """
        self.minimum = min_value
        self.maximum = max_value
        self.nbins = nbins
        self.base = base
        self.bin_width = (max_value - min_value) / nbins

    def get_index(self, value: float) -> int:
        """
        """
        if self.base:
            numerator = (np.log(value) - np.log(self.minimum)) \
                / np.log(self.base)
            denominator = (np.log(self.maximum) - np.log(self.minimum)) \
                / np.log(self.base)
        else:
            numerator = value - self.minimum
            denominator = self.maximum - self.minimum

        bin_index = int(np.floor(self.nbins * numerator / denominator))

        # Bins are indexed from 0 to nbins-1, so this check protects against
        # out-of-range indices when value == maximum.
        if bin_index == self.nbins:
            bin_index -= 1

        return bin_index

    def get_interval(self, bin_number: int) -> float:
        """
        """
        start = self.minimum + self.bin_width * bin_number
        stop = self.minimum + self.bin_width * (bin_number + 1)

        return start, stop


class CartesianROI(ABC):
    """
    Base class for manipulators with a rectangular region of interest.
    """

    def __init__(self, qx_min: float = 0, qx_max: float = 0,
                 qy_min: float = 0, qy_max: float = 0) -> None:
        """
        TODO - add docstring
        """

        # Units A^-1
        self.qx_min = qx_min
        self.qx_max = qx_max
        self.qy_min = qy_min
        self.qy_max = qy_max

        # Define data related variables
        self.data = None
        self.err_data = None
        self.qx_data = None
        self.qy_data = None
        self.mask_data = None

    @abstractmethod
    def __call__(self, data2d: Data2D = None) -> Union[float, Data1D]:
        """
        TODO - add docstring
        """
        return

    # This method might be better placed in a parent class
    def validate_and_assign_data(self, data2d: Data2D = None) -> None:
        """
        Check that the data supplied valid and assign data variables.
        """
        if not isinstance(data2d, Data2D):
            msg = "Data supplied must be of type Data2D."
            raise TypeError(msg)
        if len(data2d.detector) > 1:
            msg = f"Invalid number of detectors: {len(data2d.detector)}"
            raise ValueError(msg)

        finite_data = np.isfinite(data2d.data)
        self.data = data2d.data[finite_data]
        self.err_data = data2d.err_data[finite_data]
        self.qx_data = data2d.qx_data[finite_data]
        self.qy_data = data2d.qy_data[finite_data]
        self.mask_data = data2d.mask[finite_data]

        # No points should have zero error, if they do then assume the error is
        # the square root of the data.
        self.err_data[self.err_data == 0] = \
            np.sqrt(np.abs(self.data[self.err_data == 0]))

    @property
    def roi_mask(self):
        """
        Return a boolean array listing the elements of self.data which are
        inside the ROI. This property should only be accessed after
        CartesianROI has been called.
        """
        if any(data is None for data in [self.qx_data, self.qy_data,
                                         self.mask_data]):
            raise RuntimeError

        within_x_lims = (self.qx_data >= self.qx_min) & \
                        (self.qx_data <= self.qx_max)
        within_y_lims = (self.qy_data >= self.qy_min) & \
                        (self.qy_data <= self.qy_max)

        # Don't return masked-off data
        return within_x_lims & within_y_lims & self.mask_data


class PolarROI(ABC):
    """
    Base class for manipulators whose ROI is defined with polar coordinates.
    """

    def __init__(self, r_min: float = 0, r_max: float = 1000,
                 phi_min: float = 0, phi_max: float = 2*np.pi) -> None:
        """
        TODO - add docstring
        """

        # Units A^-1 for radii, radians for angles
        self.r_min = r_min
        self.r_max = r_max
        self.phi_min = phi_min
        self.phi_max = phi_max

        # Define data related variables
        self.data = None
        self.err_data = None
        self.q_data = None
        self.qx_data = None
        self.qy_data = None
        self.mask_data = None

    @abstractmethod
    def __call__(self, data2d: Data2D = None) -> Union[float, Data1D]:
        """
        TODO - add docstring
        """
        return

    # This method might be better placed in a parent class
    def validate_and_assign_data(self, data2d: Data2D = None) -> None:
        """
        Check that the data supplied valid and assign data variables.
        """
        if not isinstance(data2d, Data2D):
            msg = "Data supplied must be of type Data2D."
            raise TypeError(msg)
        if len(data2d.detector) > 1:
            msg = f"Invalid number of detectors: {len(data2d.detector)}"
            raise ValueError(msg)

        finite_data = np.isfinite(data2d.data)
        self.data = data2d.data[finite_data]
        self.err_data = data2d.err_data[finite_data]
        self.q_data = data2d.q_data[finite_data]
        self.qx_data = data2d.qx_data[finite_data]
        self.qy_data = data2d.qy_data[finite_data]
        self.mask_data = data2d.mask[finite_data]

        # No points should have zero error, if they do then assume the error is
        # the square root of the data.
        self.err_data[self.err_data == 0] = \
            np.sqrt(np.abs(self.data[self.err_data == 0]))


class Boxsum(CartesianROI):
    """
    Perform the sum of counts in a 2D region of interest.
    """

    def __init__(self, qx_min: float = 0, qx_max: float = 0,
                 qy_min: float = 0, qy_max: float = 0) -> None:
        super().__init__(qx_min=qx_min, qx_max=qx_max,
                         qy_min=qy_min, qy_max=qy_max)

    def __call__(self, data2d: Data2D = None) -> float:
        """
        Placeholder
        """
        self.validate_and_assign_data(data2d)
        total_sum, error, count = self._sum()

        return total_sum, error, count

    def _sum(self) -> float:
        """
        TODO - add docstring
        """

        # Currently the weights are binary, but could be fractional in future
        weights = self.roi_mask.astype(int)

        data = weights * self.data
        err_squared = weights * weights * self.err_data * self.err_data

        total_sum = np.sum(data)
        total_errors_squared = np.sum(err_squared)
        total_count = np.sum(weights)

        return total_sum, np.sqrt(total_errors_squared), total_count


class Boxavg(Boxsum):
    """
    Perform the average of counts in a 2D region of interest.
    """

    def __init__(self, qx_min: float = 0, qx_max: float = 0,
                 qy_min: float = 0, qy_max: float = 0) -> None:
        super().__init__(qx_min=qx_min, qx_max=qx_max,
                         qy_min=qy_min, qy_max=qy_max)

    def __call__(self, data2d: Data2D) -> float:
        """
        TODO - add docstring
        """
        self.validate_and_assign_data(data2d)
        total_sum, error, count = super()._sum()

        return (total_sum / count), (error / count)


class _Slab(CartesianROI):
    """
    Compute average I(Q) for a region of interest
    """

    def __init__(self, qx_min: float = 0, qx_max: float = 0, qy_min: float = 0,
                 qy_max: float = 0, nbins: int = 100, fold: bool = False):
        super().__init__(qx_min=qx_min, qx_max=qx_max,
                         qy_min=qy_min, qy_max=qy_max)
        self.nbins = nbins
        self.fold = fold

    def __call__(self, data2d: Data2D = None) -> Data1D:
        pass

    def _avg(self, data2d: Data2D, major_axis: str) -> Data1D:
        """
        TODO - add docstring
        """
        self.validate_and_assign_data(data2d)

        if major_axis == 'x':
            q_major = self.qx_data
            q_minor = self.qy_data
            minor_lims = (self.qy_min, self.qy_max)
            binning = Binning(min_value=0 if self.fold else self.qx_min,
                              max_value=self.qx_max, nbins=self.nbins)
        elif major_axis == 'y':
            q_major = self.qy_data
            q_minor = self.qx_data
            minor_lims = (self.qx_min, self.qx_max)
            binning = Binning(min_value=0 if self.fold else self.qy_min,
                              max_value=self.qy_max, nbins=self.nbins)
        else:
            msg = f"Unrecognised axis: {major_axis}"
            raise ValueError(msg)

        if self.fold:
            q_major = np.abs(q_major)

        major_weights = np.zeros((self.nbins, q_major.size))
        for m in range(self.nbins):
            # Include the value at the end of the binning range, but otherwise
            # use half-open intervals so each value belongs in only one bin.
            if m == self.nbins - 1:
                interval = 'closed'
            else:
                interval = 'half-open'
            bin_start, bin_end = binning.get_interval(bin_number=m)
            major_weights[m] \
                = weights_for_interval(array=q_major, l_bound=bin_start,
                                       u_bound=bin_end,
                                       interval_type=interval)
        minor_weights = weights_for_interval(array=q_minor,
                                             l_bound=minor_lims[0],
                                             u_bound=minor_lims[1],
                                             interval_type='closed')
        weights = major_weights * minor_weights

        q_values = np.sum(weights * q_major, axis=1)
        intensity = np.sum(weights * self.data, axis=1)
        errs_squared = np.sum((weights * self.err_data)**2, axis=1)
        bin_counts = np.sum(weights, axis=1)

        errors = np.sqrt(errs_squared)
        q_values /= bin_counts
        intensity /= bin_counts
        errors /= bin_counts

        finite = (np.isfinite(q_values) & np.isfinite(intensity))
        if not finite.any():
            msg = "Average Error: No points inside ROI to average..."
            raise ValueError(msg)

        return Data1D(x=q_values[finite], y=intensity[finite], dy=errors[finite])


class SlabX(_Slab):
    """
    Compute average I(Qx) for a region of interest
    """

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute average I(Qx) for a region of interest
        :param data2d: Data2D object
        :return: Data1D object
        """
        return self._avg(data2d, 'x')


class SlabY(_Slab):
    """
    Compute average I(Qy) for a region of interest
    """

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute average I(Qy) for a region of interest
        :param data2d: Data2D object
        :return: Data1D object
        """
        return self._avg(data2d, 'y')

