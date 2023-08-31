import numpy as np
from abc import ABC, abstractmethod
from typing import Union

from sasdata.dataloader.data_info import Data1D, Data2D


class Binning:
    """
    """

    def __init__(self, min_value, max_value, nbins, base=None):
        """
        """
        self.minimum = min_value
        self.maximum = max_value
        self.nbins = nbins
        self.base = base

    def get_index(self, value):
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


class CartesianROI(ABC):
    """
    Base class for manipulators with a rectangular region of interest.
    """

    def __init__(self, qx_min: float = 0, qx_max: float = 0,
                 qy_min: float = 0, qy_max: float = 0) -> None:
        """
        Placeholder
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
        Placeholder
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
        Placeholder
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
        Placeholder
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
        Placeholder
        """

        # Currently the weights are binary, but could be fractional in future
        weights = self.roi_mask.astype(int)

        data = weights * self.data
        err_squared = weights * weights * self.err_data * self.err_data
        # No points should have zero error, if they do then assume the worst
        err_squared[self.err_data == 0] = (weights * data)[self.err_data == 0]

        total_sum = np.sum(data)
        total_count = np.sum(weights)
        total_errors_squared = np.sum(err_squared)

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
        Placeholder
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
        Placeholder
        """
        self.validate_and_assign_data(data2d)

        # TODO - change weights
        weights = self.roi_mask.astype(int)

        if major_axis == 'x':
            q_major = self.qx_data
            binning = Binning(min_value=0 if self.fold else self.qx_min,
                              max_value=self.qx_max, nbins=self.nbins)
        elif major_axis == 'y':
            q_major = self.qy_data
            binning = Binning(min_value=0 if self.fold else self.qy_min,
                              max_value=self.qy_max, nbins=self.nbins)
        else:
            msg = f"Unrecognised axis: {major_axis}"
            raise ValueError(msg)

        q_values = np.zeros(self.nbins)
        intensity = np.zeros(self.nbins)
        errs_squared = np.zeros(self.nbins)
        bin_counts = np.zeros(self.nbins)

        for index, q_value in enumerate(q_major):
            # Skip over datapoints with no relevance
            # This should include masked datapoints.
            if weights[index] == 0:
                continue

            if self.fold and q_value < 0:
                q_value = -q_value

            q_bin = binning.get_index(q_value)
            q_values[q_bin] += weights[index] * q_value
            intensity[q_bin] += weights[index] * self.data[index]
            errs_squared[q_bin] += (weights[index] * self.err_data[index]) ** 2
            # No points should have zero error, assume the worst if they do
            if self.err_data[index] == 0.0:
                errs_squared[q_bin] += weights[index] ** 2 * abs(self.data[index])
            else:
                errs_squared[q_bin] += (weights[index] * self.err_data[index]) ** 2
            bin_counts[q_bin] += weights[index]

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

