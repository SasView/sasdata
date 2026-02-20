import math

import numpy as np
from numpy.typing import ArrayLike

from sasdata.data_util.interval import IntervalType


class DirectionalAverage:
    """
    Average along one coordinate axis of 2D data and return data for a 1D plot.
    This can also be thought of as a projection onto the major axis: 2D -> 1D.

    This class operates on a decomposed Data2D object, and returns data needed
    to construct a Data1D object. The class is instantiated with two arrays of
    orthogonal coordinate data (depending on the coordinate system, these may
    have undergone some pre-processing) and two corresponding two-element
    tuples/lists defining the lower and upper limits on the Region of Interest
    (ROI) for each coordinate axis. One of these axes is averaged along, and
    the other is divided into bins and becomes the dependent variable of the
    eventual 1D plot. These are called the minor and major axes respectively.
    When a class instance is called, it is passed the intensity and error data
    from the original Data2D object. These should not have undergone any
    coordinate system dependent pre-processing.

    Note that the old version of manipulations.py had an option for logarithmic
    binning which is used by SectorQ.
    """

    def __init__(self,
                 major_axis: ArrayLike,
                 minor_axis: ArrayLike,
                 lims: tuple[tuple[float, float] | None, tuple[float, float] | None] | None = None,
                 nbins: int = 100, base=None):
        """
        Set up direction of averaging, limits on the ROI, & the number of bins.

        :param major_axis: Coordinate data for axis onto which the 2D data is
                           projected.
        :param minor_axis: Coordinate data for the axis perpendicular to the
                           major axis.
        :param lims: Tuple (major_lims, minor_lims). Each element may be a
                     2-tuple or None. 
        :param nbins: The number of bins the major axis is divided up into.
        :param base: the base used for log, linear binning if None.
        """

        # Step 1: quick checks and parsing
        self._validate_coordinate_arrays(major_axis, minor_axis)
        major_lims, minor_lims = self._parse_lims(lims)
        self.nbins = self._coerce_nbins(nbins)
        self.base = base
        print(nbins)
        # Step 2: assign arrays and check sizes
        self.major_axis, self.minor_axis = self._assign_axes_and_check_lengths(major_axis, minor_axis)

        # Step 3: set final limits and compute bin limits
        self.major_lims, self.minor_lims = self._set_default_lims_and_bin_limits(major_lims, minor_lims)

    def _validate_coordinate_arrays(self, major_axis, minor_axis) -> None:
        """Ensure both major and minor coordinate inputs are array-like."""
        if any(not hasattr(coordinate_data, "__array__") for
                coordinate_data in (major_axis, minor_axis)):
            msg = "Must provide major & minor coordinate arrays for binning."
            raise ValueError(msg)

    def _parse_lims(self, lims):
        """
        Validate the lims parameter and return (major_lims, minor_lims).
        Accepts None or a 2-tuple (major_lims, minor_lims). Each of the two
        elements may be None or a 2-tuple of floats.
        """
        if lims is None:
            return None, None

        if not (isinstance(lims, (list, tuple)) and len(lims) == 2):
            msg = "Parameter 'lims' must be a 2-tuple (major_lims, minor_lims) or None."
            raise ValueError(msg)

        major_lims, minor_lims = lims
        return major_lims, minor_lims

    def _coerce_nbins(self, nbins):
        """Coerce nbins to int, raising a TypeError with the original message on failure."""
        try:
            return int(nbins)
        except Exception:
            msg = f"Parameter 'nbins' must be convertable to an integer via int(), got type {type(nbins)} (={nbins})"
            raise TypeError(msg)

    def _assign_axes_and_check_lengths(self, major_axis, minor_axis):
        """Assign axes to numpy arrays and check they have equal length."""
        major_arr = np.asarray(major_axis)
        minor_arr = np.asarray(minor_axis)
        if major_arr.size != minor_arr.size:
            msg = "Major and minor axes must have same length"
            raise ValueError(msg)
        return major_arr, minor_arr

    def _set_default_lims_and_bin_limits(self, major_lims, minor_lims):
        """
        Determine final major and minor limits (using data min/max if None)
        and compute bin_limits based on major_lims and self.nbins.
        Returns (major_lims_final, minor_lims_final).
        """
        # Major limits
        if major_lims is None:
            major_lims_final = (self.major_axis.min(), self.major_axis.max())
        else:
            major_lims_final = major_lims

        # Minor limits
        if minor_lims is None:
            minor_lims_final = (self.minor_axis.min(), self.minor_axis.max())
        else:
            minor_lims_final = minor_lims

        # Store and compute bin limits (nbins + 1 points for boundaries)
        self.bin_limits = np.linspace(major_lims_final[0], major_lims_final[1], self.nbins + 1)

        return major_lims_final, minor_lims_final

    @property
    def bin_widths(self) -> np.ndarray:
        """Return a numpy array of all bin widths, regardless of the point spacings."""
        return np.asarray([self.bin_width_n(i) for i in range(0, self.nbins)])

    def bin_width_n(self, bin_number: int) -> float:
        """Calculate the bin width for the nth bin.
        :param bin_number: The starting array index of the bin between 0 and self.nbins - 1.
        :return: The bin width, as a float.
        """
        lower, upper = self.get_bin_interval(bin_number)
        return upper - lower

    def get_bin_interval(self, bin_number: int) -> tuple[float, float]:

        """
        Return the lower and upper limits defining a bin, given its index.

        :param bin_number: The index of the bin (between 0 and self.nbins - 1)
        :return: A tuple of the interval limits as (lower, upper).
        """
        # Ensure bin_number is an integer and not a float or a string representation
        bin_number = int(bin_number)
        return self.bin_limits[bin_number], self.bin_limits[bin_number+1]

    def get_bin_index(self, value):
        """
        Return the index of the bin to which the supplied value belongs.
        Beware that min_value should always be numerically smaller than
        max_value. Take particular care when binning angles across the
        2pi to 0 discontinuity.

        :param value: A coordinate value in the binning interval along the major axis,
        whose bin index should be returned. Must be between min_value and max_value.

        The general formula logarithm binning is:
        bin = floor(N * (log(x) - log(min)) / (log(max) - log(min)))
        """
        if self.base:
            numerator  = self.nbins * (math.log(value, self.base) - math.log(self.major_lims[0], self.base))
            denominator = math.log(self.major_lims[1], self.base) - math.log(self.major_lims[0], self.base)
        else:
            numerator = self.nbins *(value - self.major_lims[0])
            denominator = self.major_lims[1] - self.major_lims[0]
        bin_index = int(math.floor(numerator / denominator))

        # Bins are indexed from 0 to nbins-1, so this check protects against
        # out-of-range indices when value == self.major_lims[1]
        if bin_index == self.nbins:
            bin_index -= 1

        return bin_index

    def compute_weights(self):
        """
        Return weights array for the contribution of each datapoint to each bin

        Each row of the weights array corresponds to the bin with the same
        index.
        """
        major_weights = np.zeros((self.nbins, self.major_axis.size))
        closed = IntervalType.CLOSED
        for m in range(self.nbins):
            # Include the value at the end of the binning range, but in
            # general use half-open intervals so each value belongs in only
            # one bin.
            if m == self.nbins - 1:
                interval = closed
            else:
                interval = IntervalType.HALF_OPEN
            bin_start, bin_end = self.get_bin_interval(bin_number=m)
            major_weights[m] = interval.weights_for_interval(array=self.major_axis,
                                                    l_bound=bin_start,
                                                    u_bound=bin_end)
        minor_weights = closed.weights_for_interval(array=self.minor_axis,
                                             l_bound=self.minor_lims[0],
                                             u_bound=self.minor_lims[1])
        return major_weights * minor_weights

    def __call__(self, data, err_data):
        """
        Compute the directional average of the supplied intensity & error data.

        :param data: intensity data from the origninal Data2D object.
        :param err_data: the corresponding errors for the intensity data.
        """
        weights = self.compute_weights()

        x_axis_values = np.sum(weights * self.major_axis, axis=1)
        intensity = np.sum(weights * data, axis=1)
        errs_squared = np.sum((weights * err_data)**2, axis=1)

        bin_counts = np.sum(weights, axis=1)
        # Prepare results, only compute division where bin_counts > 0
        if not np.any(bin_counts > 0):
            raise ValueError("Average Error: No bins inside ROI to average...")

        errors = np.sqrt(errs_squared)
        x_axis_values /= bin_counts
        intensity /= bin_counts
        errors /= bin_counts

        finite = np.isfinite(intensity)
        if not finite.any():
            msg = "Average Error: No points inside ROI to average..."
            raise ValueError(msg)

        return x_axis_values[finite], intensity[finite], errors[finite]
