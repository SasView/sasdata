"""
This module contains various data processors used by Sasview's slicers.
"""

from enum import Enum, auto

import numpy as np
from numpy.typing import ArrayLike

from sasdata.dataloader.data_info import Data1D, Data2D


class IntervalType(Enum):
    HALF_OPEN = auto()
    CLOSED = auto()

    def weights_for_interval(self, array, l_bound, u_bound):
        """
        Weight coordinate data by position relative to a specified interval.

        :param array: the array for which the weights are calculated
        :param l_bound: value defining the lower limit of the region of interest
        :param u_bound: value defining the upper limit of the region of interest

        If and when fractional binning is implemented (ask Lucas), this function
        will be changed so that instead of outputting zeros and ones, it gives
        fractional values instead. These will depend on how close the array value
        is to being within the interval defined.
        """

        # Whether the endpoint should be included depends on circumstance.
        # Half-open is used when binning the major axis (except for the final bin)
        # and closed used for the minor axis and the final bin of the major axis.
        if self.name.lower() == 'half_open':
            in_range = np.logical_and(l_bound <= array, array < u_bound)
        elif self.name.lower() == 'closed':
            in_range = np.logical_and(l_bound <= array, array <= u_bound)
        else:
            msg = f"Unrecognised interval_type: {self.name}"
            raise ValueError(msg)

        return np.asarray(in_range, dtype=int)



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
    binning which was only used by SectorQ. This functionality is never called
    upon by SasView however, so I haven't implemented it here (yet).
    """

    def __init__(self,
                 major_axis: ArrayLike,
                 minor_axis: ArrayLike,
                 lims: tuple[tuple[float, float] | None, tuple[float, float] | None] | None = None,
                 nbins: int = 100):
        """
        Set up direction of averaging, limits on the ROI, & the number of bins.

        :param major_axis: Coordinate data for axis onto which the 2D data is
                           projected.
        :param minor_axis: Coordinate data for the axis perpendicular to the
                           major axis.
        :param lims: Tuple (major_lims, minor_lims). Each element may be a
                     2-tuple or None. 
        :param nbins: The number of bins the major axis is divided up into.
        """

        if any(not hasattr(coordinate_data, "__array__") for
               coordinate_data in (major_axis, minor_axis)):
            msg = "Must provide major & minor coordinate arrays for binning."
            raise ValueError(msg)

        if lims is None:
            major_lims = minor_lims = None
        else:
            if not (isinstance(lims, (list, tuple)) and len(lims) == 2):
                msg = "Parameter 'lims' must be a 2-tuple (major_lims, minor_lims) or None."
                raise ValueError(msg)
            major_lims, minor_lims = lims

        if not isinstance(nbins, int):
            # TODO: Make classes that depend on this provide ints, its quite a thing to fix though
            try:
                nbins = int(nbins)
            except:
                msg = f"Parameter 'nbins' must be convertable to an integer via int(), got type {type(nbins)} (={nbins})"
                raise TypeError(msg)

        self.major_axis = np.asarray(major_axis)
        self.minor_axis = np.asarray(minor_axis)
        if self.major_axis.size != self.minor_axis.size:
            msg = "Major and minor axes must have same length"
            raise ValueError(msg)
        # In some cases all values from a given axis are part of the ROI.
        # An alternative approach may be needed for fractional weights.
        if major_lims is None:
            self.major_lims = (self.major_axis.min(), self.major_axis.max())
        else:
            self.major_lims = major_lims
        if minor_lims is None:
            self.minor_lims = (self.minor_axis.min(), self.minor_axis.max())
        else:
            self.minor_lims = minor_lims
        self.nbins = nbins
        # Assume a linear spacing for now, but allow for log, fibonacci, etc. implementations in the future
        # Add one to bin because this is for the limits, not centroids.
        self.bin_limits = np.linspace(self.major_lims[0], self.major_lims[1], self.nbins + 1)

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

        :param value: A coordinate value from somewhere along the major axis.
        """
        numerator = value - self.major_lims[0]
        denominator = self.major_lims[1] - self.major_lims[0]
        bin_index = int(np.floor(self.nbins * numerator / denominator))

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


class GenericROI:
    """
    Base class used to set up the data from a Data2D object for processing.
    This class performs any coordinate system independent setup and validation.
    """

    def __init__(self):
        """
        Assign the variables used to label the properties of the Data2D object.

        In classes inheriting from GenericROI, the variables used to define the
        boundaries of the Region Of Interest are also set up during __init__.
        """

        self.data = None
        self.err_data = None
        self.q_data = None
        self.qx_data = None
        self.qy_data = None


    def validate_and_assign_data(self, data2d: Data2D = None) -> None:
        """
        Check that the data supplied is valid and assign data to variables.
        This method must be executed before any further data processing happens

        :param data2d: A Data2D object which is the target of a child class'
                       data manipulations.
        """
        # Check that the supplied data2d is valid and usable.
        if not isinstance(data2d, Data2D):
            msg = "Data supplied must be of type Data2D."
            raise TypeError(msg)
        if len(data2d.detector) > 1:
            msg = f"Invalid number of detectors: {len(data2d.detector)}"
            raise ValueError(msg)

        # Only use data which is finite and not masked off
        valid_data = np.isfinite(data2d.data) & data2d.mask

        # Assign properties of the Data2D object to variables for reference
        # during data processing.
        self.data = data2d.data[valid_data]
        self.err_data = data2d.err_data[valid_data]
        self.q_data = data2d.q_data[valid_data]
        self.qx_data = data2d.qx_data[valid_data]
        self.qy_data = data2d.qy_data[valid_data]

        # No points should have zero error, if they do then assume the error is
        # the square root of the data. This code was added to replicate
        # previous functionality. It's a bit dodgy, so feel free to remove.
        self.err_data[self.err_data == 0] = \
            np.sqrt(np.abs(self.data[self.err_data == 0]))


class CartesianROI(GenericROI):
    """
    Base class for data manipulators with a Cartesian (rectangular) ROI.
    """

    def __init__(self, qx_range: tuple[float, float] = (0.0, 0.0), qy_range: tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Assign the variables used to label the properties of the Data2D object.
        Also establish the upper and lower bounds defining the ROI.

        The units of these parameters are A^-1
        :param qx_range: Bounds of the ROI along the Q_x direction.
        :param qy_range: Bounds of the ROI along the Q_y direction.
        """
        qx_min, qx_max = qx_range
        qy_min, qy_max = qy_range
        super().__init__()
        self.qx_min = qx_min
        self.qx_max = qx_max
        self.qy_min = qy_min
        self.qy_max = qy_max


class PolarROI(GenericROI):
    """
    Base class for data manipulators with a polar ROI.
    """

    def __init__(self, r_range: tuple[float, float], phi_range: tuple[float, float] = (0.0, 2*np.pi)) -> None:
        """
        Assign the variables used to label the properties of the Data2D object.
        Also establish the upper and lower bounds defining the ROI.

        The units are A^-1 for radial parameters, and radians for anglar ones.
        :param r_range: Tuple (r_min, r_max) defining limits for |Q| values to use during averaging.
        :param phi_range: Tuple (phi_min, phi_max) defining limits for φ in radians (in the ROI).

        Note that Phi is measured anti-clockwise from the positive x-axis.
        """
        r_min, r_max = r_range
        phi_min, phi_max = phi_range
        super().__init__()

        self.phi_data = None

        if r_min >= r_max:
            msg = "Minimum radius cannot be greater than maximum radius."
            raise ValueError(msg)
        # Units A^-1 for radii, radians for angles
        self.r_min = r_min
        self.r_max = r_max
        self.phi_min = phi_min
        self.phi_max = phi_max

    def validate_and_assign_data(self, data2d: Data2D = None) -> None:
        """
        Check that the data supplied valid and assign data variables.
        This method must be executed before any further data processing happens

        :param data2d: A Data2D object which is the target of a child class'
                       data manipulations.
        """

        # Most validation and pre-processing is taken care of by GenericROI.
        super().validate_and_assign_data(data2d)
        # Phi data can be calculated from the Cartesian Q coordinates.
        self.phi_data = np.arctan2(self.qy_data, self.qx_data)


class Boxsum(CartesianROI):
    """
    Compute the sum of the intensity within a rectangular Region Of Interest.
    """

    def __init__(self, qx_range: tuple[float, float] = (0.0, 0.0), qy_range: tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Set up the Region of Interest and its boundaries.

        The units of these parameters are A^-1
        :param qx_range: Bounds of the ROI along the Q_x direction.
        :param qy_range: Bounds of the ROI along the Q_y direction.
        """
        qx_min, qx_max = qx_range
        qy_min, qy_max = qy_range
        super().__init__(qx_range=(qx_min, qx_max),
                         qy_range=(qy_min, qy_max))

    def __call__(self, data2d: Data2D = None) -> float:
        """
        Coordinate data processing operations and return the results.

        :param data2d: The Data2D object for which the sum is calculated.
        """
        self.validate_and_assign_data(data2d)
        total_sum, error, count = self._sum()

        return total_sum, error, count

    def _sum(self) -> float:
        """
        Determine which data are inside the ROI and compute their sum.
        Also calculate the error on this calculation and the total number of
        datapoints in the region.
        """

        # Currently the weights are binary, but could be fractional in future
        interval = IntervalType.CLOSED
        x_weights = interval.weights_for_interval(array=self.qx_data,
                                         l_bound=self.qx_min,
                                         u_bound=self.qx_max)
        y_weights = interval.weights_for_interval(array=self.qy_data,
                                         l_bound=self.qy_min,
                                         u_bound=self.qy_max)
        weights = x_weights * y_weights

        data = weights * self.data
        # Not certain that the weights should be squared here, I'm just copying
        # how it was done in the old manipulations.py
        err_squared = weights * weights * self.err_data * self.err_data

        total_sum = np.sum(data)
        total_errors_squared = np.sum(err_squared)
        total_count = np.sum(weights)

        return total_sum, np.sqrt(total_errors_squared), total_count


class Boxavg(Boxsum):
    """
    Compute the average intensity within a rectangular Region Of Interest.
    """

    def __init__(self, qx_range: tuple[float, float] = (0.0, 0.0), qy_range: tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Set up the Region of Interest and its boundaries.

        The units of these parameters are A^-1
        :param qx_range: Bounds of the ROI along the Q_x direction.
        :param qy_range: Bounds of the ROI along the Q_y direction.
        """
        qx_min, qx_max = qx_range
        qy_min, qy_max = qy_range
        super().__init__(qx_range=(qx_min, qx_max),
                         qy_range=(qy_min, qy_max))

    def __call__(self, data2d: Data2D) -> float:
        """
        Coordinate data processing operations and return the results.

        :param data2d: The Data2D object for which the average is calculated.
        """
        self.validate_and_assign_data(data2d)
        total_sum, error, count = super()._sum()

        return (total_sum / count), (error / count)


class SlabX(CartesianROI):
    """
    Average I(Q_x, Q_y) along the y direction (within a ROI), giving I(Q_x).

    This class is initialised by specifying the boundaries of the ROI and is
    called by supplying a Data2D object. It returns a Data1D object.
    The averaging process can also be thought of as projecting 2D -> 1D.

    There also exists the option to "fold" the ROI, where Q data on opposite
    sides of the origin but with equal magnitudes are averaged together,
    resulting in a 1D plot with only positive Q values shown.
    """

    def __init__(self, qx_range: tuple[float, float] = (0.0, 0.0), qy_range: tuple[float, float] = (0.0, 0.0), nbins: int = 100, fold: bool = False):
        """
        Set up the ROI boundaries, the binning of the output 1D data, and fold.

        The units of these parameters are A^-1
        :param qx_range: Bounds of the ROI along the Q_x direction.
        :param qy_range: Bounds of the ROI along the Q_y direction.
        :param nbins: The number of bins data is sorted into along Q_x.
        :param fold: Whether the two halves of the ROI along Q_x should be
                     folded together during averaging.
        """
        qx_min, qx_max = qx_range
        qy_min, qy_max = qy_range
        super().__init__(qx_range=(qx_min, qx_max),
                         qy_range=(qy_min, qy_max))
        self.nbins = nbins
        self.fold = fold

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Q_x axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # SlabX is used by SasView's BoxInteractorX, which is designed so that
        # the ROI is always centred on the origin. If this ever changes, then
        # the behaviour of fold here will also need to change. Perhaps we could
        # apply a transformation to the data like the one used in WedgePhi.

        if self.fold:
            major_lims = (0, self.qx_max)
            self.qx_data = np.abs(self.qx_data)
        else:
            major_lims = (self.qx_min, self.qx_max)
        minor_lims = (self.qy_min, self.qy_max)

        directional_average = DirectionalAverage(major_axis=self.qx_data,
                                                 minor_axis=self.qy_data,
                                                 lims=(major_lims,minor_lims),
                                                 nbins=self.nbins)
        qx_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        return Data1D(x=qx_data, y=intensity, dy=error)


class SlabY(CartesianROI):
    """
    Average I(Q_x, Q_y) along the x direction (within a ROI), giving I(Q_y).

    This class is initialised by specifying the boundaries of the ROI and is
    called by supplying a Data2D object. It returns a Data1D object.
    The averaging process can also be thought of as projecting 2D -> 1D.

    There also exists the option to "fold" the ROI, where Q data on opposite
    sides of the origin but with equal magnitudes are averaged together,
    resulting in a 1D plot with only positive Q values shown.
    """

    def __init__(self, qx_range: tuple[float, float] = (0.0, 0), qy_range: tuple[float, float] = (0.0, 0), nbins: int = 100, fold: bool = False):
        """
        Set up the ROI boundaries, the binning of the output 1D data, and fold.

        The units of these parameters are A^-1
        :param qx_range: Bounds of the ROI along the Q_x direction.
        :param qy_range: Bounds of the ROI along the Q_y direction.
        :param qy_max: Upper bound of the ROI along the Q_y direction.
        :param nbins: The number of bins data is sorted into along Q_y.
        :param fold: Whether the two halves of the ROI along Q_y should be
                     folded together during averaging.
        """
        qx_min, qx_max = qx_range
        qy_min, qy_max = qy_range
        super().__init__(qx_range=(qx_min, qx_max),
                         qy_range=(qy_min, qy_max))
        self.nbins = nbins
        self.fold = fold

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Q_y axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # SlabY is used by SasView's BoxInteractorY, which is designed so that
        # the ROI is always centred on the origin. If this ever changes, then
        # the behaviour of fold here will also need to change. Perhaps we could
        # apply a transformation to the data like the one used in WedgePhi.

        if self.fold:
            major_lims = (0, self.qy_max)
            self.qy_data = np.abs(self.qy_data)
        else:
            major_lims = (self.qy_min, self.qy_max)
        minor_lims = (self.qx_min, self.qx_max)

        directional_average = DirectionalAverage(major_axis=self.qy_data,
                                                 minor_axis=self.qx_data,
                                                 lims=(major_lims,minor_lims),
                                                 nbins=self.nbins)
        qy_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        return Data1D(x=qy_data, y=intensity, dy=error)


class CircularAverage(PolarROI):
    """
    Calculate I(|Q|) by circularly averaging 2D data between 2 radial limits.

    This class is initialised by specifying lower and upper limits on the
    magnitude of Q values to consider during the averaging, though currently
    SasView always calls this class using the full range of data. When called,
    this class is supplied with a Data2D object. It returns a Data1D object
    where intensity is given as a function of Q only.
    """

    def __init__(self, r_range: tuple[float, float], nbins: int = 100) -> None:
        """
        Set up the lower and upper radial limits as well as the number of bins.

        The units are A^-1 for the radial parameters.
        :param r_min: Lower limit for |Q| values to use during averaging.
        :param r_max: Upper limit for |Q| values to use during averaging.
        :param nbins: The number of bins data is sorted into along |Q| the axis
        """
        r_min, r_max = r_range
        super().__init__(r_range=(r_min, r_max))
        self.nbins = nbins

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Q axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # Averaging takes place between radial limits
        major_lims = (self.r_min, self.r_max)
        # minor_lims is None because a full-circle angular range is used
        directional_average = DirectionalAverage(major_axis=self.q_data,
                                                 minor_axis=self.phi_data,
                                                 lims=(major_lims,None),
                                                 nbins=self.nbins)
        q_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        return Data1D(x=q_data, y=intensity, dy=error)


class Ring(PolarROI):
    """
    Calculate I(φ) by radially averaging 2D data between 2 radial limits.

    This class is initialised by specifying lower and upper limits on the
    magnitude of Q values to consider during the averaging. When called,
    this class is supplied with a Data2D object. It returns a Data1D object.
    This Data1D object gives intensity as a function of the angle from the
    positive x-axis, φ, only.
    """

    def __init__(self, r_range: tuple[float, float], nbins: int = 100) -> None:
        """
        Set up the lower and upper radial limits as well as the number of bins.

        The units are A^-1 for the radial parameters.
        :param r_min: Lower limit for |Q| values to use during averaging.
        :param r_max: Upper limit for |Q| values to use during averaging.
        :param nbins: The number of bins data is sorted into along Phi the axis
        """
        r_min, r_max = r_range
        super().__init__(r_range=(r_min, r_max))
        self.nbins = nbins

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Phi axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # Averaging takes place between radial limits
        minor_lims = (self.r_min, self.r_max)
        # major_lims is None because a full-circle angular range is used
        directional_average = DirectionalAverage(major_axis=self.phi_data,
                                                 minor_axis=self.q_data,
                                                 lims=(None,minor_lims),
                                                 nbins=self.nbins)
        phi_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        return Data1D(x=phi_data, y=intensity, dy=error)


class SectorQ(PolarROI):
    """
    Project I(Q, φ) data onto I(Q) within a region defined by Cartesian limits.

    The projection is computed by averaging together datapoints with the same
    angle φ (so long as they are within the ROI), measured anticlockwise from
    the positive x-axis.

    This class is initialised by specifying lower and upper limits on both the
    magnitude of Q and the angle φ. These four parameters specify the primary
    Region Of Interest, however there is a secondary ROI with the same |Q|
    values on the opposite side of the origin (φ + π). How this secondary ROI
    is treated depends on the value of the `fold` parameter. If fold is set to
    True, data on opposite sides of the origin are averaged together and the
    results are plotted against positive values of Q. If fold is set to False,
    the data from the two regions are graphed separeately, with the secondary
    ROI data labelled using negative Q values.

    When called, this class is supplied with a Data2D object. It returns a
    Data1D object where intensity is given as a function of Q only.
    """

    def __init__(self, r_range: tuple[float, float], phi_range: tuple[float, float] = (0.0, 2*np.pi), nbins: int = 100, fold: bool = True) -> None:
        """
        Set up the ROI boundaries, the binning of the output 1D data, and fold.

        The units are A^-1 for radial parameters, and radians for anglar ones.
        :param r_range: Tuple (r_min, r_max) defining limits for |Q| values to use during averaging.
        :param phi_range: Tuple (phi_min, phi_max) defining limits for φ in radians (in the primary ROI).
        :Defaults to full circle (0, 2*pi).
        :param nbins: The number of bins data is sorted into along the |Q| axis
        :param fold: Whether the primary and secondary ROIs should be folded
                     together during averaging.
        """
        r_min, r_max = r_range
        phi_min, phi_max = phi_range
        super().__init__(r_range=(r_min, r_max),
                         phi_range=(phi_min, phi_max))

        self.nbins = nbins
        self.fold = fold

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Q_y axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # Transform all angles to the range [0,2π) where phi_min is at zero,
        # eliminating errors when the ROI straddles the 2π -> 0 discontinuity.
        # We won't need to convert back later because we're plotting against Q.
        phi_offset = self.phi_min
        self.phi_min = 0.0
        self.phi_max = (self.phi_max - phi_offset) % (2 * np.pi)
        self.phi_data = (self.phi_data - phi_offset) % (2 * np.pi)

        major_lims = (self.r_min, self.r_max)
        minor_lims = (self.phi_min, self.phi_max)
        # Secondary region of interest covers angles on opposite side of origin
        minor_lims_alt = (self.phi_min + np.pi, self.phi_max + np.pi)

        primary_region = DirectionalAverage(major_axis=self.q_data,
                                            minor_axis=self.phi_data,
                                            lims=(major_lims,minor_lims),
                                            nbins=self.nbins)
        secondary_region = DirectionalAverage(major_axis=self.q_data,
                                              minor_axis=self.phi_data,
                                              lims=(major_lims,minor_lims_alt),
                                              nbins=self.nbins)

        primary_q, primary_I, primary_err = \
            primary_region(data=self.data, err_data=self.err_data)
        secondary_q, secondary_I, secondary_err = \
            secondary_region(data=self.data, err_data=self.err_data)

        if self.fold:
            # Combining the two regions requires re-binning; the q value
            # arrays may be unequal lengths, or the indices may correspond to
            # different q values. To average the results from >2 ROIs you would
            # need to generalise this process.
            combined_q = np.zeros(self.nbins)
            average_intensity = np.zeros(self.nbins)
            combined_err = np.zeros(self.nbins)
            bin_counts = np.zeros(self.nbins)
            for old_index, q_val in enumerate(primary_q):
                old_index = int(old_index)
                new_index = primary_region.get_bin_index(q_val)
                combined_q[new_index] += q_val
                average_intensity[new_index] += primary_I[old_index]
                combined_err[new_index] += primary_err[old_index] ** 2
                bin_counts[new_index] += 1
            for old_index, q_val in enumerate(secondary_q):
                old_index = int(old_index)
                new_index = secondary_region.get_bin_index(q_val)
                combined_q[new_index] += q_val
                average_intensity[new_index] += secondary_I[old_index]
                combined_err[new_index] += secondary_err[old_index] ** 2
                bin_counts[new_index] += 1

            combined_q /= bin_counts
            average_intensity /= bin_counts
            combined_err = np.sqrt(combined_err) / bin_counts

            finite = np.isfinite(average_intensity)

            data1d = Data1D(x=combined_q[finite], y=average_intensity[finite],
                            dy=combined_err[finite])
        else:
            # The secondary ROI is labelled with negative Q values.
            combined_q = np.append(np.flip(-1 * secondary_q), primary_q)
            combined_intensity = np.append(np.flip(secondary_I), primary_I)
            combined_error = np.append(np.flip(secondary_err), primary_err)
            data1d = Data1D(x=combined_q, y=combined_intensity,
                            dy=combined_error)

        return data1d


class WedgeQ(PolarROI):
    """
    Project I(Q, φ) data onto I(Q) within a region defined by Cartesian limits.

    The projection is computed by averaging together datapoints with the same
    angle φ (so long as they are within the ROI), measured anticlockwise from
    the positive x-axis.

    This class is initialised by specifying lower and upper limits on both the
    magnitude of Q and the angle φ.
    When called, this class is supplied with a Data2D object. It returns a
    Data1D object where intensity is given as a function of Q only.
    """

    def __init__(self, r_range: tuple[float, float], phi_range: tuple[float, float] = (0.0, 2*np.pi), nbins: int = 100) -> None:
        """
        Set up the ROI boundaries, and the binning of the output 1D data.

        The units are A^-1 for radial parameters, and radians for anglar ones.
        :param r_range: Tuple (r_min, r_max) defining limits for |Q| values to use during averaging.
        :param phi_range: Tuple (phi_min, phi_max) defining limits for φ in radians (in the primary ROI).
        :Defaults to full circle (0, 2*pi).
        :param nbins: The number of bins data is sorted into along the |Q| axis
        """
        r_min, r_max = r_range
        phi_min, phi_max = phi_range
        super().__init__(r_range=(r_min, r_max),
                         phi_range=(phi_min, phi_max))
        self.nbins = nbins

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Q_y axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # Transform all angles to the range [0,2π) where phi_min is at zero,
        # eliminating errors when the ROI straddles the 2π -> 0 discontinuity.
        # We won't need to convert back later because we're plotting against Q.
        phi_offset = self.phi_min
        self.phi_min = 0.0
        self.phi_max = (self.phi_max - phi_offset) % (2 * np.pi)
        self.phi_data = (self.phi_data - phi_offset) % (2 * np.pi)

        # Averaging takes place between radial and angular limits
        major_lims = (self.r_min, self.r_max)
        # When phi_max and phi_min have the same angle, ROI is a full circle.
        if self.phi_max == 0:
            minor_lims = None
        else:
            minor_lims = (self.phi_min, self.phi_max)

        directional_average = DirectionalAverage(major_axis=self.q_data,
                                                 minor_axis=self.phi_data,
                                                 lims=(major_lims,minor_lims),
                                                 nbins=self.nbins)
        q_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        return Data1D(x=q_data, y=intensity, dy=error)


class WedgePhi(PolarROI):
    """
    Project I(Q, φ) data onto I(φ) within a region defined by Cartesian limits.

    The projection is computed by averaging together datapoints with the same
    Q value (so long as they are within the ROI).

    This class is initialised by specifying lower and upper limits on both the
    magnitude of Q and the angle φ, measured anticlockwise from the positive
    x-axis.
    When called, this class is supplied with a Data2D object. It returns a
    Data1D object where intensity is given as a function of Q only.
    """

    def __init__(self, r_range: tuple[float, float], phi_range: tuple[float, float] = (0.0, 2*np.pi), nbins: int = 100) -> None:
        """
        Set up the ROI boundaries, and the binning of the output 1D data.

        The units are A^-1 for radial parameters, and radians for anglar ones.
        :param r_range: Tuple (r_min, r_max) defining limits for |Q| values to use during averaging.
        :param phi_range: Tuple (phi_min, phi_max) defining angular bounds in radians.
                          Defaults to full circle (0, 2*pi).
        :param nbins: The number of bins data is sorted into along the φ axis.
        """
        r_min, r_max = r_range
        phi_min, phi_max = phi_range
        super().__init__(r_range=(r_min, r_max),
                         phi_range=(phi_min, phi_max))
        self.nbins = nbins

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Q_y axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # Transform all angles to the range [0,2π) where phi_min is at zero,
        # eliminating errors when the ROI straddles the 2π -> 0 discontinuity.
        # Remember to transform back afterward as we're plotting against phi.
        phi_offset = self.phi_min
        self.phi_min = 0.0
        self.phi_max = (self.phi_max - phi_offset) % (2 * np.pi)
        self.phi_data = (self.phi_data - phi_offset) % (2 * np.pi)

        # Averaging takes place between angular and radial limits
        # When phi_max and phi_min have the same angle, ROI is a full circle.
        if self.phi_max == 0:
            major_lims = None
        else:
            major_lims = (self.phi_min, self.phi_max)
        minor_lims = (self.r_min, self.r_max)

        directional_average = DirectionalAverage(major_axis=self.phi_data,
                                                 minor_axis=self.q_data,
                                                 lims=(major_lims,minor_lims),
                                                 nbins=self.nbins)
        phi_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        # Convert angular data back to the original phi range
        phi_data += phi_offset
        # In the old manipulations.py, we also had this shift to plot the data
        # at the centre of the bins. I'm not sure why it's only angular binning
        # which gets this treatment.
        # TODO: Update this once non-linear binning options are implemented
        phi_data += directional_average.bin_widths / 2

        return Data1D(x=phi_data, y=intensity, dy=error)


class SectorPhi(WedgePhi):
    """
    Sector average as a function of phi.
    I(phi) is return and the data is averaged over Q.

    A sector is defined by r_min, r_max, phi_min, phi_max.
    The number of bin in phi also has to be defined.
    """

    # This class has only been kept around in case users are using it in
    # scripts, SectorPhi was never used by SasView. The functionality is now in
    # use through WedgeSlicer.py, so the rewritten version of this class has
    # been named WedgePhi.

################################################################################


class Ringcut(PolarROI):
    """
    Defines a ring on a 2D data set.
    The ring is defined by r_min, r_max, and
    the position of the center of the ring.

    The data returned is the region inside the ring

    Phi_min and phi_max should be defined between 0 and 2*pi
    in anti-clockwise starting from the x- axis on the left-hand side
    """

    def __init__(self, r_range: tuple[float, float] = (0.0, 0.0), phi_range: tuple[float, float] = (0.0, 2*np.pi)):

        super().__init__(r_range, phi_range)

    def __call__(self, data2D: Data2D) -> np.ndarray[bool]:
        """
        Apply the ring to the data set.
        Returns the angular distribution for a given q range

        :param data2D: Data2D object

        :return: index array in the range
        """
        super().validate_and_assign_data(data2D)

        # Calculate q_data using unmasked qx_data and qy_data
        q_data = np.sqrt(data2D.qx_data * data2D.qx_data + data2D.qy_data * data2D.qy_data)

        # check whether each data point is inside ROI
        out = (self.r_min <= q_data) & (self.r_max >= q_data)
        return out


class Boxcut(CartesianROI):
    """
    Find a rectangular 2D region of interest.
    """

    def __init__(self, x_min: float = 0.0, x_max: float = 0.0, y_min: float = 0.0, y_max: float = 0.0):
        super().__init__(x_min, x_max, y_min, y_max)

    def __call__(self, data2D: Data2D) -> np.ndarray[bool]:
        """
       Find a rectangular 2D region of interest where  data points inside the ROI are True, and False otherwise

       :param data2D: Data2D object
       :return: mask, 1d array (len = len(data))
        """
        super().validate_and_assign_data(data2D)

        # check whether each data point is inside ROI
        outx = (self.qx_min <= data2D.qx_data) & (self.qx_max > data2D.qx_data)
        outy = (self.qy_min <= data2D.qy_data) & (self.qy_max > data2D.qy_data)

        return outx & outy


class Sectorcut(PolarROI):
    """
    Defines a sector (major + minor) region on a 2D data set.
    The sector is defined by phi_min, phi_max,
    where phi_min and phi_max are defined by the right
    and left lines wrt central line.

    Phi_min and phi_max are given in units of radian
    and (phi_max-phi_min) should not be larger than pi
    """

    def __init__(self, phi_range: tuple[float, float] = (0.0, np.pi)):
        super().__init__(r_range=(0, np.inf), phi_range=phi_range)

    def __call__(self, data2D: Data2D) -> np.ndarray[bool]:
        """
        Find a rectangular 2D region of interest where  data points inside the ROI are True, and False otherwise

        :param data2D: Data2D object
        :return: mask, 1d array (len = len(data))
        """
        super().validate_and_assign_data(data2D)

        # Ensure unmasked data is used for the phi_data calculation to ensure data sizes match
        self.phi_data = np.arctan2(data2D.qy_data, data2D.qx_data)
        # Calculate q_data using unmasked qx_data and qy_data to ensure data sizes match
        q_data = np.sqrt(data2D.qx_data * data2D.qx_data + data2D.qy_data * data2D.qy_data)


        phi_offset = self.phi_min
        self.phi_min = 0.0
        self.phi_max = (self.phi_max - phi_offset) % (2 * np.pi)
        self.phi_data = (self.phi_data - phi_offset) % (2 * np.pi)
        phi_shifted = self.phi_data - np.pi


        # Determine angular bounds for both upper and lower half of image
        phi_min_angle, phi_max_angle = (self.phi_min, self.phi_max)

        # Determine regions of interest
        out_radial = (self.r_min <= q_data) & (self.r_max > q_data)
        out_upper = (phi_min_angle <= self.phi_data) & (phi_max_angle >= self.phi_data)
        out_lower = (phi_min_angle <= phi_shifted) & (phi_max_angle >= phi_shifted)

        upper_roi = out_radial & out_upper
        lower_roi = out_radial & out_lower
        out = upper_roi | lower_roi

        return out
