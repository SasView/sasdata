import numpy as np

from sasdata.dataloader.data_info import Data1D, Data2D


def weights_for_interval(array, l_bound, u_bound, interval_type='half-open'):
    """
    If and when fractional binning is implemented (ask Lucas), this function
    will be changed so that instead of outputting zeros and ones, it gives
    fractional values instead. These will depend on how close the array value
    is to being within the interval defined.
    """
    # These checks could be modified to return fractional bin weights.
    # The last value in the binning range must be included for to pass utests
    if interval_type == 'half-open':
        in_range = (l_bound <= array) & (array < u_bound)
    elif interval_type == 'closed':
        in_range = (l_bound <= array) & (array <= u_bound)
    else:
        msg = f"Unrecognised interval_type: {interval_type}"
        raise ValueError(msg)

    return np.asarray(in_range, dtype=int)


class DirectionalAverage:
    """
    TODO - write a docstring
    """

    def __init__(self, major_data, minor_data, major_lims=None,
                 minor_lims=None, nbins=100):
        """
        """
        if any(not isinstance(data, (list, np.ndarray)) for
               data in (major_data, minor_data)):
            msg = "Must provide major & minor coordinate arrays for binning."
            raise ValueError(msg)
        if any(lims is not None and len(lims) != 2 for
               lims in (major_lims, minor_lims)):
            msg = "Limits arrays must have 2 elements or be NoneType"
            raise ValueError(msg)
        if not isinstance(nbins, int):
            msg = "Parameter 'nbins' must be an integer"
            raise TypeError(msg)

        self.major_data = np.asarray(major_data)
        self.minor_data = np.asarray(minor_data)
        if major_lims is None:
            self.major_lims = (self.major_data.min(), self.major_data.max())
        else:
            self.major_lims = major_lims
        if minor_lims is None:
            self.minor_lims = (self.minor_data.min(), self.minor_data.max())
        else:
            self.minor_lims = minor_lims
        self.nbins = nbins

    @property
    def bin_width(self):
        """
        Return the bin width
        """
        return (self.major_lims[1] - self.major_lims[0]) / self.nbins

    def get_bin_interval(self, bin_number):
        """
        Return the upper and lower limits defining a given bin
        """
        bin_start = self.major_lims[0] + bin_number * self.bin_width
        bin_end = self.major_lims[0] + (bin_number + 1) * self.bin_width

        return bin_start, bin_end

    def get_bin_index(self, value):
        """
        """
        numerator = value - self.major_lims[0]
        denominator = self.major_lims[1] - self.major_lims[0]
        bin_index = int(np.floor(self.nbins * numerator / denominator))

        # Bins are indexed from 0 to nbins-1, so tihs check protects against
        # out-of-range indices when value == self.major_lims[1]
        if bin_index == self.nbins:
            bin_index -= 1

        return bin_index

    def compute_weights(self):
        """
        """
        major_weights = np.zeros((self.nbins, self.major_data.size))
        for m in range(self.nbins):
            # Include the value at the end of the binning range, but in
            # general use half-open intervals so each value begins in only
            # one bin.
            if m == self.nbins - 1:
                interval = 'closed'
            else:
                interval = 'half-open'
            bin_start, bin_end = self.get_bin_interval(bin_number=m)
            major_weights[m] = weights_for_interval(array=self.major_data,
                                                    l_bound=bin_start,
                                                    u_bound=bin_end,
                                                    interval_type=interval)
        minor_weights = weights_for_interval(array=self.minor_data,
                                             l_bound=self.minor_lims[0],
                                             u_bound=self.minor_lims[1],
                                             interval_type='closed')
        return major_weights * minor_weights

    def __call__(self, data, err_data):
        """
        """
        weights = self.compute_weights()

        x_axis_values = np.sum(weights * self.major_data, axis=1)
        intensity = np.sum(weights * data, axis=1)
        errs_squared = np.sum((weights * err_data)**2, axis=1)
        bin_counts = np.sum(weights, axis=1)

        errors = np.sqrt(errs_squared)
        x_axis_values /= bin_counts
        intensity /= bin_counts
        errors /= bin_counts

        finite = (np.isfinite(x_axis_values) & np.isfinite(intensity))
        if not finite.any():
            msg = "Average Error: No points inside ROI to average..."
            raise ValueError(msg)

        return x_axis_values[finite], intensity[finite], errors[finite]


class GenericROI:
    """
    TODO - add docstring
    """

    def __init__(self):
        """
        """
        self.data = None
        self.err_data = None
        self.q_data = None
        self.qx_data = None
        self.qy_data = None

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

        # Only use data which is finite and not masked off
        valid_data = np.isfinite(data2d.data) & data2d.mask

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
    Base class for manipulators with a rectangular region of interest.
    """

    def __init__(self, qx_min: float = 0, qx_max: float = 0,
                 qy_min: float = 0, qy_max: float = 0) -> None:
        """
        TODO - add docstring
        """

        super().__init__()
        # Units A^-1
        self.qx_min = qx_min
        self.qx_max = qx_max
        self.qy_min = qy_min
        self.qy_max = qy_max


class PolarROI(GenericROI):
    """
    Base class for manipulators whose ROI is defined with polar coordinates.
    """

    def __init__(self, r_min: float, r_max: float,
                 phi_min: float = 0, phi_max: float = 2*np.pi) -> None:
        """
        TODO - add docstring
        """

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
        """
        super().validate_and_assign_data(data2d)
        self.phi_data = np.arctan2(self.qy_data, self.qx_data)


class Boxsum(CartesianROI):
    """
    Perform the sum of counts in a 2D region of interest.
    """

    def __init__(self, qx_min: float = 0, qx_max: float = 0,
                 qy_min: float = 0, qy_max: float = 0) -> None:
        """
        TODO - add docstring
        """
        super().__init__(qx_min=qx_min, qx_max=qx_max,
                         qy_min=qy_min, qy_max=qy_max)

    def __call__(self, data2d: Data2D = None) -> float:
        """
        TODO - add docstring
        """
        self.validate_and_assign_data(data2d)
        total_sum, error, count = self._sum()

        return total_sum, error, count

    def _sum(self) -> float:
        """
        TODO - add docstring
        """

        # Currently the weights are binary, but could be fractional in future
        x_weights = weights_for_interval(array=self.qx_data,
                                         l_bound=self.qx_min,
                                         u_bound=self.qx_max,
                                         interval_type='closed')
        y_weights = weights_for_interval(array=self.qy_data,
                                         l_bound=self.qy_min,
                                         u_bound=self.qy_max,
                                         interval_type='closed')
        weights = x_weights * y_weights

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
        """
        TODO - add docstring
        """
        super().__init__(qx_min=qx_min, qx_max=qx_max,
                         qy_min=qy_min, qy_max=qy_max)

    def __call__(self, data2d: Data2D) -> float:
        """
        TODO - add docstring
        """
        self.validate_and_assign_data(data2d)
        total_sum, error, count = super()._sum()

        return (total_sum / count), (error / count)


class SlabX(CartesianROI):
    """
    Compute average I(Qx) for a region of interest
    """

    def __init__(self, qx_min: float = 0, qx_max: float = 0, qy_min: float = 0,
                 qy_max: float = 0, nbins: int = 100, fold: bool = False):
        """
        TODO - add docstring
        """
        super().__init__(qx_min=qx_min, qx_max=qx_max,
                         qy_min=qy_min, qy_max=qy_max)
        self.nbins = nbins
        self.fold = fold

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute average I(Qx) for a region of interest
        :param data2d: Data2D object
        :return: Data1D object
        """
        self.validate_and_assign_data(data2d)

        if self.fold:
            major_lims = (0, self.qx_max)
            self.qx_data = np.abs(self.qx_data)
        else:
            major_lims = (self.qx_min, self.qx_max)
        minor_lims = (self.qy_min, self.qy_max)

        directional_average = DirectionalAverage(major_data=self.qx_data,
                                                 minor_data=self.qy_data,
                                                 major_lims=major_lims,
                                                 minor_lims=minor_lims,
                                                 nbins=self.nbins)
        qx_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        return Data1D(x=qx_data, y=intensity, dy=error)


class SlabY(CartesianROI):
    """
    Compute average I(Qy) for a region of interest
    """

    def __init__(self, qx_min: float = 0, qx_max: float = 0, qy_min: float = 0,
                 qy_max: float = 0, nbins: int = 100, fold: bool = False):
        """
        TODO - add docstring
        """
        super().__init__(qx_min=qx_min, qx_max=qx_max,
                         qy_min=qy_min, qy_max=qy_max)
        self.nbins = nbins
        self.fold = fold

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute average I(Qy) for a region of interest
        :param data2d: Data2D object
        :return: Data1D object
        """
        self.validate_and_assign_data(data2d)

        if self.fold:
            major_lims = (0, self.qy_max)
            self.qy_data = np.abs(self.qy_data)
        else:
            major_lims = (self.qy_min, self.qy_max)
        minor_lims = (self.qx_min, self.qx_max)

        directional_average = DirectionalAverage(major_data=self.qy_data,
                                                 minor_data=self.qx_data,
                                                 major_lims=major_lims,
                                                 minor_lims=minor_lims,
                                                 nbins=self.nbins)
        qy_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        return Data1D(x=qy_data, y=intensity, dy=error)


class CircularAverage(PolarROI):
    """
    Perform circular averaging on 2D data

    The data returned is the distribution of counts
    as a function of Q
    """

    def __init__(self, r_min: float, r_max: float, nbins: int = 100) -> None:
        """
        TODO - add docstring
        """
        super().__init__(r_min=r_min, r_max=r_max, phi_min=0, phi_max=2*np.pi)
        self.nbins = nbins

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        TODO - add docstring
        """
        self.validate_and_assign_data(data2d)

        # Averaging takes place between radial limits
        major_lims = (self.r_min, self.r_max)
        # Average over the full angular range
        directional_average = DirectionalAverage(major_data=self.q_data,
                                                 minor_data=self.phi_data,
                                                 major_lims=major_lims,
                                                 minor_lims=None,
                                                 nbins=self.nbins)
        q_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        return Data1D(x=q_data, y=intensity, dy=error)


class Ring(PolarROI):
    """
    Defines a ring on a 2D data set.
    The ring is defined by r_min, r_max, and
    the position of the center of the ring.

    The data returned is the distribution of counts
    around the ring as a function of phi.

    Phi_min and phi_max should be defined between 0 and 2*pi
    in anti-clockwise starting from the x- axis on the left-hand side
    """

    def __init__(self, r_min: float, r_max: float, nbins: int = 100) -> None:
        """
        TODO - add docstring
        """
        super().__init__(r_min=r_min, r_max=r_max, phi_min=0, phi_max=2*np.pi)
        self.nbins = nbins

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        TODO - add docstring
        """
        self.validate_and_assign_data(data2d)

        # Averaging takes place between radial limits
        minor_lims = (self.r_min, self.r_max)
        # Average over the full angular range
        directional_average = DirectionalAverage(major_data=self.phi_data,
                                                 minor_data=self.q_data,
                                                 major_lims=None,
                                                 minor_lims=minor_lims,
                                                 nbins=self.nbins)
        phi_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        return Data1D(x=phi_data, y=intensity, dy=error)


class SectorQ(PolarROI):
    """
    Sector average as a function of Q for both wings. setting the _Sector.fold
    attribute determines whether or not the two sectors are averaged together
    (folded over) or separate.  In the case of separate (not folded), the
    qs for the "minor wing" are arbitrarily set to a negative value.
    I(Q) is returned and the data is averaged over phi.

    A sector is defined by r_min, r_max, phi_min, phi_max.
    where r_min, r_max, phi_min, phi_max >0.
    The number of bins in Q also has to be defined.
    """

    def __init__(self, r_min: float, r_max: float, phi_min: float,
                 phi_max: float, nbins: int = 100, fold: bool = True) -> None:
        """
        """
        super().__init__(r_min=r_min, r_max=r_max,
                         phi_min=phi_min, phi_max=phi_max)
        self.nbins = nbins
        self.fold = fold

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        """
        self.validate_and_assign_data(data2d)

        # Transform all angles to the range [0,2π), where phi_min is at zero.
        # We won't need to convert back later because we're plotting against Q.
        phi_offset = self.phi_min
        self.phi_min = 0.0
        self.phi_max = (self.phi_max - phi_offset) % (2 * np.pi)
        self.phi_data = (self.phi_data - phi_offset) % (2 * np.pi)

        major_lims = (self.r_min, self.r_max)
        minor_lims = (self.phi_min, self.phi_max)
        # Secondary region of interest covers angles on opposite side of origin
        minor_lims_alt = (self.phi_min + np.pi, self.phi_max + np.pi)

        primary_region = DirectionalAverage(major_data=self.q_data,
                                            minor_data=self.phi_data,
                                            major_lims=major_lims,
                                            minor_lims=minor_lims,
                                            nbins=self.nbins)
        secondary_region = DirectionalAverage(major_data=self.q_data,
                                              minor_data=self.phi_data,
                                              major_lims=major_lims,
                                              minor_lims=minor_lims_alt,
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

            finite = (np.isfinite(combined_q) & np.isfinite(average_intensity))

            data1d = Data1D(x=combined_q[finite], y=average_intensity[finite],
                            dy=combined_err[finite])
        else:
            combined_q = np.append(np.flip(-1 * secondary_q), primary_q)
            combined_intensity = np.append(np.flip(secondary_I), primary_I)
            combined_error = np.append(np.flip(secondary_err), primary_err)
            data1d = Data1D(x=combined_q, y=combined_intensity,
                            dy=combined_error)

        return data1d


class SectorPhi(PolarROI):
    """
    Sector average as a function of phi.
    I(phi) is return and the data is averaged over Q.

    A sector is defined by r_min, r_max, phi_min, phi_max.
    The number of bin in phi also has to be defined.
    """

    def __init__(self, r_min: float, r_max: float, phi_min: float,
                 phi_max: float, nbins: int = 100) -> None:
        """
        TODO - add docstring
        """
        super().__init__(r_min=r_min, r_max=r_max,
                         phi_min=phi_min, phi_max=phi_max)
        self.nbins = nbins

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        TODO - add docstring
        """
        self.validate_and_assign_data(data2d)

        # Transform all angles to the range [0,2π), where phi_min is at zero.
        # Remember to transform back afterwards
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

        directional_average = DirectionalAverage(major_data=self.phi_data,
                                                 minor_data=self.q_data,
                                                 major_lims=major_lims,
                                                 minor_lims=minor_lims,
                                                 nbins=self.nbins)
        phi_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)

        # Convert angular data back to the original phi range
        phi_data += phi_offset
        # In the old manipulations.py, we also had this shift to plot the data
        # at the centre of the bins. I'm not sure why it's only angular binning
        # which gets this treatment.
        phi_data += directional_average.bin_width / 2

        return Data1D(x=phi_data, y=intensity, dy=error)

