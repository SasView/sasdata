"""
This module contains various data processors used by Sasview's slicers.
"""


import numpy as np

from sasdata.data_util.binning import DirectionalAverage
from sasdata.data_util.interval import IntervalType
from sasdata.data_util.roi import CartesianROI, PolarROI
from sasdata.dataloader.data_info import Data1D, Data2D
from sasdata.quantities.constants import Pi, TwoPi


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
        super().__init__(qx_range=qx_range,
                         qy_range=qy_range)

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
        super().__init__(qx_range=qx_range,
                         qy_range=qy_range)

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
        super().__init__(qx_range=qx_range,
                         qy_range=qy_range)
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

    def __init__(self, qx_range: tuple[float, float] = (0.0, 0.0), qy_range: tuple[float, float] = (0.0, 0), nbins: int = 100, fold: bool = False):
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
        super().__init__(qx_range=qx_range,
                         qy_range=qy_range)
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

    def __init__(self, r_range: tuple[float, float], center: tuple[float, float] = (0.0, 0.0), nbins: int = 100) -> None:
        """
        Set up the lower and upper radial limits as well as the number of bins.

        The units are A^-1 for the radial parameters.
        :param r_min: Lower limit for |Q| values to use during averaging.
        :param r_max: Upper limit for |Q| values to use during averaging.
        :param nbins: The number of bins data is sorted into along |Q| the axis
        """
        super().__init__(r_range=r_range, center = center)
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

    def __init__(self, r_range: tuple[float, float], center: tuple[float, float] = (0.0, 0.0),  nbins: int = 100) -> None:
        """
        Set up the lower and upper radial limits as well as the number of bins.

        The units are A^-1 for the radial parameters.
        :param r_min: Lower limit for |Q| values to use during averaging.
        :param r_max: Upper limit for |Q| values to use during averaging.
        :param nbins: The number of bins data is sorted into along Phi the axis
        """
        super().__init__(r_range=r_range, center=center)
        # backward-compatible alias expected by older tests / callers
        self.nbins_phi = nbins
        # new attribute
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

    def __init__(self, r_range: tuple[float, float], phi_range: tuple[float, float] = (0.0, TwoPi), center: tuple[float, float] = (0.0, 0.0), nbins: int = 100, fold: bool = True) -> None:
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
        super().__init__(r_range=r_range, phi_range=phi_range, center = center)

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
        self.phi_max = (self.phi_max - phi_offset) % (TwoPi)
        self.phi_data = (self.phi_data - phi_offset) % (TwoPi)

        major_lims = (self.r_min, self.r_max)
        minor_lims = (self.phi_min, self.phi_max)
        # Secondary region of interest covers angles on opposite side of origin
        minor_lims_alt = (self.phi_min + Pi, self.phi_max + Pi)

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

    def __init__(self, r_range: tuple[float, float], phi_range: tuple[float, float] = (0.0, TwoPi), center: tuple[float, float] = (0.0, 0.0),nbins: int = 100) -> None:
        """
        Set up the ROI boundaries, and the binning of the output 1D data.

        The units are A^-1 for radial parameters, and radians for anglar ones.
        :param r_range: Tuple (r_min, r_max) defining limits for |Q| values to use during averaging.
        :param phi_range: Tuple (phi_min, phi_max) defining limits for φ in radians (in the primary ROI).
        :Defaults to full circle (0, 2*pi).
        :param nbins: The number of bins data is sorted into along the |Q| axis
        """
        super().__init__(r_range=r_range, phi_range=phi_range, center = center)
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
        self.phi_max = (self.phi_max - phi_offset) % (TwoPi)
        self.phi_data = (self.phi_data - phi_offset) % (TwoPi)

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

    def __init__(self, r_range: tuple[float, float], phi_range: tuple[float, float] = (0.0, TwoPi), center: tuple[float, float] = (0.0, 0.0), nbins: int = 100) -> None:
        """
        Set up the ROI boundaries, and the binning of the output 1D data.

        The units are A^-1 for radial parameters, and radians for anglar ones.
        :param r_range: Tuple (r_min, r_max) defining limits for |Q| values to use during averaging.
        :param phi_range: Tuple (phi_min, phi_max) defining angular bounds in radians.
                          Defaults to full circle (0, 2*pi).
        :param nbins: The number of bins data is sorted into along the φ axis.
        """
        super().__init__(r_range=r_range, phi_range=phi_range, center = center)
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
        self.phi_max = (self.phi_max - phi_offset) % (TwoPi)
        self.phi_data = (self.phi_data - phi_offset) % (TwoPi)

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
    # Backwards-compatible constructor that accepts legacy keyword names
    # (r_min, r_max, phi_min, phi_max) and forwards them to the modern
    # initializer used by the parent classes.
    def __init__(self, r_min: float, r_max: float,
                phi_min: float = 0.0, phi_max: float = TwoPi,
                center: tuple[float, float] = (0.0, 0.0),
                nbins: int = 100) -> None:

    # Forward to WedgePhi using the tuple-based it expects.

        super().__init__(r_range=(self.r_min, self.r_max),phi_range=(self.phi_min, self.phi_max), center=center,nbins=self.nbins)
        
        # Ensure legacy attribute names exist on the instance (defensive).
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.nbins = int(nbins)

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

    def __init__(self, r_range: tuple[float, float] = (0.0, 0.0), phi_range: tuple[float, float] = (0.0, TwoPi), center: tuple[float, float] = (0.0, 0.0)):

        super().__init__(r_range, phi_range, center)

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

    def __init__(self, qx_range: tuple[float, float] = (0.0, 0.0), qy_range: tuple[float, float] = (0.0, 0.0)):
        super().__init__(qx_range=qx_range, qy_range=qy_range)

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

    def __init__(self, phi_range: tuple[float, float] = (0.0, Pi), center: tuple[float, float] = (0.0, 0.0)):
        super().__init__(r_range=(0, np.inf), phi_range=phi_range, center=center)

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
        self.phi_max = (self.phi_max - phi_offset) % (TwoPi)
        self.phi_data = (self.phi_data - phi_offset) % (TwoPi)
        phi_shifted = self.phi_data - Pi

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
