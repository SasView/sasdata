"""
This module contains various data processors used by Sasview's slicers.
"""


import math

import numpy as np

from sasdata.data_util.binning import DirectionalAverage
from sasdata.data_util.interval import IntervalType
from sasdata.data_util.roi import CartesianROI, PolarROI
from sasdata.dataloader.data_info import Data1D, Data2D
from sasdata.quantities.constants import Pi, TwoPi


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

    def __init__(self, qx_range: tuple[float, float] = (0.0, 0.0), qy_range: tuple[float, float] = (0.0, 0.0), nbins: int = 100, fold: bool = False, base: float = None):
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
        self.base = base

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
                                                 nbins=self.nbins, base=self.base)
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

    def __init__(self, qx_range: tuple[float, float] = (0.0, 0.0), qy_range: tuple[float, float] = (0.0, 0.0), nbins: int = 100, fold: bool = False, base: float = None):
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
        self.base =base

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
                                                 nbins=self.nbins, base =self.base)
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

    def __init__(self, r_range: tuple[float, float], center: tuple[float, float] = (0.0, 0.0), nbins: int = 100, base: float = None) -> None:
        """
        Set up the lower and upper radial limits as well as the number of bins.

        The units are A^-1 for the radial parameters.
        :param r_min: Lower limit for |Q| values to use during averaging.
        :param r_max: Upper limit for |Q| values to use during averaging.
        :param nbins: The number of bins data is sorted into along |Q| the axis
        """
        super().__init__(r_range=r_range, center = center)
        self.nbins = nbins
        self.base =  base



    def __call__(self, data2D, ismask=False):
        """
        Perform circular averaging on the data. Uses DirectionalAverage for
        bin construction and weights, and computes dx (d_q) using get_dq_data
        averaged with those weights so behavior matches the legacy implementation.

        :param data2D: Data2D object
        :param ismask: If True, respect data2D.mask (skip masked points). If False, ignore mask.
        :return: Data1D object with x (bin centers), y (intensity), dy and dx (if available)
        """
        # Work on unmasked finite arrays first (matches legacy filtering)
        finite_mask = np.isfinite(data2D.data)
        if not np.any(finite_mask):
            raise RuntimeError(f"Circular averaging: invalid q_data: {data2D.q_data}")

        data_all = data2D.data[finite_mask]
        q_all = data2D.q_data[finite_mask]
        qx_all = data2D.qx_data[finite_mask]
        qy_all = data2D.qy_data[finite_mask]
        err_all = data2D.err_data[finite_mask] if data2D.err_data is not None else None
        mask_all = data2D.mask[finite_mask]

        # Optional mask handling: legacy used an ismask flag to optionally skip masked points
        if ismask:
            sel = mask_all
        else:
            sel = np.ones_like(mask_all, dtype=bool)

        # Selected arrays used for binning & averaging
        major_axis = q_all[sel]
        phi_axis = np.arctan2(qy_all[sel], qx_all[sel])
        data_vals = data_all[sel]
        err_vals = err_all[sel] if err_all is not None else None

        # Prepare dq_data if available, aligned to the finite mask and selection
        dq_vals = None
        if getattr(data2D, "dqx_data", None) is not None and getattr(data2D, "dqy_data", None) is not None:
            dq_full = get_dq_data(data2D)  # already uses np.isfinite(data2D.data)
            dq_vals = dq_full[sel]

        # Set up DirectionalAverage with full-circle phi range
        major_lims = (self.r_min, self.r_max)
        minor_lims = (0.0, TwoPi)
        directional_average = DirectionalAverage(
            major_axis=major_axis,
            minor_axis=phi_axis,
            lims=(major_lims, minor_lims),
            nbins=self.nbins,
            base=self.base,
        )

        # Compute weights, then produce averaged intensity/error via DirectionalAverage
        weights = directional_average.compute_weights()
        _, intensity, error = directional_average(data=data_vals, err_data=err_vals)

        # Determine populated bins and produce legacy-style bin centres for x
        populated = np.sum(weights, axis=1) > 0
        if not np.any(populated):
            raise ValueError("Average Error: No points inside ROI to average...")

        bin_centres = directional_average.bin_limits[:-1] + directional_average.bin_widths / 2.0
        x = bin_centres[populated]

        # Compute dx (d_q) per bin by averaging dq_vals with the same weights
        if dq_vals is not None:
            dq_weighted = np.sum(weights * dq_vals, axis=1)
            counts = np.sum(weights, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dx_full = dq_weighted / counts
            dx = dx_full[populated]
        else:
            dx = None

        return Data1D(x=x, y=intensity, dy=error, dx=dx)

class Ring(PolarROI):
    """
    Calculate I(φ) by radially averaging 2D data between 2 radial limits.

    This class is initialised by specifying lower and upper limits on the
    magnitude of Q values to consider during the averaging. When called,
    this class is supplied with a Data2D object. It returns a Data1D object.
    This Data1D object gives intensity as a function of the angle from the
    positive x-axis, φ, only.
    """

    def __init__(self, r_range: tuple[float, float], center: tuple[float, float] = (0.0, 0.0),  nbins: int = 100, base: float = None) -> None:
        """
        Set up the lower and upper radial limits as well as the number of bins.

        The units are A^-1 for the radial parameters.
        :param r_min: Lower limit for |Q| values to use during averaging.
        :param r_max: Upper limit for |Q| values to use during averaging.
        :param nbins: The number of bins data is sorted into along Phi the axis
        """
        super().__init__(r_range=r_range, center=center)
        # backward-compatible alias expected by older tests / callers
        #self.nbins_phi = nbins
        # new attribute
        self.nbins = nbins
        self.base = base

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
        phi_bins = np.zeros(self.nbins)
        phi_counts = np.zeros(self.nbins)
        phi_values = np.zeros(self.nbins)
        phi_err = np.zeros(self.nbins)

        # Shift to apply to calculated phi values in order
        # to center first bin at zero
        phi_shift = Pi / self.nbins

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
            i_phi = int(math.floor((self.nbins) *
                                   (phi_value + phi_shift) / (2 * Pi)))

            # Take care of the edge case at phi = 2pi.
            if i_phi >= self.nbins:
                i_phi = 0
            phi_bins[i_phi] += frac * data[npt]

            if err_data is None or err_data[npt] == 0.0:
                if data_n < 0:
                    data_n = -data_n
                phi_err[i_phi] += frac * frac * math.fabs(data_n)
            else:
                phi_err[i_phi] += frac * frac * err_data[npt] * err_data[npt]
            phi_counts[i_phi] += frac

        for i in range(self.nbins):
            phi_bins[i] = phi_bins[i] / phi_counts[i]
            phi_err[i] = math.sqrt(phi_err[i]) / phi_counts[i]
            phi_values[i] = 2.0 * math.pi / self.nbins * (1.0 * i)

        idx = (np.isfinite(phi_bins))

        if not idx.any():
            msg = "Average Error: No points inside ROI to average..."
            raise ValueError(msg)

        return Data1D(x=phi_values[idx], y=phi_bins[idx], dy=phi_err[idx])


    '''
    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Phi axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # half-bin shift so the first bin is centered at zero
        phi_shift = np.pi / self.nbins
        shifted_phi = (self.phi_data +Pi+ phi_shift) % (TwoPi)

        self.phi_data=self.phi_data+Pi
        self.phi_min = 0.0
        self.phi_max = TwoPi

        minor_lims = (self.r_min, self.r_max)
        major_lims = (self.phi_min, self.phi_max)


        # major_lims is None because a full-circle angular range is used
        directional_average = DirectionalAverage(major_axis=shifted_phi,
                                                 minor_axis=self.q_data,
                                                 lims=(major_lims,minor_lims),
                                                 nbins=self.nbins, base=self.base)
        # Reuse DirectionalAverage's weights, then compute the same sums/divisions
        weights = directional_average.compute_weights()

        # Projected x values (averaged shifted phi per bin) -- not used as final x,
        # but computed here to mirror DirectionalAverage internal behaviour.
        x_axis_values = np.sum(weights * shifted_phi, axis=1)

        intensity = np.sum(weights * self.data, axis=1)
        errs_squared = np.sum((weights * self.err_data) ** 2, axis=1)

        bin_counts = np.sum(weights, axis=1)
        if not np.any(bin_counts > 0):
            raise ValueError("Average Error: No bins inside ROI to average...")

        errors = np.sqrt(errs_squared)

        # Only compute divisions where bin_counts > 0 (others will become NaN/inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_axis_values = x_axis_values / bin_counts
            intensity = intensity / bin_counts
            errors = errors / bin_counts

        # Legacy reported x values are the unshifted bin starts (i.e. 2*pi*i/nbins)
        phi_values = np.linspace(0.0, TwoPi, self.nbins, endpoint=False)

        finite = np.isfinite(intensity)
        if not finite.any():
            raise ValueError("Average Error: No points inside ROI to average...")

        return Data1D(x=phi_values[finite], y=intensity[finite], dy=errors[finite])
    '''

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

    def __init__(self, r_range: tuple[float, float], phi_range: tuple[float, float] = (0.0, TwoPi), center: tuple[float, float] = (0.0, 0.0), nbins: int = 100, fold: bool = True, base: float = None) -> None:
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
        self.base = base

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Q_y axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # Detect legacy phi convention (atan2 + pi -> values in [0, 2pi))
        phi_vals = self.phi_data
        try:
            min_phi = np.nanmin(phi_vals)
        except Exception:
            min_phi = None

        if min_phi is not None and min_phi >= 0.0:
            # Convert legacy shifted values back to standard atan2 range [-pi, pi]
            self.phi_data = phi_vals - np.pi

        # Transform all angles to the range [0,2π) where phi_min is at zero,
        # eliminating errors when the ROI straddles the 2π -> 0 discontinuity.
        # We won't need to convert back later because we're plotting against Q.
        phi_offset = self.phi_min
        self.phi_min = 0.0
        self.phi_max = (self.phi_max - phi_offset ) % (TwoPi)
        self.phi_data = (self.phi_data - phi_offset ) % (TwoPi)

        major_lims = (self.r_min, self.r_max)
        minor_lims = (self.phi_min, self.phi_max)
        # Secondary region of interest covers angles on opposite side of origin
        minor_lims_alt = (self.phi_min + Pi, self.phi_max + Pi)

        primary_region = DirectionalAverage(major_axis=self.q_data,
                                            minor_axis=self.phi_data,
                                            lims=(major_lims,minor_lims),
                                            nbins=self.nbins, base=self.base)
        secondary_region = DirectionalAverage(major_axis=self.q_data,
                                              minor_axis=self.phi_data,
                                              lims=(major_lims,minor_lims_alt),
                                              nbins=self.nbins, base=self.base)

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

    def __init__(self, r_range: tuple[float, float], phi_range: tuple[float, float] = (0.0, TwoPi), center: tuple[float, float] = (0.0, 0.0), nbins: int = 100, base: float = None) -> None:
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
        self.base = base

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Q_y axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # Detect legacy phi convention (atan2 + pi -> values in [0, 2pi))
        phi_vals = self.phi_data
        try:
            min_phi = np.nanmin(phi_vals)
        except Exception:
            min_phi = None

        if min_phi is not None and min_phi >= 0.0:
            # Convert legacy shifted values back to standard atan2 range [-pi, pi]
            phi_vals = phi_vals + np.pi

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
                                                 nbins=self.nbins, base=self.base)
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

    def __init__(self, r_range: tuple[float, float], phi_range: tuple[float, float] = (0.0, TwoPi), center: tuple[float, float] = (0.0, 0.0), nbins: int = 100, base: float = None) -> None:
        """
        Set up the ROI boundaries, and the binning of the output 1D data.

        The units are A^-1 for radial parameters, and radians for anglar ones.
        :param r_range: Tuple (r_min, r_max) defining limits for |Q| values to use during averaging.
        :param phi_range: Tuple (phi_min, phi_max) defining angular bounds in radians.
                          Defaults to full circle (0, 2*pi).
        :param nbins: The number of bins data is sorted into along the φ axis.
        """

        super().__init__(r_range=r_range, phi_range=phi_range, center = center)
        print(nbins)
        self.nbins = nbins
        self.base = base

    def __call__(self, data2d: Data2D = None) -> Data1D:
        """
        Compute the 1D average of 2D data, projecting along the Q_y axis.

        :param data2d: The Data2D object for which the average is computed.
        :return: Data1D object for plotting.
        """
        self.validate_and_assign_data(data2d)

        # Detect legacy phi convention (atan2 + pi -> values in [0, 2pi))
        phi_vals = self.phi_data
        try:
            min_phi = np.nanmin(phi_vals)
        except Exception:
            min_phi = None

        if min_phi is not None and min_phi >= 0.0:
            # Convert legacy shifted values back to standard atan2 range [-pi, pi]
            self.phi_data = phi_vals -  np.pi

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
                                                 nbins=self.nbins, base = self.base)
        phi_data, intensity, error = \
            directional_average(data=self.data, err_data=self.err_data)


        # Compute phi bin starts to match legacy behaviour (Ring / old SectorPhi)
        # phi_min has been normalized to 0 earlier; phi_offset stores original start.
        if self.phi_max == 0:
            # full circle
            full_phi = np.linspace(0.0, TwoPi, self.nbins, endpoint=False)
        else:
            full_phi = np.linspace(self.phi_min, self.phi_max, self.nbins, endpoint=False)

        # Shift back to original phi range
        full_phi = (full_phi + phi_offset) % (TwoPi)

       # Determine which bins were populated using the weights (preserves full bin index space)
        weights = directional_average.compute_weights()
        populated = np.sum(weights, axis=1) > 0

        # Bounds checking: ensure we have matching counts between populated bins and returned intensity
        if np.sum(populated) == 0:
            raise ValueError("Average Error: No points inside ROI to average...")

        # Construct phi values at (legacy) bin centers for only populated bins, matching order of returned intensity
        phi_centers = full_phi[populated] + directional_average.bin_widths[populated] / 2.0

        # intensity and error returned by DirectionalAverage are already filtered to the populated/finite bins
        return Data1D(x=phi_centers, y=intensity, dy=error)


        '''
        # Convert angular data back to the original phi range
        phi_data += phi_offset
        # In the old manipulations.py, we also had this shift to plot the data
        # at the centre of the bins. I'm not sure why it's only angular binning
        # which gets this treatment.
        # TODO: Update this once non-linear binning options are implemented
        weights = directional_average.compute_weights()
        populated = np.sum(weights, axis=1) > 0
        phi_data += directional_average.bin_widths[populated] / 2

        return Data1D(x=phi_data, y=intensity, dy=error)
        '''

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

        super().__init__(r_range=(r_min, r_max), phi_range=(phi_min, phi_max), center=center, nbins=nbins)



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
