

import numpy as np
from numpy._typing import ArrayLike

from sasdata.quantities.quantity import Quantity


class NDRebin:
    """
    N-dimensional rebinning of data into regular bins, with optional
    fractional binning and error propagation.

    Provide values at points with ND coordinates.
    The coordinates may not be in a nice grid.
    The data can be in any array shape.

    The coordinates are in the same shape plus one dimension,
        preferably the first dimension, which is the ND coordinate
        position

    Rebin that data into a regular grid.

    Note that this does lose some information from the underlying data,
        as you are essentially averaging multiple measurements into one bin.

    Note that once can use this function to perform integrations over
        one or more dimensions by setting the num_bins to 1 or the 
        step_size to infinity for one or more axes. The integration will
        be performed from the lower to upper bound of that axis.

    Parameters
    ----------
    data : Quantity[ArrayLike]
        Data values in an Nd array.
    coords : Quantity[ArrayLike]
        The coordinates corresponding to each data point, same size of data
        plus one more dimension with the same length as the
        dimensionality of the space (Ndim)
    data_errs : Quantity[ArrayLike], optional
        Errors on data. Optional, the same size as data.
    axes : ArrayLike | None = None
        The axes of the coordinate system we are binning
        into. Defaults to diagonal (e.g. (1,0,0), (0,1,0), and 
        (0,0,1) for 3D data). A list of Ndim element vectors
    upper : ArrayLike | None = None
        The upper limits along each axis. Defaults to the largest
        values in the data if no limits are provided.
        A 1D list of Ndims values.
    lower : ArrayLike | None = None
        The lower limits along each axis. Defaults to the smallest
        values in the data if no limits are provided.
        A 1D list of Ndims values.
    step_size : ArrayLike | None = None
        The size of steps along each axis. Supercedes
        num_bins. A list of length Ndim.
    num_bins : ArrayLike | None = None
        The number of bins along each axis. Superceded by
        step_size if step_size is provided. At least one of step_size
        or num_bins must be provided.
    fractional : bool = False
        Whether to perform fractional binning or not. Defaults 
        to false.
        -If false, measurements are binned into one bin,
        the one they fall within. Roughly a "nearest neighbor"
        approach.
        -If true, fractional binning will be applied, where
        the value of a measurement is distributed to its 2^Ndim
        nearest neighbors weighted by proximity. For example, if
        a point falls exactly between two bins, its value will be
        given to both bins with 50% weight. This is roughly a
        "linear interpolation" approach. Tends to do better at 
        reducing sharp peaks and edges if data is sampled unevenly.
        However, this is roughly 2^Ndim times slower since you have
        to address each bin 2^Ndim more times.
    normalization : bool = True
        Whether to normalize (average) the data or not. If false,
        the data are just summed into each bin. If true, the weighted
        average of all points added to a bin is computed.

    Attributes
    ----------
    binned_data : 
        has size num_bins and is NDimensional, contains
        the binned data
    bin_centers_list : 
        is a list of 1D vectors, contains the
        axes of the binned data. The coordinates of bin [i,j,k]
        is given by 
        bin_centers_list[0][i]*axes[i]+bin_centers_list[1][j]*axes[j]+
        bin_centers_list[0][k]*axes[k]
    binned_data_errs : 
        has size num_bins and is NDimensional, contains
        the propagated errors of the binned_data
    bins_list : 
        is a list of 1D vectors, is similar to bin_centers_list,
        but instead contains the edges of the bins, so it is 1 longer
        in each dimension
    step_size : 
        is a list of Ndims numbers, contains the step size
        along each dimensino
    num_bins : 
        is a list of Ndims numbers, contains the number
        of bins along each dimension

    Methods
    -------
    run(self):
        Bin the data into the defined bins.

    Typical usage
    -------------
    .. code-block::
    # test syntax 1
    Ndims = 4
    Nvals = int(1e4)
    qmat = np.random.rand(Ndims, Nvals)
    Imat = np.random.rand(Nvals)

    rebin = NDRebin(Imat, qmat,
        step_size=0.1*np.random.rand(Ndims)+0.05,
        lower=0.1*np.random.rand(Ndims)+0.0,
        upper=0.1*np.random.rand(Ndims)+0.9)
    rebin.run()

    Ibin = rebin.binned_data
    qbin = rebin.bin_centers_list


    # test syntax 2
    Ndims = 2
    Nvals = int(1e4)
    qmat = np.random.rand(Ndims, 100, Nvals)
    Imat = np.random.rand(100, Nvals)
    Imat_errs = np.random.rand(100, Nvals)

    rebin = NDRebin(Imat, qmat,
        data_errs = Imat_errs,
        num_bins=[10,20],
        axes = np.eye(2),
        fractional=True)
    rebin.run()

    Ibin = rebin.binned_data
    qbin = rebin.bin_centers_list
    Ibin_errs = rebin.binned_data_errs
    bins_list = rebin.bins_list
    step_size = rebin.step_size
    num_bins = rebin.num_bins
    """

    def __init__(
        self,
        data: Quantity[ArrayLike],
        coords: Quantity[ArrayLike],
        data_errs: Quantity[ArrayLike] | None = None,
        axes: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        lower: ArrayLike | None = None,
        step_size: ArrayLike | None = None,
        num_bins: ArrayLike | None = None,
        fractional: bool = False,
        normalize: bool = True,
    ):
        self.data = data
        self.coords = coords
        self.data_errs = data_errs
        self.axes = axes
        self.upper = upper
        self.lower = lower
        self.step_size = step_size
        self.num_bins = num_bins
        self.fractional = fractional
        self.normalize = normalize

        # Internal attributes initialised later
        self.Nvals: int | None = None
        self.Ndims: int | None = None
        self.data_flat = None
        self.errors_flat = None
        self.coords_flat = None
        self.bins_list = None
        self.bin_centers_list = None
        self.bin_inds = None
        self.binned_data = None
        self.binned_data_errs = None
        self.n_samples = None

        self._prepared = False   # flag to avoid double-prepare

    def __call__(self):
        self.run()
        return self.binned_data

    def run(self) -> None:
        """Bin the data into the defined bins."""
        if not self._prepared:
            self._prepare()

        if self.fractional:
            self._calculate_fractional_bins()
        else:
            self._calculate_bins()

        self._norm_data()

    def _prepare(self) -> None:
        """Compute derived quantities: shapes, flattened data, bins, indices."""
        if self._prepared:
            return

        # check the size of the data and coords inputs
        # and define Ndims and Nvals
        self._check_data_coords()

        # flatten the input data and errors
        self._check_data_errs()

        # flatten the coords
        self._flatten_coords()

        # handle optional axes
        if self.axes is None:
            # make axes if not provided
            self._make_axes()
        else:
            # project into specified axes
            self._project_axes()

        # build the limits
        self._build_limits()

        # make the bins
        self._make_bins()

        # make the bin indices
        self._create_bin_inds()

        self._prepared = True

    def _check_data_coords(self):
        """Compute Nvals and Ndims and validate shapes."""
        # Identify number of points
        self.Nvals = int(self.data.size)

        # Identify number of dimensions
        Ndims = self.coords.size / self.Nvals

        # if Ndims is not an integer value we have a problem
        if not float(Ndims).is_integer():
            raise ValueError("The coords have to have the same shape as "
                            "the data, plus one more dimension which is "
                            "length Ndims")
        self.Ndims = int(Ndims)

    def _check_data_errs(self):
        # flatten input data to 1D of length Nvals
        self.data_flat = self.data.reshape(-1)
        if self.data_errs is None:
            self.errors_flat = 0*self.data_flat   # no errors
        else:
            self.errors_flat = self.data_errs.reshape(-1)

        if self.errors_flat.shape != self.data_flat.shape:
            raise ValueError("Data and errors have to have the same shape.")

    def _flatten_coords(self):
        # if 1D, need to add a size 1 dimension index to coords
        if self.Ndims == 1:
            self.coords = self.coords.reshape(-1, 1)

        # check if the first axis of coords is the dimensions axis
        if self.coords.shape[0] == self.Ndims:
            # first check if it is the first axis
            self.dim_axis = 0
        elif self.coords.shape[-1] == self.Ndims:
            # now check if it is the last axis
            self.dim_axis = -1
        else:
            # search if any axis is size Ndims
            self.dim_axis = next(i for i, s in enumerate(self.coords.shape) if s == self.Ndims)

        if not self.coords.shape[self.dim_axis] == self.Ndims:
            raise ValueError("The coords have to have one dimension which is "
                            "the dimensionality of the space")

        # flatten coords to size Nvals x Ndims
        moved = np.moveaxis(self.coords, self.dim_axis, 0)
        self.coords_flat = moved.reshape(self.Ndims, -1).T

    def _make_axes(self):
        # if axes are not provided, default to identity
        if self.axes is None:
            self.axes = np.eye(self.Ndims)

    def _project_axes(self):
        # now project the data into the axes
        self.axes_inv = np.linalg.inv(self.axes)
        self.coords_flat = np.tensordot(self.coords_flat, self.axes_inv, axes=([1], [0]))

    def _build_limits(self):
        # if limits were not provided, default to the min and max
        # coord in each dimension
        coords = self.coords_flat
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        lower = mins if self.lower is None else self.lower
        upper = maxs if self.upper is None else self.upper

        # if provided just one limit for 1D as a scalar, make it a list
        # for formatting purposes
        self.lower = np.atleast_1d(self.lower)
        self.upper = np.atleast_1d(self.upper)

        # validate limits sizes
        if self.lower.size != self.Ndims:
            raise ValueError("Lower limits must be None or a 1D iterable of length Ndims.")
        if self.upper.size != self.Ndims:
            raise ValueError("Upper limits must be None or a 1D iterable of length Ndims.")

        # if individual limits are nan, inf, none, etc, replace with min/max
        finite_lower = np.isfinite(lower)
        finite_upper = np.isfinite(upper)
        lower = np.where(finite_lower, lower, mins)
        upper = np.where(finite_upper, upper, maxs)

        # if any of the limits are in the wrong order, flip them
        self.lower = np.minimum(lower, upper)
        self.upper = np.maximum(lower, upper)

    def _make_bins(self):
        # bins_list is a Ndims long list of vectors which are the edges of
        #   each bin. Each vector is num_bins[i]+1 long
        self.bins_list = []

        # bin_centers_list is a Ndims long list of vectors which are the centers of
        #   each bin. Each vector is num_bins[i] long
        self.bin_centers_list = []

        # create the bins in each dimension
        if self.step_size is None:
            self._step_size_from_num_bins()
        else:
            self._num_bins_from_step_size()

    def _step_size_from_num_bins(self):
        # if step_size was not specified, derive from num_bins
        self.step_size = []
        # if provided just one num_bin for 1D as a scalar, make it a list
        # for formatting purposes
        self.num_bins = np.atleast_1d(self.num_bins)
        if self.num_bins.size != self.Ndims:
            raise ValueError("num_bins must be None or a 1D iterable of length Ndims.")
        for ind in range(self.Ndims):
            these_bins = np.linspace(self.lower[ind], self.upper[ind], self.num_bins[ind]+1)
            these_centers = (these_bins[:-1] + these_bins[1:]) / 2.0
            this_step_size = these_bins[1] - these_bins[0]

            self.bins_list.append(these_bins)
            self.bin_centers_list.append(these_centers)
            self.step_size.append(this_step_size)

    def _num_bins_from_step_size(self):
        # if num_bins was not specified, derive from step_size
        self.num_bins = []
        # if provided just one step_size for 1D as a scalar, make it a list
        # for formatting purposes
        self.step_size = np.atleast_1d(self.step_size)
        if self.step_size.size != self.Ndims:
            raise ValueError("step_size must be None or a 1D iterable of length Ndims.")
        for ind in range(self.Ndims):
            if self.lower[ind] == self.upper[ind]:
                # min and max of limits are the same, i.e. data has to be exactly this
                these_bins = np.array([self.lower[ind], self.lower[ind]])
            else:
                these_bins = np.arange(self.lower[ind], self.upper[ind], self.step_size[ind])
            if these_bins[-1] != self.upper[ind]:
                these_bins = np.append(these_bins, self.upper[ind])
            these_centers = (these_bins[:-1] + these_bins[1:]) / 2.0
            this_num_bins = these_bins.size-1

            self.bins_list.append(these_bins)
            self.bin_centers_list.append(these_centers)
            self.num_bins.append(this_num_bins)

    def _create_bin_inds(self):
        # create the bin inds for each data point as a Nvals x Ndims long vector
        self.bin_inds = np.zeros((self.Nvals, self.Ndims))
        for ind in range(self.Ndims):
            this_min = self.bins_list[ind][0]
            this_step = self.step_size[ind]
            self.bin_inds[:, ind] = (self.coords_flat[:,ind] - this_min) / this_step
            # any that are outside the bin limits should be removed
            self.bin_inds[self.coords_flat[:, ind]< self.bins_list[ind][0],  ind] = np.nan
            self.bin_inds[self.coords_flat[:, ind]==self.bins_list[ind][-1], ind] = self.num_bins[ind]-1
            self.bin_inds[self.coords_flat[:, ind]> self.bins_list[ind][-1], ind] = np.nan

    def _calculate_bins(self):
        # this is a non-vector way of binning the data:
        # for ind in range(Nvals):
        #     this_bin_ind = bin_inds[ind,:]
        #     if not np.isnan(this_bin_ind).any():
        #         this_bin_ind = this_bin_ind.astype(int)
        #         binned_data[*this_bin_ind] = binned_data[*this_bin_ind] + data_flat[ind]
        #         binned_data_errs[*this_bin_ind] = binned_data_errs[*this_bin_ind] + errors_flat[ind]**2
        #         n_samples[*this_bin_ind] = n_samples[*this_bin_ind] + 1


        # and here is a vector equivalent
        # -------------------------------------------------------------
        # Inputs:
        #   bin_inds       : (Nvals, Ndims) array of indices, some rows may contain NaN
        #   data_flat      : (Nvals,)  values to accumulate
        #   errors_flat    : (Nvals,)  errors to accumulate (squared)
        #   binned_data    : Ndims-dimensional array (output)
        #   binned_data_errs : Ndims-dimensional array (output)
        #   n_samples      : Ndims-dimensional array (output)
        # -------------------------------------------------------------

        # 1. Identify valid rows (no NaNs)
        valid = ~np.isnan(self.bin_inds).any(axis=1)

        # 2. Convert valid bins to integer indices
        inds_int = self.bin_inds[valid].astype(int)

        # 3. Map multidimensional indices → flat indices
        flat_idx = np.ravel_multi_index(inds_int.T, dims=self.num_bins)

        # 4. Use bincount to accumulate in a vectorized way
        size = np.prod(self.num_bins)

        bd_sum = np.bincount(flat_idx, weights=self.data_flat[valid], minlength=size)
        err_sum = np.bincount(flat_idx, weights=self.errors_flat[valid]**2, minlength=size)
        ns_sum = np.bincount(flat_idx, minlength=size)

        # 5. Reshape and add into the original arrays
        self.binned_data = bd_sum.reshape(self.num_bins)
        self.binned_data_errs = err_sum.reshape(self.num_bins)
        self.n_samples = ns_sum.reshape(self.num_bins)

    def _calculate_fractional_bins(self):

        # more convenient to work with half shifted inds
        # bin_inds_frac for bin i is between i-0.5 and i+0.5 and the bin center
        # is at i.
        bin_inds_frac = self.bin_inds - 0.5

        # 1. Identify valid rows (no NaNs)
        valid = ~np.isnan(bin_inds_frac).any(axis=1)
        valid_inds = bin_inds_frac[valid]
        partial_weights = 1.-np.mod(valid_inds, 1)
        data_valid = self.data_flat[valid]
        errs_valid = self.errors_flat[valid]

        # In 1D, for a point at x between bin centers at x_i and x_{i+1},
        # wx_{i+1}=(x-x_i)/dx partial weight goes to bin i+1
        # and wx_i=1-w_{i+1} partial weight goes to bin i.
        # bin_inds = (x-(x_1-dx/2))/dx = (x-x_1)/dx+0.5. Therefore
        # bin_inds_frac = (x-x_1)/dx, so wx_{i+1} = mod(idx,1) and
        # wx_i = 1-mod(idx,1)

        # for each dimension, double the amount of subpoints
        for ind in range(self.Ndims):
            # bins on the edge only go in one bin on that axis
            edge_mask = np.logical_not(
                np.logical_or(valid_inds[:, ind]<0,
                              valid_inds[:, ind]>self.num_bins[ind]-1)
                              )
            partial_weights[~edge_mask, ind] = 1.0
            # will be where the bin goes
            arr_mod = valid_inds[edge_mask]
            arr_mod[:, ind] += 1.
            valid_inds = np.vstack([valid_inds, arr_mod])
            # how close it is to that bin
            arr_mod = partial_weights[edge_mask]
            arr_mod[:, ind] = 1. - arr_mod[:, ind]
            partial_weights = np.vstack([partial_weights, arr_mod])
            # the value and uncertainty
            data_valid = np.concatenate([data_valid, data_valid[edge_mask]])
            errs_valid = np.concatenate([errs_valid, errs_valid[edge_mask]])

        # any bins that ended up outside just get clamped
        for ind in range(self.Ndims):
            valid_inds[valid_inds[:, ind]<0, ind] = 0
            valid_inds[valid_inds[:, ind]>self.num_bins[ind]-1, ind] = self.num_bins[ind]-1

        # weights are the product of partial weights
        weights = np.prod(partial_weights, axis=1)

        # 2. Convert valid bins to integer indices
        inds_int = valid_inds.astype(int)

        # 3. Map multidimensional indices → flat indices
        flat_idx = np.ravel_multi_index(inds_int.T, dims=self.num_bins)

        # 4. Use bincount to accumulate in a vectorized way
        size = np.prod(self.num_bins)

        bd_sum = np.bincount(flat_idx, weights=weights*data_valid, minlength=size)
        err_sum = np.bincount(flat_idx, weights=(weights**2)*(errs_valid**2), minlength=size)
        ns_sum = np.bincount(flat_idx, weights=weights, minlength=size)

        # 5. Reshape and add into the original arrays
        self.binned_data = bd_sum.reshape(self.num_bins)
        self.binned_data_errs = err_sum.reshape(self.num_bins)
        self.n_samples = ns_sum.reshape(self.num_bins)

    def _norm_data(self):
        # normalize binned_data by the number of times sampled
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.normalize:
                self.binned_data = np.divide(self.binned_data, self.n_samples)
                self.binned_data_errs = np.divide(np.sqrt(self.binned_data_errs), self.n_samples)
            else:
                self.binned_data_errs = np.sqrt(self.binned_data_errs)

        # any bins with no samples is nan
        mask = self.n_samples == 0
        self.binned_data[mask] = np.nan
        self.binned_data_errs[mask] = np.nan
