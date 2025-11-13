
import numpy as np
from numpy._typing import ArrayLike

from typing import List, Optional, Sequence

from sasdata.quantities.quantity import Quantity



def NDrebin(data: Quantity[ArrayLike],
            coords: Quantity[ArrayLike],
            axes: list[Quantity[ArrayLike]],
            limits: Optional[List[Sequence[float]]] = None,
            step_size: Optional[List[Sequence[float]]] = None,
            num_bins: Optional[List[Sequence[int]]] = None
        ):
    """
    I provide values at points with coordinates
    The points may not be in a nice grid

    I want to rebin that data into a regular grid

    :data: The data at each point
    :coords: The locations of each data point, same size of data
        plus one more dimension with the same length as the
        dimensionality of the space (Ndim)
    :axes: The axes of the coordinate system we are binning
        into. Defaults to diagonal (e.g. (1,0,0), (0,1,0), and 
        (0,0,1) for 3D data). A list of Ndim element vectors
    :limits: The limits along each axis. Defaults to the smallest
        and largest values in the data if no limits are provided.
        A list of 2 element vectors.
    :step_size: The size of steps along each axis. Supercedes
        num_bins. A list of length Ndim.
    :num_bins: The number of bins along each axis. Superceded by
        step_size if step_size is provided. At least one of step_size
        or num_bins must be provided.
    """

    # Identify number of points
    Nvals = data.size

    # Identify number of dimensions
    Ndims = coords.size / Nvals

    # if Ndims is not an integer value we have a problem
    if not Ndims.is_integer():
        raise ValueError(f"The coords have to have the same shape as "
                         "the data, plus one more dimension which is "
                         "length Ndims")

    # flatten input data to 1D of length Nvals
    data_flat = data.reshape(-1)

    # check if the first axis of coords is the dimensions axis
    if coords.shape[0] == Ndims:
        # first check if it is the first axis
        dim_axis = 0
    elif coords.shape[-1] == Ndims:
        # now check if it is the last axis
        dim_axis = -1
    else:
        # search if any axis is size Ndims
        dim_axis = next(i for i, s in enumerate(coords.shape) if s == Ndims)
    
    if not coords.shape[dim_axis] == Ndims:
        raise ValueError(f"The coords have to have one dimension which is "
                         "the dimensionality of the space")

    # flatten coords to size Nvals x Ndims
    moved = np.moveaxis(coords, dim_axis, 0)
    coords_flat = moved.reshape(Ndims, -1).T


    # if axes are not provided, default to identity
    if axes is None:
        axes = np.eye(Ndims)


    # now project the data into the axes
    axes_inv = np.linalg.inv(axes)
    coords_flat = np.tensordot(coords_flat, axes_inv, axes=([1], [0]))


    # if limits were not provided, default to the min and max 
    # coord in each dimension
    if limits is None:
        limits = np.zeros((2,Ndims))
        for ind in range(Ndims):
            limits[0,ind] = np.min(coords_flat[:,ind])
            limits[1,ind] = np.max(coords_flat[:,ind])

    # clean up limits
    for ind in range(Ndims):
        # if individual limits are nan, inf, none, etc, replace with min/max
        if not np.isfinite(limits[0,ind]):
            limits[0,ind] = np.min(coords_flat[:,ind])
        if not np.isfinite(limits[1,ind]):
            limits[1,ind] = np.max(coords_flat[:,ind])
        # if any of the limits are in the wrong order, flip them
        if limits[0,ind] > limits[1,ind]:
            temp = limits[0,ind]
            limits[0,ind] = limits[1,ind]
            limits[1,ind] = temp


    # bins_list is a Ndims long list of vectors which are the edges of
    #   each bin. Each vector is num_bins[i]+1 long
    bins_list = []

    # bin_centers_list is a Ndims long list of vectors which are the centers of
    #   each bin. Each vector is num_bins[i] long
    bin_centers_list = []

    # create the bins in each dimension
    if step_size is None:
        # if step_size was not specified, derive from num_bins
        step_size = []
        for ind in range(Ndims):
            these_bins = np.linspace(limits[0,ind], limits[1,ind], num_bins[ind]+1)
            these_centers = (these_bins[:-1] + these_bins[1:]) / 2.0
            this_step_size = these_bins[1] - these_bins[0]

            bins_list.append(these_bins)
            bin_centers_list.append(these_centers)
            step_size.append(this_step_size)
    else:
        # else use step_size and derive num_bins
        num_bins = []
        for ind in range(Ndims):
            these_bins = np.arange(limits[0,ind], limits[1,ind], step_size[ind])
            if these_bins[-1] != limits[1,ind]:
                these_bins = np.append(these_bins, limits[1,ind])
            these_centers = (these_bins[:-1] + these_bins[1:]) / 2.0
            this_num_bins = these_bins.size-1

            bins_list.append(these_bins)
            bin_centers_list.append(these_centers)
            num_bins.append(this_num_bins)


    
    # create the bin inds for each data point as a Nvals x Ndims long vector
    bin_inds = np.zeros((Nvals, Ndims))
    for ind in range(Ndims):
            this_min = bins_list[ind][0]
            this_step = step_size[ind]
            bin_inds[:, ind] = np.floor((coords_flat[:,ind] - this_min) / this_step)
            # any that are outside the bin limits should be removed
            bin_inds[bin_inds<0, ind] = np.nan
            bin_inds[bin_inds>num_bins[ind], ind] = np.nan


    # create the binned data matrix of size num_bins[0] x num_bins[1] x ...
    binned_data = np.zeros(num_bins)
    binned_data_errs = np.zeros(num_bins)
    errors_flat = 0*data_flat   # TODO add error propagation
    n_samples = np.zeros(num_bins)

    # now bin the data!
    for ind in range(Nvals):
        binned_data[bin_inds[ind,:]] = binned_data[bin_inds[ind,:]] + data_flat[ind]
        binned_data_errs[bin_inds[ind,:]] = binned_data_errs[bin_inds[ind,:]] + errors_flat[ind]**2
        n_samples[bin_inds[ind,:]] = n_samples[bin_inds[ind,:]] + 1
       

    # normalize binned_data by the number of times sampled
    binned_data = np.divide(binned_data, n_samples)
    binned_data_errs = np.divide(np.sqrt(binned_data_errs), n_samples)

    # any bins with no samples is nan
    binned_data[n_samples==0] = np.nan
    binned_data_errs[n_samples==0] = np.nan


    return binned_data, bin_centers_list
