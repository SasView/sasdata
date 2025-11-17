
import time
from collections.abc import Sequence

import numpy as np
from numpy._typing import ArrayLike

from sasdata.quantities.quantity import Quantity


def NDrebin(data: Quantity[ArrayLike],
            coords: Quantity[ArrayLike],
            data_errs: Quantity[ArrayLike] | None = None,
            axes: list[Quantity[ArrayLike]] | None = None,
            upper: list[Sequence[float]] | None = None,
            lower: list[Sequence[float]] | None = None,
            step_size: list[Sequence[float]] | None = None,
            num_bins: list[Sequence[int]] | None = None,
            fractional: bool | None = False
        ):
    """
    Provide values at points with ND coordinates.
    The coordinates may not be in a nice grid.
    The data can be in any array shape.

    The coordinates are in the same shape plus one dimension,
        preferably the first dimension, which is the ND coordinate
        position

    Rebin that data into a regular grid.

    Note that this does lose some information from the underlying data,
        as you are essentially averaging multiple measurements into one bin.

    :data: The data at each point
    :coords: The locations of each data point, same size of data
        plus one more dimension with the same length as the
        dimensionality of the space (Ndim)
    :data_errs: Optional, the same size as data, the uncertainties on data
    :axes: The axes of the coordinate system we are binning
        into. Defaults to diagonal (e.g. (1,0,0), (0,1,0), and 
        (0,0,1) for 3D data). A list of Ndim element vectors
    :upper: The upper limits along each axis. Defaults to the largest
        values in the data if no limits are provided.
        A 1D list of Ndims values.
    :lower: The lower limits along each axis. Defaults to the smallest
        values in the data if no limits are provided.
        A 1D list of Ndims values.
    :step_size: The size of steps along each axis. Supercedes
        num_bins. A list of length Ndim.
    :num_bins: The number of bins along each axis. Superceded by
        step_size if step_size is provided. At least one of step_size
        or num_bins must be provided.
    :fractional: Whether to perform fractional binning or not. Defaults 
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


    Returns: binned_data, bin_centers_list
    :binned_data: has size num_bins and is NDimensional, contains
        the binned data
    :bin_centers_list: is a list of 1D vectors, contains the
        axes of the binned data. The coordinates of bin [i,j,k]
        is given by 
        bin_centers_list[0][i]*axes[i]+bin_centers_list[1][j]*axes[j]+
        bin_centers_list[0][k]*axes[k]
    :binned_data_errs: has size num_bins and is NDimensional, contains
        the propagated errors of the binned_data
    :bins_list: is a list of 1D vectors, is similar to bin_centers_list,
        but instead contains the edges of the bins, so it is 1 longer
        in each dimension
    :step_size: is a list of Ndims numbers, contains the step size
        along each dimensino
    :num_bins: is a list of Ndims numbers, contains the number
        of bins along each dimension


    An example call might be:
    # test syntax 1
    Ndims = 4
    Nvals = int(1e5)
    qmat = np.random.rand(Ndims, Nvals)
    Imat = np.random.rand(Nvals)

    Ibin, qbin, *rest = NDrebin(Imat, qmat,
        step_size=0.1*np.random.rand(Ndims)+0.05,
        lower=0.1*np.random.rand(Ndims)+0.0,
        upper=0.1*np.random.rand(Ndims)+0.9
    )
    results = NDrebin(Imat, qmat,
        step_size=0.1*np.random.rand(Ndims)+0.05,
        lower=0.1*np.random.rand(Ndims)+0.0,
        upper=0.1*np.random.rand(Ndims)+0.9
    )
    Ibin = results[0]
    qbin = results[1]
    bins_list = results[2]
    step_size = results[3]
    num_bins = results[4]

    
    # test syntax 2
    Ndims = 2
    Nvals = int(1e5)
    qmat = np.random.rand(Ndims, 100, Nvals)
    Imat = np.random.rand(100, Nvals)
    Imat_errs = np.random.rand(100, Nvals)

    binned_data, bin_centers_list, binned_data_errs, bins_list, step_size, num_bins \
    = NDrebin(Imat, qmat,
        data_errs = Imat_errs,
        num_bins=[10,20],
        axes = np.eye(2),
        fractional=True
    )

    """

    # Identify number of points
    Nvals = data.size

    # Identify number of dimensions
    Ndims = coords.size / Nvals

    # if Ndims is not an integer value we have a problem
    if not Ndims.is_integer():
        raise ValueError("The coords have to have the same shape as "
                         "the data, plus one more dimension which is "
                         "length Ndims")
    Ndims = int(Ndims)

    # flatten input data to 1D of length Nvals
    data_flat = data.reshape(-1)
    if data_errs is None:
        no_errs = True
        errors_flat = 0*data_flat   # no errors
    else:
        no_errs = False
        errors_flat = data_errs.reshape(-1)

    if errors_flat.shape != data_flat.shape:
        ValueError("Data and errors have to have the same shape.")

    # if 1D, need to add a size 1 dimension index to coords
    if Ndims == 1:
        coords = coords.reshape(-1, 1)

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
        raise ValueError("The coords have to have one dimension which is "
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
    if upper is None:
        upper = np.zeros(Ndims)
        for ind in range(Ndims):
            upper[ind] = np.max(coords_flat[:,ind])
    if lower is None:
        lower = np.zeros(Ndims)
        for ind in range(Ndims):
            lower[ind] = np.min(coords_flat[:,ind])

    # if provided just one limit for 1D as a scalar, make it a list
    # for formatting purposes
    lower = np.atleast_1d(lower)
    upper = np.atleast_1d(upper)
    # if not isinstance(lower, (list, tuple)):
    #     lower = [lower]
    # print(lower)
    # if not isinstance(upper, (list, tuple)):
    #     upper = [upper]

    # clean up limits
    if lower.size != Ndims:
        ValueError("Lower limits must be None or a 1D iterable of length Ndims.")
    if upper.size != Ndims:
        ValueError("Upper limits must be None or a 1D iterable of length Ndims.")
    for ind in range(Ndims):
        # if individual limits are nan, inf, none, etc, replace with min/max
        if not np.isfinite(lower[ind]):
            lower[ind] = np.min(coords_flat[:,ind])
        if not np.isfinite(upper[ind]):
            upper[ind] = np.max(coords_flat[:,ind])
        # if any of the limits are in the wrong order, flip them
        if lower[ind] > upper[ind]:
            temp = lower[ind]
            lower[ind] = upper[ind]
            upper[ind] = temp


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
        # if provided just one num_bin for 1D as a scalar, make it a list
        # for formatting purposes
        # if not isinstance(num_bins, (list, tuple)):
        #     num_bins = [num_bins]
        num_bins = np.atleast_1d(num_bins)
        if num_bins.size != Ndims:
            ValueError("num_bins must be None or a 1D iterable of length Ndims.")
        for ind in range(Ndims):
            these_bins = np.linspace(lower[ind], upper[ind], num_bins[ind]+1)
            these_centers = (these_bins[:-1] + these_bins[1:]) / 2.0
            this_step_size = these_bins[1] - these_bins[0]

            bins_list.append(these_bins)
            bin_centers_list.append(these_centers)
            step_size.append(this_step_size)
    else:
        # else use step_size and derive num_bins
        num_bins = []
        # if provided just one step_size for 1D as a scalar, make it a list
        # for formatting purposes
        # if not isinstance(step_size, (list, tuple)):
        #     step_size = [step_size]
        step_size = np.atleast_1d(step_size)
        if step_size.size != Ndims:
            ValueError("step_size must be None or a 1D iterable of length Ndims.")
        for ind in range(Ndims):
            if lower[ind] == upper[ind]:
                # min and max of limits are the same, i.e. data has to be exactly this
                these_bins = np.array([lower[ind], lower[ind]])
            else:
                these_bins = np.arange(lower[ind], upper[ind], step_size[ind])
            if these_bins[-1] != upper[ind]:
                these_bins = np.append(these_bins, upper[ind])
            these_centers = (these_bins[:-1] + these_bins[1:]) / 2.0
            this_num_bins = these_bins.size-1

            bins_list.append(these_bins)
            bin_centers_list.append(these_centers)
            num_bins.append(this_num_bins)


    # create the binned data matrix of size num_bins[0] x num_bins[1] x ...
    # binned_data = np.zeros(num_bins)
    # binned_data_errs = np.zeros(num_bins)
    # n_samples = np.zeros(num_bins)

    # create the bin inds for each data point as a Nvals x Ndims long vector
    bin_inds = np.zeros((Nvals, Ndims))
    for ind in range(Ndims):
        this_min = bins_list[ind][0]
        this_step = step_size[ind]
        bin_inds[:, ind] = (coords_flat[:,ind] - this_min) / this_step
        # any that are outside the bin limits should be removed
        bin_inds[coords_flat[:, ind]<bins_list[ind][0], ind] = np.nan
        bin_inds[coords_flat[:, ind]==bins_list[ind][-1], ind] = num_bins[ind]-1
        bin_inds[coords_flat[:, ind]>bins_list[ind][-1], ind] = np.nan

    if not fractional:
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
        valid = ~np.isnan(bin_inds).any(axis=1)

        # 2. Convert valid bins to integer indices
        inds_int = bin_inds[valid].astype(int)

        # 3. Map multidimensional indices → flat indices
        flat_idx = np.ravel_multi_index(inds_int.T, dims=num_bins)

        # 4. Use bincount to accumulate in a vectorized way
        size = np.prod(num_bins)

        bd_sum = np.bincount(flat_idx, weights=data_flat[valid], minlength=size)
        err_sum = np.bincount(flat_idx, weights=errors_flat[valid]**2, minlength=size)
        ns_sum = np.bincount(flat_idx, minlength=size)

        # 5. Reshape and add into the original arrays
        binned_data = bd_sum.reshape(num_bins)
        binned_data_errs = err_sum.reshape(num_bins)
        n_samples = ns_sum.reshape(num_bins)


    else:
        # more convenient to work with half shifted inds
        bin_inds = bin_inds - 0.5

        # 1. Identify valid rows (no NaNs)
        valid = ~np.isnan(bin_inds).any(axis=1)
        valid_inds = bin_inds[valid]
        partial_weights = 1.-np.mod(valid_inds, 1)


        # for each dimension, double the amount of subpoints
        # for a point at x_i, 1-x_i goes to
        for ind in range(Ndims):
            # will be where the bin goes
            arr_mod = valid_inds.copy()
            arr_mod[:, ind] += 1.
            valid_inds = np.vstack([valid_inds, arr_mod])
            # how close it is to that bin
            arr_mod = partial_weights.copy()
            arr_mod[:, ind] = 1. - arr_mod[:, ind]
            partial_weights = np.vstack([partial_weights, arr_mod])


        # any bins that ended up outside just get clamped
        for ind in range(Ndims):
            valid_inds[valid_inds[:, ind]<0, ind] = 0
            valid_inds[valid_inds[:, ind]>num_bins[ind]-1, ind] = num_bins[ind]-1

        # weights are the product of partial weights
        weights = np.prod(partial_weights, axis=1)

        # need to tile the data and errs to weight them for each bin
        data_valid = np.tile(data_flat[valid], 2**Ndims)
        errs_valid = np.tile(errors_flat[valid], 2**Ndims)

        # 2. Convert valid bins to integer indices
        inds_int = valid_inds.astype(int)

        # 3. Map multidimensional indices → flat indices
        flat_idx = np.ravel_multi_index(inds_int.T, dims=num_bins)

        # 4. Use bincount to accumulate in a vectorized way
        size = np.prod(num_bins)

        bd_sum = np.bincount(flat_idx, weights=weights*data_valid, minlength=size)
        err_sum = np.bincount(flat_idx, weights=(weights**2)*(errs_valid**2), minlength=size)
        ns_sum = np.bincount(flat_idx, weights=weights, minlength=size)

        # 5. Reshape and add into the original arrays
        binned_data = bd_sum.reshape(num_bins)
        binned_data_errs = err_sum.reshape(num_bins)
        n_samples = ns_sum.reshape(num_bins)




    # normalize binned_data by the number of times sampled
    with np.errstate(divide='ignore', invalid='ignore'):
        binned_data = np.divide(binned_data, n_samples)
        binned_data_errs = np.divide(np.sqrt(binned_data_errs), n_samples)

    # any bins with no samples is nan
    binned_data[n_samples==0] = np.nan
    binned_data_errs[n_samples==0] = np.nan


    if no_errs:
        return binned_data, bin_centers_list, bins_list, step_size, num_bins
    else:
        return binned_data, bin_centers_list, binned_data_errs, bins_list, step_size, num_bins









if __name__ == "__main__":
    # a bunch of local testing
    import matplotlib.pyplot as plt
    import numpy as np

    if True:
        # Parameters for the Gaussian function
        mu = 0.0          # Mean
        sigma = 1.0       # Standard deviation
        noise = 0.1
        Nvals = int(1e6)

        # Generate Nvals random values between -5 and 5
        x = -5 + (5 + 5) * np.random.rand(Nvals)   # shape: (Nvals,)

        # Build qmat equivalent to MATLAB:
        # qmat = [x(:)'; 0*x(:)'; 0*x(:)']
        # → 3 x Nvals array
        qmat = np.vstack([
            x,
            np.zeros_like(x),
            np.zeros_like(x)
        ])

        # Evaluate the Gaussian function
        I_1 = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

        # Add uniform noise in [-noise, +noise]
        I_1 = I_1 - noise + 2 * noise * np.random.rand(Nvals)

        # Fiducial
        xreal = np.linspace(-5, 5, 1001)
        Ireal = np.exp(-((xreal - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

        # NDrebin
        import time
        start = time.perf_counter()
        Ibin, qbin, *rest = NDrebin(I_1, qmat,
            step_size=[0.1,np.inf,np.inf]
        )
        end = time.perf_counter()
        print(f"Computed {Nvals} points in {end - start:.6f} seconds")

        start = time.perf_counter()
        Ibin2, qbin2, *rest = NDrebin(I_1, qmat,
            step_size=[0.1,np.inf,np.inf],
            fractional = True
        )
        end = time.perf_counter()
        print(f"Computed {Nvals} points with fractional binning in {end - start:.6f} seconds")

        # Plot
        plt.figure()
        plt.plot(np.squeeze(qbin[0]), np.squeeze(Ibin), 'o', linewidth=2, label='sum')
        plt.plot(np.squeeze(qbin2[0]), np.squeeze(Ibin2), 'o', linewidth=2, label='fractional')
        plt.plot(xreal, Ireal, 'k-', linewidth=2, label='analytic')

        plt.xlabel('x')
        plt.ylabel('I')
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)





    if True:
        # Parameters of the 2D Gaussian
        mu = np.array([0.15, 0.0, 0.0])                 # Mean vector
        sigma = np.array([0.015, 0.055, 0.05])              # Std dev in x and y
        noise = 0.
        SDD = 2.7
        k_0 = 2*np.pi/5
        pix_x = 1./128.
        pix_y = 1./128.

        # Generate our 2D detector grid
        x = np.arange(-64,64)*pix_x
        y = np.arange(-64,64)*pix_y

        [xmesh, ymesh] = np.meshgrid(x, y)

        # calculate qx, qy, qz
        qx = k_0*xmesh/np.sqrt(xmesh**2+ymesh**2+SDD**2)
        qy = k_0*ymesh/np.sqrt(xmesh**2+ymesh**2+SDD**2)
        qz = k_0-k_0*SDD/np.sqrt(xmesh**2+ymesh**2+SDD**2)

        # qmat
        qmat0 = np.stack([qx,qy,qz], axis=2)

        # now rotate about y
        angle_list = np.pi/180*np.linspace(-15,15,int(30/.25))
        qmat = np.zeros((len(x), len(y), len(angle_list), 3))
        for ind in range(len(angle_list)):
            new_qmat = np.copy(qmat0)
            new_qmat[:,:,0] = np.cos(angle_list[ind])*qmat0[:,:,0] - \
                np.sin(angle_list[ind])*qmat0[:,:,2]
            new_qmat[:,:,2] = np.sin(angle_list[ind])*qmat0[:,:,0] + \
                np.cos(angle_list[ind])*qmat0[:,:,2]
            qmat[:,:,ind,:] = qmat0


        # Evaluate Gaussian:
        #     G(x,y) = (1/(2πσxσy)) * exp(-[(x-μx)^2/(2σx^2) + (y-μy)^2/(2σy^2)])
        I_2D = (
            np.exp(
                -((qmat[:,:,:,0] - mu[0])**2) / (2 * sigma[0]**2)
                -((qmat[:,:,:,1] - mu[1])**2) / (2 * sigma[1]**2)
                -((qmat[:,:,:,2] - mu[2])**2) / (2 * sigma[2]**2)
            ) /
            (2 * np.pi * sigma[0] * sigma[1] * sigma[2])
        )

        # Add uniform noise
        I_2D = I_2D - noise + 2 * noise * np.random.rand(*I_2D.shape)

        # Rebin in 2D.
        # You can choose finite steps for both x and y depending on how you want bins defined.
        import time
        start = time.perf_counter()
        Ibin, qbin, *rest = NDrebin(I_2D, qmat,
            step_size=[0.006, 0.006, np.inf]
        )
        end = time.perf_counter()
        print(f"Computed {qmat.size/3} points in {end - start:.6f} seconds")

        start = time.perf_counter()
        Ibin2, qbin2, *rest = NDrebin(I_2D, qmat,
            step_size=[0.0035, 0.0035, np.inf],
            fractional=True
        )
        end = time.perf_counter()
        print(f"Computed {qmat.size/3} points with fractional binning in {end - start:.6f} seconds")


        # Fiducial 2D
        [xmesh, ymesh] = np.meshgrid(qbin2[0], qbin2[1])
        Ireal = (
            np.exp(
                -((xmesh - mu[0])**2) / (2 * sigma[0]**2)
                -((ymesh - mu[1])**2) / (2 * sigma[1]**2)
            ) /
            (2 * np.pi * sigma[0] * sigma[1])
        )

        # Plot a 1D slice of the binned data along x (y bins aggregated)
        plt.figure(figsize=(4,8))

        plt.subplot(3, 1, 1)
        plt.pcolormesh(qbin[0], qbin[1], np.squeeze(Ibin.T), shading='nearest')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('sum')
        plt.colorbar()
        plt.tight_layout()

        plt.subplot(3, 1, 2)
        plt.pcolormesh(qbin2[0], qbin2[1], np.squeeze(Ibin2.T), shading='nearest')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('sum')
        plt.colorbar()
        plt.tight_layout()

        plt.subplot(3, 1, 3)
        plt.pcolormesh(xmesh, ymesh, Ireal, shading='nearest')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('real')
        plt.colorbar()
        plt.tight_layout()
        plt.show(block=False)







    # test ND gaussian
    Ndims = 3
    mu = np.zeros(Ndims)                 # Mean vector
    sigma = np.random.rand(Ndims)              # Std dev in x and y
    noise = 0.1
    Nvals = int(1e6)

    # Generate random points (x, y) in a 2D square
    qmat = -5 + 10 * np.random.rand(Ndims, Nvals)

    # Evaluate 2D Gaussian:
    #     G(x,y) = (1/(2πσxσy)) * exp(-[(x-μx)^2/(2σx^2) + (y-μy)^2/(2σy^2)])
    exp_op = np.zeros(Nvals)
    sigma_tot = 1
    for ind in range(Ndims):
        exp_op = exp_op -((qmat[ind,:] - mu[ind])**2) / (2 * sigma[ind]**2)
        sigma_tot = sigma_tot * sigma[ind]
    I_ND = (
        np.exp(exp_op) /
        (2 * np.pi * sigma_tot)
    )

    # Add uniform noise
    I_ND = I_ND - noise + 2 * noise * np.random.rand(1,Nvals)

    # Rebin in 2D.
    # You can choose finite steps for both x and y depending on how you want bins defined.
    import time
    start = time.perf_counter()
    Ibin, qbin, *rest = NDrebin(I_ND, qmat,
        step_size=0.2*np.random.rand(Ndims)+0.1,
        lower=[1,2,3],
        upper=[9,8,7]
    )
    end = time.perf_counter()
    print(f"Computed {Nvals} points in {end - start:.6f} seconds")

    start = time.perf_counter()
    Ibin, qbin, *rest = NDrebin(I_ND, qmat,
        step_size=0.2*np.random.rand(Ndims)+0.1,
        lower=[1,2,3],
        upper=[9,8,7],
        fractional=True
    )
    end = time.perf_counter()
    print(f"Computed {Nvals} points with fractional binning in {end - start:.6f} seconds")



    # test syntax
    Ndims = 4
    Nvals = int(1e5)
    qmat = np.random.rand(Ndims, Nvals)
    Imat = np.random.rand(Nvals)

    Ibin, qbin, *rest = NDrebin(Imat, qmat,
        step_size=0.1*np.random.rand(Ndims)+0.05,
        lower=0.1*np.random.rand(Ndims)+0.0,
        upper=0.1*np.random.rand(Ndims)+0.9
    )
    results = NDrebin(Imat, qmat,
        step_size=0.1*np.random.rand(Ndims)+0.05,
        lower=0.1*np.random.rand(Ndims)+0.0,
        upper=0.1*np.random.rand(Ndims)+0.9
    )
    Ibin = results[0]
    qbin = results[1]
    bins_list = results[2]
    step_size = results[3]
    num_bins = results[4]

    # test syntax
    Ndims = 2
    Nvals = int(1e5)
    qmat = np.random.rand(Ndims, 100, Nvals)
    Imat = np.random.rand(100, Nvals)
    Imat_errs = np.random.rand(100, Nvals)

    binned_data, bin_centers_list, binned_data_errs, bins_list, step_size, num_bins \
    = NDrebin(Imat, qmat,
        data_errs = Imat_errs,
        num_bins=[10,20],
        axes = np.eye(2),
        fractional=True
    )

    input()
