import time

import numpy as np
from matplotlib import pyplot as plt

from sasdata.transforms.NDrebin import NDrebin


def test_1D_exact(show_plots: bool):
    # Parameters for the Gaussian function
    mu = 0.0          # Mean
    sigma = 1.0       # Standard deviation

    # fiducial
    xreal = np.linspace(-5, 5, 11)
    Ireal = np.exp(-((xreal - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    # rebin to the exact same bins
    Ibin, qbin, *rest = NDrebin(Ireal, xreal,
                                lower=-5.5, upper=5.5, num_bins=11)

    assert all(Ibin == Ireal)
    assert all(qbin[0] == xreal)

    # Plot
    if show_plots:
        plt.figure()
        plt.plot(qbin[0], Ibin, 'o', linewidth=2, label='bin')
        plt.plot(xreal, Ireal, 'k-', linewidth=2, label='exact')

        plt.xlabel('x')
        plt.ylabel('I')
        plt.legend()
        plt.tight_layout()
        plt.show()


    # rebin to the exact same bins with fractional
    Ibin, qbin, *rest = NDrebin(Ireal, xreal, lower=-5.5, upper=5.5, num_bins=11, fractional=True)

    assert all(Ibin == Ireal)

    # Plot
    if show_plots:
        plt.figure()
        plt.plot(qbin[0], Ibin, 'o', linewidth=2, label='fractional bin')
        plt.plot(xreal, Ireal, 'k-', linewidth=2, label='exact')

        plt.xlabel('x')
        plt.ylabel('I')
        plt.legend()
        plt.tight_layout()
        plt.show()


def test_syntax():
    # test syntax
    Ndims = 4
    Nvals = int(1e4)
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
    Nvals = int(1e4)
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


# test ND gaussian
def test_ND():
    Ndims = 4
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
    start = time.perf_counter()
    Ibin, qbin, *rest = NDrebin(I_ND, qmat,
        step_size=0.2*np.random.rand(Ndims)+0.1,
        lower=[1,2,3,0],
        upper=[9,8,7,9.5]
    )
    end = time.perf_counter()
    print(f"Computed {Nvals} points in {end - start:.6f} seconds")

    start = time.perf_counter()
    Ibin, qbin, *rest = NDrebin(I_ND, qmat,
        step_size=0.2*np.random.rand(Ndims)+0.1,
        lower=[1,2,3,0],
        upper=[9,8,7,9.5],
        fractional=True
    )
    end = time.perf_counter()
    print(f"Computed {Nvals} points with fractional binning in {end - start:.6f} seconds")



if __name__ == "__main__":
    test_1D_exact(show_plots=False)
    test_syntax()
    test_ND()
