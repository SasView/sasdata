import time

import numpy as np
from matplotlib import pyplot as plt

from sasdata.transforms.NDrebin import NDRebin


def test_1D_exact(show_plots: bool):
    # Parameters for the Gaussian function
    mu = 0.0          # Mean
    sigma = 1.0       # Standard deviation

    # fiducial
    xreal = np.linspace(-5, 5, 11)
    Ireal = np.exp(-((xreal - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    # rebin to the exact same bins
    rebin = NDRebin(Ireal, xreal,
                    lower=-5.5, upper=5.5, num_bins=11)
    rebin.run()
    Ibin = rebin.binned_data
    qbin = rebin.bin_centers_list

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
    rebin = NDRebin(Ireal, xreal,
                    lower=-5.5, upper=5.5, num_bins=11, fractional=True)
    rebin.run()
    Ibin = rebin.binned_data
    qbin = rebin.bin_centers_list

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



def test_2D(show_plots: bool):
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
    start = time.perf_counter()
    rebin = NDRebin(I_2D, qmat,
        step_size=[0.006, 0.006, np.inf])
    rebin.run()
    Ibin = rebin.binned_data
    qbin = rebin.bin_centers_list
    end = time.perf_counter()
    print(f"Computed {qmat.size/3} points in {end - start:.6f} seconds")

    start = time.perf_counter()
    rebin = NDRebin(I_2D, qmat,
        step_size=[0.0035, 0.0035, np.inf],
        fractional=True)
    rebin.run()
    Ibin2 = rebin.binned_data
    qbin2 = rebin.bin_centers_list
    end = time.perf_counter()
    print(f"Computed {qmat.size/3} points with fractional binning in {end - start:.6f} seconds")


    if show_plots:
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
        plt.show()


def test_syntax():
    # test syntax
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


    # test syntax
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
    rebin = NDRebin(I_ND, qmat,
        step_size=0.2*np.random.rand(Ndims)+0.1,
        lower=[1,2,3,0],
        upper=[9,8,7,9.5]
    )
    rebin.run()
    Ibin = rebin.binned_data
    qbin = rebin.bin_centers_list
    end = time.perf_counter()
    print(f"Computed {Nvals} points in {end - start:.6f} seconds")

    start = time.perf_counter()
    rebin = NDRebin(I_ND, qmat,
        step_size=0.2*np.random.rand(Ndims)+0.1,
        lower=[1,2,3,0],
        upper=[9,8,7,9.5],
        fractional=True
    )
    rebin.run()
    Ibin = rebin.binned_data
    qbin = rebin.bin_centers_list
    end = time.perf_counter()
    print(f"Computed {Nvals} points with fractional binning in {end - start:.6f} seconds")



if __name__ == "__main__":
    test_1D_exact(show_plots=True)
    test_2D(show_plots=True)
    test_syntax()
    test_ND()
