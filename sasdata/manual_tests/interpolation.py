import matplotlib.pyplot as plt
import numpy as np

from sasdata.quantities import units
from sasdata.quantities.plotting import quantity_plot
from sasdata.quantities.quantity import NamedQuantity
from sasdata.transforms.rebinning import InterpolationOptions, calculate_interpolation_matrix_1d


def linear_interpolation_check():

    for from_bins in [(-10, 10, 10),
                      (-10, 10, 1000),
                      (-15, 5, 10),
                      (15,5, 10)]:
        for to_bins in [
            (-15, 0, 10),
            (-15, 15, 10),
            (0, 20, 100)]:

            plt.figure()

            x = NamedQuantity("x", np.linspace(*from_bins), units=units.meters)
            y = x**2

            quantity_plot(x, y)

            new_x = NamedQuantity("x_new", np.linspace(*to_bins), units=units.meters)

            rebin_mat = calculate_interpolation_matrix_1d(x, new_x, order=InterpolationOptions.LINEAR)

            new_y = y @ rebin_mat

            quantity_plot(new_x, new_y)

            print(new_y.history.summary())

    plt.show()




linear_interpolation_check()