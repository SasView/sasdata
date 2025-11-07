""" Dev docs: Demo to show the behaviour of the re-binning methods """

import matplotlib.pyplot as plt
import numpy as np

from sasdata.slicing.meshes.voronoi_mesh import voronoi_mesh
from sasdata.slicing.slicers.AnularSector import AnularSector

if __name__ == "__main__":
    q_range = 1.5
    demo1 = True
    demo2 = True

    # Demo of sums, annular sector over some not very circular data

    if demo1:

        x = (2 * q_range) * (np.random.random(400) - 0.5)
        y = (2 * q_range) * (np.random.random(400) - 0.5)

        display_mesh = voronoi_mesh(x, y)


        def lobe_test_function(x, y):
            return 1 + np.sin(x*np.pi/q_range)*np.sin(y*np.pi/q_range)


        random_lobe_data = lobe_test_function(x, y)

        plt.figure("Input Dataset 1")
        display_mesh.show_data(random_lobe_data, actually_show=False)

        data_order_0 = []
        data_order_neg1 = []

        sizes = np.linspace(0.1, 1, 100)

        for index, size in enumerate(sizes):
            q0 = 0.75 - 0.6*size
            q1 = 0.75 + 0.6*size
            phi0 = np.pi/2 - size
            phi1 = np.pi/2 + size

            rebinner = AnularSector(q0, q1, phi0, phi1)

            data_order_neg1.append(rebinner.sum(x, y, random_lobe_data, order=-1))
            data_order_0.append(rebinner.sum(x, y, random_lobe_data, order=0))

            if index % 10 == 0:
                plt.figure("Regions 1")
                rebinner.bin_mesh.show(actually_show=False)

        plt.title("Regions")

        plt.figure("Sum of region, dataset 1")

        plt.plot(sizes, data_order_neg1)
        plt.plot(sizes, data_order_0)

        plt.legend(["Order -1", "Order 0"])
        plt.title("Sum over region")


    # Demo of averaging, annular sector over ring shaped data

    if demo2:

        x, y = np.meshgrid(np.linspace(-q_range, q_range, 41), np.linspace(-q_range, q_range, 41))
        x = x.reshape(-1)
        y = y.reshape(-1)

        display_mesh = voronoi_mesh(x, y)


        def ring_test_function(x, y):
            r = np.sqrt(x**2 + y**2)
            return np.log(np.sinc(r*1.5)**2)


        grid_ring_data = ring_test_function(x, y)

        plt.figure("Input Dataset 2")
        display_mesh.show_data(grid_ring_data, actually_show=False)

        data_order_0 = []
        data_order_neg1 = []

        sizes = np.linspace(0.1, 1, 100)

        for index, size in enumerate(sizes):
            q0 = 0.25
            q1 = 1.25

            phi0 = np.pi/2 - size
            phi1 = np.pi/2 + size

            rebinner = AnularSector(q0, q1, phi0, phi1)

            data_order_neg1.append(rebinner.average(x, y, grid_ring_data, order=-1))
            data_order_0.append(rebinner.average(x, y, grid_ring_data, order=0))

            if index % 10 == 0:
                plt.figure("Regions 2")
                rebinner.bin_mesh.show(actually_show=False)

        plt.title("Regions")

        plt.figure("Average of region 2")

        plt.plot(sizes, data_order_neg1)
        plt.plot(sizes, data_order_0)

        plt.legend(["Order -1", "Order 0"])
        plt.title("Sum over region")

    plt.show()

