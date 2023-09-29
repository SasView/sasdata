""" Dev docs: Demo to show the behaviour of the re-binning methods """

import numpy as np

import matplotlib.pyplot as plt

from sasdata.data_util.slicing.slicers.AnularSector import AnularSector
from sasdata.data_util.slicing.meshes.mesh import Mesh
from sasdata.data_util.slicing.meshes.voronoi_mesh import voronoi_mesh



if __name__ == "__main__":
    q_range = 1.5


    x = (2*q_range)*(np.random.random(400)-0.5)
    y = (2*q_range)*(np.random.random(400)-0.5)

    display_mesh = voronoi_mesh(x, y)

    # Demo of sums, annular sector over some not very circular data


    def lobe_test_function(x, y):
        return 1 + np.sin(x*np.pi/q_range)*np.sin(y*np.pi/q_range)


    random_lobe_data = lobe_test_function(x, y)

    plt.figure("Input Dataset 1")
    display_mesh.show_data(random_lobe_data, actually_show=False)

    data_order_0 = []

    sizes = np.linspace(0.1, 1, 100)

    for index, size in enumerate(sizes):
        q0 = 0.75 - 0.6*size
        q1 = 0.75 + 0.6*size
        phi0 = np.pi/2 - size
        phi1 = np.pi/2 + size

        rebinner = AnularSector(q0, q1, phi0, phi1, order=0)

        data_order_0.append(rebinner.sum(x, y, random_lobe_data))

        if index % 10 == 0:
            plt.figure("Regions")
            rebinner.bin_mesh.show(actually_show=False)

    plt.figure("Data")

    plt.plot(sizes, data_order_0)

    plt.show()


    # Demo of averaging, annular sector over ring shaped data