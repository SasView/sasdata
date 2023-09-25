"""
Meshes used in testing along with some expected values
"""

import numpy as np

from sasdata.data_util.slicing.meshes.voronoi_mesh import voronoi_mesh
from sasdata.data_util.slicing.meshes.mesh import Mesh
from sasdata.data_util.slicing.meshes.meshmerge import meshmerge

coords = np.arange(-4, 5)
grid_mesh = voronoi_mesh(*np.meshgrid(coords, coords))


item_1 = np.array([
    [-3.5, -0.5],
    [-0.5,  3.5],
    [ 0.5,  3.5],
    [ 3.5, -0.5],
    [ 0.0,  1.5]], dtype=float)

item_2 = np.array([
    [-1.0, -2.0],
    [-2.0, -2.0],
    [-2.0, -1.0],
    [-1.0, -1.0]], dtype=float)

mesh_points = np.concatenate((item_1, item_2), axis=0)
cells = [[0,1,2,3,4],[5,6,7,8]]

shape_mesh = Mesh(mesh_points, cells)

# Subset of the mappings that meshmerge should include
# This can be read off the plots generated below
expected_shape_mappings = [
    (98, -1),
    (99, -1),
    (12, 0),
    (1, -1),
    (148, 1),
    (149, 1),
    (110, 1),
    (144, -1),
    (123, -1)]


expected_grid_mappings = [
    (89, 1),
    (146, 29),
    (66, 34),
    (112, 45)
]


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    combined_mesh, _, _ = meshmerge(grid_mesh, shape_mesh)

    plt.figure()
    combined_mesh.show(actually_show=False, show_labels=True, color='k')
    grid_mesh.show(actually_show=False, show_labels=True, color='r')

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    plt.figure()
    combined_mesh.show(actually_show=False, show_labels=True, color='k')
    shape_mesh.show(actually_show=False, show_labels=True, color='r')

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

    plt.show()
