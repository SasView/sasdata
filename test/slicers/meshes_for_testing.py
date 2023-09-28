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
    (100, -1),
    (152, -1),
    (141, -1),
    (172, -1),
    (170, -1),
    (0, -1),
    (1, -1),
    (8, 0),
    (9, 0),
    (37, 0),
    (83, 0),
    (190, 1),
    (186, 1),
    (189, 1),
    (193, 1)
]

expected_grid_mappings = [
    (89, 0),
    (90, 1),
    (148, 16),
    (175, 35),
    (60, 47),
    (44, 47),
    (80, 60)
]



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    combined_mesh, _, _ = meshmerge(grid_mesh, shape_mesh)

    plt.figure()
    combined_mesh.show(actually_show=False, show_labels=True, color='k')
    grid_mesh.show(actually_show=False, show_labels=True, color='r')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.figure()
    combined_mesh.show(actually_show=False, show_labels=True, color='k')
    shape_mesh.show(actually_show=False, show_labels=True, color='r')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.show()
