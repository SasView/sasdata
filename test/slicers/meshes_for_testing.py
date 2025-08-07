"""
Meshes used in testing along with some expected values
"""

import numpy as np

from sasdata.slicing.meshes.mesh import Mesh
from sasdata.slicing.meshes.meshmerge import meshmerge
from sasdata.slicing.meshes.voronoi_mesh import voronoi_mesh

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

#
# Mesh location tests
#

location_test_mesh_points = np.array([
    [0, 0],  # 0
    [0, 1],  # 1
    [0, 2],  # 2
    [1, 0],  # 3
    [1, 1],  # 4
    [1, 2],  # 5
    [2, 0],  # 6
    [2, 1],  # 7
    [2, 2]], dtype=float)

location_test_mesh_cells = [
    [0, 1, 4, 3],
    [1, 2, 5, 4],
    [3, 4, 7, 6],
    [4, 5, 8, 7]]

location_test_mesh = Mesh(location_test_mesh_points, location_test_mesh_cells)

test_coords = 0.25 + 0.5*np.arange(4)
location_test_points_x, location_test_points_y = np.meshgrid(test_coords, test_coords)

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

    plt.figure()
    location_test_mesh.show(actually_show=False, show_labels=True)
    plt.scatter(location_test_points_x, location_test_points_y)

    plt.show()
