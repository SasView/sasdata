import numpy as np

from sasdata.data_util.slicing.meshes.mesh import Mesh
from sasdata.data_util.slicing.meshes.delaunay_mesh import delaunay_mesh
from sasdata.data_util.slicing.meshes.util import closed_loop_edges


import time

def meshmerge(mesh_a: Mesh, mesh_b: Mesh) -> tuple[Mesh, np.ndarray, np.ndarray]:
    """ Take two lists of polygons and find their intersections

    Polygons in each of the input variables should not overlap i.e. a point in space should be assignable to
    at most one polygon in mesh_a and at most one polygon in mesh_b

    Mesh topology should be sensible, otherwise bad things might happen, also, the cells of the input meshes
    must be in order (which is assumed by the mesh class constructor anyway).

    :returns:
        1) A triangulated mesh based on both sets of polygons together
        2) The indices of the mesh_a polygon that corresponds to each triangle, -1 for nothing
        3) The indices of the mesh_b polygon that corresponds to each triangle, -1 for nothing

    """

    t0 = time.time()

    # Find intersections of all edges in mesh one with edges in mesh two
    # TODO: Speed this up

    # Fastest way might just be to calculate the intersections of all lines on edges,
    # see whether we need filtering afterwards

    edges_a = np.array(mesh_a.edges, dtype=int)
    edges_b = np.array(mesh_b.edges, dtype=int)

    edge_a_1 = mesh_a.points[edges_a[:, 0], :]
    edge_a_2 = mesh_a.points[edges_a[:, 1], :]
    edge_b_1 = mesh_b.points[edges_b[:, 0], :]
    edge_b_2 = mesh_b.points[edges_b[:, 1], :]

    a_grid, b_grid = np.mgrid[0:mesh_a.n_edges, 0:mesh_b.n_edges]
    a_grid = a_grid.reshape(-1)
    b_grid = b_grid.reshape(-1)

    p1 = edge_a_1[a_grid, :]
    p2 = edge_a_2[a_grid, :]
    p3 = edge_b_1[b_grid, :]
    p4 = edge_b_2[b_grid, :]

    #
    # Solve the equations
    #
    #    z_a1 + s delta_z_a = z_b1 + t delta_z_b
    #
    # for z = (x, y)
    #

    start_point_diff = p1 - p3

    delta1 = p2 - p1
    delta3 = p4 - p3

    deltas = np.concatenate(([-delta1], [delta3]), axis=0)
    deltas = np.moveaxis(deltas, 0, 2)

    st = np.linalg.solve(deltas, start_point_diff)

    # Find the points where s and t are in (0, 1)

    intersection_inds = np.logical_and(
        np.logical_and(0 < st[:, 0], st[:, 0] < 1),
        np.logical_and(0 < st[:, 1], st[:, 1] < 1))

    start_points_for_intersections = p1[intersection_inds, :]
    deltas_for_intersections = delta1[intersection_inds, :]

    points_to_add = start_points_for_intersections + st[intersection_inds, 0].reshape(-1,1) * deltas_for_intersections

    t1 = time.time()
    print("Edge intersections:", t1 - t0)

    # Build list of all input points, in a way that we can check for coincident points


    points = np.concatenate((
                mesh_a.points,
                mesh_b.points,
                points_to_add
                ))


    # Remove coincident points

    points = np.unique(points, axis=0)

    # Triangulate based on these intersections

    output_mesh = delaunay_mesh(points[:, 0], points[:, 1])


    t2 = time.time()
    print("Delaunay:", t2 - t1)


    # Find centroids of all output triangles, and find which source cells they belong to

    ## step 1) Assign -1 to all cells of original meshes
    assignments_a = -np.ones(output_mesh.n_cells, dtype=int)
    assignments_b = -np.ones(output_mesh.n_cells, dtype=int)

    ## step 2) Find centroids of triangulated mesh (just needs to be a point inside, but this is a good one)
    centroids = []
    for cell in output_mesh.cells:
        centroid = np.sum(output_mesh.points[cell, :]/3, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    t3 = time.time()
    print("Centroids:", t3 - t2)


    ## step 3) Perform checks based on winding number method (see wikipedia Point in Polygon).
    #
    # # TODO: Brute force search is sllllloooooooowwwwww - keeping track of which points are where would be better
    # for mesh, assignments in [
    #         (mesh_a, assignments_a),
    #         (mesh_b, assignments_b)]:
    #
    #     for centroid_index, centroid in enumerate(centroids):
    #         for cell_index, cell in enumerate(mesh.cells):
    #
    #             # Bounding box check
    #             points = mesh.points[cell, :]
    #             if np.any(centroid < np.min(points, axis=0)): # x or y less than any in polygon
    #                 continue
    #
    #             if np.any(centroid > np.max(points, axis=0)): # x or y greater than any in polygon
    #                 continue
    #
    #             # Winding number check - count directional crossings of vertical half line from centroid
    #             winding_number = 0
    #             for i1, i2 in closed_loop_edges(cell):
    #                 p1 = mesh.points[i1, :]
    #                 p2 = mesh.points[i2, :]
    #
    #                 # if the section xs do not straddle the x=centroid_x coordinate, then the
    #                 # edge cannot cross the half line.
    #                 # If it does, then remember which way it was
    #                 # * Careful about ends
    #                 # * Also, note that the p1[0] == p2[0] -> (no contribution) case is covered by the strict inequality
    #                 if p1[0] > centroid[0] >= p2[0]:
    #                     left_right = -1
    #                 elif p2[0] > centroid[0] >= p1[0]:
    #                     left_right = 1
    #                 else:
    #                     continue
    #
    #                 # Find the y point that it crosses x=centroid at
    #                 # note: denominator cannot be zero because of strict inequality above
    #                 gradient = (p2[1] - p1[1]) / (p2[0] - p1[0])
    #                 x_delta = centroid[0] - p1[0]
    #                 y = p1[1] + x_delta * gradient
    #
    #                 if y > centroid[1]:
    #                     winding_number += left_right
    #
    #
    #             if abs(winding_number) > 0:
    #                 # Do assignment of input cell to output triangle index
    #                 assignments[centroid_index] = cell_index
    #                 break # point is assigned
    #
    #         # end cell loop
    #
    #     # end centroid loop

    assignments_a = mesh_a.locate_points(centroids[:, 0], centroids[:, 1])
    assignments_b = mesh_b.locate_points(centroids[:, 0], centroids[:, 1])

    t4 = time.time()
    print("Assignments:", t4 - t3)

    return output_mesh, assignments_a, assignments_b


def main():
    from voronoi_mesh import voronoi_mesh

    n1 = 100
    n2 = 100

    m1 = voronoi_mesh(np.random.random(n1), np.random.random(n1))
    m2 = voronoi_mesh(np.random.random(n2), np.random.random(n2))


    mesh, assignement1, assignement2 = meshmerge(m1, m2)

    mesh.show()


if __name__ == "__main__":
    main()