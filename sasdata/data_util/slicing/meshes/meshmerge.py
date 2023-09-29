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

    new_x = []
    new_y = []
    for edge_a in mesh_a.edges:
        for edge_b in mesh_b.edges:

            p1 = mesh_a.points[edge_a[0]]
            p2 = mesh_a.points[edge_a[1]]
            p3 = mesh_b.points[edge_b[0]]
            p4 = mesh_b.points[edge_b[1]]

            # Bounding box check

            # First edge entirely to left of other
            if max((p1[0], p2[0])) < min((p3[0], p4[0])):
                continue

            # First edge entirely below other
            if max((p1[1], p2[1])) < min((p3[1], p4[1])):
                continue

            # First edge entirely to right of other
            if min((p1[0], p2[0])) > max((p3[0], p4[0])):
                continue

            # First edge entirely above other
            if min((p1[1], p2[1])) > max((p3[1], p4[1])):
                continue

            #
            # Parametric description of intersection in terms of position along lines
            #
            # Simultaneous eqns (to reflect current wiki notation)
            # s(x2 - x1) - t(x4 - x3) = x3 - x1
            # s(y2 - y1) - t(y4 - y3) = y3 - y1
            #
            # in matrix form:
            # m.(s,t) = v
            #


            m = np.array([
                [p2[0] - p1[0], p3[0] - p4[0]],
                [p2[1] - p1[1], p3[1] - p4[1]]])

            v = np.array([p3[0] - p1[0], p3[1] - p1[1]])

            if np.linalg.det(m) == 0:
                # Lines don't intersect, or are colinear in a way that doesn't matter
                continue

            st = np.linalg.solve(m, v)

            # As the purpose of this is finding new points for the merged mesh, we don't
            # want new points if they are right at the end of the lines, hence non-strict
            # inequalities here
            if np.any(st <= 0) or np.any(st >= 1):
                # Exclude intection points, that are not on the *segments*
                continue

            x = p1[0] + (p2[0] - p1[0])*st[0]
            y = p1[1] + (p2[1] - p1[1])*st[0]

            new_x.append(x)
            new_y.append(y)


    t1 = time.time()
    print("Edge intersections:", t1 - t0)

    # Build list of all input points, in a way that we can check for coincident points


    points = np.concatenate((
                mesh_a.points,
                mesh_b.points,
                np.array((new_x, new_y)).T
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
