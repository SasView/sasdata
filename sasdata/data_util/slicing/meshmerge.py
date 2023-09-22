from typing import Sequence
from scipy.spatial import Delaunay

import numpy as np

from dataclasses import dataclass

from sasdata.data_util.slicing.mesh import Mesh

import matplotlib.pyplot as plt

def meshmerge(mesh_a: Mesh, mesh_b: Mesh) -> tuple[Mesh, np.ndarray, np.ndarray]:
    """ Take two lists of polygons and find their intersections

    Polygons in each of the input variables should not overlap i.e. a point in space should be assignable to
    at most one polygon in mesh_a and at most one polygon in mesh_b

    Mesh topology should be sensible, otherwise bad things might happen

    :returns:
        1) A triangulated mesh based on both sets of polygons together
        2) The indices of the mesh_a polygon that corresponds to each triangle, -1 for nothing
        3) The indices of the mesh_b polygon that corresponds to each triangle, -1 for nothing

    """

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



    # Build list of all input points, in a way that we can check for coincident points

    # plt.scatter(mesh_a.points[:,0], mesh_a.points[:,1])
    # plt.scatter(mesh_b.points[:,0], mesh_b.points[:,1])
    # plt.scatter(new_x, new_y)
    #
    # mesh_a.show(False)
    # mesh_b.show(False, color=(.8, .5, 0))
    #
    # plt.xlim([0,1])
    # plt.ylim([0,1])
    #
    # plt.show()

    points = np.concatenate((
                mesh_a.points,
                mesh_b.points,
                np.array((new_x, new_y)).T
                ))

    # plt.scatter(points[:,0], points[:,1])
    # plt.show()

    # Remove coincident points

    points = np.unique(points, axis=0)

    # Triangulate based on these intersections

    # Find centroids of all output triangles, and find which source cells they belong to

    ## Assign -1 to all cells
    ## Find centroids - they're just the closed voronoi cells?
    ## Check whether within bounding box
    ## If in bounding box, check cell properly using winding number, if inside, assign


def simple_intersection():
    mesh_a = Mesh(
                np.array([[0, 0.5],[1,0.5]], dtype=float),
                [[0, 1]], [])

    mesh_b = Mesh(
        np.array([[0.5, 0], [0.5, 1]], dtype=float),
        [[0, 1]], [])

    meshmerge(mesh_a, mesh_b)



def simple_intersection_2():
    mesh_a = Mesh(
                np.array([[4,3],[1,3]], dtype=float),
                [[0, 1]], [])

    mesh_b = Mesh(
        np.array([[3, 4], [3, 1]], dtype=float),
        [[0, 1]], [])

    meshmerge(mesh_a, mesh_b)
def main():
    from voronoi_mesh import voronoi_mesh

    n1 = 100
    n2 = 100

    m1 = voronoi_mesh(np.random.random(n1), np.random.random(n1))
    m2 = voronoi_mesh(np.random.random(n2), np.random.random(n2))


    meshmerge(m1, m2)

if __name__ == "__main__":
    main()
    # simple_intersection()