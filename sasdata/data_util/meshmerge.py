from typing import Sequence
from scipy.spatial import Delaunay

import numpy as np

from dataclasses import dataclass

@dataclass
class Mesh:
    points: np.ndarray
    edges: Sequence[Sequence[int]] # List of pairs of points forming edges
    cells: Sequence[Sequence[int]] # List of edges constituting a cell


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

    new_points = []
    for edge_a in mesh_a.edges:
        for edge_b in mesh_b.edges:
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

            p1 = mesh_a.points[edge_a[0]]
            p2 = mesh_a.points[edge_a[1]]
            p3 = mesh_b.points[edge_b[0]]
            p4 = mesh_b.points[edge_b[1]]

            m = np.array([
                [p2[0] - p1[0], p3[0] - p4[0]],
                [p2[1] - p1[1], p3[1] - p4[1]]])

            v = np.array([p3[0] - p1[0], p3[1] - p1[1]])

            if np.linalg.det(m) == 0:
                # Lines don't intersect
                break

            st = np.linalg.solve(m, v)

            # As the purpose of this is finding new points for the merged mesh, we don't
            # want new points if they are right at the end of the lines, hence non strict
            # inequalities here
            if np.any(st <= 0) or np.any(st >= 1):
                # Exclude intection points, that are not on the *segments*
                break

            x = p1[0] + (p2[0] - p1[1])*st[0]
            y = p1[1] + (p2[1] - p1[1])*st[1]

            new_points.append((x, y))

    # Build list of all input points, in a way that we can check for coincident points



    # Remove coincident points


    # Triangulate based on these intersections

    # Find centroids of all output triangles, and find which source cells they belong to

    ## Assign -1 to all cells
    ## Find centroids
    ## Check whether within bounding box
    ## If in bounding box, check cell properly using winding number, if inside, assign
