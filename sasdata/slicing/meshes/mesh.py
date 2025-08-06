from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection

from sasdata.slicing.meshes.util import closed_loop_edges


class Mesh:
    def __init__(self,
                 points: np.ndarray,
                 cells: Sequence[Sequence[int]]):

        """
        Object representing a mesh.

        Parameters are the values:
            mesh points
            map from edge to points
            map from cells to edges

        it is done this way to ensure a non-redundant representation of cells and edges,
        however there are no checks for the topology of the mesh, this is assumed to be done by
        whatever creates it. There are also no checks for ordering of cells.

        :param points: points in 2D forming vertices of the mesh
        :param cells: ordered lists of indices of points forming each cell (face)

        """

        self.points = points
        self.cells = cells

        # Get edges

        edges = set()
        for cell_index, cell in enumerate(cells):

            for a, b in closed_loop_edges(cell):
                # make sure the representation is unique
                if a > b:
                    edges.add((a, b))
                else:
                    edges.add((b, a))

        self.edges = list(edges)

        # Associate edges with faces

        edge_lookup = {edge: i for i, edge in enumerate(self.edges)}
        self.cells_to_edges = []
        self.cells_to_edges_signs = []

        for cell in cells:

            this_cell_data = []
            this_sign_data = []

            for a, b in closed_loop_edges(cell):
                # make sure the representation is unique
                if a > b:
                    this_cell_data.append(edge_lookup[(a, b)])
                    this_sign_data.append(1)
                else:
                    this_cell_data.append(edge_lookup[(b, a)])
                    this_sign_data.append(-1)

            self.cells_to_edges.append(this_cell_data)
            self.cells_to_edges_signs.append(this_sign_data)

        # Counts for elements
        self.n_points = self.points.shape[0]
        self.n_edges = len(self.edges)
        self.n_cells = len(self.cells)

        # Areas
        self._areas = None


    @property
    def areas(self):
        """ Areas of cells """

        if self._areas is None:
            # Calculate areas
            areas = []
            for cell in self.cells:
                # Use triangle shoelace formula, basically calculate the
                # determinant based on of triangles with one point at 0,0
                a_times_2 = 0.0
                for i1, i2 in closed_loop_edges(cell):
                    p1 = self.points[i1, :]
                    p2 = self.points[i2, :]
                    a_times_2 += p1[0]*p2[1] - p1[1]*p2[0]

                areas.append(0.5*np.abs(a_times_2))

            # Save in cache
            self._areas = np.array(areas)

        # Return cache
        return self._areas


    def show(self, actually_show=True, show_labels=False, **kwargs):
        """ Show on a plot """
        ax = plt.gca()
        segments = [[self.points[edge[0]], self.points[edge[1]]] for edge in self.edges]
        line_collection = LineCollection(segments=segments, **kwargs)
        ax.add_collection(line_collection)

        if show_labels:
            text_color = kwargs["color"] if "color" in kwargs else 'k'
            for i, cell in enumerate(self.cells):
                xy = np.sum(self.points[cell, :], axis=0)/len(cell)
                ax.text(xy[0], xy[1], str(i), horizontalalignment="center", verticalalignment="center", color=text_color)

        x_limits = [np.min(self.points[:,0]), np.max(self.points[:,0])]
        y_limits = [np.min(self.points[:,1]), np.max(self.points[:,1])]

        plt.xlim(x_limits)
        plt.ylim(y_limits)

        if actually_show:
            plt.show()

    def locate_points(self, x: np.ndarray, y: np.ndarray):
        """ Find the cells that contain the specified points"""

        x = x.reshape(-1)
        y = y.reshape(-1)

        # The most simple implementation is not particularly fast, especially in python
        #
        # Less obvious, but hopefully faster strategy
        #
        # Ultimately, checking the inclusion of a point within a polygon
        # requires checking the crossings of a half line with the polygon's
        # edges.
        #
        # A fairly efficient thing to do is to check every edge for crossing
        # the axis parallel lines x=point_x.
        # Then these edges that cross can map back to the polygons they're in
        # and a final check for inclusion can be done with the edge sign property
        # and some explicit checking of the
        #
        # Basic idea is:
        #  1) build a matrix for each point-edge pair
        #     True if the edge crosses the half-line above a point
        #  2) for each cell get the winding number by evaluating the
        #     sum of the component edges, weighted 1/-1 according to direction


        edges = np.array(self.edges)

        edge_xy_1 = self.points[edges[:, 0], :]
        edge_xy_2 = self.points[edges[:, 1], :]

        edge_x_1 = edge_xy_1[:, 0]
        edge_x_2 = edge_xy_2[:, 0]



        # Make an n_edges-by-n_inputs boolean matrix that indicates which of the
        # edges cross x=points_x line
        crossers = np.logical_xor(
                        edge_x_1.reshape(-1, 1) < x.reshape(1, -1),
                        edge_x_2.reshape(-1, 1) < x.reshape(1, -1))

        # Calculate the gradients, some might be infs, but none that matter will be
        # TODO: Disable warnings
        gradients = (edge_xy_2[:, 1] - edge_xy_1[:, 1]) / (edge_xy_2[:, 0] - edge_xy_1[:, 0])

        # Distance to crossing points edge 0
        delta_x = x.reshape(1, -1) - edge_x_1.reshape(-1, 1)

        # Signed distance from point to y (doesn't really matter which sign)
        delta_y = gradients.reshape(-1, 1) * delta_x + edge_xy_1[:, 1:] - y.reshape(1, -1)

        score_matrix = np.logical_and(delta_y > 0, crossers)

        output = -np.ones(len(x), dtype=int)
        for cell_index, (cell_edges, sign) in enumerate(zip(self.cells_to_edges, self.cells_to_edges_signs)):
            cell_score = np.sum(score_matrix[cell_edges, :] * np.array(sign).reshape(-1, 1), axis=0)
            points_in_cell = np.abs(cell_score) == 1
            output[points_in_cell] = cell_index

        return output

    def show_data(self,
                  data: np.ndarray,
                  cmap='winter',
                  mesh_color='white',
                  show_mesh=False,
                  actually_show=True,
                  density=False):

        """ Show with data """

        colormap = cm.get_cmap(cmap, 256)

        data = data.reshape(-1)

        if density:
            data = data / self.areas

        cmin = np.min(data)
        cmax = np.max(data)

        color_index_map = np.array(255 * (data - cmin) / (cmax - cmin), dtype=int)

        for cell, color_index in zip(self.cells, color_index_map):

            color = colormap(color_index)

            plt.fill(self.points[cell, 0], self.points[cell, 1], color=color, edgecolor=None)

        if show_mesh:
            self.show(actually_show=False, color=mesh_color)

        if actually_show:
            self.show()


if __name__ == "__main__":
    from test.slicers.meshes_for_testing import location_test_mesh, location_test_points_x, location_test_points_y

    cell_indices = location_test_mesh.locate_points(location_test_points_x, location_test_points_y)

    print(cell_indices)

    for i in range(location_test_mesh.n_cells):
        inds = cell_indices == i
        plt.scatter(
            location_test_points_x.reshape(-1)[inds],
            location_test_points_y.reshape(-1)[inds])

    location_test_mesh.show()
