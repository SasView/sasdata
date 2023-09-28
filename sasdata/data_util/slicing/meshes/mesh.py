from typing import Sequence

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection

from sasdata.data_util.slicing.meshes.util import closed_loop_edges

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

        for cell in cells:

            this_cell_data = []

            for a, b in closed_loop_edges(cell):
                # make sure the representation is unique
                if a > b:
                    this_cell_data.append(edge_lookup[(a, b)])
                else:
                    this_cell_data.append(edge_lookup[(b, a)])

            self.cells_to_edges.append(this_cell_data)

        # Counts for elements
        self.n_points = self.points.shape[0]
        self.n_edges = len(self.edges)
        self.n_cells = len(self.cells)

        # Areas
        self._areas = None

    def find_locations(self, points):
        """ Find indices of cells containing the input points """

        


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
            self._areas = np.ndarray(areas)

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

    def show_data(self, data: np.ndarray, cmap='winter', mesh_color='white', show_mesh=True, actually_show=True):
        """ Show with data """

        colormap = cm.get_cmap(cmap, 256)

        cmin = np.min(data)
        cmax = np.max(data)

        color_index_map = np.array(255 * (data - cmin) / (cmax - cmin), dtype=int)

        for cell, color_index in zip(self.cells, color_index_map):

            color = colormap(color_index)

            plt.fill(self.points[cell, 0], self.points[cell, 1], color=color)

        if show_mesh:
            self.show(actually_show=False, color=mesh_color)

        if actually_show:
            self.show()