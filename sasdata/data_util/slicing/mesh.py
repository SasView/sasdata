from typing import Sequence

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Mesh:
    def __init__(self, points: np.ndarray, edges: Sequence[Sequence[int]], cells: Sequence[Sequence[int]]):
        self.points = points
        self.edges = edges
        self.cells = cells

        self._cells_to_points = None


    def show(self, actually_show=True, **kwargs):

        ax = plt.gca()
        segments = [[self.points[edge[0]], self.points[edge[1]]] for edge in self.edges]
        line_collection = LineCollection(segments=segments, **kwargs)
        ax.add_collection(line_collection)

        if actually_show:
            plt.show()

    def show_data(self, data: np.ndarray):
        raise NotImplementedError("Show data not implemented")