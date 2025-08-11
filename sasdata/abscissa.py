from abc import ABC, abstractmethod

import numpy as np
from numpy._typing import ArrayLike

from quantities.quantity import Quantity


class Abscissa(ABC):

    def __init__(self, axes: list[Quantity]):
        self._axes = axes
        self._dimensionality = len(axes)
    @property
    def dimensionality(self) -> int:
        """ Dimensionality of this data """
        return self._dimensionality

    @property
    @abstractmethod
    def is_grid(self) -> bool:
        """ Are these coordinates using a grid representation
        ( as opposed to a general list representation)

        is_grid = True: implies that the corresponding ordinate is n-dimensional tensor
        is_grid = False: implies that the corresponding ordinate is a 1D list

        If the data is one dimensional, is_grid=True

        """


    @property
    def axes(self) -> list[Quantity]:
        """ Axes of the data:

        If it's an (n1-by-n2-by-n3...) grid (is_grid=True): give the values for each axis, returning a list like
          [Quantity(length n1), Quantity(length n2), Quantity(length n3) ... ]

        If it is not grid data (is_grid=False), but n points on a general mesh, give one array for each dimension
          [Quantity(length n), Quantity(length n), Quantity(length n) ... ]

        """

        return self._axes

    @staticmethod
    def determine(axes_data: list[Quantity[ArrayLike]], ordinate_data: Quantity[ArrayLike]) -> "Abscissa":
        """ Get an Abscissa object that fits the combination of axes and data"""

        # Different possibilites:
        #   1: axes_data[i].shape == axes_data[j].shape == ordinate_data.shape
        #    1a: axis_data[i] is 1D =>
        #      1a-i:  len(axes_data) == 1 => Grid type (trivially)
        #      1a-ii: len(axes_data) != 1 => Mesh type
        #    1b: axis_data[i] dimensionality matches len(axis_data) => Grid type
        #    1c: other => Error
        #   2: (len(axes_data[0]), len(axes_data[1])... ) == ordinate_data.shape => Grid type
        #   3: None of the above => Error

        ordinate_shape = np.array(ordinate_data.value).shape

        if len(ordinate_shape) == 1 and all([np.array(axis.value).shape == ordinate_shape for axis in axes_data]):
            return ScatterAbscissa(axes_data)

        elif
class GridAbscissa(Abscissa):

    @property
    def is_grid(self):
        return True

class ScatterAbscissa(Abscissa):

    @property
    def is_grid(self):
        return False

