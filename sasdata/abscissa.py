from abc import ABC, abstractmethod

import numpy as np
from numpy._typing import ArrayLike

from quantities.quantity import Quantity
from exceptions import InterpretationError
from util import is_increasing


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
    def _determine_error_message(axis_arrays: list[np.ndarray], ordinate_shape: tuple):
        """ Error message for the `.determine` function"""

        shape_string = ", ".join([str(axis.shape) for axis in axis_arrays])

        return f"Cannot interpret array shapes axis: [{shape_string}], ordinate: {ordinate_shape}"

    @staticmethod
    def determine(axis_data: list[Quantity[ArrayLike]], ordinate_data: Quantity[ArrayLike]) -> "Abscissa":
        """ Get an Abscissa object that fits the combination of axes and data"""

        # Different posibilites:
        #   1: axes_data[i].shape == axes_data[j].shape == ordinate_data.shape
        #    1a: axis_data[i] is 1D =>
        #      1a-i:  len(axes_data) == 1 => Grid type or Scatter type depending on sortedness
        #      1a-ii: len(axes_data) != 1 => Scatter type
        #    1b: axis_data[i] dimensionality matches len(axis_data) => Meshgrid type
        #    1c: other => Error
        #   2: (len(axes_data[0]), len(axes_data[1])... ) == ordinate_data.shape => Grid type
        #   3: None of the above => Error

        ordinate_shape = np.array(ordinate_data.value).shape
        axis_arrays = [np.array(axis.value) for axis in axis_data]

        # 1:
        if all([axis.shape == ordinate_shape for axis in axis_arrays]):
            # 1a:
            if all([len(axis.shape)== 1 for axis in axis_arrays]):
                # 1a-i:
                if len(axis_arrays) == 1:
                    # Is it sorted
                    if is_increasing(axis_arrays[0]):
                        return GridAbscissa(axis_data)
                    else:
                        return ScatterAbscissa(axis_data)
                # 1a-ii
                else:
                    return ScatterAbscissa(axis_data)
            # 1b
            elif all([len(axis.shape) == len(axis_arrays) for axis in axis_arrays]):

                return MeshgridAbscissa(axis_data)

            else:
                raise InterpretationError(Abscissa._determine_error_message(axis_arrays, ordinate_shape))

        elif all([len(axis.shape)== 1 for axis in axis_arrays]) and \
                tuple([axis.shape[0] for axis in axis_arrays]) == ordinate_shape:

            # Require that they are sorted
            if all([is_increasing(axis) for axis in axis_arrays]):

                return GridAbscissa(axis_data)

            else:
                raise InterpretationError("Grid axes are not sorted")

        else:
            raise InterpretationError(Abscissa._determine_error_message(axis_arrays, ordinate_shape))

class GridAbscissa(Abscissa):

    @property
    def is_grid(self):
        return True

class MeshgridAbscissa(Abscissa):

    @property
    def is_grid(self):
        return True

class ScatterAbscissa(Abscissa):

    @property
    def is_grid(self):
        return False

