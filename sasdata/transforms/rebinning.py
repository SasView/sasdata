""" Algorithms for interpolation and rebinning """
from typing import TypeVar

import numpy as np
from numpy._typing import ArrayLike

from sasdata.quantities.quantity import Quantity
from scipy.sparse import coo_matrix

from enum import Enum

class InterpolationOptions(Enum):
    NEAREST_NEIGHBOUR = 0
    LINEAR = 1



def calculate_interpolation_matrix_1d(input_axis: Quantity[ArrayLike],
                                      output_axis: Quantity[ArrayLike],
                                      mask: ArrayLike | None = None,
                                      order: InterpolationOptions = InterpolationOptions.NEAREST_NEIGHBOUR,
                                      is_density=False):

    # We want the input values in terms of the output units, will implicitly check compatability

    working_units = output_axis.units

    input_x = input_axis.in_units_of(working_units)
    output_x = output_axis.in_units_of(working_units)

    # Get the array indices that will map the array to a sorted one
    input_sort = np.argsort(input_x)
    output_sort = np.argsort(output_x)

    input_unsort = np.arange(len(output_x), dtype=int)[input_sort]
    output_unsort = np.arange(len(input_x), dtype=int)[output_sort]

    sorted_in = input_x[input_sort]
    sorted_out = output_x[output_sort]

    match order:
        case InterpolationOptions.NEAREST_NEIGHBOUR:

            # COO Sparse matrix definition data
            i_entries = []
            j_entries = []

            crossing_points = 0.5*(sorted_out[1:] + sorted_out[:-1])

            # Find the output values nearest to each of the input values
            n_i = len(sorted_in)
            n_j = len(sorted_out)
            i=0
            for k, crossing_point in enumerate(crossing_points):
                while i < n_i and sorted_in[i] < crossing_point:
                    i_entries.append(i)
                    j_entries.append(k)
                    i += 1

            # All the rest in the last bin
            while i < n_i:
                i_entries.append(i)
                j_entries.append(n_j-1)
                i += 1

            i_entries = input_unsort[np.array(i_entries, dtype=int)]
            j_entries = output_unsort[np.array(j_entries, dtype=int)]
            values = np.ones_like(i_entries, dtype=float)

            return coo_matrix((values, (i_entries, j_entries)), shape=(n_i, n_j))


        case _:
            raise ValueError(f"Unsupported interpolation order: {order}")

def calculate_interpolation_matrix(input_axes: list[Quantity[ArrayLike]],
                                   output_axes: list[Quantity[ArrayLike]],
                                   data: ArrayLike | None = None,
                                   mask: ArrayLike | None = None):

    pass



def rebin(data: Quantity[ArrayLike],
          axes: list[Quantity[ArrayLike]],
          new_axes: list[Quantity[ArrayLike]],
          mask: ArrayLike | None = None,
          interpolation_order: int = 1):

    """ This algorithm is only for operations that preserve dimensionality,
    i.e. non-projective rebinning.
    """

    pass