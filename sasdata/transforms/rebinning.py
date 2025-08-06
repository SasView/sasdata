""" Algorithms for interpolation and rebinning """

from enum import Enum

import numpy as np
from numpy._typing import ArrayLike
from scipy.sparse import coo_matrix

from sasdata.quantities.quantity import Quantity


class InterpolationOptions(Enum):
    NEAREST_NEIGHBOUR = 0
    LINEAR = 1
    CUBIC = 3

class InterpolationError(Exception):
    """ We probably want to raise exceptions because interpolation is not appropriate/well-defined,
    not the same as numerical issues that will raise ValueErrors"""


def calculate_interpolation_matrix_1d(input_axis: Quantity[ArrayLike],
                                      output_axis: Quantity[ArrayLike],
                                      mask: ArrayLike | None = None,
                                      order: InterpolationOptions = InterpolationOptions.LINEAR,
                                      is_density=False):

    """ Calculate the matrix that converts values recorded at points specified by input_axis to
    values recorded at points specified by output_axis"""

    # We want the input values in terms of the output units, will implicitly check compatability
    # TODO: incorporate mask

    working_units = output_axis.units

    input_x = input_axis.in_units_of(working_units)
    output_x = output_axis.in_units_of(working_units)

    # Get the array indices that will map the array to a sorted one
    input_sort = np.argsort(input_x)
    output_sort = np.argsort(output_x)

    input_unsort = np.arange(len(input_x), dtype=int)[input_sort]
    output_unsort = np.arange(len(output_x), dtype=int)[output_sort]

    sorted_in = input_x[input_sort]
    sorted_out = output_x[output_sort]

    n_in = len(sorted_in)
    n_out = len(sorted_out)

    conversion_matrix = None # output

    match order:
        case InterpolationOptions.NEAREST_NEIGHBOUR:

            # COO Sparse matrix definition data
            i_entries = []
            j_entries = []

            crossing_points = 0.5*(sorted_out[1:] + sorted_out[:-1])

            # Find the output values nearest to each of the input values
            i=0
            for k, crossing_point in enumerate(crossing_points):
                while i < n_in and sorted_in[i] < crossing_point:
                    i_entries.append(i)
                    j_entries.append(k)
                    i += 1

            # All the rest in the last bin
            while i < n_in:
                i_entries.append(i)
                j_entries.append(n_out-1)
                i += 1

            i_entries = input_unsort[np.array(i_entries, dtype=int)]
            j_entries = output_unsort[np.array(j_entries, dtype=int)]
            values = np.ones_like(i_entries, dtype=float)

            conversion_matrix = coo_matrix((values, (i_entries, j_entries)), shape=(n_in, n_out))

        case InterpolationOptions.LINEAR:

            # Leverage existing linear interpolation methods to get the mapping
            # do a linear interpolation on indices
            #   the floor should give the left bin
            #   the ceil should give the right bin
            #   the fractional part should give the relative weightings

            input_indices = np.arange(n_in, dtype=int)
            output_indices = np.arange(n_out, dtype=int)

            fractional = np.interp(x=sorted_out, xp=sorted_in, fp=input_indices, left=0, right=n_in-1)

            left_bins = np.floor(fractional).astype(int)
            right_bins = np.ceil(fractional).astype(int)

            right_weight = fractional % 1
            left_weight = 1 - right_weight

            # There *should* be no repeated entries for both i and j in the main part, but maybe at the ends
            # If left bin is the same as right bin, then we only want one entry, and the weight should be 1

            same = left_bins == right_bins
            not_same = ~same

            same_bins = left_bins[same] # could equally be right bins, they're the same

            same_indices = output_indices[same]
            not_same_indices = output_indices[not_same]

            j_entries_sorted = np.concatenate((same_indices, not_same_indices, not_same_indices))
            i_entries_sorted = np.concatenate((same_bins, left_bins[not_same], right_bins[not_same]))

            i_entries = input_unsort[i_entries_sorted]
            j_entries = output_unsort[j_entries_sorted]

            # weights don't need to be unsorted # TODO: check this is right, it should become obvious if we use unsorted data
            weights = np.concatenate((np.ones_like(same_bins, dtype=float), left_weight[not_same], right_weight[not_same]))

            conversion_matrix = coo_matrix((weights, (i_entries, j_entries)), shape=(n_in, n_out))

        case InterpolationOptions.CUBIC:
            # Cubic interpolation, much harder to implement because we can't just cheat and use numpy

            input_indices = np.arange(n_in, dtype=int)
            output_indices = np.arange(n_out, dtype=int)

            # Find the location of the largest value in sorted_in that
            # is less than every value of sorted_out
            lower_bound = (
                np.sum(np.where(np.less.outer(sorted_in, sorted_out), 1, 0), axis=0) - 1
            )

            # We're using the Finite Difference Cubic Hermite spline
            # https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Interpolation_on_an_arbitrary_interval
            # https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Finite_difference

            x1 = sorted_in[lower_bound]  # xₖ on the wiki
            x2 = sorted_in[lower_bound + 1]  # xₖ₊₁ on the wiki

            x0 = sorted_in[lower_bound[lower_bound - 1 >= 0] - 1]  # xpₖ₋₁ on the wiki
            x0 = np.hstack([np.zeros(x1.size - x0.size), x0])

            x3 = sorted_in[
                lower_bound[lower_bound + 2 < sorted_in.size] + 2
            ]  # xₖ₊₂ on the wiki
            x3 = np.hstack([x3, np.zeros(x2.size - x3.size)])

            t = (sorted_out - x1) / (x2 - x1)  # t on the wiki

            y0 = (
                -t * (x1 - x2) * (t**2 - 2 * t + 1) / (2 * x0 - 2 * x1)
            )  # The coefficient to pₖ₋₁ on the wiki
            y1 = (
                -t * (t**2 - 2 * t + 1) * (x0 - 2 * x1 + x2)
                + (x0 - x1) * (3 * t**3 - 5 * t**2 + 2)
            ) / (2 * (x0 - x1))  # The coefficient to pₖ
            y2 = (
                t
                * (
                    -t * (t - 1) * (x1 - 2 * x2 + x3)
                    + (x2 - x3) * (-3 * t**2 + 4 * t + 1)
                )
                / (2 * (x2 - x3))
            )  # The coefficient to pₗ₊₁
            y3 = t**2 * (t - 1) * (x1 - x2) / (2 * (x2 - x3))  # The coefficient to pₖ₊₂

            conversion_matrix = np.zeros((n_in, n_out))

            (row, column) = np.indices(conversion_matrix.shape)

            mask1 = row == lower_bound[column]

            conversion_matrix[np.roll(mask1, -1, axis=0)] = y0
            conversion_matrix[mask1] = y1
            conversion_matrix[np.roll(mask1, 1, axis=0)] = y2

            # Special boundary condition for y3
            pick = np.roll(mask1, 2, axis=0)
            pick[0:1, :] = 0
            if pick.any():
                conversion_matrix[pick] = y3

        case _:
            raise InterpolationError(f"Unsupported interpolation order: {order}")

    if mask is None:
        return conversion_matrix, None

    else:
        # Create a new mask

        # Convert to numerical values
        # Conservative masking: anything touched by the previous mask is now masked
        new_mask = (np.array(mask, dtype=float) @ conversion_matrix) != 0.0

        return conversion_matrix, new_mask


def calculate_interpolation_matrix_2d_axis_axis(input_1: Quantity[ArrayLike],
                                                input_2: Quantity[ArrayLike],
                                                output_1: Quantity[ArrayLike],
                                                output_2: Quantity[ArrayLike],
                                                mask,
                                                order: InterpolationOptions = InterpolationOptions.LINEAR,
                                                is_density: bool = False):

    # This is just the same 1D matrices things

    match order:
        case InterpolationOptions.NEAREST_NEIGHBOUR:
            pass

        case InterpolationOptions.LINEAR:
            pass

        case InterpolationOptions.CUBIC:
            pass

        case _:
            pass


def calculate_interpolation_matrix(input_axes: list[Quantity[ArrayLike]],
                                   output_axes: list[Quantity[ArrayLike]],
                                   data: ArrayLike | None = None,
                                   mask: ArrayLike | None = None):

    # TODO: We probably should delete this, but lets keep it for now

    if len(input_axes) not in (1, 2):
        raise InterpolationError("Interpolation is only supported for 1D and 2D data")

    if len(input_axes) == 1 and len(output_axes) == 1:
        # Check for dimensionality
        input_axis = input_axes[0]
        output_axis = output_axes[0]

        if len(input_axis.value.shape) == 1:
            if len(output_axis.value.shape) == 1:
                calculate_interpolation_matrix_1d()

    if len(output_axes) != len(input_axes):
        # Input or output axes might be 2D matrices
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
