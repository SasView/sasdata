"""Tests for math operations"""

import numpy as np
import pytest

from sasdata.quantities import units
from sasdata.quantities.quantity import NamedQuantity, matinv, tensordot, trace, transpose

order_list = [[0, 1, 2, 3], [0, 2, 1], [1, 0], [0, 1], [2, 0, 1], [3, 1, 2, 0]]


@pytest.mark.parametrize("order", order_list)
def test_transpose(order: list[int]):
    """Check that the transpose operation changes the order of indices correctly for raw data and quantities - uses sizes as way of tracking"""

    input_shape = tuple([i + 1 for i in range(len(order))])
    expected_shape = tuple([i + 1 for i in order])

    input_mat = np.zeros(input_shape)
    input_quantity = NamedQuantity("testmat", np.zeros(input_shape), units=units.none)

    measured_mat = transpose(input_mat, axes=tuple(order))
    measured_quantity = transpose(input_quantity, axes=tuple(order))

    assert measured_mat.shape == expected_shape
    assert measured_quantity.value.shape == expected_shape


@pytest.mark.parametrize(
    "matrix, offset, expected_trace",
    [
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 0, 15),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 1, 8),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 2, 3),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), -1, 12),
    ],
)
def test_trace_offset(matrix, offset, expected_trace):
    """Check that the trace operation correctly identifies the offset value for raw data and quantities."""
    assert (trace(matrix, offset=offset) == expected_trace).all()
    assert (trace(NamedQuantity("testmat", matrix, units=units.none), offset).value == expected_trace).all()


@pytest.mark.parametrize(
    "matrix, axis1, axis2, expected_trace",
    [
        (np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]), 0, 1, np.array([4, 6])),
        (np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]), 1, 2, np.array([5, 5, 5])),
        (np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]), 0, 2, np.array([3, 7])),
    ],
)
def test_trace_axes(matrix, axis1, axis2, expected_trace):
    """Check that the trace operation correctly identifies the offset value for raw data and quantities."""
    assert (trace(matrix, axis1=axis1, axis2=axis2) == expected_trace).all()
    assert (
        trace(NamedQuantity("testmat", matrix, units=units.none), axis1=axis1, axis2=axis2).value == expected_trace
    ).all()


@pytest.mark.parametrize(
    "matrix, expected_inverse",
    [
        (np.array([[1]]), np.array([[1]])),
        (np.array([[-2.0, 1.0], [1.5, -0.5]]), np.array([[1, 2], [3, 4]])),
    ],
)
def test_inverse(matrix, expected_inverse):
    """Check that the matinv operation correctly inverse for raw data and quantities."""
    print(matinv(matrix))
    print(expected_inverse)
    assert (matinv(matrix) == expected_inverse).all()
    assert (matinv(NamedQuantity("testmat", matrix, units=units.none)).value == expected_inverse).all()


rng_seed = 1979
tensor_product_with_identity_sizes = (4, 6, 5)


@pytest.mark.parametrize(
    "x, x_unit",
    [
        (NamedQuantity("x", np.random.rand(*tensor_product_with_identity_sizes), units=units.meters), units.meters),
        ((np.random.rand(*tensor_product_with_identity_sizes), units.none)),
    ],
)
@pytest.mark.parametrize("index, size", [tup for tup in enumerate(tensor_product_with_identity_sizes)])
def test_tensor_product_with_identity_quantities(x, x_unit, index, size):
    """Check the correctness of the tensor product by multiplying by the identity (quantity, quantity)"""
    np.random.seed(rng_seed)
    y = NamedQuantity("y", np.eye(size), units.seconds)

    z = tensordot(x, y, index, 0)

    # Check units
    assert z.units == x_unit * units.seconds

    # Expected sizes - last index gets moved to end
    output_order = [i for i in (0, 1, 2) if i != index] + [index]
    output_sizes = [tensor_product_with_identity_sizes[i] for i in output_order]

    assert z.value.shape == tuple(output_sizes)

    # Restore original order and check
    reverse_order = [-1, -1, -1]
    for to_index, from_index in enumerate(output_order):
        reverse_order[from_index] = to_index

    z_reordered = transpose(z, axes=tuple(reverse_order))

    assert z_reordered.value.shape == tensor_product_with_identity_sizes

    # Check values

    try:
        mat_in = x.in_si()
    except AttributeError:
        mat_in = x

    mat_out = transpose(z, axes=tuple(reverse_order)).in_si()

    assert np.all(np.abs(mat_in - mat_out) < 1e-10)


@pytest.mark.parametrize(
    "x, x_unit",
    [
        (NamedQuantity("x", np.random.rand(*tensor_product_with_identity_sizes), units=units.meters), units.meters),
    ],
)
@pytest.mark.parametrize("index, size", [tup for tup in enumerate(tensor_product_with_identity_sizes)])
def test_tensor_product_with_identity_quantity_matrix(x, x_unit, index, size):
    """Check the correctness of the tensor product by multiplying by the identity (quantity, matrix)"""
    np.random.seed(rng_seed)
    y = np.eye(size)

    z = tensordot(x, y, index, 0)

    assert z.units == x_unit

    # Expected sizes - last index gets moved to end
    output_order = [i for i in (0, 1, 2) if i != index] + [index]
    output_sizes = [tensor_product_with_identity_sizes[i] for i in output_order]

    assert z.value.shape == tuple(output_sizes)

    # Restore original order and check
    reverse_order = [-1, -1, -1]
    for to_index, from_index in enumerate(output_order):
        reverse_order[from_index] = to_index

    z_reordered = transpose(z, axes=tuple(reverse_order))

    assert z_reordered.value.shape == tensor_product_with_identity_sizes

    # Check values

    try:
        mat_in = x.in_si()
    except AttributeError:
        mat_in = x

    mat_out = transpose(z, axes=tuple(reverse_order)).in_si()

    assert np.all(np.abs(mat_in - mat_out) < 1e-10)
