""" Tests for math operations """

import pytest

import numpy as np
from sasdata.quantities.quantity import NamedQuantity, tensordot, transpose
from sasdata.quantities import units

order_list = [
    [0, 1, 2, 3],
    [0, 2, 1],
    [1, 0],
    [0, 1],
    [2, 0, 1],
    [3, 1, 2, 0]
]

@pytest.mark.parametrize("order", order_list)
def test_transpose_raw(order: list[int]):
    """ Check that the transpose operation changes the order of indices correctly - uses sizes as way of tracking"""

    input_shape = tuple([i+1 for i in range(len(order))])
    expected_shape = tuple([i+1 for i in order])

    input_mat = np.zeros(input_shape)

    measured_mat = transpose(input_mat, axes=tuple(order))

    assert measured_mat.shape == expected_shape


@pytest.mark.parametrize("order", order_list)
def test_transpose_raw(order: list[int]):
    """ Check that the transpose operation changes the order of indices correctly - uses sizes as way of tracking"""
    input_shape = tuple([i + 1 for i in range(len(order))])
    expected_shape = tuple([i + 1 for i in order])

    input_mat = NamedQuantity("testmat", np.zeros(input_shape), units=units.none)

    measured_mat = transpose(input_mat, axes=tuple(order))

    assert measured_mat.value.shape == expected_shape


rng_seed = 1979
tensor_product_with_identity_sizes = (4,6,5)

@pytest.mark.parametrize("index, size", [tup for tup in enumerate(tensor_product_with_identity_sizes)])
def test_tensor_product_with_identity_quantities(index, size):
    """ Check the correctness of the tensor product by multiplying by the identity (quantity, quantity)"""
    np.random.seed(rng_seed)

    x = NamedQuantity("x", np.random.rand(*tensor_product_with_identity_sizes), units=units.meters)
    y = NamedQuantity("y", np.eye(size), units.seconds)

    z = tensordot(x, y, index, 0)

    # Check units
    assert z.units == units.meters * units.seconds

    # Expected sizes - last index gets moved to end
    output_order = [i for i in (0, 1, 2) if i != index] + [index]
    output_sizes = [tensor_product_with_identity_sizes[i] for i in output_order]

    assert z.value.shape == tuple(output_sizes)

    # Restore original order and check
    reverse_order = [-1, -1, -1]
    for to_index, from_index in enumerate(output_order):
        reverse_order[from_index] = to_index

    z_reordered = transpose(z, axes = tuple(reverse_order))

    assert z_reordered.value.shape == tensor_product_with_identity_sizes

    # Check values

    mat_in = x.in_si()
    mat_out = transpose(z, axes=tuple(reverse_order)).in_si()

    assert np.all(np.abs(mat_in - mat_out) < 1e-10)


@pytest.mark.parametrize("index, size", [tup for tup in enumerate(tensor_product_with_identity_sizes)])
def test_tensor_product_with_identity_quantity_matrix(index, size):
    """ Check the correctness of the tensor product by multiplying by the identity (quantity, matrix)"""
    np.random.seed(rng_seed)

    x = NamedQuantity("x", np.random.rand(*tensor_product_with_identity_sizes), units.meters)
    y = np.eye(size)

    z = tensordot(x, y, index, 0)

    assert z.units == units.meters

    # Expected sizes - last index gets moved to end
    output_order = [i for i in (0, 1, 2) if i != index] + [index]
    output_sizes = [tensor_product_with_identity_sizes[i] for i in output_order]

    assert z.value.shape == tuple(output_sizes)

    # Restore original order and check
    reverse_order = [-1, -1, -1]
    for to_index, from_index in enumerate(output_order):
        reverse_order[from_index] = to_index

    z_reordered = transpose(z, axes = tuple(reverse_order))

    assert z_reordered.value.shape == tensor_product_with_identity_sizes

    # Check values

    mat_in = x.in_si()
    mat_out = transpose(z, axes=tuple(reverse_order)).in_si()

    assert np.all(np.abs(mat_in - mat_out) < 1e-10)


@pytest.mark.parametrize("index, size", [tup for tup in enumerate(tensor_product_with_identity_sizes)])
def test_tensor_product_with_identity_matrix_quantity(index, size):
    """ Check the correctness of the tensor product by multiplying by the identity (matrix, quantity)"""
    np.random.seed(rng_seed)

    x = np.random.rand(*tensor_product_with_identity_sizes)
    y = NamedQuantity("y", np.eye(size), units.seconds)

    z = tensordot(x, y, index, 0)

    assert z.units == units.seconds


    # Expected sizes - last index gets moved to end
    output_order = [i for i in (0, 1, 2) if i != index] + [index]
    output_sizes = [tensor_product_with_identity_sizes[i] for i in output_order]

    assert z.value.shape == tuple(output_sizes)

    # Restore original order and check
    reverse_order = [-1, -1, -1]
    for to_index, from_index in enumerate(output_order):
        reverse_order[from_index] = to_index

    z_reordered = transpose(z, axes = tuple(reverse_order))

    assert z_reordered.value.shape == tensor_product_with_identity_sizes

    # Check values

    mat_in = x
    mat_out = transpose(z, axes=tuple(reverse_order)).in_si()

    assert np.all(np.abs(mat_in - mat_out) < 1e-10)
