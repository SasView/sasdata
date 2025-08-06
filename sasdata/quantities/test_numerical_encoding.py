""" Tests for the encoding and decoding of numerical data"""

import numpy as np
import pytest

from sasdata.quantities.numerical_encoding import numerical_decode, numerical_encode


@pytest.mark.parametrize("value", [-100.0, -10.0, -1.0, 0.0, 0.5, 1.0, 10.0, 100.0, 1e100])
def test_float_encode_decode(value: float):

    assert isinstance(value, float) # Make sure we have the right inputs

    encoded = numerical_encode(value)
    decoded = numerical_decode(encoded)

    assert isinstance(decoded, float)
    assert value == decoded

@pytest.mark.parametrize("value", [-100, -10, -1, 0, 1, 10, 100, 1000000000000000000000000000000000])
def test_int_encode_decode(value: int):

    assert isinstance(value, int)  # Make sure we have the right inputs

    encoded = numerical_encode(value)
    decoded = numerical_decode(encoded)

    assert isinstance(decoded, int)
    assert value == decoded

@pytest.mark.parametrize("shape", [
    (2,3,4),
    (1,2),
    (10,5,10),
    (1,),
    (4,),
    (0, ) ])
def test_numpy_float_encode_decode(shape):
    np.random.seed(1776)
    test_matrix = np.random.rand(*shape)

    encoded = numerical_encode(test_matrix)
    decoded = numerical_decode(encoded)

    assert decoded.dtype == test_matrix.dtype
    assert decoded.shape == test_matrix.shape
    assert np.all(decoded == test_matrix)

@pytest.mark.parametrize("dtype", [int, float, complex])
def test_numpy_dtypes_encode_decode(dtype):
    test_matrix = np.zeros((3,3), dtype=dtype)

    encoded = numerical_encode(test_matrix)
    decoded = numerical_decode(encoded)

    assert decoded.dtype == test_matrix.dtype

@pytest.mark.parametrize("dtype", [int, float, complex])
@pytest.mark.parametrize("shape, n, m", [
    ((8, 8), (1,3,5),(2,5,7)),
    ((6, 8), (1,0,5),(0,5,0)),
    ((6, 1), (1, 0, 5), (0, 0, 0)),
])
def test_coo_matrix_encode_decode(shape, n, m, dtype):

    values = np.arange(10)
