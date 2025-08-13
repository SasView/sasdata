from collections.abc import Callable
from typing import TypeVar

import numpy as np

T = TypeVar("T")

def cache[T](fun: Callable[[], T]):
    """ Decorator to store values """

    cache_state = [False, None]

    def wrapper() -> T:
        if not cache_state[0]:
            cache_state[0] = True
            cache_state[1] = fun()

        return cache_state[1]

    return wrapper

def is_increasing(data: np.ndarray):
    """ Check if a 1D array is sorted in strictly increasing order"""
    return np.all(data[1:] > data[:-1])

def is_decreasing(data: np.ndarray):
    """ Check if a 1D array is sorted in strictly decreasing order"""
    return np.all(data[1:] < data[:-1])

def is_sorted(data: np.ndarray):
    """ Check if a 1D array is strictly sorted """
    return is_increasing(data) or is_decreasing(data)