from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")

def closed_loop_edges(values: Sequence[T]) -> tuple[T, T]:
    """ Generator for a closed loop of edge pairs """
    for pair in zip(values, values[1:]):
        yield pair

    yield values[-1], values[0]
