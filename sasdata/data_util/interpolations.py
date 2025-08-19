"""
Interpolation functions for 1d data sets.
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Union


def linear(x_interp: ArrayLike, x: ArrayLike, y: ArrayLike, dy: Optional[ArrayLike] = None)\
        -> tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Computes linear interpolation of dataset (x, y) at the points x_interp.
    Error propagation is implemented when dy is provided.
    Requires that min(x) <= x_interp <= max(x)

    TODO: reductus package has a similar function in err1d if we add the additional dependency
    """
    x_interp = np.array(x_interp)
    sort = np.argsort(x)
    x = np.array(x)[sort]
    y = np.array(y)[sort]
    dy = np.array(dy)[sort] if (dy is not None and len(dy) == len(y)) else None

    # find out where the interpolated points fit into the existing data
    index_2 = np.searchsorted(x, x_interp)
    index_1 = index_2 - 1

    # linear interpolation of new y points
    y_interp_1 = y[index_1] * (x_interp - x[index_2]) / (x[index_1] - x[index_2])
    y_interp_2 = y[index_2] * (x_interp - x[index_1]) / (x[index_2] - x[index_1])
    y_interp = y_interp_1 + y_interp_2

    # error propagation
    if dy is not None:
        dy_interp_1 = dy[index_1] ** 2 * ((x_interp - x[index_2]) / (x[index_1] - x[index_2])) ** 2
        dy_interp_2 = dy[index_2] ** 2 * ((x_interp - x[index_1]) / (x[index_2] - x[index_1])) ** 2
        dy_interp = np.sqrt(dy_interp_1 + dy_interp_2)
    else:
        dy_interp = None

    return y_interp, dy_interp


def linear_scales(x_interp: ArrayLike,
                  x: ArrayLike,
                  y: ArrayLike,
                  dy: Optional[ArrayLike] = None,
                  scale: Optional[str] = "linear") -> tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Perform linear interpolation on different scales.
    Error propagation is implemented when dy is provided.

    Scale is set to "linear" by default.
    Setting scale to "log" will perform the interpolation of (log(x), log(y)) at log(x_interp); log(y_interp) will be
    converted back to y_interp in the return.

    Returns (y_interp, dy_interp | None)
    """
    x = np.array(x)
    y = np.array(y)

    if scale == "linear":
        result = linear(x_interp=x_interp, x=x, y=y, dy=dy)
        return result

    elif scale == "log":
        dy = np.array(dy) / y if (dy is not None and len(dy) == len(x)) else None
        x_interp = np.log(x_interp)
        x = np.log(x)
        y = np.log(y)
        result = linear(x_interp=x_interp, x=x, y=y, dy=dy)
        y_interp = np.exp(result[0])
        dy_interp = None if result[1] is None else y_interp * result[1]
        return y_interp, dy_interp


