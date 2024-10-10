""" Algorithms for interpolation and rebinning """
from typing import TypeVar

from numpy._typing import ArrayLike

from sasdata.quantities.quantity import Quantity

def rebin(data: Quantity[ArrayLike], axes: list[Quantity[ArrayLike]], new_axes: list[Quantity[ArrayLike]], interpolation_order=1):
    pass