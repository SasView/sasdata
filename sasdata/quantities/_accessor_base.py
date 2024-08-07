from typing import TypeVar

from sasdata.quantities.quantity import Quantity
import sasdata.quantities.units as units


T = TypeVar("T")

class Accessor[T]:
    """ Base class """
    def __init__(self, value_target: str, unit_target: str):
        self._value_target = value_target
        self._unit_target = unit_target

    @property
    def quantity(self) -> Quantity[T]:
        raise NotImplementedError("Not implemented yet")
