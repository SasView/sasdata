from typing import Collection, Sequence, TypeVar, Generic, Self
from dataclasses import dataclass

from numpy._typing import ArrayLike

from sasdata.quantities.units import Unit


class UnitError(Exception):
    """ Errors caused by unit specification not being correct """


QuantityType = TypeVar("QuantityType")

class Quantity[QuantityType]:
    def __init__(self, value: QuantityType, units: Unit):
        self.value = value
        self.units = units

    def in_units_of(self, units: Unit) -> QuantityType:
        if self.units.equivalent(units):
            return (units.scale / self.units.scale) * self.value
        else:
            raise UnitError(f"Target units ({units}) not compatible with existing units ({self.units}).")

    def __mul__(self: Self, other: ArrayLike | Self ) -> Self:
        if isinstance(other, Quantity):
            pass

        else:
            pass

    def __truediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, Quantity):
            pass

        else:
            pass

    def __rdiv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, Quantity):
            pass

        else:
            pass
    def __add__(self: Self, other: Self) -> Self:
        if isinstance(other, Quantity):
            pass

    def __sub__(self: Self, other: Self) -> Self:
        if isinstance(other, Quantity):
            pass

