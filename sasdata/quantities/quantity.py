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
            return (self.units.scale / units.scale) * self.value
        else:
            raise UnitError(f"Target units ({units}) not compatible with existing units ({self.units}).")

    def __mul__(self: Self, other: ArrayLike | Self ) -> Self:
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.units * other.units)

        else:
            return Quantity(self.value * other, self.units)

    def __rmul__(self: Self, other: ArrayLike | Self):
        if isinstance(other, Quantity):
            return Quantity(other.value * self.value, other.units * self.units)

        else:
            return Quantity(other * self.value, self.units)


    def __truediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.units / other.units)

        else:
            return Quantity(self.value / other, self.units)

    def __rtruediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.units / other.units)

        else:
            return Quantity(self.value / other, self.units)

    def __add__(self: Self, other: Self | ArrayLike) -> Self:
        if isinstance(other, Quantity):
            if self.units.equivalent(other.units):
                return Quantity(self.value + (other.value * other.scale)/self.scale, self.units)
            else:
                raise UnitError(f"Units do not have the same dimensionality: {self.units} vs {other.units}")

        else:
            raise UnitError(f"Cannot perform addition/subtraction non-quantity {type(other)} with quantity")

    # Don't need __radd__ because only quantity/quantity operations should be allowed

    def __neg__(self):
        return Quantity(-self.value, self.units)

    def __sub__(self: Self, other: Self | ArrayLike) -> Self:
        return self + (-other)

    def __rsub__(self: Self, other: Self | ArrayLike) -> Self:
        return (-self) + other

    def __pow__(self: Self, other: int):
        return Quantity(self.value**other, self.units**other)
