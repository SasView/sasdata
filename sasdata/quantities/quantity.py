from typing import Collection, Sequence, TypeVar, Generic, Self
from dataclasses import dataclass

from numpy._typing import ArrayLike

from sasdata.quantities.operations import Operation, Variable
from sasdata.quantities.units import Unit

import hashlib


class UnitError(Exception):
    """ Errors caused by unit specification not being correct """


QuantityType = TypeVar("QuantityType")

class BaseQuantity[QuantityType]:
    def __init__(self, value: QuantityType, units: Unit):
        self.value = value
        self.units = units

    def in_units_of(self, units: Unit) -> QuantityType:
        if self.units.equivalent(units):
            return (self.units.scale / units.scale) * self.value
        else:
            raise UnitError(f"Target units ({units}) not compatible with existing units ({self.units}).")

    def __mul__(self: Self, other: ArrayLike | Self ) -> Self:
        if isinstance(other, BaseQuantity):
            return BaseQuantity(self.value * other.value, self.units * other.units)

        else:
            return BaseQuantity(self.value * other, self.units)

    def __rmul__(self: Self, other: ArrayLike | Self):
        if isinstance(other, BaseQuantity):
            return BaseQuantity(other.value * self.value, other.units * self.units)

        else:
            return BaseQuantity(other * self.value, self.units)

    def __truediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, BaseQuantity):
            return BaseQuantity(self.value / other.value, self.units / other.units)

        else:
            return BaseQuantity(self.value / other, self.units)

    def __rtruediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, BaseQuantity):
            return BaseQuantity(self.value / other.value, self.units / other.units)

        else:
            return BaseQuantity(self.value / other, self.units)

    def __add__(self: Self, other: Self | ArrayLike) -> Self:
        if isinstance(other, BaseQuantity):
            if self.units.equivalent(other.units):
                return BaseQuantity(self.value + (other.value * other.units.scale) / self.units.scale, self.units)
            else:
                raise UnitError(f"Units do not have the same dimensionality: {self.units} vs {other.units}")

        else:
            raise UnitError(f"Cannot perform addition/subtraction non-quantity {type(other)} with quantity")

    # Don't need __radd__ because only quantity/quantity operations should be allowed

    def __neg__(self):
        return BaseQuantity(-self.value, self.units)

    def __sub__(self: Self, other: Self | ArrayLike) -> Self:
        return self + (-other)

    def __rsub__(self: Self, other: Self | ArrayLike) -> Self:
        return (-self) + other

    def __pow__(self: Self, other: int):
        return BaseQuantity(self.value ** other, self.units ** other)

    @staticmethod
    def parse(number_or_string: str | ArrayLike, unit: str, absolute_temperature: False):
        pass


class Quantity[QuantityType](BaseQuantity[QuantityType]):
    def with_uncertainty(self, uncertainty: BaseQuantity[QuantityType]):
        return UncertainQuantity(self.value, self.units, uncertainty=uncertainty)


class NamedQuantity[QuantityType](BaseQuantity[QuantityType]):
    def __init__(self, value: QuantityType, units: Unit, name: str):
        super().__init__(value, units)
        self.name = name

    def with_uncertainty(self, uncertainty: BaseQuantity[QuantityType]):
        return UncertainNamedQuantity(self.value, self.units, uncertainty=uncertainty, name=self.name)


class UncertainBaseQuantity[QuantityType](BaseQuantity[QuantityType]):
    pass

class UncertainQuantity[QuantityType](BaseQuantity[QuantityType]):
    def __init__(self, value: QuantityType, units: Unit, uncertainty: BaseQuantity[QuantityType]):
        super().__init__(value, units)
        self.uncertainty = uncertainty

        hash_value = hashlib.md5(value, uncertainty)


class UncertainNamedQuantity[QuantityType](BaseQuantity[QuantityType]):
    def __init__(self, value: QuantityType, units: Unit, uncertainty: BaseQuantity[QuantityType], name: str):
        super().__init__(value, units)
        self.uncertainty = uncertainty
        self.name = name

        self.history = Variable(self.name)
