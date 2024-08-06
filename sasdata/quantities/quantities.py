from typing import Collection, Sequence, TypeVar, Generic, Self
from dataclasses import dataclass

from numpy._typing import ArrayLike

from sasdata.quantities.units_base import Unit

QuantityType = TypeVar("QuantityType")

class Quantity[QuantityType]:
    def __init__(self, value: QuantityType, units: Unit):
        self.value = value
        self.units = units

    def in_units_of(self, units: Unit) -> QuantityType:
        pass

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


class ExpressionMethod:
    pass


class SetExpressionMethod(ExpressionMethod):
    pass


class AnyExpressionMethod(ExpressionMethod):
    pass


class ForceExpressionMethod(ExpressionMethod):
    pass


class UnitToken:
    def __init__(self, unit: Collection[NamedUnit], method: ExpressionMethod):
        pass

unit_dictionary = {
    "Amps": Unit(1, Dimensions(current=1), UnitName("A")),
    "Coulombs": Unit(1, Dimensions(current=1, time=1), UnitName("C"))
}

@dataclass
class Disambiguator:
    A: Unit = unit_dictionary["Amps"]
    C: Unit = unit_dictionary["Coulombs"]

def parse_units(unit_string: str, disambiguator: Disambiguator = Disambiguator()) -> Unit:
    pass
