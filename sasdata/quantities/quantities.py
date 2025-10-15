from typing import Collection, Sequence, TypeVar, Generic
from dataclasses import dataclass

from numpy._typing import ArrayLike


class Dimensions:
    """

    Note that some SI Base units are

    For example, moles and angular measures are dimensionless from this perspective, and candelas are

    """
    def __init__(self,
                 length: int = 0,
                 time: int = 0,
                 mass: int = 0,
                 current: int = 0,
                 temperature: int = 0):

        self.length = length
        self.time = time
        self.mass = mass
        self.current = current
        self.temperature = temperature

    def __mul__(self, other: "Dimensions"):

        if not isinstance(other, Dimensions):
            return NotImplemented

        return Dimensions(
            self.length + other.length,
            self.time + other.time,
            self.mass + other.mass,
            self.current + other.current,
            self.temperature + other.temperature)

    def __truediv__(self, other: "Dimensions"):

        if not isinstance(other, Dimensions):
            return NotImplemented

        return Dimensions(
            self.length - other.length,
            self.time - other.time,
            self.mass - other.mass,
            self.current - other.current,
            self.temperature - other.temperature)

    def __pow__(self, power: int):

        if not isinstance(power, int):
            return NotImplemented

        return Dimensions(
            self.length * power,
            self.time * power,
            self.mass * power,
            self.current * power,
            self.temperature * power)

    def __eq__(self, other: "Dimensions"):
        if isinstance(other, Dimensions):
            return (self.length == other.length and
                    self.time == other.time and
                    self.mass == other.mass and
                    self.current == other.current and
                    self.temperature == other.temperature)

        return NotImplemented



@dataclass
class UnitName:
    ascii_name: str
    unicode_name: str | None = None

    @property
    def best_name(self):
        if self.unicode_name is None:
            return self.ascii_name
        else:
            return self.unicode_name

class Unit:
    def __init__(self,
                 si_scaling_factor: float,
                 dimensions: Dimensions,
                 name: UnitName | None = None):

        self.scale = si_scaling_factor
        self.dimensions = dimensions
        self.name = name

    def _components(self, tokens: Sequence["UnitToken"]):
        pass

    def __mul__(self, other: "Unit"):
        if not isinstance(other, Unit):
            return NotImplemented

        return Unit(self.scale * other.scale, self.dimensions * other.dimensions)

    def __truediv__(self, other: "Unit"):
        if not isinstance(other, Unit):
            return NotImplemented

        return Unit(self.scale / other.scale, self.dimensions / other.dimensions)

    def __pow__(self, power: int):
        if not isinstance(power, int):
            return NotImplemented

        return Unit(self.scale**power, self.dimensions**power)

    def equivalent(self, other: "Unit"):
        return self.dimensions == other.dimensions


# QuantityType = TypeVar("QuantityType")
class Quantity[QuantityType]:
    def __init__(self, value: QuantityType, units: Unit):
        self.value = value
        self.units = units

    def in_units_of(self, units: Unit) -> QuantityType:
        pass

    def __mul__(self, other: ArrayLike | "Quantity" ) -> "Quantity":
        if isinstance(other, Quantity):
            pass

        else:
            pass

    def __truediv__(self, other: float | "Quantity") -> "Quantity":
        if isinstance(other, Quantity):
            pass

        else:
            pass

    def __rdiv__(self, other: float | "Quantity") -> "Quantity":
        if isinstance(other, Quantity):
            pass

        else:
            pass
    def __add__(self, other: "Quantity") -> "Quantity":
        if isinstance(other, Quantity):
            pass

    def __sub__(self, other: "Quantity") -> "Quantity":
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
