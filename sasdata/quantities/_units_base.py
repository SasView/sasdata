from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Self

import numpy as np
from unicode_superscript import int_as_unicode_superscript


class DimensionError(Exception):
    pass

class Dimensions:
    """

    Note that some SI Base units are not useful from the perspecive of the sasview project, and make things
    behave badly. In particular: moles and angular measures are dimensionless, and candelas are really a weighted
    measure of power.

    We do however track angle and amount, because its really useful for formatting units

    """
    def __init__(self,
                 length: int = 0,
                 time: int = 0,
                 mass: int = 0,
                 current: int = 0,
                 temperature: int = 0,
                 moles_hint: int = 0,
                 angle_hint: int = 0):

        self.length = length
        self.time = time
        self.mass = mass
        self.current = current
        self.temperature = temperature
        self.moles_hint = moles_hint
        self.angle_hint = angle_hint

    @property
    def is_dimensionless(self):
        """ Is this dimension dimensionless (ignores moles_hint and angle_hint) """
        return self.length == 0 and self.time == 0 and self.mass == 0 and self.current == 0 and self.temperature == 0

    def __mul__(self: Self, other: Self):

        if not isinstance(other, Dimensions):
            return NotImplemented

        return Dimensions(
            self.length + other.length,
            self.time + other.time,
            self.mass + other.mass,
            self.current + other.current,
            self.temperature + other.temperature,
            self.moles_hint + other.moles_hint,
            self.angle_hint + other.angle_hint)

    def __truediv__(self: Self, other: Self):

        if not isinstance(other, Dimensions):
            return NotImplemented

        return Dimensions(
            self.length - other.length,
            self.time - other.time,
            self.mass - other.mass,
            self.current - other.current,
            self.temperature - other.temperature,
            self.moles_hint - other.moles_hint,
            self.angle_hint - other.angle_hint)

    def __pow__(self, power: int | float):

        if not isinstance(power, (int, float)):
            return NotImplemented

        frac = Fraction(power).limit_denominator(500) # Probably way bigger than needed, 10 would probably be fine
        denominator = frac.denominator
        numerator = frac.numerator

        # Throw errors if dimension is not a multiple of the denominator

        if self.length % denominator != 0:
            raise DimensionError(f"Cannot apply power of {frac} to unit with length dimensionality {self.length}")

        if self.time % denominator != 0:
            raise DimensionError(f"Cannot apply power of {frac} to unit with time dimensionality {self.time}")

        if self.mass % denominator != 0:
            raise DimensionError(f"Cannot apply power of {frac} to unit with mass dimensionality {self.mass}")

        if self.current % denominator != 0:
            raise DimensionError(f"Cannot apply power of {frac} to unit with current dimensionality {self.current}")

        if self.temperature % denominator != 0:
            raise DimensionError(f"Cannot apply power of {frac} to unit with temperature dimensionality {self.temperature}")

        if self.moles_hint % denominator != 0:
            raise DimensionError(f"Cannot apply power of {frac} to unit with moles hint dimensionality of {self.moles_hint}")

        if self.angle_hint % denominator != 0:
            raise DimensionError(f"Cannot apply power of {frac} to unit with angle hint dimensionality of {self.angle_hint}")

        return Dimensions(
            (self.length * numerator) // denominator,
            (self.time * numerator) // denominator,
            (self.mass * numerator) // denominator,
            (self.current * numerator) // denominator,
            (self.temperature * numerator) // denominator,
            (self.moles_hint * numerator) // denominator,
            (self.angle_hint * numerator) // denominator)

    def __eq__(self: Self, other: Self):
        if isinstance(other, Dimensions):
            return (self.length == other.length and
                    self.time == other.time and
                    self.mass == other.mass and
                    self.current == other.current and
                    self.temperature == other.temperature and
                    self.moles_hint == other.moles_hint and
                    self.angle_hint == other.angle_hint)

        return NotImplemented

    def __hash__(self):
        """ Unique representation of units using Godel like encoding"""

        two_powers = 0
        if self.length < 0:
            two_powers += 1

        if self.time < 0:
            two_powers += 2

        if self.mass < 0:
            two_powers += 4

        if self.current < 0:
            two_powers += 8

        if self.temperature < 0:
            two_powers += 16

        if self.moles_hint < 0:
            two_powers += 32

        if self.angle_hint < 0:
            two_powers += 64

        return 2**two_powers * 3**abs(self.length) * 5**abs(self.time) * \
            7**abs(self.mass) * 11**abs(self.current) * 13**abs(self.temperature) * \
            17**abs(self.moles_hint) * 19**abs(self.angle_hint)

    def __repr__(self):
        tokens = []
        for name, size in [
            ("length", self.length),
            ("time", self.time),
            ("mass", self.mass),
            ("current", self.current),
            ("temperature", self.temperature),
            ("amount", self.moles_hint),
            ("angle", self.angle_hint)]:

            if size == 0:
                pass
            elif size == 1:
                tokens.append(f"{name}")
            else:
                tokens.append(f"{name}{int_as_unicode_superscript(size)}")

        return ' '.join(tokens)

    def si_repr(self):
        tokens = []
        for name, size in [
            ("kg", self.mass),
            ("m", self.length),
            ("s", self.time),
            ("A", self.current),
            ("K", self.temperature),
            ("mol", self.moles_hint)]:

            if size == 0:
                pass
            elif size == 1:
                tokens.append(f"{name}")
            else:
                tokens.append(f"{name}{int_as_unicode_superscript(size)}")

        match self.angle_hint:
            case 0:
                pass
            case 2:
                tokens.append("sr")
            case -2:
                tokens.append("sr" + int_as_unicode_superscript(-1))
            case _:
                tokens.append("rad" + int_as_unicode_superscript(self.angle_hint))

        return ''.join(tokens)


class Unit:
    def __init__(self,
                 si_scaling_factor: float,
                 dimensions: Dimensions):

        self.scale = si_scaling_factor
        self.dimensions = dimensions

    def _components(self, tokens: Sequence["UnitToken"]):
        pass

    def __mul__(self: Self, other: "Unit"):
        if isinstance(other, Unit):
            return Unit(self.scale * other.scale, self.dimensions * other.dimensions)
        elif isinstance(other, (int, float)):
            return Unit(other * self.scale, self.dimensions)
        return NotImplemented

    def __truediv__(self: Self, other: "Unit"):
        if isinstance(other, Unit):
            return Unit(self.scale / other.scale, self.dimensions / other.dimensions)
        elif isinstance(other, (int, float)):
            return Unit(self.scale / other, self.dimensions)
        else:
            return NotImplemented

    def __rtruediv__(self: Self, other: "Unit"):
        if isinstance(other, Unit):
            return Unit(other.scale / self.scale, other.dimensions / self.dimensions)
        elif isinstance(other, (int, float)):
            return Unit(other / self.scale, self.dimensions ** -1)
        else:
            return NotImplemented

    def __pow__(self, power: int | float):
        if not isinstance(power, int | float):
            return NotImplemented

        return Unit(self.scale**power, self.dimensions**power)


    def equivalent(self: Self, other: "Unit"):
        return self.dimensions == other.dimensions

    def __eq__(self: Self, other: "Unit"):
        return self.equivalent(other) and np.abs(np.log(self.scale/other.scale)) < 1e-5

    def si_equivalent(self):
        """ Get the SI unit corresponding to this unit"""
        return Unit(1, self.dimensions)

    def _format_unit(self, format_process: list["UnitFormatProcessor"]):
        for processor in format_process:
            pass

    def __repr__(self):
        if self.scale == 1:
            # We're in SI
            return self.dimensions.si_repr()

        else:
            return f"Unit[{self.scale}, {self.dimensions}]"

    @staticmethod
    def parse(unit_string: str) -> "Unit":
        pass

class NamedUnit(Unit):
    """ Units, but they have a name, and a symbol

    :si_scaling_factor: Number of these units per SI equivalent
    :param dimensions: Dimensions object representing the dimensionality of these units
    :param name: Name of unit - string without unicode
    :param ascii_symbol: Symbol for unit without unicode
    :param symbol: Unicode symbol
    """
    def __init__(self,
                 si_scaling_factor: float,
                 dimensions: Dimensions,
                 name: str | None = None,
                 ascii_symbol: str | None = None,
                 latex_symbol: str | None = None,
                 symbol: str | None = None):

        super().__init__(si_scaling_factor, dimensions)
        self.name = name
        self.ascii_symbol = ascii_symbol
        self.symbol = symbol
        self.latex_symbol = latex_symbol if latex_symbol is not None else ascii_symbol

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        """Match other units exactly or match strings against ANY of our names"""
        match other:
            case str():
                return self.name == other or self.name == f"{other}s" or self.ascii_symbol == other or self.symbol == other
            case NamedUnit():
                return self.name == other.name \
                    and self.ascii_symbol == other.ascii_symbol and self.symbol == other.symbol
            case Unit():
                return self.equivalent(other) and np.abs(np.log(self.scale/other.scale)) < 1e-5
            case _:
                return False


    def startswith(self, prefix: str) -> bool:
        """Check if any representation of the unit begins with the prefix string"""
        prefix = prefix.lower()
        return (self.name is not None and self.name.lower().startswith(prefix)) \
                or (self.ascii_symbol is not None and self.ascii_symbol.lower().startswith(prefix)) \
                or (self.symbol is not None and self.symbol.lower().startswith(prefix))


class ArbitraryUnit(NamedUnit):
    """A unit for an unknown quantity

    While this library attempts to handle all known SI units, it is
    likely that users will want to express quantities of arbitrary
    units (for example, calculating donuts per person for a meeting).
    The arbitrary unit allows for these unforseeable quantities."""

    def __init__(self,
                 numerator: str | list[str] | dict[str, int],
                 denominator: None | list[str] | dict[str, int]= None):
        match numerator:
            case str():
                self._numerator = {numerator: 1}
            case list():
                self._numerator = {}
                for key in numerator:
                    if key in self._numerator:
                        self._numerator[key] += 1
                    else:
                        self._numerator[key] = 1
            case dict():
                self._numerator = numerator
            case _:
                raise TypeError
        match denominator:
            case None:
                self._denominator = {}
            case str():
                self._denominator = {denominator: 1}
            case list():
                self._denominator = {}
                for key in denominator:
                    if key in self._denominator:
                        self._denominator[key] += 1
                    else:
                        self._denominator[key] = 1
            case dict():
                self._denominator = denominator
            case _:
                raise TypeError
        self._unit = Unit(1, Dimensions())  # Unitless

        super().__init__(si_scaling_factor=1, dimensions=self._unit.dimensions, symbol=self._name())

    def _name(self):
        match (self._numerator, self._denominator):
            case ({}, {}):
                return ""
            case (_, {}):
                return " ".join(self._numerator)
            case ({}, _):
                return "1 / " + " ".join(self._denominator)
            case _:
                return " ".join(self._numerator) + " / " + " ".join(self._denominator)

    def __eq__(self, other):
        match other:
            case ArbitraryUnit():
                return self._numerator == other._numerator and self._denominator == other._denominator and self._unit == other._unit
            case Unit():
                return not self._numerator and not self._denominator and self._unit == other


    def __mul__(self: Self, other: "Unit"):
        match other:
            case ArbitraryUnit():
                num = dict(self._numerator)
                for key in other._numerator:
                    if key in num:
                        num[key] += other._numerator[key]
                    else:
                        num[key] = other._numerator[key]
                den = dict(self._denominator)
                for key in other._denominator:
                    if key in den:
                        den[key] += other._denominator[key]
                    else:
                        den[key] = other._denominator[key]
                result = ArbitraryUnit(num, den)
                result._unit *= other._unit
                return result
            case NamedUnit() | Unit() | int() | float():
                result = ArbitraryUnit(self._numerator, self._denominator)
                result._unit *= other
                return result
            case _:
                return NotImplemented

    def __rmul__(self: Self, other):
        return self * other

    def __truediv__(self: Self, other: "Unit"):
        match other:
            case ArbitraryUnit():
                num = dict(self._numerator)
                for key in other._denominator:
                    if key in num:
                        num[key] += other._denominator[key]
                    else:
                        num[key] = other._denominator[key]
                den = dict(self._denominator)
                for key in other._numerator:
                    if key in den:
                        den[key] += other._numerator[key]
                    else:
                        den[key] = other._numerator[key]
                result = ArbitraryUnit(num, den)
                result._unit /= other._unit
                return result
            case NamedUnit() | Unit() | int() | float():
                result = ArbitraryUnit(self._numerator, self._denominator)
                result._unit /= other
                return result
            case _:
                return NotImplemented

    def __rtruediv__(self: Self, other: "Unit"):
        # if isinstance(other, Unit):
        #     return Unit(other.scale / self.scale, other.dimensions / self.dimensions)
        # elif isinstance(other, (int, float)):
        #     return Unit(other / self.scale, self.dimensions ** -1)
        # else:
            return NotImplemented

    def __pow__(self, power: int):
        match power:
            case int() | float():
                num = {key: value * power for key, value in self._numerator.items()}
                den = {key: value * power for key, value in self._denominator.items()}
                result = ArbitraryUnit(num, den)
                result._unit = self._unit ** power
                return result
            case _:
                return NotImplemented


    def equivalent(self: Self, other: "Unit"):
        match other:
            case ArbitraryUnit():
                return self._unit.equivalent(other._unit) and sorted(self._numerator) == sorted(other._numerator) and sorted(self._denominator) == sorted(other._denominator)
            case _:
                return False

    def si_equivalent(self):
        """ Get the SI unit corresponding to this unit"""
        """FIXME: TODO"""
        return Unit(1, self.dimensions)

    def _format_unit(self, format_process: list["UnitFormatProcessor"]):
        """FIXME: TODO"""
        for processor in format_process:
            pass

    def __repr__(self):
        """FIXME: TODO"""
        result = self._name()
        if self._unit.__repr__():
            result += f" {self._unit.__repr__()}"
        return result

    @staticmethod
    def parse(unit_string: str) -> "Unit":
        """FIXME: TODO"""
        pass
#
# Parsing plan:
#  Require unknown amounts of units to be explicitly positive or negative?
#
#



@dataclass
class ProcessedUnitToken:
    """ Mid processing representation of formatted units """
    base_string: str
    exponent_string: str
    latex_exponent_string: str
    exponent: int

class UnitFormatProcessor:
    """ Represents a step in the unit processing pipeline"""
    def apply(self, scale, dimensions) -> tuple[ProcessedUnitToken, float, Dimensions]:
        """ This will be called to deal with each processing stage"""

class RequiredUnitFormatProcessor(UnitFormatProcessor):
    """ This unit is required to exist in the formatting """
    def __init__(self, unit: Unit, power: int = 1):
        self.unit = unit
        self.power = power
    def apply(self, scale, dimensions) -> tuple[float, Dimensions, ProcessedUnitToken]:
        new_scale = scale / (self.unit.scale * self.power)
        new_dimensions = self.unit.dimensions / (dimensions**self.power)
        token = ProcessedUnitToken(self.unit, self.power)

        return new_scale, new_dimensions, token
class GreedyAbsDimensionUnitFormatProcessor(UnitFormatProcessor):
    """ This processor minimises the dimensionality of the unit by multiplying by as many
    units of the specified type as needed """
    def __init__(self, unit: Unit):
        self.unit = unit

    def apply(self, scale, dimensions) -> tuple[ProcessedUnitToken, float, Dimensions]:
        pass

class GreedyAbsDimensionUnitFormatProcessor(UnitFormatProcessor):
    pass

class UnitGroup:
    """ A group of units that all have the same dimensionality """
    def __init__(self, name: str, units: list[NamedUnit]):
        self.name = name
        self.units = sorted(units, key=lambda unit: unit.scale)

