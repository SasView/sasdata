from dataclasses import dataclass
from typing import Sequence, Self, TypeVar

import numpy as np

from sasdata.quantities.unicode_superscript import int_as_unicode_superscript

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

    def __pow__(self, power: int):

        if not isinstance(power, int):
            return NotImplemented

        return Dimensions(
            self.length * power,
            self.time * power,
            self.mass * power,
            self.current * power,
            self.temperature * power,
            self.moles_hint * power,
            self.angle_hint * power)

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
        s = ""
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
                s += f"{name}"
            else:
                s += f"{name}{int_as_unicode_superscript(size)}"

        return s

class Unit:
    def __init__(self,
                 si_scaling_factor: float,
                 dimensions: Dimensions):

        self.scale = si_scaling_factor
        self.dimensions = dimensions

    def _components(self, tokens: Sequence["UnitToken"]):
        pass

    def __mul__(self: Self, other: "Unit"):
        if not isinstance(other, Unit):
            return NotImplemented

        return Unit(self.scale * other.scale, self.dimensions * other.dimensions)

    def __truediv__(self: Self, other: "Unit"):
        if not isinstance(other, Unit):
            return NotImplemented

        return Unit(self.scale / other.scale, self.dimensions / other.dimensions)

    def __rtruediv__(self: Self, other: "Unit"):
        if isinstance(other, Unit):
            return Unit(other.scale / self.scale, other.dimensions / self.dimensions)
        elif isinstance(other, (int, float)):
            return Unit(other / self.scale, self.dimensions ** -1)
        else:
            return NotImplemented

    def __pow__(self, power: int):
        if not isinstance(power, int):
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
        return f"Unit[{self.scale}, {self.dimensions}]"

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
                 symbol: str | None = None):

        super().__init__(si_scaling_factor, dimensions)
        self.name = name
        self.ascii_symbol = ascii_symbol
        self.symbol = symbol

    def __repr__(self):
        return self.name

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
    def __init__(self, name: str, units: list[Unit]):
        self.name = name
        self.units = sorted(units, key=lambda unit: unit.scale)
