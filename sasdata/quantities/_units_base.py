from dataclasses import dataclass
from typing import Sequence, Self, TypeVar

import numpy as np

from sasdata.quantities.unicode_superscript import int_as_unicode_superscript


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

    def __mul__(self: Self, other: Self):

        if not isinstance(other, Dimensions):
            return NotImplemented

        return Dimensions(
            self.length + other.length,
            self.time + other.time,
            self.mass + other.mass,
            self.current + other.current,
            self.temperature + other.temperature)

    def __truediv__(self: Self, other: Self):

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

    def __eq__(self: Self, other: Self):
        if isinstance(other, Dimensions):
            return (self.length == other.length and
                    self.time == other.time and
                    self.mass == other.mass and
                    self.current == other.current and
                    self.temperature == other.temperature)

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

        return 2**two_powers * 3**abs(self.length) * 5**abs(self.time) * \
            7**abs(self.mass) * 11**abs(self.current) * 13**abs(self.temperature)

    def __repr__(self):
        s = ""
        for name, size in [
            ("L", self.length),
            ("T", self.time),
            ("M", self.mass),
            ("C", self.current),
            ("K", self.temperature)]:

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
                 dimensions: Dimensions,
                 name: str | None = None,
                 ascii_symbol: str | None = None,
                 symbol: str | None = None):

        self.scale = si_scaling_factor
        self.dimensions = dimensions
        self.name = name
        self.ascii_symbol = ascii_symbol
        self.symbol = symbol

    def _components(self, tokens: Sequence["UnitToken"]):
        pass

    def __mul__(self: Self, other: Self):
        if not isinstance(other, Unit):
            return NotImplemented

        return Unit(self.scale * other.scale, self.dimensions * other.dimensions)

    def __truediv__(self: Self, other: Self):
        if not isinstance(other, Unit):
            return NotImplemented

        return Unit(self.scale / other.scale, self.dimensions / other.dimensions)

    def __rtruediv__(self: Self, other: Self):
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

    def equivalent(self: Self, other: Self):
        return self.dimensions == other.dimensions

    def __eq__(self: Self, other: Self):
        return self.equivalent(other) and np.abs(np.log(self.scale/other.scale)) < 1e-5

class UnitGroup:
    def __init__(self, name: str, units: list[Unit]):
        self.name = name
        self.units = sorted(units, key=lambda unit: unit.scale)
