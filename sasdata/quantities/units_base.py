from dataclasses import dataclass
from typing import Sequence, Self, TypeVar

import numpy as np


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
        return hash((self.length, self.time, self.mass, self.current, self.temperature))

class Unit:
    def __init__(self,
                 si_scaling_factor: float,
                 dimensions: Dimensions):

        self.scale = si_scaling_factor
        self.dimensions = dimensions

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
