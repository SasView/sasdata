import re
import sys
from collections import defaultdict, namedtuple
from fractions import Fraction
from typing import Self

import numpy as np

_ascii_version = "0123456789-"
_unicode_version = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"


def int_as_unicode_superscript(number: int):
    string = str(number)

    for old, new in zip(_ascii_version, _unicode_version):
        string = string.replace(old, new)

    return string


class DimensionError(Exception):
    pass


class Dimensions:
    """

    Note that some SI Base units are not useful from the perspecive of the sasview project, and make things
    behave badly. In particular: moles and angular measures are dimensionless, and candelas are really a weighted
    measure of power.

    We do however track angle and amount, because its really useful for formatting units

    """

    def __init__(
        self,
        length: int = 0,
        time: int = 0,
        mass: int = 0,
        current: int = 0,
        temperature: int = 0,
        moles_hint: int = 0,
        angle_hint: int = 0,
    ):

        self.length = length
        self.time = time
        self.mass = mass
        self.current = current
        self.temperature = temperature
        self.moles_hint = moles_hint
        self.angle_hint = angle_hint

    @property
    def is_dimensionless(self):
        """Is this dimension dimensionless (ignores moles_hint and angle_hint)"""
        return (
            self.length == 0
            and self.time == 0
            and self.mass == 0
            and self.current == 0
            and self.temperature == 0
        )

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
            self.angle_hint + other.angle_hint,
        )

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
            self.angle_hint - other.angle_hint,
        )

    def __pow__(self, power: int | float):

        if not isinstance(power, (int, float)):
            return NotImplemented

        frac = Fraction(power).limit_denominator(
            500
        )  # Probably way bigger than needed, 10 would probably be fine
        denominator = frac.denominator
        numerator = frac.numerator

        # Throw errors if dimension is not a multiple of the denominator

        if self.length % denominator != 0:
            raise DimensionError(
                f"Cannot apply power of {frac} to unit with length dimensionality {self.length}"
            )

        if self.time % denominator != 0:
            raise DimensionError(
                f"Cannot apply power of {frac} to unit with time dimensionality {self.time}"
            )

        if self.mass % denominator != 0:
            raise DimensionError(
                f"Cannot apply power of {frac} to unit with mass dimensionality {self.mass}"
            )

        if self.current % denominator != 0:
            raise DimensionError(
                f"Cannot apply power of {frac} to unit with current dimensionality {self.current}"
            )

        if self.temperature % denominator != 0:
            raise DimensionError(
                f"Cannot apply power of {frac} to unit with temperature dimensionality {self.temperature}"
            )

        if self.moles_hint % denominator != 0:
            raise DimensionError(
                f"Cannot apply power of {frac} to unit with moles hint dimensionality of {self.moles_hint}"
            )

        if self.angle_hint % denominator != 0:
            raise DimensionError(
                f"Cannot apply power of {frac} to unit with angle hint dimensionality of {self.angle_hint}"
            )

        return Dimensions(
            (self.length * numerator) // denominator,
            (self.time * numerator) // denominator,
            (self.mass * numerator) // denominator,
            (self.current * numerator) // denominator,
            (self.temperature * numerator) // denominator,
            (self.moles_hint * numerator) // denominator,
            (self.angle_hint * numerator) // denominator,
        )

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Dimensions):
            return (
                self.length == other.length
                and self.time == other.time
                and self.mass == other.mass
                and self.current == other.current
                and self.temperature == other.temperature
                and self.moles_hint == other.moles_hint
                and self.angle_hint == other.angle_hint
            )

        return NotImplemented

    def __hash__(self):
        """Unique representation of units using Godel like encoding"""

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

        return (
            2**two_powers
            * 3 ** abs(self.length)
            * 5 ** abs(self.time)
            * 7 ** abs(self.mass)
            * 11 ** abs(self.current)
            * 13 ** abs(self.temperature)
            * 17 ** abs(self.moles_hint)
            * 19 ** abs(self.angle_hint)
        )

    def __repr__(self):
        tokens = []
        for name, size in [
            ("length", self.length),
            ("time", self.time),
            ("mass", self.mass),
            ("current", self.current),
            ("temperature", self.temperature),
            ("amount", self.moles_hint),
            ("angle", self.angle_hint),
        ]:

            if size == 0:
                pass
            elif size == 1:
                tokens.append(f"{name}")
            else:
                tokens.append(f"{name}{int_as_unicode_superscript(size)}")

        return " ".join(tokens)

    def si_repr(self):
        tokens = []
        for name, size in [
            ("kg", self.mass),
            ("m", self.length),
            ("s", self.time),
            ("A", self.current),
            ("K", self.temperature),
            ("mol", self.moles_hint),
        ]:

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

        return "".join(tokens)


class Unit:
    def __init__(self, si_scaling_factor: float, dimensions: Dimensions):

        self.scale = si_scaling_factor
        self.dimensions = dimensions

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
            return Unit(other / self.scale, self.dimensions**-1)
        else:
            return NotImplemented

    def __pow__(self, power: int | float):
        if not isinstance(power, int | float):
            return NotImplemented

        return Unit(self.scale**power, self.dimensions**power)

    def equivalent(self: Self, other: "Unit"):
        return self.dimensions == other.dimensions

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(other, Unit):
            return (
                self.equivalent(other)
                and np.abs(np.log(self.scale / other.scale)) < 1e-5
            )
        return False

    def si_equivalent(self):
        """Get the SI unit corresponding to this unit"""
        return Unit(1, self.dimensions)

    def __repr__(self):
        if self.scale == 1:
            # We're in SI
            return self.dimensions.si_repr()

        else:
            return f"Unit[{self.scale}, {self.dimensions}]"


class NamedUnit(Unit):
    """Units, but they have a name, and a symbol

    :si_scaling_factor: Number of these units per SI equivalent
    :param dimensions: Dimensions object representing the dimensionality of these units
    :param name: Name of unit - string without unicode
    :param ascii_symbol: Symbol for unit without unicode
    :param symbol: Unicode symbol
    """

    def __init__(
        self,
        si_scaling_factor: float,
        dimensions: Dimensions,
        name: str | None = None,
        ascii_symbol: str | None = None,
        latex_symbol: str | None = None,
        symbol: str | None = None,
    ):

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
                return (
                    self.name == other
                    or self.name == f"{other}s"
                    or self.ascii_symbol == other
                    or self.symbol == other
                )
            case NamedUnit():
                return (
                    self.name == other.name
                    and self.ascii_symbol == other.ascii_symbol
                    and self.symbol == other.symbol
                )
            case Unit():
                return (
                    self.equivalent(other)
                    and np.abs(np.log(self.scale / other.scale)) < 1e-5
                )
            case _:
                return False

    def startswith(self, prefix: str) -> bool:
        """Check if any representation of the unit begins with the prefix string"""
        prefix = prefix.lower()
        return (
            (self.name is not None and self.name.lower().startswith(prefix))
            or (
                self.ascii_symbol is not None
                and self.ascii_symbol.lower().startswith(prefix)
            )
            or (self.symbol is not None and self.symbol.lower().startswith(prefix))
        )


class UnknownUnit(NamedUnit):
    """A unit for an unknown quantity

    While this library attempts to handle all known SI units, it is
    likely that users will want to express quantities of arbitrary
    units (for example, calculating donuts per person for a meeting).
    The arbitrary unit allows for these unforseeable quantities."""

    def __init__(
        self,
        numerator: str | list[str] | dict[str, int | float],
        denominator: None | list[str] | dict[str, int | float] = None,
    ):
        if numerator is None:
            return TypeError
        self._numerator = UnknownUnit._parse_arg(numerator)
        self._denominator = UnknownUnit._parse_arg(denominator)
        self._unit = NamedUnit(1, Dimensions(), "")  # Unitless

        super().__init__(
            si_scaling_factor=1, dimensions=self._unit.dimensions, symbol=self._name()
        )

    @staticmethod
    def _parse_arg(
        arg: str | list[str] | dict[str, int | float] | None,
    ) -> dict[str, int | float]:
        """Parse the different possibilities for constructor arguments

        Both the numerator and the denominator could be a string, a
        list of strings, or a dict.  Parse any of these values into a
        dictionary of names and powers.

        """
        match arg:
            case None:
                return {}
            case str():
                return {UnknownUnit._valid_name(arg): 1}
            case list():
                result: dict[str, int | float] = {}
                for key in arg:
                    if key in result:
                        result[key] += 1
                    else:
                        UnknownUnit._valid_name(key)
                        result[key] = 1
                return result
            case dict():
                for key in arg:
                    UnknownUnit._valid_name(key)
                return arg
            case _:
                raise TypeError

    @staticmethod
    def _valid_name(name: str) -> str:
        """Confirms that the name of a unit is appropriate

        This mostly confirms that the unit does not contain math
        operators that would act on other units, like / or ^
        """

        if re.search(r"[*/^\s]", name):
            raise RuntimeError(
                f'Unit name "{name}" contains invalid characters (*, /, ^, or whitespace)'
            )

        return name

    def _name(self):
        num = []
        for key, value in self._numerator.items():
            if value == 1:
                num.append(key)
            else:
                num.append(f"{key}^{value}")
        den = []
        for key, value in self._denominator.items():
            den.append(f"{key}^{-value}")
        num.sort()
        den.sort()
        return " ".join(num + den)

    def __eq__(self, other):
        match other:
            case UnknownUnit():
                return (
                    self._numerator == other._numerator
                    and self._denominator == other._denominator
                    and self._unit == other._unit
                )
            case Unit():
                return (
                    not self._numerator
                    and not self._denominator
                    and self._unit == other
                )
            case _:
                return False

    def __mul__(self: Self, other: "Unit"):
        match other:
            case UnknownUnit():
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
                result = UnknownUnit(num, den)
                result._unit *= other._unit
                return result._reduce()
            case NamedUnit() | Unit() | int() | float():
                result = UnknownUnit(self._numerator, self._denominator)
                result._unit *= other
                return result
            case _:
                return NotImplemented

    def __rmul__(self: Self, other):
        return self * other

    def __truediv__(self: Self, other: "Unit") -> "UnknownUnit":
        match other:
            case UnknownUnit():
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
                result = UnknownUnit(num, den)
                result._unit /= other._unit
                return result._reduce()
            case NamedUnit() | Unit() | int() | float():
                result = UnknownUnit(self._numerator, self._denominator)
                result._unit /= other
                return result
            case _:
                return NotImplemented

    def __rtruediv__(self: Self, other: "Unit") -> "UnknownUnit":
        return (self / other) ** -1

    def __pow__(self, power: int | float) -> "UnknownUnit":
        match power:
            case int() | float():
                num = {key: value * power for key, value in self._numerator.items()}
                den = {key: value * power for key, value in self._denominator.items()}
                if power < 0:
                    num, den = den, num
                    num = {k: -v for k, v in num.items()}
                    den = {k: -v for k, v in den.items()}

                result = UnknownUnit(num, den)
                result._unit = self._unit**power
                return result
            case _:
                return NotImplemented

    def equivalent(self: Self, other: "Unit"):
        match other:
            case UnknownUnit():
                return (
                    self._unit.equivalent(other._unit)
                    and sorted(self._numerator) == sorted(other._numerator)
                    and sorted(self._denominator) == sorted(other._denominator)
                )
            case _:
                return False

    def _reduce(self):
        """Remove redundant units"""
        for k in self._denominator:
            if k in self._numerator:
                common = min(self._numerator[k], self._denominator[k])
                self._numerator[k] -= common
                self._denominator[k] -= common
        dead_nums = [k for k in self._numerator if self._numerator[k] == 0]
        for k in dead_nums:
            del self._numerator[k]
        dead_dens = [k for k in self._denominator if self._denominator[k] == 0]
        for k in dead_dens:
            del self._denominator[k]
        return self

    def __str__(self):
        result = self._name()
        if type(self._unit) is NamedUnit and self._unit.name.strip():
            result += f" {self._unit.name.strip()}"
        if type(self._unit) is Unit and str(self._unit).strip():
            result += f" {str(self._unit).strip()}"
        return result

    def __repr__(self):
        return str(self)


class UnitGroup:
    """A group of units that all have the same dimensionality"""

    def __init__(self, name: str, units: list[NamedUnit]):
        self.name = name
        self.units = sorted(units, key=lambda unit: unit.scale)


Magnitude = namedtuple(
    "Magnitude", ["symbol", "special_symbol", "latex_symbol", "name", "scale"]
)

bigger_magnitudes: list[Magnitude] = [
    Magnitude("E", None, None, "exa", 1e18),
    Magnitude("P", None, None, "peta", 1e15),
    Magnitude("T", None, None, "tera", 1e12),
    Magnitude("G", None, None, "giga", 1e9),
    Magnitude("M", None, None, "mega", 1e6),
    Magnitude("k", None, None, "kilo", 1e3),
]

smaller_magnitudes: list[Magnitude] = [
    Magnitude("m", None, None, "milli", 1e-3),
    Magnitude("u", "µ", r"\mu", "micro", 1e-6),
    Magnitude("n", None, None, "nano", 1e-9),
    Magnitude("p", None, None, "pico", 1e-12),
    Magnitude("f", None, None, "femto", 1e-15),
    Magnitude("a", None, None, "atto", 1e-18),
]

unusual_magnitudes: list[Magnitude] = [
    Magnitude("d", None, None, "deci", 1e-1),
    Magnitude("c", None, None, "centi", 1e-2),
]

all_magnitudes = bigger_magnitudes + smaller_magnitudes

UnitData = namedtuple(
    "UnitData",
    [
        "symbol",
        "special_symbol",
        "latex_symbol",
        "singular",
        "plural",
        "scale",
        "length",
        "time",
        "mass",
        "current",
        "temperature",
        "moles_hint",
        "angle_hint",
        "magnitudes",
    ],
)

# Length, time, mass, current, temperature
base_si_units: list[UnitData] = [
    UnitData(
        "m",
        None,
        None,
        "meter",
        "meters",
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        all_magnitudes + unusual_magnitudes,
    ),
    UnitData(
        "s", None, None, "second", "seconds", 1, 0, 1, 0, 0, 0, 0, 0, smaller_magnitudes
    ),
    UnitData(
        "g", None, None, "gram", "grams", 1e-3, 0, 0, 1, 0, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "A", None, None, "ampere", "amperes", 1, 0, 0, 0, 1, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "K", None, None, "kelvin", "kelvin", 1, 0, 0, 0, 0, 1, 0, 0, all_magnitudes
    ),
]

derived_si_units: list[UnitData] = [
    UnitData(
        "Hz", None, None, "hertz", "hertz", 1, 0, -1, 0, 0, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "N", None, None, "newton", "newtons", 1, 1, -2, 1, 0, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "Pa", None, None, "pascal", "pascals", 1, -1, -2, 1, 0, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "J", None, None, "joule", "joules", 1, 2, -2, 1, 0, 0, 0, 0, all_magnitudes
    ),
    UnitData("W", None, None, "watt", "watts", 1, 2, -3, 1, 0, 0, 0, 0, all_magnitudes),
    UnitData(
        "C", None, None, "coulomb", "coulombs", 1, 0, 1, 0, 1, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "V", None, None, "volts", "volts", 1, 2, -3, 1, -1, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "Ohm", "Ω", r"\Omega", "ohm", "ohms", 1, 2, -3, 1, -2, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "F", None, None, "farad", "farads", 1, -2, 4, -1, 2, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "S", None, None, "siemens", "siemens", 1, -2, 3, -1, 2, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "Wb", None, None, "weber", "webers", 1, 2, -2, 1, -1, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "T", None, None, "tesla", "tesla", 1, 0, -2, 1, -1, 0, 0, 0, all_magnitudes
    ),
    UnitData(
        "H", None, None, "henry", "henry", 1, 2, -2, 1, -2, 0, 0, 0, all_magnitudes
    ),
]

non_si_dimensioned_units: list[UnitData] = [
    UnitData(
        "Ang", "Å", r"\AA", "angstrom", "angstroms", 1e-10, 1, 0, 0, 0, 0, 0, 0, []
    ),
    UnitData("micron", None, None, "micron", "microns", 1e-6, 1, 0, 0, 0, 0, 0, 0, []),
    UnitData("min", None, None, "minute", "minutes", 60, 0, 1, 0, 0, 0, 0, 0, []),
    UnitData(
        "rpm",
        None,
        None,
        "revolutions per minute",
        "revolutions per minute",
        1 / 60,
        0,
        -1,
        0,
        0,
        0,
        0,
        0,
        [],
    ),
    UnitData("h", None, None, "hour", "hours", 3600, 0, 1, 0, 0, 0, 0, 0, []),
    UnitData("d", None, None, "day", "days", 3600 * 24, 0, 1, 0, 0, 0, 0, 0, []),
    UnitData(
        "y", None, None, "year", "years", 3600 * 24 * 365.2425, 0, 1, 0, 0, 0, 0, 0, []
    ),
    UnitData(
        "deg", None, None, "degree", "degrees", 180 / np.pi, 0, 0, 0, 0, 0, 0, 1, []
    ),
    UnitData("rad", None, None, "radian", "radians", 1, 0, 0, 0, 0, 0, 0, 1, []),
    UnitData(
        "rot", None, None, "rotation", "rotations", 2 * np.pi, 0, 0, 0, 0, 0, 0, 1, []
    ),
    UnitData("sr", None, None, "stradian", "stradians", 1, 0, 0, 0, 0, 0, 0, 2, []),
    UnitData("l", None, None, "litre", "litres", 1e-3, 3, 0, 0, 0, 0, 0, 0, []),
    UnitData(
        "eV",
        None,
        None,
        "electronvolt",
        "electronvolts",
        1.602176634e-19,
        2,
        -2,
        1,
        0,
        0,
        0,
        0,
        all_magnitudes,
    ),
    UnitData(
        "au",
        None,
        None,
        "atomic mass unit",
        "atomic mass units",
        1.660538921e-27,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        [],
    ),
    UnitData(
        "mol",
        None,
        None,
        "mole",
        "moles",
        6.02214076e23,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        smaller_magnitudes,
    ),
    UnitData(
        "kgForce", None, None, "kg force", "kg force", 9.80665, 1, -2, 1, 0, 0, 0, 0, []
    ),
    UnitData(
        "C", None, None, "degree Celsius", "degrees Celsius", 1, 0, 0, 0, 0, 1, 0, 0, []
    ),
    UnitData(
        "miles", None, None, "mile", "miles", 1760 * 3 * 0.3048, 1, 0, 0, 0, 0, 0, 0, []
    ),
    UnitData("yrd", None, None, "yard", "yards", 3 * 0.3048, 1, 0, 0, 0, 0, 0, 0, []),
    UnitData("ft", None, None, "foot", "feet", 0.3048, 1, 0, 0, 0, 0, 0, 0, []),
    UnitData("in", None, None, "inch", "inches", 0.0254, 1, 0, 0, 0, 0, 0, 0, []),
    UnitData("lb", None, None, "pound", "pounds", 0.45359237, 0, 0, 1, 0, 0, 0, 0, []),
    UnitData(
        "lbf",
        None,
        None,
        "pound force",
        "pounds force",
        4.448222,
        1,
        -2,
        1,
        0,
        0,
        0,
        0,
        [],
    ),
    UnitData(
        "oz", None, None, "ounce", "ounces", 0.45359237 / 16, 0, 0, 1, 0, 0, 0, 0, []
    ),
    UnitData(
        "psi",
        None,
        None,
        "pound force per square inch",
        "pounds force per square inch",
        4.448222 / (0.0254**2),
        -1,
        -2,
        1,
        0,
        0,
        0,
        0,
        [],
    ),
]

non_si_dimensionless_units: list[UnitData] = [
    UnitData("none", None, None, "none", "none", 1, 0, 0, 0, 0, 0, 0, 0, []),
    UnitData(
        "percent", "%", r"\%", "percent", "percent", 0.01, 0, 0, 0, 0, 0, 0, 0, []
    ),
]

non_si_units: list[UnitData] = non_si_dimensioned_units + non_si_dimensionless_units

# TODO:
# Add Hartree? Rydberg? Bohrs?
# Add CGS

# Two stages of aliases, to make sure units don't get lost

aliases_1 = {"A": ["Amps", "amps"], "C": ["Coulombs", "coulombs"]}

aliases_2 = {
    "y": ["yr", "year"],
    "d": ["day"],
    "h": ["hr", "hour"],
    "Ang": ["A", "Å"],
    "au": ["amu"],
    "percent": ["%"],
    "deg": ["degr", "Deg", "degree", "degrees", "Degrees"],
    "none": ["Counts", "counts", "cnts", "Cnts", "a.u.", "fraction", "Fraction"],
    "K": ["C"],  # Ugh, cansas
}


all_units: list[UnitData] = base_si_units + derived_si_units + non_si_units

encoding = "utf-8"


def format_name(name: str):
    return name.lower().replace(" ", "_")


this = sys.modules[__name__]

### Begin live patching

symbol_lookup: dict[str, NamedUnit] = {}
unit_types_temp = defaultdict(list)  # Keep track of unit types
unit_types = defaultdict(list)

for unit_def in all_units:
    formatted_plural = format_name(unit_def.plural)
    formatted_singular = format_name(unit_def.singular)

    dimensions = Dimensions(
        unit_def.length,
        unit_def.time,
        unit_def.mass,
        unit_def.current,
        unit_def.temperature,
        unit_def.moles_hint,
        unit_def.angle_hint,
    )
    current_unit = NamedUnit(
        unit_def.scale,
        Dimensions(
            unit_def.length,
            unit_def.time,
            unit_def.mass,
            unit_def.current,
            unit_def.temperature,
            unit_def.moles_hint,
            unit_def.angle_hint,
        ),
        name=formatted_plural,
        ascii_symbol=unit_def.symbol,
        latex_symbol=unit_def.latex_symbol,
        symbol=(
            unit_def.symbol
            if unit_def.special_symbol is None
            else unit_def.special_symbol
        ),
    )
    setattr(this, formatted_plural, current_unit)

    symbol_lookup[unit_def.symbol] = current_unit
    if unit_def.special_symbol is not None:
        symbol_lookup[unit_def.special_symbol] = current_unit

    unit_types_temp[hash(dimensions)].append(
        (
            unit_def.symbol,
            unit_def.special_symbol,
            formatted_singular,
            formatted_plural,
            unit_def.scale,
            dimensions,
        )
    )

    unit_types[hash(dimensions)].append(formatted_plural)

    for mag in unit_def.magnitudes:
        # Work out the combined symbol, accounts for unicode or not
        combined_special_symbol = (
            mag.symbol if mag.special_symbol is None else mag.special_symbol
        ) + (
            unit_def.symbol
            if unit_def.special_symbol is None
            else unit_def.special_symbol
        )

        combined_symbol = mag.symbol + unit_def.symbol

        # Combined unit name
        combined_name_singular = f"{mag.name}{formatted_singular}"
        combined_name_plural = f"{mag.name}{formatted_plural}"

        combined_scale = unit_def.scale * mag.scale

        latex_symbol = None
        if unit_def.latex_symbol is not None and mag.latex_symbol is not None:
            latex_symbol = f"{{{mag.latex_symbol}}}{unit_def.latex_symbol}"
        elif unit_def.latex_symbol is not None:
            latex_symbol = f"{mag.symbol}{unit_def.latex_symbol}"
        elif mag.latex_symbol is not None:
            latex_symbol = f"{{{mag.latex_symbol}}}{unit_def.symbol}"

        # Units
        dimensions = Dimensions(
            unit_def.length,
            unit_def.time,
            unit_def.mass,
            unit_def.current,
            unit_def.temperature,
            unit_def.moles_hint,
            unit_def.angle_hint,
        )

        current_unit = NamedUnit(
            combined_scale,
            Dimensions(
                unit_def.length,
                unit_def.time,
                unit_def.mass,
                unit_def.current,
                unit_def.temperature,
                unit_def.moles_hint,
                unit_def.angle_hint,
            ),
            name=combined_name_plural,
            ascii_symbol=combined_symbol,
            latex_symbol=latex_symbol,
            symbol=combined_special_symbol,
        )
        setattr(this, combined_name_plural, current_unit)

        symbol_lookup[combined_symbol] = current_unit
        symbol_lookup[combined_special_symbol] = current_unit

        unit_types_temp[hash(dimensions)].append(
            (
                combined_symbol,
                combined_special_symbol,
                combined_name_singular,
                combined_name_plural,
                combined_scale,
                dimensions,
            )
        )

        unit_types[hash(dimensions)].append(combined_name_plural)


# Higher dimensioned types
#

length_units = unit_types_temp[hash(Dimensions(length=1))]
time_units = unit_types_temp[hash(Dimensions(time=1))]
mass_units = unit_types_temp[hash(Dimensions(mass=1))]
amount_units = unit_types_temp[hash(Dimensions(moles_hint=1))]


# Length based
for symbol, special_symbol, singular, plural, scale, _ in length_units:
    for prefix, power, name, unicode_suffix in [
        ("square_", 2, plural, "²"),
        ("cubic_", 3, plural, "³"),
        ("per_", -1, singular, "⁻¹"),
        ("per_square_", -2, singular, "⁻²"),
        ("per_cubic_", -3, singular, "⁻³"),
    ]:
        dimensions = Dimensions(length=power)
        unit_name = prefix + name
        unit_special_symbol = (
            symbol if special_symbol is None else special_symbol
        ) + unicode_suffix
        unit_symbol = symbol + f"^{power}"
        setattr(
            this,
            unit_name,
            NamedUnit(
                scale**power,
                Dimensions(length=power),
                name=unit_name,
                ascii_symbol=unit_symbol,
                symbol=unit_special_symbol,
            ),
        )

        unit_types[hash(dimensions)].append(unit_name)

# Speed and acceleration
for (
    length_symbol,
    length_special_symbol,
    _,
    length_name,
    length_scale,
    _,
) in length_units:
    for time_symbol, time_special_symbol, time_name, _, time_scale, _ in time_units:
        speed_name = length_name + "_per_" + time_name
        accel_name = length_name + "_per_square_" + time_name

        speed_dimensions = Dimensions(length=1, time=-1)
        accel_dimensions = Dimensions(length=1, time=-2)

        length_special = (
            length_special_symbol
            if length_special_symbol is not None
            else length_symbol
        )
        time_special = (
            time_special_symbol if time_special_symbol is not None else time_symbol
        )

        setattr(
            this,
            speed_name,
            NamedUnit(
                length_scale / time_scale,
                Dimensions(length=1, time=-1),
                name=speed_name,
                ascii_symbol=f"{length_symbol}/{time_symbol}",
                symbol=f"{length_special}{time_special}⁻¹",
            ),
        )
        setattr(
            this,
            accel_name,
            NamedUnit(
                length_scale / time_scale**2,
                Dimensions(length=1, time=-2),
                name=speed_name,
                ascii_symbol=f"{length_symbol}/{time_symbol}^2",
                symbol=f"{length_special}{time_special}⁻²",
            ),
        )

        unit_types[hash(speed_dimensions)].append(speed_name)
        unit_types[hash(accel_dimensions)].append(accel_name)

# Density
for (
    length_symbol,
    length_special_symbol,
    length_name,
    _,
    length_scale,
    _,
) in length_units:
    for mass_symbol, mass_special_symbol, _, mass_name, mass_scale, _ in mass_units:
        name = mass_name + "_per_cubic_" + length_name

        dimensions = Dimensions(length=-3, mass=1)

        mass_special = (
            mass_symbol if mass_special_symbol is None else mass_special_symbol
        )
        length_special = (
            length_symbol if length_special_symbol is None else length_special_symbol
        )

        setattr(
            this,
            name,
            NamedUnit(
                mass_scale / length_scale**3,
                Dimensions(length=-3, mass=1),
                name=name,
                ascii_symbol=f"{mass_symbol} {length_symbol}^-3",
                symbol=f"{mass_special}{length_special}⁻³",
            ),
        )

        unit_types[hash(dimensions)].append(name)

# Concentration
for (
    length_symbol,
    length_special_symbol,
    length_name,
    _,
    length_scale,
    _,
) in length_units:
    for (
        amount_symbol,
        amount_special_symbol,
        _,
        amount_name,
        amount_scale,
        _,
    ) in amount_units:
        name = amount_name + "_per_cubic_" + length_name

        dimensions = Dimensions(length=-3, moles_hint=1)

        length_special = (
            length_symbol if length_special_symbol is None else length_special_symbol
        )
        amount_special = (
            amount_symbol if amount_special_symbol is None else amount_special_symbol
        )

        setattr(
            this,
            name,
            NamedUnit(
                amount_scale / length_scale**3,
                Dimensions(length=-3, moles_hint=1),
                name=name,
                ascii_symbol=f"{amount_symbol} {length_symbol}^-3",
                symbol=f"{amount_special}{length_special}⁻³",
            ),
        )

        unit_types[hash(dimensions)].append(name)

# TODO: Torque, Momentum, Entropy

#
# Add aliases to symbol lookup table
#

# Apply the alias transforms sequentially
for aliases in [aliases_1, aliases_2]:
    for base_name in aliases:
        alias_list = aliases[base_name]
        for alias in alias_list:
            symbol_lookup[alias] = symbol_lookup[base_name]

#
# Collections of units by type
#

dimension_names = [
    ("length", Dimensions(length=1)),
    ("area", Dimensions(length=2)),
    ("volume", Dimensions(length=3)),
    ("inverse_length", Dimensions(length=-1)),
    ("inverse_area", Dimensions(length=-2)),
    ("inverse_volume", Dimensions(length=-3)),
    ("time", Dimensions(time=1)),
    ("rate", Dimensions(time=-1)),
    ("speed", Dimensions(length=1, time=-1)),
    ("acceleration", Dimensions(length=1, time=-2)),
    ("density", Dimensions(length=-3, mass=1)),
    ("force", Dimensions(1, -2, 1, 0, 0)),
    ("pressure", Dimensions(-1, -2, 1, 0, 0)),
    ("energy", Dimensions(2, -2, 1, 0, 0)),
    ("power", Dimensions(2, -3, 1, 0, 0)),
    ("charge", Dimensions(0, 1, 0, 1, 0)),
    ("potential", Dimensions(2, -3, 1, -1, 0)),
    ("resistance", Dimensions(2, -3, 1, -2, 0)),
    ("capacitance", Dimensions(-2, 4, -1, 2, 0)),
    ("conductance", Dimensions(-2, 3, -1, 2, 0)),
    ("magnetic_flux", Dimensions(2, -2, 1, -1, 0)),
    ("magnetic_flux_density", Dimensions(0, -2, 1, -1, 0)),
    ("inductance", Dimensions(2, -2, 1, -2, 0)),
    ("temperature", Dimensions(temperature=1)),
    ("dimensionless", Dimensions()),
    ("angle", Dimensions(angle_hint=1)),
    ("solid_angle", Dimensions(angle_hint=2)),
    ("amount", Dimensions(moles_hint=1)),
    ("concentration", Dimensions(length=-3, moles_hint=1)),
]

for dimension_name, dimensions in dimension_names:
    setattr(
        this,
        dimension_name,
        UnitGroup(
            name=dimension_name,
            units=[getattr(this, x) for x in unit_types[hash(dimensions)]],
        ),
    )

setattr(this, "unit_group_names", [x for x, _ in dimension_names])

setattr(this, "unit_groups", {x: getattr(this, x) for x, _ in dimension_names})
