"""
Builds a data file containing details of units
"""

from collections import defaultdict, namedtuple

import numpy as np
from _autogen_warning import warning_text
from _units_base import Dimensions

Magnitude = namedtuple("Magnitude", ["symbol", "special_symbol", "latex_symbol", "name", "scale"])

bigger_magnitudes: list[Magnitude] = [
    Magnitude("E", None, None, "exa", 1e18),
    Magnitude("P", None, None, "peta", 1e15),
    Magnitude("T", None, None, "tera", 1e12),
    Magnitude("G", None, None, "giga", 1e9),
    Magnitude("M", None, None, "mega", 1e6),
    Magnitude("k", None, None, "kilo", 1e3) ]

smaller_magnitudes: list[Magnitude] = [
    Magnitude("m", None, None, "milli", 1e-3),
    Magnitude("u", "µ", r"\mu", "micro", 1e-6),
    Magnitude("n", None, None, "nano", 1e-9),
    Magnitude("p", None, None, "pico", 1e-12),
    Magnitude("f", None, None, "femto", 1e-15),
    Magnitude("a", None, None, "atto", 1e-18)]

unusual_magnitudes: list[Magnitude] = [
    Magnitude("d", None, None, "deci", 1e-1),
    Magnitude("c", None, None, "centi", 1e-2)
]

all_magnitudes = bigger_magnitudes + smaller_magnitudes

UnitData = namedtuple("UnitData", ["symbol", "special_symbol", "latex_symbol", "singular", "plural", "scale", "length", "time", "mass", "current", "temperature", "moles_hint", "angle_hint", "magnitudes"])

# Length, time, mass, current, temperature
base_si_units = [
    UnitData("m", None, None, "meter", "meters", 1, 1, 0, 0, 0, 0, 0, 0, all_magnitudes + unusual_magnitudes),
    UnitData("s", None, None, "second", "seconds", 1, 0, 1, 0, 0, 0, 0, 0, smaller_magnitudes),
    UnitData("g", None, None, "gram", "grams", 1e-3, 0, 0, 1, 0, 0, 0, 0, all_magnitudes),
    UnitData("A", None, None, "ampere", "amperes", 1, 0, 0, 0, 1, 0, 0, 0, all_magnitudes),
    UnitData("K", None, None, "kelvin", "kelvin", 1, 0, 0, 0, 0, 1, 0, 0, all_magnitudes) ]

derived_si_units = [
    UnitData("Hz", None, None, "hertz", "hertz", 1, 0, -1, 0, 0, 0, 0, 0, all_magnitudes),
    UnitData("N", None, None, "newton", "newtons", 1, 1, -2, 1, 0, 0, 0, 0, all_magnitudes),
    UnitData("Pa", None, None, "pascal", "pascals", 1, -1, -2, 1, 0, 0, 0, 0, all_magnitudes),
    UnitData("J", None, None, "joule", "joules", 1, 2, -2, 1, 0, 0, 0, 0, all_magnitudes),
    UnitData("W", None, None, "watt", "watts", 1, 2, -3, 1, 0, 0, 0, 0, all_magnitudes),
    UnitData("C", None, None, "coulomb", "coulombs", 1, 0, 1, 0, 1, 0, 0, 0, all_magnitudes),
    UnitData("V", None, None, "volts", "volts", 1, 2, -3, 1, -1, 0, 0, 0, all_magnitudes),
    UnitData("Ohm", "Ω", r"\Omega", "ohm", "ohms", 1, 2, -3, 1, -2, 0, 0, 0, all_magnitudes),
    UnitData("F", None, None, "farad", "farads", 1, -2, 4, -1, 2, 0, 0, 0, all_magnitudes),
    UnitData("S", None, None, "siemens", "siemens", 1, -2, 3, -1, 2, 0, 0, 0, all_magnitudes),
    UnitData("Wb", None, None, "weber", "webers", 1, 2, -2, 1, -1, 0, 0, 0, all_magnitudes),
    UnitData("T", None, None, "tesla", "tesla", 1, 0, -2, 1, -1, 0, 0, 0, all_magnitudes),
    UnitData("H", None, None, "henry", "henry", 1, 2, -2, 1, -2, 0, 0, 0, all_magnitudes),
]

non_si_dimensioned_units: list[tuple[str, str | None, str, str, float, int, int, int, int, int, int, int, list]] = [
    UnitData("Ang", "Å", r"\AA", "angstrom", "angstroms", 1e-10, 1, 0, 0, 0, 0, 0, 0, []),
    UnitData("micron", None, None, "micron", "microns", 1e-6, 1, 0, 0, 0, 0, 0, 0, []),
    UnitData("min", None, None, "minute", "minutes", 60, 0, 1, 0, 0, 0, 0, 0, []),
    UnitData("h", None, None, "hour", "hours", 3600, 0, 1, 0, 0, 0, 0, 0, []),
    UnitData("d", None, None, "day", "days", 3600*24, 0, 1, 0, 0, 0, 0, 0, []),
    UnitData("y", None, None, "year", "years", 3600*24*365.2425, 0, 1, 0, 0, 0, 0, 0, []),
    UnitData("deg", None, None, "degree", "degrees", 180/np.pi, 0, 0, 0, 0, 0, 0, 1, []),
    UnitData("rad", None, None, "radian", "radians", 1, 0, 0, 0, 0, 0, 0, 1, []),
    UnitData("rot", None, None, "rotation", "rotations", 2*np.pi, 0, 0, 0, 0, 0, 0, 1, []),
    UnitData("sr", None, None, "stradian", "stradians", 1, 0, 0, 0, 0, 0, 0, 2, []),
    UnitData("l", None, None, "litre", "litres", 1e-3, 3, 0, 0, 0, 0, 0, 0, []),
    UnitData("eV", None, None, "electronvolt", "electronvolts", 1.602176634e-19, 2, -2, 1, 0, 0, 0, 0, all_magnitudes),
    UnitData("au", None, None, "atomic mass unit", "atomic mass units", 1.660538921e-27, 0, 0, 1, 0, 0, 0, 0, []),
    UnitData("mol", None, None, "mole", "moles", 6.02214076e23, 0, 0, 0, 0, 0, 1, 0, smaller_magnitudes),
    UnitData("kgForce", None, None, "kg force", "kg force",  9.80665, 1, -2, 1, 0, 0, 0, 0, []),
    UnitData("C", None, None, "degree Celsius", "degrees Celsius", 1, 0, 0, 0, 0, 1, 0, 0, []),
    UnitData("miles", None, None, "mile", "miles", 1760*3*0.3048, 1, 0, 0, 0, 0, 0, 0, []),
    UnitData("yrd", None, None, "yard", "yards", 3*0.3048, 1, 0, 0, 0, 0, 0, 0, []),
    UnitData("ft", None, None, "foot", "feet", 0.3048, 1, 0, 0, 0, 0, 0, 0, []),
    UnitData("in", None, None, "inch", "inches", 0.0254, 1, 0, 0, 0, 0, 0, 0, []),
    UnitData("lb", None, None, "pound", "pounds", 0.45359237, 0, 0, 1, 0, 0, 0, 0, []),
    UnitData("lbf", None, None, "pound force", "pounds force", 4.448222, 1, -2, 1, 0, 0, 0, 0, []),
    UnitData("oz", None, None, "ounce", "ounces", 0.45359237/16, 0, 0, 1, 0, 0, 0, 0, []),
    UnitData("psi", None, None, "pound force per square inch", "pounds force per square inch", 4.448222/(0.0254**2), -1, -2, 1, 0, 0, 0, 0, []),
]

non_si_dimensionless_units: list[tuple[str, str | None, str, str, float, int, int, int, int, int, int, int, list]] = [
    UnitData("none", None, None, "none", "none", 1, 0, 0, 0, 0, 0, 0, 0, []),
    UnitData("percent", "%", r"\%", "percent", "percent", 0.01, 0, 0, 0, 0, 0, 0, 0, [])
]

non_si_units = non_si_dimensioned_units + non_si_dimensionless_units

# TODO:
# Add Hartree? Rydberg? Bohrs?
# Add CGS

# Two stages of aliases, to make sure units don't get lost

aliases_1 = {
    "A": ["Amps", "amps"],
    "C": ["Coulombs", "coulombs"]
}

aliases_2 = {
    "y": ["yr", "year"],
    "d": ["day"],
    "h": ["hr", "hour"],
    "Ang": ["A", "Å"],
    "au": ["amu"],
    "percent": ["%"],
    "deg": ["degr", "Deg", "degree", "degrees", "Degrees"],
    "none": ["Counts", "counts", "cnts", "Cnts", "a.u.", "fraction", "Fraction"],
    "K": ["C"] # Ugh, cansas
}



all_units = base_si_units + derived_si_units + non_si_units

encoding = "utf-8"

def format_name(name: str):
    return name.lower().replace(" ", "_")

with open("units.py", 'w', encoding=encoding) as fid:

    # Write warning header
    fid.write('"""'+(warning_text%"_build_tables.py, _units_base.py")+'"""')

    # Write in class definitions
    fid.write("\n\n"
              "#\n"
              "# Included from _units_base.py\n"
              "#\n\n")

    with open("_units_base.py") as base:
        for line in base:
            # unicode_superscript is a local module when called from
            # _unit_tables.py but a submodule of sasdata.quantities
            # when called from units.py.  This condition patches the
            # line when the copy is made.
            if line.startswith("from unicode_superscript"):
                fid.write(line.replace("from unicode_superscript", "\nfrom sasdata.quantities.unicode_superscript"))
            else:
                fid.write(line)

    # Write in unit definitions
    fid.write("\n\n"
              "#\n"
              "# Specific units\n"
              "#\n\n")

    symbol_lookup = {}
    unit_types_temp = defaultdict(list) # Keep track of unit types
    unit_types = defaultdict(list)

    for unit_def in all_units:

        formatted_plural = format_name(unit_def.plural)
        formatted_singular = format_name(unit_def.singular)

        dimensions = Dimensions(unit_def.length, unit_def.time, unit_def.mass, unit_def.current, unit_def.temperature, unit_def.moles_hint, unit_def.angle_hint)
        fid.write(f"{formatted_plural} = NamedUnit({unit_def.scale}, Dimensions({unit_def.length}, {unit_def.time}, {unit_def.mass}, {unit_def.current}, {unit_def.temperature}, {unit_def.moles_hint}, {unit_def.angle_hint}),"
                      f"name='{formatted_plural}',"
                      f"ascii_symbol='{unit_def.symbol}',"
                      f"{'' if unit_def.latex_symbol is None else f"""latex_symbol=r'{unit_def.latex_symbol}',""" }"
                      f"symbol='{unit_def.symbol if unit_def.special_symbol is None else unit_def.special_symbol}')\n")

        symbol_lookup[unit_def.symbol] = formatted_plural
        if unit_def.special_symbol is not None:
            symbol_lookup[unit_def.special_symbol] = formatted_plural

        unit_types_temp[hash(dimensions)].append(
            (unit_def.symbol, unit_def.special_symbol, formatted_singular, formatted_plural, unit_def.scale, dimensions))

        unit_types[hash(dimensions)].append(formatted_plural)

        for mag in unit_def.magnitudes:

            # Work out the combined symbol, accounts for unicode or not
            combined_special_symbol = (mag.symbol if mag.special_symbol is None else mag.special_symbol) + \
                              (unit_def.symbol if unit_def.special_symbol is None else unit_def.special_symbol)

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
            dimensions = Dimensions(unit_def.length, unit_def.time, unit_def.mass, unit_def.current, unit_def.temperature, unit_def.moles_hint, unit_def.angle_hint)
            fid.write(f"{combined_name_plural} = NamedUnit({combined_scale}, "
                      f"Dimensions({unit_def.length}, {unit_def.time}, {unit_def.mass}, {unit_def.current}, {unit_def.temperature}, {unit_def.moles_hint}, {unit_def.angle_hint}),"
                      f"name='{combined_name_plural}',"
                      f"ascii_symbol='{combined_symbol}',"
                      f"{'' if latex_symbol is None else f"""latex_symbol=r'{latex_symbol}',""" }"
                      f"symbol='{combined_special_symbol}')\n")

            symbol_lookup[combined_symbol] = combined_name_plural
            symbol_lookup[combined_special_symbol] = combined_name_plural

            unit_types_temp[hash(dimensions)].append(
                (combined_symbol, combined_special_symbol, combined_name_singular,
                 combined_name_plural, combined_scale, dimensions))

            unit_types[hash(dimensions)].append(combined_name_plural)

    #
    # Higher dimensioned types
    #

    length_units = unit_types_temp[hash(Dimensions(length=1))]
    time_units = unit_types_temp[hash(Dimensions(time=1))]
    mass_units = unit_types_temp[hash(Dimensions(mass=1))]
    amount_units = unit_types_temp[hash(Dimensions(moles_hint=1))]

    # Length based
    for symbol, special_symbol, singular, plural, scale, _ in length_units:
        for prefix, power, name, unicode_suffix in [
              ("square_", 2, plural, '²'),
              ("cubic_", 3, plural, '³'),
              ("per_", -1, singular, '⁻¹'),
              ("per_square_", -2, singular,'⁻²'),
              ("per_cubic_", -3, singular,'⁻³')]:

            dimensions = Dimensions(length=power)
            unit_name = prefix + name
            unit_special_symbol = (symbol if special_symbol is None else special_symbol) + unicode_suffix
            unit_symbol = symbol + f"^{power}"
            fid.write(f"{unit_name} = NamedUnit({scale**power}, Dimensions(length={power}), "
                      f"name='{unit_name}', "
                      f"ascii_symbol='{unit_symbol}', "
                      f"symbol='{unit_special_symbol}')\n")

            unit_types[hash(dimensions)].append(unit_name)

    # Speed and acceleration
    for length_symbol, length_special_symbol, _, length_name, length_scale, _ in length_units:
        for time_symbol, time_special_symbol, time_name, _, time_scale, _ in time_units:
            speed_name = length_name + "_per_" + time_name
            accel_name = length_name + "_per_square_" + time_name

            speed_dimensions = Dimensions(length=1, time=-1)
            accel_dimensions = Dimensions(length=1, time=-2)

            length_special = length_special_symbol if length_special_symbol is not None else length_symbol
            time_special = time_special_symbol if time_special_symbol is not None else time_symbol

            fid.write(f"{speed_name} "
                      f"= NamedUnit({length_scale / time_scale}, "
                      f"Dimensions(length=1, time=-1), "
                      f"name='{speed_name}', "
                      f"ascii_symbol='{length_symbol}/{time_symbol}', "
                      f"symbol='{length_special}{time_special}⁻¹')\n")

            fid.write(f"{accel_name} = NamedUnit({length_scale / time_scale**2}, "
                      f"Dimensions(length=1, time=-2), "
                      f"name='{accel_name}', "
                      f"ascii_symbol='{length_symbol}/{time_symbol}^2', "
                      f"symbol='{length_special}{time_special}⁻²')\n")

            unit_types[hash(speed_dimensions)].append(speed_name)
            unit_types[hash(accel_dimensions)].append(accel_name)

    # Density
    for length_symbol, length_special_symbol, length_name, _, length_scale, _ in length_units:
        for mass_symbol, mass_special_symbol, _, mass_name, mass_scale, _ in mass_units:

            name = mass_name + "_per_cubic_" + length_name

            dimensions = Dimensions(length=-3, mass=1)

            mass_special = mass_symbol if mass_special_symbol is None else mass_special_symbol
            length_special = length_symbol if length_special_symbol is None else length_special_symbol

            fid.write(f"{name} "
                      f"= NamedUnit({mass_scale / length_scale**3}, "
                      f"Dimensions(length=-3, mass=1), "
                      f"name='{name}', "
                      f"ascii_symbol='{mass_symbol} {length_symbol}^-3', "
                      f"symbol='{mass_special}{length_special}⁻³')\n")

            unit_types[hash(dimensions)].append(name)

    # Concentration
    for length_symbol, length_special_symbol, length_name, _, length_scale, _ in length_units:
        for amount_symbol, amount_special_symbol, _, amount_name, amount_scale, _ in amount_units:

            name = amount_name + "_per_cubic_" + length_name

            dimensions = Dimensions(length=-3, moles_hint=1)

            length_special = length_symbol if length_special_symbol is None else length_special_symbol
            amount_special = amount_symbol if amount_special_symbol is None else amount_special_symbol

            fid.write(f"{name} "
                      f"= NamedUnit({amount_scale / length_scale**3}, "
                      f"Dimensions(length=-3, moles_hint=1), "
                      f"name='{name}', "
                      f"ascii_symbol='{amount_symbol} {length_symbol}^-3', "
                      f"symbol='{amount_special}{length_special}⁻³')\n")

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
    # Write out the symbol lookup table
    #
    fid.write("\n#\n# Lookup table from symbols to units\n#\n\n")
    fid.write("symbol_lookup = {\n")
    for k in symbol_lookup:
        if k != "none":
            fid.write(f'        "{k}": {symbol_lookup[k]},\n')
    fid.write("}\n\n")

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

    fid.write("\n#\n# Units by type\n#\n\n")

    for dimension_name, dimensions in dimension_names:


        fid.write(f"\n"
                  f"{dimension_name} = UnitGroup(\n"
                  f"  name = '{dimension_name}',\n"
                  f"  units = [\n")

        for unit_name in unit_types[hash(dimensions)]:
            fid.write("    " + unit_name + ",\n")

        fid.write("])\n")


    # List of dimensions
    fid.write("\n\n")
    fid.write("unit_group_names = [\n")
    for dimension_name, _ in dimension_names:
        fid.write(f"    '{dimension_name}',\n")
    fid.write("]\n\n")

    fid.write("unit_groups = {\n")
    for dimension_name, _ in dimension_names:
        fid.write(f"    '{dimension_name}': {dimension_name},\n")
    fid.write("}\n\n")


with open("accessors.py", 'w', encoding=encoding) as fid:


    fid.write('"""'+(warning_text%"_build_tables.py, _accessor_base.py")+'"""\n\n')

    with open("_accessor_base.py") as base:
        for line in base:
            fid.write(line)

    for dimension_name, dimensions in dimension_names:

        accessor_name = dimension_name.capitalize().replace("_", "") + "Accessor"

        fid.write(f"\n"
                  f"class {accessor_name}[T](QuantityAccessor[T]):\n"
                  f"    dimension_name = '{dimension_name}'\n"
                  f"\n")

        for unit_name in unit_types[hash(dimensions)]:
            fid.write(f"    @property\n"
                      f"    def {unit_name}(self) -> T:\n"
                      f"        quantity = self.quantity\n"
                      f"        if quantity is None:\n"
                      f"            return None\n"
                      f"        else:\n"
                      f"            return quantity.in_units_of(units.{unit_name})\n"
                      f"\n")

        fid.write("\n")

with open("si.py", 'w') as fid:

    si_unit_names = [values.plural for values in base_si_units + derived_si_units if values.plural != "grams"] + ["kilograms"]
    si_unit_names.sort()

    fid.write('"""'+(warning_text%"_build_tables.py")+'"""\n\n')
    fid.write("from sasdata.quantities.units import (\n")

    for name in si_unit_names:
        fid.write(f"    {name},\n")

    fid.write(")\n")
    fid.write("\nall_si = [\n")

    for name in si_unit_names:
        fid.write(f"    {name},\n")

    fid.write("]\n")
