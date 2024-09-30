"""
Builds a data file containing details of units
"""

import numpy as np
from collections import defaultdict
from _units_base import Dimensions, Unit
from _autogen_warning import warning_text

bigger_magnitudes = [
    ("E", None, "exa", 1e18),
    ("P", None, "peta", 1e15),
    ("T", None, "tera", 1e12),
    ("G", None, "giga", 1e9),
    ("M", None, "mega", 1e6),
    ("k", None, "kilo", 1e3) ]

smaller_magnitudes = [
    ("m", None, "milli", 1e-3),
    ("u", "µ", "micro", 1e-6),
    ("n", None, "nano", 1e-9),
    ("p", None, "pico", 1e-12),
    ("f", None, "femto", 1e-15),
    ("a", None, "atto", 1e-18)]

unusual_magnitudes = [
    ("d", None, "deci", 1e-1),
    ("c", None, "centi", 1e-2)
]

all_magnitudes = bigger_magnitudes + smaller_magnitudes

# Length, time, mass, current, temperature
base_si_units = [
    ("m", None, "meter", "meters", 1, 1, 0, 0, 0, 0, 0, 0, all_magnitudes + unusual_magnitudes),
    ("s", None, "second", "seconds", 1, 0, 1, 0, 0, 0, 0, 0, smaller_magnitudes),
    ("g", None, "gram", "grams", 1e-3, 0, 0, 1, 0, 0, 0, 0, all_magnitudes),
    ("A", None, "ampere", "amperes", 1, 0, 0, 0, 1, 0, 0, 0, all_magnitudes),
    ("K", None, "kelvin", "kelvin", 1, 0, 0, 0, 0, 1, 0, 0, all_magnitudes) ]

derived_si_units = [
    ("Hz", None, "hertz", "hertz", 1, 0, -1, 0, 0, 0, 0, 0, all_magnitudes),
    ("N", None, "newton", "newtons", 1, 1, -2, 1, 0, 0, 0, 0, all_magnitudes),
    ("Pa", None, "pascal", "pascals", 1, -1, -2, 1, 0, 0, 0, 0, all_magnitudes),
    ("J", None, "joule", "joules", 1, 2, -2, 1, 0, 0, 0, 0, all_magnitudes),
    ("W", None, "watt", "watts", 1, 2, -3, 1, 0, 0, 0, 0, all_magnitudes),
    ("C", None, "coulomb", "coulombs", 1, 0, 1, 0, 1, 0, 0, 0, all_magnitudes),
    ("V", None, "volts", "volts", 1, 2, -3, 1, -1, 0, 0, 0, all_magnitudes),
    ("Ohm", "Ω", "ohm", "ohms", 1, 2, -3, 1, -2, 0, 0, 0, all_magnitudes),
    ("F", None, "farad", "farads", 1, -2, 4, -1, 2, 0, 0, 0, all_magnitudes),
    ("S", None, "siemens", "siemens", 1, -2, 3, -1, 2, 0, 0, 0, all_magnitudes),
    ("Wb", None, "weber", "webers", 1, 2, -2, 1, -1, 0, 0, 0, all_magnitudes),
    ("T", None, "tesla", "tesla", 1, 0, -2, 1, -1, 0, 0, 0, all_magnitudes),
    ("H", None, "henry", "henry", 1, 2, -2, 1, -2, 0, 0, 0, all_magnitudes),
]

non_si_dimensioned_units: list[tuple[str, str | None, str, str, float, int, int, int, int, int, int, int, list]] = [
    ("Ang", "Å", "angstrom", "angstroms", 1e-10, 1, 0, 0, 0, 0, 0, 0, []),
    ("min", None, "minute", "minutes", 60, 0, 1, 0, 0, 0, 0, 0, []),
    ("h", None, "hour", "hours", 360, 0, 1, 0, 0, 0, 0, 0, []),
    ("d", None, "day", "days", 360*24, 0, 1, 0, 0, 0, 0, 0, []),
    ("y", None, "year", "years", 360*24*365.2425, 0, 1, 0, 0, 0, 0, 0, []),
    ("deg", None, "degree", "degrees", 180/np.pi, 0, 0, 0, 0, 0, 0, 1, []),
    ("rad", None, "radian", "radians", 1, 0, 0, 0, 0, 0, 0, 1, []),
    ("sr", None, "stradian", "stradians", 1, 0, 0, 0, 0, 0, 0, 2, []),
    ("l", None, "litre", "litres", 1e-3, 3, 0, 0, 0, 0, 0, 0, []),
    ("eV", None, "electronvolt", "electronvolts", 1.602176634e-19, 2, -2, 1, 0, 0, 0, 0, all_magnitudes),
    ("au", None, "atomic mass unit", "atomic mass units", 1.660538921e-27, 0, 0, 1, 0, 0, 0, 0, []),
    ("mol", None, "mole", "moles", 6.02214076e23, 0, 0, 0, 0, 0, 1, 0, smaller_magnitudes),
    ("kgForce", None, "kg force", "kg force",  9.80665, 1, -2, 1, 0, 0, 0, 0, []),
    ("C", None, "degree Celsius", "degrees Celsius", 1, 0, 0, 0, 0, 1, 0, 0, []),
    ("miles", None, "mile", "miles", 1760*3*0.3048, 1, 0, 0, 0, 0, 0, 0, []),
    ("yrd", None, "yard", "yards", 3*0.3048, 1, 0, 0, 0, 0, 0, 0, []),
    ("ft", None, "foot", "feet", 0.3048, 1, 0, 0, 0, 0, 0, 0, []),
    ("in", None, "inch", "inches", 0.0254, 1, 0, 0, 0, 0, 0, 0, []),
    ("lb", None, "pound", "pounds", 0.45359237, 0, 0, 1, 0, 0, 0, 0, []),
    ("lbf", None, "pound force", "pounds force", 4.448222, 1, -2, 1, 0, 0, 0, 0, []),
    ("oz", None, "ounce", "ounces", 0.45359237/16, 0, 0, 1, 0, 0, 0, 0, []),
    ("psi", None, "pound force per square inch", "pounds force per square inch", 4.448222/(0.0254**2), -1, -2, 1, 0, 0, 0, 0, []),
]

non_si_dimensionless_units: list[tuple[str, str | None, str, str, float, int, int, int, int, int, int, int, list]] = [
    ("none", None, "none", "none", 1, 0, 0, 0, 0, 0, 0, 0, []),
    ("percent", "%", "percent", "percent", 0.01, 0, 0, 0, 0, 0, 0, 0, [])
]

non_si_units = non_si_dimensioned_units + non_si_dimensionless_units

# TODO:
# Add Hartree? Rydberg? Bohrs?
# Add CGS

aliases = {
    "y": ["yr", "year"],
    "d": ["day"],
    "h": ["hr", "hour"],
    "Ang": ["A", "Å"],
    "au": ["a.u.", "amu"],
    "percent": ["%"],
    "deg": ["degr", "Deg", "degrees", "Degrees"],
    "none": ["Counts", "counts", "cnts", "Cnts"]
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

    with open("_units_base.py", 'r') as base:
        for line in base:
            fid.write(line)

    # Write in unit definitions
    fid.write("\n\n"
              "#\n"
              "# Specific units \n"
              "#\n\n")

    symbol_lookup = {}
    unit_types_temp = defaultdict(list) # Keep track of unit types
    unit_types = defaultdict(list)

    for unit_def in all_units:

        try:
            symbol, special_symbol, singular, plural, scale, length, time, \
                mass, current, temperature, moles_hint, angle_hint, magnitudes = unit_def
        except Exception as e:
            print(unit_def)
            raise e

        formatted_plural = format_name(plural)
        formatted_singular = format_name(singular)

        dimensions = Dimensions(length, time, mass, current, temperature, moles_hint, angle_hint)
        fid.write(f"{formatted_plural} = NamedUnit({scale}, Dimensions({length}, {time}, {mass}, {current}, {temperature}, {moles_hint}, {angle_hint}),"
                      f"name='{formatted_plural}',"
                      f"ascii_symbol='{symbol}',"
                      f"symbol='{symbol if special_symbol is None else special_symbol}')\n")

        symbol_lookup[symbol] = formatted_plural
        if special_symbol is not None:
            symbol_lookup[special_symbol] = formatted_plural

        unit_types_temp[hash(dimensions)].append(
            (symbol, special_symbol, formatted_singular, formatted_plural, scale, dimensions))

        unit_types[hash(dimensions)].append(formatted_plural)

        for mag_symbol, mag_special_symbol, name, mag_scale in magnitudes:

            # Work out the combined symbol, accounts for unicode or not
            combined_special_symbol = (mag_symbol if mag_special_symbol is None else mag_special_symbol) + \
                              (symbol if special_symbol is None else special_symbol)

            combined_symbol = mag_symbol + symbol

            # Combined unit name
            combined_name_singular = f"{name}{formatted_singular}"
            combined_name_plural = f"{name}{formatted_plural}"

            combined_scale = scale * mag_scale

            # Units
            dimensions = Dimensions(length, time, mass, current, temperature, moles_hint, angle_hint)
            fid.write(f"{combined_name_plural} = NamedUnit({combined_scale}, "
                      f"Dimensions({length}, {time}, {mass}, {current}, {temperature}),"
                      f"name='{combined_name_plural}',"
                      f"ascii_symbol='{combined_symbol}',"
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

            fid.write(f"{speed_name} "
                      f"= NamedUnit({length_scale / time_scale}, "
                      f"Dimensions(length=1, time=-1), "
                      f"name='{speed_name}', "
                      f"ascii_symbol='{length_symbol}/{time_symbol}', "
                      f"symbol='{length_special_symbol}{time_special_symbol}⁻¹')\n")

            fid.write(f"{accel_name} = NamedUnit({length_scale / time_scale}, "
                      f"Dimensions(length=1, time=-2), "
                      f"name='{accel_name}', "
                      f"ascii_symbol='{length_symbol}/{time_symbol}^2', "
                      f"symbol='{length_special_symbol}{time_special_symbol}⁻²')\n")

            unit_types[hash(speed_dimensions)].append(speed_name)
            unit_types[hash(accel_dimensions)].append(accel_name)

    # Density
    for length_symbol, length_special_symbol, length_name, _, length_scale, _ in length_units:
        for mass_symbol, mass_special_symbol, _, mass_name, mass_scale, _ in mass_units:

            name = mass_name + "_per_cubic_" + length_name

            dimensions = Dimensions(length=-3, mass=1)

            fid.write(f"{name} "
                      f"= NamedUnit({mass_scale / length_scale**3}, "
                      f"Dimensions(length=-3, mass=1), "
                      f"name='{name}', "
                      f"ascii_symbol='{mass_symbol} {length_symbol}^-3', "
                      f"symbol='{mass_special_symbol}{length_special_symbol}⁻³')\n")

            unit_types[hash(dimensions)].append(name)

    # Concentration
    for length_symbol, length_special_symbol, length_name, _, length_scale, _ in length_units:
        for amount_symbol, amount_special_symbol, _, amount_name, amount_scale, _ in amount_units:

            name = amount_name + "_per_cubic_" + length_name

            dimensions = Dimensions(length=-3, moles_hint=1)

            fid.write(f"{name} "
                      f"= NamedUnit({amount_scale / length_scale**3}, "
                      f"Dimensions(length=-3, moles_hint=1), "
                      f"name='{name}', "
                      f"ascii_symbol='{amount_symbol} {length_symbol}^-3', "
                      f"symbol='{amount_special_symbol}{length_special_symbol}⁻³')\n")

            unit_types[hash(dimensions)].append(name)

    # TODO: Torque, Momentum, Entropy

    #
    # Add aliases to symbol lookup table
    #

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

    fid.write("\n#\n# Units by type \n#\n\n")

    for dimension_name, dimensions in dimension_names:


        fid.write(f"\n"
                  f"{dimension_name} = UnitGroup(\n"
                  f"  name = '{dimension_name}', \n"
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

    with open("_accessor_base.py", 'r') as base:
        for line in base:
            fid.write(line)

    for dimension_name, dimensions in dimension_names:

        accessor_name = dimension_name.capitalize().replace("_", "") + "Accessor"

        fid.write(f"\n"
                  f"class {accessor_name}[T](QuantityAccessor[T]):\n"
                  f"    dimension_name = '{dimension_name}'\n"
                  f"    \n")

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

    fid.write('"""'+(warning_text%"_build_tables.py")+'"""\n\n')
    si_unit_names = [values[3] for values in base_si_units + derived_si_units if values[3] != "grams"] + ["kilograms"]

    for name in si_unit_names:

        fid.write(f"from sasdata.quantities.units import {name}\n")

    fid.write("\nall_si = [\n")
    for name in si_unit_names:
        fid.write(f"    {name},\n")
    fid.write("]\n")