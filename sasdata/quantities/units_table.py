"""
Builds a data file containing details of units
"""

import numpy as np

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

all_magnitudes = bigger_magnitudes + smaller_magnitudes

# Length, time, mass, current, temperature
base_si_units = [
    ("m", None, "meter", "meters", 1, 1, 0, 0, 0, 0, all_magnitudes),
    ("s", None, "second", "seconds", 1, 0, 1, 0, 0, 0, smaller_magnitudes),
    ("g", None, "gram", "grams", 1, 0, 0, 1, 0, 0, all_magnitudes),
    ("A", None, "amp", "amps", 1, 0, 0, 0, 1, 0, all_magnitudes),
    ("K", None, "kelvin", "kelvin", 1, 0, 0, 0, 0, 1, all_magnitudes) ]

derived_si_units = [
    ("Hz", None, "hertz", "hertz", 1, 0, -1, 0, 0, 0, all_magnitudes),
    ("N", None, "newton", "newtons", 1, 1, -2, 1, 0, 0, all_magnitudes),
    ("Pa", None, "pascal", "pascals", 1, -1, -2, 1, 0, 0, all_magnitudes),
    ("J", None, "joule", "joules", 1, 2, -2, 1, 0, 0, all_magnitudes),
    ("W", None, "watt", "watts", 1, 2, -3, 1, 0, 0, all_magnitudes),
    ("C", None, "coulomb", "coulombs", 1, 0, 1, 0, 1, 0, all_magnitudes),
    ("V", None, "volts", "volts", 1, 2, -3, 1, -1, 0, all_magnitudes),
    ("Ohm", "Ω", "ohm", "ohms", 1, 2, -3, 1, -2, 0, all_magnitudes),
    ("F", None, "farad", "farads", 1, -2, 4, -1, 2, 0, all_magnitudes),
    ("S", None, "siemens", "siemens", 1, -2, 3, -1, 2, 0, all_magnitudes),
    ("Wb", None, "weber", "webers", 1, 2, -2, 1, -1, 0, all_magnitudes),
    ("T", None, "tesla", "tesla", 1, 2, -2, 1, -1, 0, all_magnitudes),
    ("H", None, "henry", "henry", 1, 2, -2, 1, -2, 0, all_magnitudes),
    ("C", None, "degree Celsius", "degrees Celsius", 1, 0, 0, 0, 0, 1, [])
]

non_si_units = [
    ("A", None, "angstrom", "angstroms", 1e-10, 1, 0, 0, 0, 0, []),
    ("min", None, "minute", "minutes", 60, 0, 1, 0, 0, 0, []),
    ("hr", None, "hour", "hours", 360, 0, 1, 0, 0, 0, []),
    ("d", None, "day", "days", 360*24, 0, 1, 0, 0, 0, []),
    ("day", None, "day", "days", 360*24, 0, 1, 0, 0, 0, []),
    ("y", None, "year", "years", 360*24*365.2425, 0, 1, 0, 0, 0, []),
    ("yr", None, "year", "years", 360*24*365.2425, 0, 1, 0, 0, 0, []),
    ("deg", None, "degree", "degrees", 180/np.pi, 0, 0, 0, 0, 0, []),
    ("rad", None, "radian", "radians", 1, 0, 0, 0, 0, 0, []),
    ("sr", None, "stradian", "stradians", 1, 0, 0, 0, 0, 0, [])
]

all_units = base_si_units + derived_si_units + non_si_units

encoding = "utf-8"

with open("unit_data.txt", mode='w', encoding=encoding) as fid:
    for symbol, special_symbol, singular, plural, scale, length, time, mass, current, temperature, magnitudes in all_units:
        fid.write(f"'{symbol}', '{special_symbol}', '{singular}', '{plural}', ")
        fid.write(f"{scale}, {length}, {time}, {mass}, {current}, {temperature}\n")

        for mag_symbol, mag_special_symbol, name, mag_scale in magnitudes:

            combined_symbol = (mag_symbol if mag_special_symbol is None else mag_special_symbol) + \
                              (symbol if special_symbol is None else special_symbol)

            fid.write(f"'{mag_symbol}{symbol}', '{combined_symbol}', '{name}{singular}', '{name}{plural}', ")
            fid.write(f"{scale * mag_scale}, {length}, {time}, {mass}, {current}, {temperature}\n")
