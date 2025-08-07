# NOTE: This module will probably be a lot more involved once how this is getting into the configuration will be sorted.

import sasdata.quantities.units as unit
from sasdata.dataset_types import unit_kinds
from sasdata.quantities.units import NamedUnit

default_units = {
    "Q": [unit.per_nanometer, unit.per_angstrom, unit.per_meter],
    "I": [unit.per_centimeter, unit.per_meter],
    "dQ": "Q",
    "dI": "I",
}


def defaults_or_fallback(column_name: str) -> list[NamedUnit]:
    value = default_units.get(column_name, unit_kinds[column_name].units)
    if isinstance(value, str):
        return defaults_or_fallback(value)
    return value


def first_default_for_fallback(column_name: str) -> NamedUnit:
    return defaults_or_fallback(column_name)[0]


def get_default_unit(column_name: str, unit_group: unit.UnitGroup):
    value = first_default_for_fallback(column_name)
    # Fallback to the first unit in the unit group if we don't have a default registered.
    if value is None:
        return unit_group.units[0]
    return value
