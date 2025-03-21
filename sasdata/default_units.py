# NOTE: This module will probably be a lot more involved once how this is getting into the configuration will be sorted.

from sasdata.quantities.units import NamedUnit
import sasdata.quantities.units as unit
from sasdata.dataset_types import unit_kinds

default_units = {
    'Q': [unit.per_nanometer, unit.per_angstrom, unit.per_meter],
    'I': [unit.per_centimeter, unit.per_meter],
    'dQ': 'Q',
    'dI': 'I'
}

def defaults_or_fallback(column_name: str) -> list[NamedUnit]:
    value = default_units.get(column_name, unit_kinds[column_name].units)
    if isinstance(value, str):
        return defaults_or_fallback(value)
    return value

def first_default_for_fallback(column_name: str) -> NamedUnit:
    return defaults_or_fallback(column_name)[0]
