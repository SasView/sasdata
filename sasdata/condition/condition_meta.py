from dataclasses import dataclass
from typing import Union, Optional

import numpy as np

from sasdata.data_util.nxsunit import Converter, standardize_units

SCALAR_TYPE = Union[float, int, str]
VALUE_TYPE = Union[str, float, list[str], np.ndarray]


class ConditionBase:
    pass


@dataclass
class Scalar:
    # A scalar value for a condition
    _name: str
    _value: SCALAR_TYPE
    _unit: Optional[str]
    _uncertainty: Optional[VALUE_TYPE]
    _desired_unit: Optional[str]

    def __init__(self, **kwargs):
        # Data class automatically assigns class values
        super().__init__(**kwargs)
        self._converter = Converter(self._unit)
        self._desired_unit = self._unit

    def convert(self, new_unit: str):
        if self.are_units_sensible(new_unit):
            self._desired_unit = new_unit
            self._converter(self._value, new_unit)
        else:
            # TODO: warn the user in a meaningful way (no Exceptions!)
            pass

    def are_units_sensible(self, units: str):
        """A check to see if the units passed to the method make sense based on the condition type used."""
        if self._converter:
            compatible = self._converter.get_compatible_units()
            std_units = standardize_units(units)
            if len(compatible) == len(std_units):
                for comp, unit in zip(compatible, std_units):
                    if unit not in comp:
                        return False
            else:
                return False
        return True


class Vector:
    def __init__(self, vals: dict[str, dict[str, VALUE_TYPE]]):
        self.values: list[Scalar] = []
        for name, val_dict in vals.items():
            value = val_dict.get('value', 0.0)
            unit = val_dict.get('unit', None)
            unc = val_dict.get('uncertainty', None)
            self.values.append(Scalar(name=name, value=value, unit=unit, uncertainty=unc))
