from dataclasses import dataclass
from typing import Union, Optional

import numpy as np

from sasdata.data_util.nxsunit import Converter

SCALAR_TYPE = Union[float, int, str]
VALUE_TYPE = Union[str, float, list[str], np.ndarray]


@dataclass
class ConditionScalar:
    """A scalar value for a condition"""
    # The name of the condition
    name: str
    # The value for the condition
    value: SCALAR_TYPE
    # The uncertainty for the condition
    uncertainty: Optional[VALUE_TYPE]
    # The description of the scalar quantity
    description: Optional[str]


class ConditionVector:
    def __init__(self, vals: dict[str, dict[str, VALUE_TYPE]]):
        self.values: list[ConditionScalar] = []
        for name, val_dict in vals.items():
            value = val_dict.get('value', 0.0)
            unc = val_dict.get('uncertainty', None)
            desc = val_dict.get('description', name)
            self.values.append(ConditionScalar(name=name, value=value, uncertainty=unc, description=desc))


@dataclass
class ConditionUnit(ConditionScalar):
    unit: str

    def __init__(self, **kwargs):
        # Data class automatically assigns class values
        super().__init__(**kwargs)
        self._converter = Converter(self.unit)
        self._desired_unit: str = self.unit

    def convert(self, new_unit: str):
        if self._converter.are_units_sensible(new_unit):
            self._desired_unit = new_unit
            self._converter(self.value, new_unit)
        else:
            # TODO: warn the user in a meaningful way, but do not throw an Exception
            pass
