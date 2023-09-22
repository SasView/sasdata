"""
A condition a data set was subject to. This could be a sample environment, instrumental, or
"""

from typing import Union, Optional
import numpy as np

from sasdata.data_util.nxsunit import Converter

value_type = Union[str, float, list[str], np.ndarray]


class Condition:

    def __init__(self, value: value_type, uncertainty: Optional[value_type] = None, units: Optional[str] = None,
                 ll: Optional[value_type] = None, ul: Optional[value_type] = None):
        self._value: value_type = value
        self._uncertainty: value_type = uncertainty
        self.desired_units = units
        self._base_units: str = units
        self._converter = Converter(self._base_units) if self._base_units else None
        self.upper_limit: value_type = ul
        self.lower_limit: value_type = ll

    @property
    def value(self):
        return self._converter(self._value, self.desired_units)
