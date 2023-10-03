"""
A Condition that the data set was subject to during the course of a measurement.

A Condition could be a sample environment condition, such as temperature, pressure, or magnet field, an instrumental
setting, such as detector distance, or wavelength, a sample condition, such as concentration, or contrast, or any other
condition, such as total measurement time, relative time to another measurement, instrument used, etc.
"""

from typing import Union, Optional
import numpy as np

from sasdata.data_util.nxsunit import Converter

value_type = Union[str, float, list[str], np.ndarray]


class Condition:
    # The cansas_class variable defines the NeXuS class the Condition should be saved under. Sub-classing of Condition
    #   will help for fine-grained canSAS classing.
    _can_sas_class: str = "NXnote"
    # The _name variable defines the name that all similar conditions should have.
    _name: str = "generic"

    def __init__(self, value: value_type, uncertainty: Optional[value_type] = None, units: Optional[str] = None):
        """The Condition class initialization routine. This creates a Condition object from the values passed to it.

        :param name: The name of the Condition.
        :param value: The mean value of the Condition, at the time of the measurement.
        :param uncertainty: (Optional) The uncertainty of the value. This is assumed to be the value one
            standard-deviation from the mean value.
        :param units: (Optional) The units of the value, in a form that can be interpreted by the Converter class.
        """
        # Base properties related to the condition. Internal variables should be immutable.
        self._value: value_type = value
        self._uncertainty: value_type = uncertainty
        self._desired_units: str = units
        self._base_units: str = units
        try:
            self._converter: Optional[Converter] = Converter(self._base_units)
        except ValueError:
            # For unrecognized units, the Converter class throws a ValueError.
            self._converter = None

        self._suggested_views = []
        self.selected_view = None

    @property
    def value(self):
        """A property to calculate the converted condition value, to maintain the immutability of self._value."""
        return self._converter(self._value, self.units) if self._converter else self._value

    @property
    def uncertainty(self):
        """A property to calculate the converted uncertainty, to maintain the immutability or self._uncertainty.
        This assumes the percent uncertainty is constant relative to the scaled value."""
        return self._converter(self._uncertainty, self.units) if self._converter else self._uncertainty

    @property
    def name(self):
        """A property to fetch the condition name to ensure the immutability of self._name."""
        return self._name

    @property
    def units(self):
        """A property to fetch the condition units."""
        return self._desired_units if self._desired_units else self._base_units

    @units.setter
    def units(self, units: str):
        """A generic property to allow extra operations to be performed when modifying units."""
        self._desired_units = units

    def revert_to_base_units(self):
        """Sets the units property to None, which, in turn, sets _desired_units to None, forcing the converter and
        other operations to use _base_units."""
        self.units = None
