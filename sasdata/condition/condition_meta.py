from dataclasses import dataclass
from typing import Union, Optional

import numpy as np

from sasdata.data_util.nxsunit import Converter

SCALAR_TYPE = Union[float, int, str]
VALUE_TYPE = Union[str, float, list[str], np.ndarray]


@dataclass
class ConditionScalar:
    """A scalar value for a condition.  A condition can be any property that differentiates the resulting data set from
    other data sets. This can include sample environment conditions, sample characteristics, instrumental conditions,
    or even user-defined values."""
    # TODO: Allow for extensible parameter names
    # The name of the condition
    name: str
    # The value for the condition
    value: SCALAR_TYPE
    # The uncertainty for the condition
    uncertainty: Optional[VALUE_TYPE]
    # The description of the scalar quantity
    description: Optional[str]


@dataclass
class ConditionScalable(ConditionScalar):
    unit: str

    def __init__(self, **kwargs):
        # Data class automatically assigns class values
        super().__init__(**kwargs)
        self._converter = Converter(self.unit)
        self._desired_unit: str = self.unit

    def convert(self, new_unit: str):
        """Scale the value and all related parameter values to the new units.

        :param new_unit: A string representation of the units the parameter should be scaled to."""
        if self._converter.are_units_sensible(new_unit):
            self._desired_unit = new_unit
            self.value = self._converter(self.value, new_unit)
            self.uncertainty = self._converter(self.uncertainty, new_unit)
        else:
            # TODO: warn the user in a meaningful way
            pass


class ConditionVector:
    """A ConditionVector defines all Conditions pertinent to a single Data object."""
    def __init__(self, value_map: dict[str, dict[str, VALUE_TYPE]]):
        """Initialize the ConditionVector using the value map passed in.
        :param value_map: A dictionary mapping the parameter name to a dictionary of name:value pairs associated with
            the parameter.
        """
        # self._conditions should be created on instantiation and be considered non-mutable.
        self._conditions: dict[str, ConditionScalar] = self._create_condition_dict(value_map)
        # Extra conditions are mutable user-defined conditions. If a key appears in both maps, the one in this map will
        #   be returned, *but*
        self.extra_conditions: dict[str, ConditionScalar] = {}

    @staticmethod
    def _create_condition_dict(value_map: dict[str, dict[str, VALUE_TYPE]]) -> dict[str, ConditionScalar]:
        """Create and return a condition dictionary using the supplied value map.
        :param value_map: A dictionary mapping the parameter name to a dictionary of name:value pairs associated with
            the parameter.
        """
        conditions = {}
        for name, val_dict in value_map.items():
            # TODO: Add more specific sub-conditions to keep predictable names
            if 'unit' in val_dict.keys():
                conditions[name] = ConditionScalable(**val_dict)
            else:
                conditions[name] = ConditionScalar(**val_dict)
        return conditions

    def add_conditions(self, value_map: dict[str, dict[str, VALUE_TYPE]]):
        """Add any number of user-defined conditions to the condition vector.
        :param value_map: A dictionary mapping the parameter name to a dictionary of name:value pairs associated with
            the parameter.
        """
        self.extra_conditions.update(self._create_condition_dict(value_map))

    def get_condition(self, name: str) -> Optional[ConditionScalar]:
        """A pass-through for ConditionVector.get_condition_by_name(name), with a simplified name."""
        return self.get_condition_by_name(name)

    def get_condition_by_name(self, name: str) -> Optional[ConditionScalar]:
        """Get a condition by its name. If a user-defined condition is present, that value will be selected
        preferentially over the value given in the data. This allows the end-user to override the value,
        regardless of what was read from the data file.

        :param name: The condition name that is desired.
        :return: The ConditionScalar instance with name==name or None if no condition has that name."""
        return self.extra_conditions.get(name, self._conditions.get(name, None))

    def get_default_condition(self, name: str) -> Optional[ConditionScalar]:
        """Get a condition by its name, as it was read from the data file.

        :param name: The condition name that is desired.
        :return: The ConditionScalar instance with name==name or None if no condition has that name."""
        return self._conditions.get(name, None)
