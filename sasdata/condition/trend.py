from typing import Union
from sasdata.data.data import Data
from sasdata.condition.condition_meta import VALUE_TYPE


class Trend:
    """A class to bundle Data based on specific conditions.

    *self._data* A list of sasdata.data.data.Data objects that the trend will be generated from.
    *self._conditions: A list of condition names that will be used to search the data sets to generate the trend.
    *self._trend* A dictionary mapping the conditions to the data set with those exact conditions
        e.g. If the conditions are ['time', 'temperature'] the trend would look something like:
        self._trend = {
            0912309812048: {
              293.15: Data1,
              303.15: Data2
            },
            0912309812058: {
              293.15: Data3,
              303.15: Data4
            },
            ...
        }
        The first level of the trend are the timestamps (epoch time in this case), the second level are temperatures
          (in Kelvin in this case), and the temperatures are mapped to the Data object with those specific conditions.
    *self._untrendy_data* A list of Data objects that do not have all conditions required for the trend.

    """
    def __init__(self, data: list[Data], conditions: list[str]):
        """Instantiate a Trend object using the given data and conditions.

        :param data: A list of Data objects to create a trend.
        :param conditions: An order list of condition names to generate the trend.
        """
        # All data that could be used in this trend
        self._data: list[Data] = data
        # A list of condition names
        self._conditions: list[str] = conditions
        # A hierarchical mapping of the condition names to conditional values to the data set
        self._trend = {}

        # Data sets that do not have any of the conditions provided
        self._untrendy_data = []
        self.map_trend()

    @property
    def trend(self):
        return self._trend

    @property
    def untrendy(self):
        return self._untrendy_data

    def add_condition(self, condition: str, location: int):
        self._conditions.insert(location, condition)
        self.map_trend()

    def add_data(self, data_sets: Union[Data, list[Data]]):
        data_sets = list(data_sets)
        self._data.extend(data_sets)
        self.map_trend()

    def map_trend(self):
        """Create a data trend using the existing data and conditions."""
        for data_set in self._data:
            conditions = []
            for condition in self._conditions:
                cond = data_set.conditions.get_condition_by_name(condition)
                if not cond:
                    self._untrendy_data.append(data_set)
                    continue
                conditions.append(cond.value)
            self._trend['values'] = self._trend_append_recursive(self._trend, conditions, data_set)

    def _trend_append_recursive(self, trend: dict, vals: list[VALUE_TYPE], data: Data) -> dict[VALUE_TYPE: Data]:
        """A recursive mapping a series of conditions to the Data set those conditions are associated with."""
        if not vals:
            return data
        for value in vals:
            new_trend = trend.get(value, {})
            trend[value] = self._trend_append_recursive(new_trend, vals[1:], data)
        return trend
