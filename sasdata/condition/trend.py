from typing import Union
from sasdata.data.data import Data


class Trend:
    """A class to bundle Data based on specific conditions"""
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
        # TODO: This should have a more well-defined structure
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
        # TODO: Write this
        #  Step 1: Get data sets that have all condition names in them.
        #  Step 2: Get the condition values
        #  Step 3: Ensure condition units all match (need to define a default unit set)
        #  Step 4: Create the final mapping (in whatever form that will take) and populate self._trend
        #  Step 5: Add unused data sets to self._untrendy_data
        pass
