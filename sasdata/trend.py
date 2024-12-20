#!/usr/bin/env python

from dataclasses import dataclass
from typing import Self
from sasdata.data import SasData

# Axis strs refer to the name of their associated NamedQuantity.

@dataclass
class Trend:
    data: list[SasData]
    trend_axis: str

    # Designed to take in a particular value of the trend axis, and return the SasData object that matches it.
    # TODO: Not exaclty sure what item's type will be. It could depend on where it is pointing to.
    def __getitem__(self, item) -> SasData:
        raise NotImplementedError()

    # TODO: Assumes there are at least 2 items in data. Is this reasonable to assume? Should there be error handling for
    # situations where this may not be the case?
    def all_axis_match(self, axis: str) -> bool:
        reference_data = self.data[0]
        for datum in self.data[1::]:
            contents = datum._data_contents
            axis_datum = [content for content in contents if content.name == axis][0]
            if axis_datum != datum:
                return False
        return True

    # TODO: Not sure if this should return a new trend, or just mutate the existing trend
    # TODO: May be some details on the method as well.
    def interpolate(self, axis: str) -> Self:
        raise NotImplementedError()
