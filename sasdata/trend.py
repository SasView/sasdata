#!/usr/bin/env python

from dataclasses import dataclass
from typing import Self
from sasdata.data import SasData
from sasdata.data_backing import Dataset, Group
import numpy as np
from sasdata.quantities.quantity import NamedQuantity
from sasdata.transforms.rebinning import calculate_interpolation_matrix

# Axis strs refer to the name of their associated NamedQuantity.

# TODO: This probably shouldn't be here but will keep it here for now.
# TODO: Not sure how to type hint the return.
def get_metadatum_from_path(data: SasData, metadata_path: list[str]):
    current_group = data._raw_metadata
    for path_item in metadata_path:
        current_item = current_group.children.get(path_item, None)
        if current_item is None or (isinstance(current_item, Dataset) and path_item != metadata_path[-1]):
            raise ValueError('Path does not lead to valid a metadatum.')
        elif isinstance(current_item, Group):
            current_group = current_item
        else:
            return current_item.data
    raise ValueError('End of path without finding a dataset.')


@dataclass
class Trend:
    data: list[SasData]
    # This is going to be a path to a specific metadatum.
    #
    # TODO: But what if the trend axis will be a particular NamedQuantity? Will probably need to think on this.
    trend_axis: list[str]

    # Designed to take in a particular value of the trend axis, and return the SasData object that matches it.
    # TODO: Not exaclty sure what item's type will be. It could depend on where it is pointing to.
    def __getitem__(self, item) -> SasData:
        for datum in self.data:
            metadatum = get_metadatum_from_path(datum, self.trend_axis)
            if metadatum == item:
                return datum
        raise KeyError()
    @property
    def trend_axes(self) -> list[float]:
        return [get_metadatum_from_path(datum, self.trend_axis) for datum in self.data]

    def data_axes(self, data: SasData, axis: str) -> list[NamedQuantity]:
        return  [content for content in data._data_contents if content.name == axis]

    def reference_data_axis(self, data: SasData, axis: str) -> NamedQuantity:
        return self.data_axes(data, axis)[0]

    # TODO: Assumes there are at least 2 items in data. Is this reasonable to assume? Should there be error handling for
    # situations where this may not be the case?
    def all_axis_match(self, axis: str) -> bool:
        reference_data = self.data[0]
        data_axis = self.reference_data_axis(reference_data, axis)
        for datum in self.data[1::]:
            contents = datum._data_contents
            axis_datum = [content for content in contents if content.name == axis][0]
            # FIXME: Linter is complaining about typing.
            if not np.all(np.isclose(axis_datum.value, data_axis.value)):
                return False
        return True

    # TODO: For now, return a new trend, but decide later. Shouldn't be too hard to change.
    def interpolate(self, axis: str) -> Self:
        new_data: list[SasData] = []
        reference_data = self.data[0]
        # TODO: I don't like the repetition here. Can probably abstract a function for this ot make it clearer.
        data_axis = self.reference_data_axis(reference_data, axis)
        for i, datum in enumerate(self.data):
            if i == 0:
                # This is already the reference axis; no need to interpolate it.
                continue
            # TODO: Again, repetition
            axis_datum = [content for content in datum._data_contents if content.name == axis][0]
            # TODO: There are other options which may need to be filled (or become new params to this method)
            mat = calculate_interpolation_matrix(axis_datum, data_axis)
            new_quantities: list[NamedQuantity]  = []
            for quantity in datum._data_contents:
                if quantity.name == axis_datum.name:
                    continue
                new_quantities.append(quantity @ mat)

            new_datum = SasData(datum.name,
                                new_quantities,
                                datum._raw_metadata)
            new_data.append(new_datum)
        new_trend = Trend(new_data,
                          self.trend_axis)
        return new_trend
