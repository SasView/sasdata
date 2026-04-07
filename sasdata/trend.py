
from dataclasses import dataclass

import numpy as np

from sasdata.data import SasData
from sasdata.quantities.quantity import Quantity
from sasdata.transforms.rebinning import calculate_interpolation_matrix_1d

# Axis strs refer to the name of their associated NamedQuantity.

# TODO: This probably shouldn't be here but will keep it here for now. --> In sasdta/data.py?
# TODO: Similarity/relation to __getitem__ in SasData class?
# TODO: Or a method of Metadata class?
# TODO: Not sure how to type hint the return.
def get_metadatum_from_path(data: SasData, metadata_path: list[str]):
    current_node = data.metadata.raw
    for path_item in metadata_path:
        if isinstance(current_node.contents, list):
            # Search through list of MetaNodes
            current_item = None
            for node in current_node.contents:
                if node.name == path_item:
                    current_item = node
                    break
        else:
            # Not a list, can't navigate further
            raise ValueError('Path does not lead to a valid metadatum.')

        if current_item is None:
            raise ValueError('Path does not lead to a valid metadatum.')

        # Check if we're at the end of the path
        if path_item == metadata_path[-1]:
            return current_item.contents

        current_node = current_item
    raise ValueError('End of path without finding a dataset.')

@dataclass
class Trend:
    data: list[SasData]
    trend_axes: dict[str, list[str] | list]  # Path or manual values

    def __post_init__(self):

        # First, filter out invalid data items
        self._filter_and_validate_data()

        # Validate data length matches manual value lists
        self._validate_manual_values()

        # Validate metadata paths
        self._validate_metadata_paths()

    def _filter_and_validate_data(self):
        """Filter out non-SasData objects and validate data integrity"""
        valid_data = []
        invalid_indices = []

        for i, datum in enumerate(self.data):
            if not isinstance(datum, SasData):
                invalid_indices.append(i)
                continue

            # Check if datum has metadata
            if not hasattr(datum, 'metadata') or datum.metadata is None:
                invalid_indices.append(i)
                continue

            # Check if datum has raw metadata
            if not hasattr(datum.metadata, 'raw') or datum.metadata.raw is None:
                invalid_indices.append(i)
                continue

            valid_data.append(datum)

        # Update data with only valid items
        self.data = valid_data

        # Warn about filtered items
        if invalid_indices:
            print(f"Warning: Removed data items at indices {invalid_indices} - not SasData objects or missing/invalid metadata")

        # Additional validation
        if not self.data:
            raise ValueError("No valid data items remain after filtering")

        if len(self.data) < 2:
            print(f"Warning: Only {len(self.data)} valid data items remain")

    def _validate_manual_values(self):
        """Ensure manual value lists match data length"""
        for axis_name, axis_config in self.trend_axes.items():
            if isinstance(axis_config, list) and not isinstance(axis_config[0], str):
                # This is a manual value list (not a path)
                if len(axis_config) != len(self.data):
                    raise ValueError(f"Manual values for axis '{axis_name}' must have same length as data ({len(self.data)} items, got {len(axis_config)})")

    def _validate_metadata_paths(self):
        """Validate metadata paths"""
        for axis_name, axis_config in self.trend_axes.items():
            if isinstance(axis_config, list) and len(axis_config) > 0 and isinstance(axis_config[0], str):
                # This is a metadata path
                for i, datum in enumerate(self.data):
                    try:
                        get_metadatum_from_path(datum, axis_config)
                    except ValueError as e:
                        raise ValueError(f"trend_axes['{axis_name}'] path {axis_config} invalid for data item {i}: {e}")

    def get_trend_values(self, axis_name: str) -> list:
        """Get values for a named trend axis"""
        if axis_name not in self.trend_axes:
            raise KeyError(f"Axis '{axis_name}' not found")

        axis_config = self.trend_axes[axis_name]

        if isinstance(axis_config, list) and len(axis_config) > 0 and isinstance(axis_config[0], str):
            # Metadata path - extract from data
            return [get_metadatum_from_path(datum, axis_config) for datum in self.data]
        else:
            # Manual values - return as-is
            return axis_config.copy()  # Return copy to prevent modification

    def add_manual_axis(self, axis_name: str, values: list):
        """Add a new manual trend axis"""
        if len(values) != len(self.data):
            raise ValueError(f"Manual values must have same length as data ({len(self.data)} items, got {len(values)})")

        self.trend_axes[axis_name] = values.copy()

    def add_metadata_axis(self, axis_name: str, path: list[str]):
        """Add a new metadata trend axis"""
        # Validate the path first
        for i, datum in enumerate(self.data):
            try:
                get_metadatum_from_path(datum, path)
            except ValueError as e:
                raise ValueError(f"Path {path} invalid for data item {i}: {e}")

        self.trend_axes[axis_name] = path

    @property
    def axis_names(self) -> list[str]:
        return list(self.trend_axes.keys())

    def is_manual_axis(self, axis_name: str) -> bool:
        """Check if an axis uses manual values or metadata path"""
        if axis_name not in self.trend_axes:
            raise KeyError(f"Axis '{axis_name}' not found")

        axis_config = self.trend_axes[axis_name]
        return not (isinstance(axis_config, list) and len(axis_config) > 0 and isinstance(axis_config[0], str))

    # TODO: Assumes there are at least 2 items in data. Is this reasonable to assume? Should there be error handling for
    # situations where this may not be the case?
    def all_axis_match(self, axis: str) -> bool:
        reference_data = self.data[0]
        data_axis = reference_data[axis]
        for datum in self.data[1::]:
            axis_datum = datum[axis]
            # FIXME: Linter is complaining about typing.
            if not np.all(np.isclose(axis_datum.value, data_axis.value)):
                return False
        return True

    # TODO: For now, return a new trend, but decide later. Shouldn't be too hard to change.
    def interpolate(self, axis: str) -> "Trend":
        new_data: list[SasData] = []
        reference_data = self.data[0]
        # TODO: I don't like the repetition here. Can probably abstract a function for this ot make it clearer.
        data_axis = reference_data[axis]
        for i, datum in enumerate(self.data):
            if i == 0:
                # This is already the reference axis; no need to interpolate it.
                continue
            # TODO: Again, repetition
            axis_datum = datum[axis]
            # TODO: There are other options which may need to be filled (or become new params to this method)
            mat, _ = calculate_interpolation_matrix_1d(axis_datum, data_axis)
            new_quantities: dict[str, Quantity] = {}
            for name, quantity in datum._data_contents.items():
                if name == axis:
                    new_quantities[name] = data_axis
                    continue
                new_quantities[name] = quantity @ mat

            new_datum = SasData(
                name=datum.name,
                data_contents=new_quantities,
                dataset_type=datum.dataset_type,
                metadata=datum.metadata,
            )
            new_data.append(new_datum)
        new_trend = Trend(new_data,
                          self.trend_axes)
        return new_trend
