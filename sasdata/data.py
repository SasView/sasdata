from enum import Enum
from typing import TypeVar, Any, Self
from dataclasses import dataclass

import numpy as np

from sasdata.dataset_types import DatasetType, one_dim, two_dim
from sasdata.quantities.quantity import NamedQuantity, Quantity
from sasdata.metadata import Metadata
from sasdata.quantities.accessors import AccessorTarget
from sasdata.data_backing import Group, key_tree


class SasData:
    def __init__(self, name: str,
                 data_contents: dict[str, Quantity],
                 dataset_type: DatasetType,
                 raw_metadata: Group,
                 verbose: bool=False):

        self.name = name
        # validate data contents
        if not all([key in dataset_type.optional or key in dataset_type.required for key in data_contents.keys()]):
            raise ValueError("Columns don't match the dataset type")
        self._data_contents = data_contents
        self._raw_metadata = raw_metadata
        self._verbose = verbose

        self.metadata = Metadata(AccessorTarget(raw_metadata, verbose=verbose))

        # TODO: Could this be optional?
        self.dataset_type: DatasetType = dataset_type

        # Components that need to be organised after creation
        self.mask = None # TODO: fill out
        self.model_requirements = None # TODO: fill out

    # TODO: Handle the other data types.
    @property
    def ordinate(self) -> Quantity:
        if self.dataset_type == one_dim or self.dataset_type == two_dim:
            return self._data_contents['I']
        # TODO: seesans
        # Let's ignore that this method can return None for now.
        return None

    @property
    def abscissae(self) -> Quantity:
        if self.dataset_type == one_dim:
            return self._data_contents['Q']
        return None

    def __getitem__(self, item: str):
        return self._data_contents[item]

    def summary(self, indent = "  ", include_raw=False):
        s = f"{self.name}\n"

        for data in self._data_contents:
            s += f"{indent}{data}\n"

        s += f"Metadata:\n"
        s += "\n"
        s += self.metadata.summary()

        if include_raw:
            s += key_tree(self._raw_metadata)

        return s
