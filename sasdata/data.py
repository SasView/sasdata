from enum import Enum
from typing import TypeVar, Any, Self
from dataclasses import dataclass

import numpy as np

from quantities.quantity import NamedQuantity
from sasdata.metadata import Metadata
from sasdata.quantities.accessors import AccessorTarget
from sasdata.data_backing import Group, key_tree


class SasData:
    def __init__(self, name: str,
                 data_contents: list[NamedQuantity],
                 raw_metadata: Group,
                 verbose: bool=False):

        self.name = name
        self._data_contents = data_contents
        self._raw_metadata = raw_metadata
        self._verbose = verbose

        self.metadata = Metadata(AccessorTarget(raw_metadata, verbose=verbose))

        # Components that need to be organised after creation
        self.ordinate: NamedQuantity[np.ndarray] = None # TODO: fill out
        self.abscissae: list[NamedQuantity[np.ndarray]] = None # TODO: fill out
        self.mask = None # TODO: fill out
        self.model_requirements = None # TODO: fill out

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