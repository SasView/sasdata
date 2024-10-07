from enum import Enum
from typing import TypeVar, Any, Self
from dataclasses import dataclass

from quantities.quantity import NamedQuantity
from sasdata.metadata import Metadata
from sasdata.quantities.accessors import AccessorTarget
from sasdata.data_backing import Group, key_tree


class SasData:
    def __init__(self, name: str, data_contents: list[NamedQuantity], raw_metadata: Group, verbose: bool=False):
        self.name = name
        self._data_contents = data_contents
        self._raw_metadata = raw_metadata
        self._verbose = verbose

        self.metadata = Metadata(AccessorTarget(raw_metadata, verbose=verbose))

        # TO IMPLEMENT

        # abscissae: list[NamedQuantity[np.ndarray]]
        # ordinate: NamedQuantity[np.ndarray]
        # other: list[NamedQuantity[np.ndarray]]
        #
        # metadata: Metadata
        # model_requirements: ModellingRequirements

    def summary(self, indent = "  ", include_raw=False):
        s = f"{self.name}\n"

        for data in self._data_contents:
            s += f"{indent}{data}\n"

        s += f"{indent}Metadata:\n"
        s += self.metadata.summary()

        if include_raw:
            s += key_tree(self._raw_metadata)

        return s