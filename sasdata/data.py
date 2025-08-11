from enum import Enum
from typing import TypeVar, Any, Self
from dataclasses import dataclass

import numpy as np

from sasdata.quantities.quantity import NamedQuantity
from sasdata.metadata import Metadata
from sasdata.quantities.accessors import AccessorTarget
from sasdata.data_backing import Group, key_tree


class SasData:
    """ General object containing data in the SasView ecosystem"""

    def __init__(self,
                 name: str,
                 ordinate: Quantity,
                 mask: Quantity,
                 abscissae: list[Quantity],
                 dependents: list["SasData"]):

        self._ordinate = ordinate
        self._abscissae = abscissae
        self._mask = mask

    @property
    def ordinate(self) -> Quantity:
        return self._ordinate

    @property
    def abscissae(self) -> list[Quantity]:
        return self._abscissae

    @property
    def mask(self) -> Quantity:
        return self._mask

    def scatter_data(self):
        """ Return data in the coordinate/value form [(x1, x2, x3, y)...]"""


class SasDerivedMeasurement(SasData):
    """ General object sas measurement that has not come directly from a file,
    for example, the difference between two datasets"""


    def __init__(self,
                 name: str,
                 ordinate: Quantity,
                 abscissae: list[Quantity],
                 dependents: list["SasData"],
                 metadata: DerivedMetadata):

        super().__init__(
            name=name,
            ordinate=ordinate,
            abscissae=abscissae,
            dependents=dependents)

        self.metadata = metadata




class SasMeasurement(SasData):
    def __init__(self, name: str,
                 data_contents: dict[str, Quantity],
                 dataset_type: DatasetType,
                 metadata: Metadata,
                 verbose: bool=False):

    def __init__(
        self,
        name: str,
        data_contents: dict[str, Quantity],
        dataset_type: DatasetType,
        metadata: Metadata,
        verbose: bool = False,
    ):
        self.name = name
        # validate data contents
        if not all([key in dataset_type.optional or key in dataset_type.required for key in data_contents]):
            raise ValueError("Columns don't match the dataset type")
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
