from abc import ABC, abstractmethod

from functools import singledispatch
import numpy as np
from typing import Self

from sasdata.data import SasData
from sasdata.metadata import Metadata
from sasdata import dataset_types
from sasdata.quantities.quantity import Operation


class ModellingRequirements(ABC):
    """Requirements that need to be passed to any modelling step"""

    dimensionality: int
    operation: Operation

    def __add__(self, other: Self) -> Self:
        return self.compose(other)

    @singledispatch
    def compose(self, other: Self) -> Self:
        return compose(self, other)

    @abstractmethod
    def from_qi_transformation(
        self, data: np.ndarray, metadata: Metadata
    ) -> np.ndarray:
        """Transformation for going from qi to this data"""
        pass


class ComposeRequirements(ModellingRequirements):
    """Composition of two models"""

    first: ModellingRequirements
    second: ModellingRequirements

    def __init__(self, fst, snd):
        self.first = fst
        self.second = snd

    def from_qi_transformation(
        self, data: np.ndarray, metadata: Metadata
    ) -> np.ndarray:
        """Perform both transformations in order"""
        return self.second.from_qi_transformation(
            self.first.from_qi_transformation(data, metadata), metadata
        )


class SesansModel(ModellingRequirements):
    """Perform Hankel transform for SESANS"""

    def from_qi_transformation(
        self, data: np.ndarray, metadata: Metadata
    ) -> np.ndarray:
        """Perform Hankel transform"""
        # FIXME: Actually do the Hankel transform
        return data


class SmearModel(ModellingRequirements):
    """Perform a slit smearing"""

    def from_qi_transformation(
        self, data: np.ndarray, metadata: Metadata
    ) -> np.ndarray:
        """Perform smearing transfor"""
        # FIXME: Actually do the smearing transform
        return data


class NullModel(ModellingRequirements):
    """A model that does nothing"""

    def from_qi_transformation(
        self, data: np.ndarray, _metadata: Metadata
    ) -> np.ndarray:
        """Do nothing"""
        return data


def compose(
    a: ModellingRequirements, b: ModellingRequirements
) -> ModellingRequirements:
    return ComposeRequirements(a, b)


def guess_requirements(data: SasData) -> ModellingRequirements:
    """Use names of axes and units to guess what kind of processing needs to be done"""
    if data.dataset_type == dataset_types.sesans:
        return SmearModel() + SesansModel()
    pass


@singledispatch
def compose(
    first: ModellingRequirements, second: ModellingRequirements
) -> ModellingRequirements:
    return ComposeRequirements(first, second)
