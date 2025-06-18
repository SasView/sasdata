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
        # Compose uses the reversed order
        return compose(other, self)

    @abstractmethod
    def preprocess_q(self, data: np.ndarray, metadata: Metadata) -> np.ndarray:
        """Transform the Q values before processing in the model"""
        pass

    @abstractmethod
    def postprocess_iq(self, data: np.ndarray, metadata: Metadata) -> np.ndarray:
        """Transform the I(Q) values after running the model"""
        pass


class ComposeRequirements(ModellingRequirements):
    """Composition of two models"""

    first: ModellingRequirements
    second: ModellingRequirements

    def __init__(self, fst, snd):
        self.first = fst
        self.second = snd

    def preprocess_q(self, data: np.ndarray, metadata: Metadata) -> np.ndarray:
        """Perform both transformations in order"""
        return self.second.preprocess_q(
            self.first.preprocess_q(data, metadata), metadata
        )

    def postprocess_iq(self, data: np.ndarray, metadata: Metadata) -> np.ndarray:
        """Perform both transformations in order"""
        return self.second.postprocess_iq(
            self.first.postprocess_iq(data, metadata), metadata
        )


class SesansModel(ModellingRequirements):
    """Perform Hankel transform for SESANS"""

    def from_qi_transformation(
        self, data: np.ndarray, metadata: Metadata
    ) -> np.ndarray:
        """Perform Hankel transform"""
        # FIXME: Actually do the Hankel transform
        return data
    def postprocess_iq(self, data: np.ndarray, metadata: Metadata) -> np.ndarray:
        """
        Apply the SESANS transform to the computed I(q)
        """


class SmearModel(ModellingRequirements):
    """Perform a slit smearing"""

    def preprocess_q(self, data: np.ndarray, metadata: Metadata) -> np.ndarray:
        """Perform smearing transform"""
        # FIXME: Actually do the smearing transform
        return data

    def postprocess_iq(self, data: np.ndarray, metadata: Metadata) -> np.ndarray:
        """Perform smearing transform"""
        # FIXME: Actually do the smearing transform
        return data


class NullModel(ModellingRequirements):
    """A model that does nothing"""

    def compose(self, other: ModellingRequirements) -> ModellingRequirements:
        return other

    def preprocess_q(self, data: np.ndarray, _metadata: Metadata) -> np.ndarray:
        """Do nothing"""
        return data

    def postprocess_iq(self, data: np.ndarray, metadata: Metadata) -> np.ndarray:
        """Do nothing"""
        return data


def compose(
    a: ModellingRequirements, b: ModellingRequirements
) -> ModellingRequirements:
    return ComposeRequirements(a, b)


def guess_requirements(data: SasData) -> ModellingRequirements:
    """Use names of axes and units to guess what kind of processing needs to be done"""
    if data.dataset_type == dataset_types.sesans:
        return SesansModel()
    pass


@singledispatch
def compose(
    second: ModellingRequirements, first: ModellingRequirements
) -> ModellingRequirements:
    """Compose to models together

    This function uses a reverse order so that it can perform dispatch on
    the *second* term, since the classes already had a chance to dispatch
    on the first parameter

    """
    return ComposeRequirements(first, second)


@compose.register
def _(second: NullModel, first: ModellingRequirements) -> ModellingRequirements:
    return first
