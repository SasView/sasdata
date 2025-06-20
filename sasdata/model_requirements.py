from abc import ABC, abstractmethod

from functools import singledispatch
import numpy as np
from typing import Self
from scipy.special import j0

from sasdata.data import SasData
from sasdata.metadata import Metadata
from sasdata.quantities import units
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
    def preprocess_q(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Transform the Q values before processing in the model"""
        pass

    @abstractmethod
    def postprocess_iq(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Transform the I(Q) values after running the model"""
        pass


class ComposeRequirements(ModellingRequirements):
    """Composition of two models"""

    first: ModellingRequirements
    second: ModellingRequirements

    def __init__(self, fst, snd):
        self.first = fst
        self.second = snd

    def preprocess_q(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform both transformations in order"""
        return self.second.preprocess_q(
            self.first.preprocess_q(data, metadata), metadata
        )

    def postprocess_iq(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform both transformations in order"""
        return self.second.postprocess_iq(
            self.first.postprocess_iq(data, metadata), metadata
        )


class SesansModel(ModellingRequirements):
    """Perform Hankel transform for SESANS"""

    def preprocess_q(self, SElength: np.ndarray, full_data: SasData) -> np.ndarray:
        """Calculate the q values needed to perform the Hankel transform

        Note: this is undefined for the case when SElengths contains
        exactly one element and that values is zero.

        """
        # FIXME: Actually do the Hankel transform
        SElength = np.asarray(SElength)
        if len(SElength) == 1:
            q_min, q_max = 0.01 * 2 * pi / SElength[-1], 10 * 2 * pi / SElength[0]
        else:
            # TODO: Why does q_min depend on the number of correlation lengths?
            # TODO: Why does q_max depend on the correlation step size?
            q_min = 0.1 * 2 * np.pi / (np.size(SElength) * SElength[-1])
            q_max = 2 * np.pi / (SElength[1] - SElength[0])

        # TODO: Possibly make this adjustable
        log_spacing = 1.0003
        self.q = np.exp(np.arange(np.log(q_min), np.log(q_max), np.log(log_spacing)))

        dq = np.diff(self.q)
        dq = np.insert(dq, 0, dq[0])

        self.H0 = dq / (2 * np.pi) * self.q

        self.H = np.outer(self.q, SElength)
        j0(self.H, out=self.H)
        self.H *= (dq * self.q / (2 * np.pi)).reshape((-1, 1))

        reptheta = np.outer(
            self.q,
            full_data._data_contents["Wavelength"].in_units_of(units.angstroms)
            / (2 * np.pi),
        )
        # Note: Using inplace update with reptheta => arcsin(reptheta).
        # When q L / 2 pi > 1 that means wavelength is too large to
        # reach that q value at any angle. These should produce theta = NaN
        # without any warnings.
        #
        # Reverse the condition to protect against NaN. We can't use
        # theta > zaccept since all comparisons with NaN return False.
        zaccept = [
            x.terms["zmax"] for x in full_data.metadata.process if "zmax" in x.terms
        ][0]
        with np.errstate(invalid="ignore"):
            mask = ~(np.arcsin(reptheta) <= zaccept.in_units_of(units.radians))
        self.H[mask] = 0

        return self.q

    def postprocess_iq(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """
        Apply the SESANS transform to the computed I(q)
        """
        G0 = np.dot(self.H0, data)
        G = np.dot(self.H.T, data)
        P = G - G0
        return P


class SmearModel(ModellingRequirements):
    """Perform a slit smearing"""

    def preprocess_q(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform smearing transform"""
        # FIXME: Actually do the smearing transform
        return data

    def postprocess_iq(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform smearing transform"""
        # FIXME: Actually do the smearing transform
        return data


class NullModel(ModellingRequirements):
    """A model that does nothing"""

    def compose(self, other: ModellingRequirements) -> ModellingRequirements:
        return other

    def preprocess_q(self, data: np.ndarray, _full_data: SasData) -> np.ndarray:
        """Do nothing"""
        return data

    def postprocess_iq(self, data: np.ndarray, _full_data: SasData) -> np.ndarray:
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
    """Null model is the identity element of composition"""
    return first


@compose.register
def _(second: SesansModel, first: ModellingRequirements) -> ModellingRequirements:
    match first:
        case SmearModel():
            # To the first approximation, there is no slit smearing in SESANS data
            return second
        case _:
            return ComposeRequirements(first, second)
