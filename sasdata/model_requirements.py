from abc import ABC, abstractmethod

from functools import singledispatch
import numpy as np
from typing import Self
from scipy.special import j0, erf

from sasdata.data import SasData
from sasdata.quantities import units
from sasdata.quantities.quantity import Quantity
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
    def preprocess_q(self, data: Quantity[np.ndarray], full_data: SasData) -> np.ndarray:
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

    def preprocess_q(self, data: Quantity[np.ndarray], full_data: SasData) -> np.ndarray:
        """Perform both transformations in order"""
        return self.second.preprocess_q(
            self.first.preprocess_q(data, full_data), full_data
        )

    def postprocess_iq(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform both transformations in order"""
        return self.second.postprocess_iq(
            self.first.postprocess_iq(data, full_data), full_data
        )


class SesansModel(ModellingRequirements):
    """Perform Hankel transform for SESANS"""

    def preprocess_q(self, spin_echo_length: Quantity[np.ndarray], full_data: SasData) -> np.ndarray:
        """Calculate the q values needed to perform the Hankel transform

        Note: this is undefined for the case when spin_echo_lengths contains
        exactly one element and that values is zero.

        """
        # FIXME: Actually do the Hankel transform
        spin_echo_length = spin_echo_length.in_units_of(units.angstroms)
        if len(spin_echo_length) == 1:
            q_min, q_max = 0.01 * 2 * np.pi / spin_echo_length[-1], 10 * 2 * np.pi / spin_echo_length[0]
        else:
            # TODO: Why does q_min depend on the number of correlation lengths?
            # TODO: Why does q_max depend on the correlation step size?
            q_min = 0.1 * 2 * np.pi / (np.size(spin_echo_length) * spin_echo_length[-1])
            q_max = 2 * np.pi / (spin_echo_length[1] - spin_echo_length[0])

        # TODO: Possibly make this adjustable
        log_spacing = 1.0003
        self.q = np.exp(np.arange(np.log(q_min), np.log(q_max), np.log(log_spacing)))

        dq = np.diff(self.q)
        dq = np.insert(dq, 0, dq[0])

        self.H0 = dq / (2 * np.pi) * self.q

        self.H = np.outer(self.q, spin_echo_length)
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
        G0 = self.H0 @ data
        G = self.H.T @ data
        P = G - G0

        return P


MINIMUM_RESOLUTION = 1e-8
MINIMUM_ABSOLUTE_Q = 0.02  # relative to the minimum q in the data
# According to (Barker & Pedersen 1995 JAC), 2.5 sigma is a good limit.
# According to simulations with github.com:scattering/sansresolution.git
# it is better to use asymmetric bounds (2.5, 3.0)
PINHOLE_N_SIGMA = (2.5, 3.0)

class PinholeModel(ModellingRequirements):
    """Perform a pin hole smearing"""

    def __init__(self, nsigma: (float, float) = PINHOLE_N_SIGMA):
        self.nsigma_low, self.nsigma_high = nsigma

    def preprocess_q(self, q: Quantity[np.ndarray], full_data: SasData) -> np.ndarray:
        """Perform smearing transform"""
        # FIXME: Actually do the smearing transform
        self.q, self.q_width = q.in_units_of_with_standard_error(units.per_angstrom)
        q_min = np.min(self.q - self.nsigma_low * self.q_width)
        q_max = np.max(self.q + self.nsigma_high * self.q_width)


        self.q_calc = linear_extrapolation(self.q, q_min, q_max)

        # Protect against models which are not defined for very low q.  Limit
        # the smallest q value evaluated (in absolute) to 0.02*min
        cutoff = MINIMUM_ABSOLUTE_Q*np.min(self.q)
        self.q_calc = self.q_calc[abs(self.q_calc) >= cutoff]

        # Build weight matrix from calculated q values
        self.weight_matrix = pinhole_resolution(
            self.q_calc, self.q, np.maximum(self.q_width, MINIMUM_RESOLUTION),
            nsigma=(self.nsigma_low, self.nsigma_high))

        return np.abs(self.q_calc)


    def postprocess_iq(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform smearing transform"""
        print(self.weight_matrix.shape)
        return self.weight_matrix.T @ data


class NullModel(ModellingRequirements):
    """A model that does nothing"""

    def compose(self, other: ModellingRequirements) -> ModellingRequirements:
        return other

    def preprocess_q(self, data: Quantity[np.ndarray], _full_data: SasData) -> np.ndarray:
        """Do nothing"""
        return data

    def postprocess_iq(self, data: np.ndarray, _full_data: SasData) -> np.ndarray:
        """Do nothing"""
        return data

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
        case PinholeModel():
            # To the first approximation, there is no slit smearing in SESANS data
            return second
        case _:
            return ComposeRequirements(first, second)


def linear_extrapolation(q, q_min, q_max):
    """
    Extrapolate *q* out to [*q_min*, *q_max*] using the step size in *q* as
    a guide.  Extrapolation below uses about the same size as the first
    interval.  Extrapolation above uses about the same size as the final
    interval.

    Note that extrapolated values may be negative.
    """
    q = np.sort(q)
    if q_min + 2*MINIMUM_RESOLUTION < q[0]:
        delta = q[1] - q[0] if len(q) > 1 else 0
        n_low = int(np.ceil((q[0]-q_min) / delta)) if delta > 0 else 15
        q_low = np.linspace(q_min, q[0], n_low+1)[:-1]
    else:
        q_low = []
    if q_max - 2*MINIMUM_RESOLUTION > q[-1]:
        delta = q[-1] - q[-2] if len(q) > 1 else 0
        n_high = int(np.ceil((q_max-q[-1]) / delta)) if delta > 0 else 15
        q_high = np.linspace(q[-1], q_max, n_high+1)[1:]
    else:
        q_high = []
    return np.concatenate([q_low, q, q_high])


def pinhole_resolution(q_calc: np.ndarray, q: np.ndarray, q_width: np.ndarray, nsigma=PINHOLE_N_SIGMA) -> np.ndarray:
    r"""
    Compute the convolution matrix *W* for pinhole resolution 1-D data.

    Each row *W[i]* determines the normalized weight that the corresponding
    points *q_calc* contribute to the resolution smeared point *q[i]*.  Given
    *W*, the resolution smearing can be computed using *dot(W,q)*.

    Note that resolution is limited to $\pm 2.5 \sigma$.[1]  The true resolution
    function is a broadened triangle, and does not extend over the entire
    range $(-\infty, +\infty)$.  It is important to impose this limitation
    since some models fall so steeply that the weighted value in gaussian
    tails would otherwise dominate the integral.

    *q_calc* must be increasing.  *q_width* must be greater than zero.

    [1] Barker, J. G., and J. S. Pedersen. 1995. Instrumental Smearing Effects
    in Radially Symmetric Small-Angle Neutron Scattering by Numerical and
    Analytical Methods. Journal of Applied Crystallography 28 (2): 105--14.
    https://doi.org/10.1107/S0021889894010095.
    """
    # The current algorithm is a midpoint rectangle rule.  In the test case,
    # neither trapezoid nor Simpson's rule improved the accuracy.
    edges = bin_edges(q_calc)
    #edges[edges < 0.0] = 0.0 # clip edges below zero
    cdf = erf((edges[:, None] - q[None, :]) / (np.sqrt(2.0)*q_width)[None, :])
    weights = cdf[1:] - cdf[:-1]
    # Limit q range to (-2.5,+3) sigma
    try:
        nsigma_low, nsigma_high = nsigma
    except TypeError:
        nsigma_low = nsigma_high = nsigma
    qhigh = q + nsigma_high*q_width
    qlow = q - nsigma_low*q_width  # linear limits
    ##qlow = q*q/qhigh  # log limits
    weights[q_calc[:, None] < qlow[None, :]] = 0.
    weights[q_calc[:, None] > qhigh[None, :]] = 0.
    weights /= np.sum(weights, axis=0)[None, :]
    return weights

def bin_edges(x):
    """
    Determine bin edges from bin centers, assuming that edges are centered
    between the bins.

    Note: this uses the arithmetic mean, which may not be appropriate for
    log-scaled data.
    """
    if len(x) < 2 or (np.diff(x) < 0).any():
        raise ValueError("Expected bins to be an increasing set")
    edges = np.hstack([
        x[0]  - 0.5*(x[1]  - x[0]),  # first point minus half first interval
        0.5*(x[1:] + x[:-1]),        # mid points of all central intervals
        x[-1] + 0.5*(x[-1] - x[-2]), # last point plus half last interval
        ])
    return edges
