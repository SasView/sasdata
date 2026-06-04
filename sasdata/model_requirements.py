from abc import ABC, abstractmethod
from functools import singledispatch
from typing import Self

import numpy as np
from scipy.special import erf, j0

from sasdata import dataset_types
from sasdata.data import SasData
from sasdata.quantities import units
from sasdata.quantities.quantity import Operation, Quantity


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
            self.first.preprocess_q(data, full_data), full_data
        )

    def postprocess_iq(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform both transformations in order"""
        return self.second.postprocess_iq(
            self.first.postprocess_iq(data, full_data), full_data
        )


class SesansModel(ModellingRequirements):
    """Perform Hankel transform for SESANS"""

    def preprocess_q(
        self, spin_echo_length: np.ndarray, full_data: SasData
    ) -> np.ndarray:
        """Calculate the q values needed to perform the Hankel transform

        Note: this is undefined for the case when spin_echo_lengths contains
        exactly one element and that values is zero.

        """
        if len(spin_echo_length) == 1:
            q_min, q_max = (
                0.01 * 2 * np.pi / spin_echo_length[-1],
                10 * 2 * np.pi / spin_echo_length[0],
            )
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

    def __init__(self, q_width: np.ndarray, nsigma: (float, float) = PINHOLE_N_SIGMA):
        self.q_width = q_width
        self.nsigma_low, self.nsigma_high = nsigma

    def preprocess_q(self, q: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform smearing transform"""
        self.q = q
        q_min = np.min(self.q - self.nsigma_low * self.q_width)
        q_max = np.max(self.q + self.nsigma_high * self.q_width)

        self.q_calc = linear_extrapolation(self.q, q_min, q_max)

        # Protect against models which are not defined for very low q.  Limit
        # the smallest q value evaluated (in absolute) to 0.02*min
        cutoff = MINIMUM_ABSOLUTE_Q * np.min(self.q)
        self.q_calc = self.q_calc[abs(self.q_calc) >= cutoff]

        # Build weight matrix from calculated q values
        self.weight_matrix = pinhole_resolution(
            self.q_calc,
            self.q,
            np.maximum(self.q_width, MINIMUM_RESOLUTION),
            nsigma=(self.nsigma_low, self.nsigma_high),
        )

        return np.abs(self.q_calc)

    def postprocess_iq(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform smearing transform"""
        return self.weight_matrix.T @ data


class SlitModel(ModellingRequirements):
    """Perform a slit smearing"""

    def __init__(
        self,
        q_length: np.ndarray,
        q_width: np.ndarray,
        nsigma: (float, float) = PINHOLE_N_SIGMA,
    ):
        self.q_length = q_length
        self.q_width = q_width
        self.nsigma_low, self.nsigma_high = nsigma

    def preprocess_q(self, q: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform smearing transform"""
        self.q = q
        q_min = np.min(self.q - self.nsigma_low * self.q_width)
        q_max = np.max(self.q + self.nsigma_high * self.q_width)

        self.q_calc = slit_extend_q(self.q, self.q_width, self.q_length)

        # Protect against models which are not defined for very low q.  Limit
        # the smallest q value evaluated (in absolute) to 0.02*min
        cutoff = MINIMUM_ABSOLUTE_Q * np.min(self.q)
        self.q_calc = self.q_calc[abs(self.q_calc) >= cutoff]

        # Build weight matrix from calculated q values
        self.weight_matrix = slit_resolution(
            self.q_calc, self.q, self.q_length, self.q_width
        )

        return np.abs(self.q_calc)

    def postprocess_iq(self, data: np.ndarray, full_data: SasData) -> np.ndarray:
        """Perform smearing transform"""
        return self.weight_matrix.T @ data


class NullModel(ModellingRequirements):
    """A model that does nothing"""

    def compose(self, other: ModellingRequirements) -> ModellingRequirements:
        return other

    def preprocess_q(
        self, data: Quantity[np.ndarray], _full_data: SasData
    ) -> np.ndarray:
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
        case PinholeModel() | SlitModel():
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
    delta_low = q[1] - q[0] if len(q) > 1 else 0
    n_low = int(np.ceil((q[0] - q_min) / delta_low)) if delta_low > 0 else 15
    q_low = np.linspace(q_min, q[0], n_low + 1)[:-1] if q_min + 2 * MINIMUM_RESOLUTION >= q[0] else []

    delta_high = q[-1] - q[-2] if len(q) > 1 else 0
    n_high = int(np.ceil((q_max - q[-1]) / delta_high)) if delta_high > 0 else 15
    q_high = np.linspace(q[-1], q_max, n_high + 1)[1:] if q_max - 2 * MINIMUM_RESOLUTION <= q[-1] else []
    return np.concatenate([q_low, q, q_high])


def pinhole_resolution(
    q_calc: np.ndarray, q: np.ndarray, q_width: np.ndarray, nsigma=PINHOLE_N_SIGMA
) -> np.ndarray:
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
    # edges[edges < 0.0] = 0.0 # clip edges below zero
    cdf = erf((edges[:, None] - q[None, :]) / (np.sqrt(2.0) * q_width)[None, :])
    weights = cdf[1:] - cdf[:-1]
    # Limit q range to (-2.5,+3) sigma
    try:
        nsigma_low, nsigma_high = nsigma
    except TypeError:
        nsigma_low = nsigma_high = nsigma
    qhigh = q + nsigma_high * q_width
    qlow = q - nsigma_low * q_width  # linear limits
    ##qlow = q*q/qhigh  # log limits
    weights[q_calc[:, None] < qlow[None, :]] = 0.0
    weights[q_calc[:, None] > qhigh[None, :]] = 0.0
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
    edges = np.hstack(
        [
            x[0] - 0.5 * (x[1] - x[0]),  # first point minus half first interval
            0.5 * (x[1:] + x[:-1]),  # mid points of all central intervals
            x[-1] + 0.5 * (x[-1] - x[-2]),  # last point plus half last interval
        ]
    )
    return edges


def slit_resolution(q_calc, q, width, length, n_length=30):
    r"""
    Build a weight matrix to compute *I_s(q)* from *I(q_calc)*, given
    $q_\perp$ = *width* (in the high-resolution axis) and $q_\parallel$
    = *length* (in the low resolution axis).  *n_length* is the number
    of steps to use in the integration over $q_\parallel$ when both
    $q_\perp$ and $q_\parallel$ are non-zero.

    Each $q$ can have an independent width and length value even though
    current instruments use the same slit setting for all measured points.

    If slit length is large relative to width, use:

    .. math::

        I_s(q_i) = \frac{1}{\Delta q_\perp}
            \int_0^{\Delta q_\perp}
                I\left(\sqrt{q_i^2 + q_\perp^2}\right) \,dq_\perp

    If slit width is large relative to length, use:

    .. math::

        I_s(q_i) = \frac{1}{2 \Delta q_\parallel}
            \int_{-\Delta q_\parallel}^{\Delta q_\parallel}
                I\left(|q_i + q_\parallel|\right) \,dq_\parallel

    For a mixture of slit width and length use:

    .. math::

        I_s(q_i) = \frac{1}{2 \Delta q_\parallel \Delta q_\perp}
            \int_{-\Delta q_\parallel}^{\Delta q_\parallel}
            \int_0^{\Delta q_\perp}
                I\left(\sqrt{(q_i + q_\parallel)^2 + q_\perp^2}\right)
                \,dq_\perp dq_\parallel

    **Definition**

    We are using the mid-point integration rule to assign weights to each
    element of a weight matrix $W$ so that

    .. math::

        I_s(q) = W\,I(q_\text{calc})

    If *q_calc* is at the mid-point, we can infer the bin edges from the
    pairwise averages of *q_calc*, adding the missing edges before
    *q_calc[0]* and after *q_calc[-1]*.

    For $q_\parallel = 0$, the smeared value can be computed numerically
    using the $u$ substitution

    .. math::

        u_j = \sqrt{q_j^2 - q^2}

    This gives

    .. math::

        I_s(q) \approx \sum_j I(u_j) \Delta u_j

    where $I(u_j)$ is the value at the mid-point, and $\Delta u_j$ is the
    difference between consecutive edges which have been first converted
    to $u$.  Only $u_j \in [0, \Delta q_\perp]$ are used, which corresponds
    to $q_j \in \left[q, \sqrt{q^2 + \Delta q_\perp}\right]$, so

    .. math::

        W_{ij} = \frac{1}{\Delta q_\perp} \Delta u_j
               = \frac{1}{\Delta q_\perp} \left(
                    \sqrt{q_{j+1}^2 - q_i^2} - \sqrt{q_j^2 - q_i^2} \right)
            \ \text{if}\  q_j \in \left[q_i, \sqrt{q_i^2 + q_\perp^2}\right]

    where $I_s(q_i)$ is the theory function being computed and $q_j$ are the
    mid-points between the calculated values in *q_calc*.  We tweak the
    edges of the initial and final intervals so that they lie on integration
    limits.

    (To be precise, the transformed midpoint $u(q_j)$ is not necessarily the
    midpoint of the edges $u((q_{j-1}+q_j)/2)$ and $u((q_j + q_{j+1})/2)$,
    but it is at least in the interval, so the approximation is going to be
    a little better than the left or right Riemann sum, and should be
    good enough for our purposes.)

    For $q_\perp = 0$, the $u$ substitution is simpler:

    .. math::

        u_j = \left|q_j - q\right|

    so

    .. math::

        W_{ij} = \frac{1}{2 \Delta q_\parallel} \Delta u_j
            = \frac{1}{2 \Delta q_\parallel} (q_{j+1} - q_j)
            \ \text{if}\ q_j \in
                \left[q-\Delta q_\parallel, q+\Delta q_\parallel\right]

    However, we need to support cases were $u_j < 0$, which means using
    $2 (q_{j+1} - q_j)$ when $q_j \in \left[0, q_\parallel-q_i\right]$.
    This is not an issue for $q_i > q_\parallel$.

    For both $q_\perp > 0$ and $q_\parallel > 0$ we perform a 2 dimensional
    integration with

    .. math::

        u_{jk} = \sqrt{q_j^2 - (q + (k\Delta q_\parallel/L))^2}
            \ \text{for}\ k = -L \ldots L

    for $L$ = *n_length*.  This gives

    .. math::

        W_{ij} = \frac{1}{2 \Delta q_\perp q_\parallel}
            \sum_{k=-L}^L \Delta u_{jk}
                \left(\frac{\Delta q_\parallel}{2 L + 1}\right)


    """

    # The current algorithm is a midpoint rectangle rule.
    q_edges = bin_edges(q_calc)  # Note: requires q > 0
    weights = np.zeros((len(q), len(q_calc)), "d")

    for i, (qi, w, l) in enumerate(zip(q, width, length)):
        if w == 0.0 and l == 0.0:
            # Perfect resolution, so return the theory value directly.
            # Note: assumes that q is a subset of q_calc.  If qi need not be
            # in q_calc, then we can do a weighted interpolation by looking
            # up qi in q_calc, then weighting the result by the relative
            # distance to the neighbouring points.
            weights[i, :] = q_calc == qi
        elif l == 0:
            weights[i, :] = _q_perp_weights(q_edges, qi, w)
        elif w == 0 and qi >= l:
            in_x = 1.0 * ((q_calc >= qi - l) & (q_calc <= qi + l))
            weights[i, :] = in_x * np.diff(q_edges) / (2 * l)
        elif w == 0:
            in_x = 1.0 * ((q_calc >= qi - l) & (q_calc <= qi + l))
            abs_x = 1.0 * (q_calc < abs(qi - l))
            weights[i, :] = (in_x + abs_x) * np.diff(q_edges) / (2 * l)
        else:
            weights[i, :] = _q_perp_weights(
                q_edges, qi + np.arange(-n_length, n_length + 1) * l / n_length, w
            )
            weights[i, :] /= 2 * n_length + 1

    return weights.T


def _q_perp_weights(q_edges, qi, w):
    q_edges = np.reshape(q_edges, (1, -1))
    qi = np.reshape(qi, (-1, 1))
    # Convert bin edges from q to u
    u_limit = np.sqrt(qi**2 + w**2)
    u_edges = q_edges**2 - qi**2
    u_edges[q_edges < abs(qi)] = 0.0
    u_edges[q_edges > u_limit] = np.repeat(
        u_limit**2 - qi**2, u_edges.shape[1], axis=1
    )[q_edges > u_limit]
    return (np.diff(np.sqrt(u_edges), axis=1) / w).sum(axis=0)


def slit_extend_q(q, width, length):
    """
    Given *q*, *width* and *length*, find a set of sampling points *q_calc* so
    that each point I(q) has sufficient support from the underlying
    function.
    """
    q_min, q_max = np.min(q - length), np.max(np.sqrt((q + length) ** 2 + width**2))

    return geometric_extrapolation(q, q_min, q_max)


def geometric_extrapolation(q, q_min, q_max, points_per_decade=None):
    r"""
    Extrapolate *q* to [*q_min*, *q_max*] using geometric steps, with the
    average geometric step size in *q* as the step size.

    if *q_min* is zero or less then *q[0]/10* is used instead.

    *points_per_decade* sets the ratio between consecutive steps such
    that there will be $n$ points used for every factor of 10 increase
    in *q*.

    If *points_per_decade* is not given, it will be estimated as follows.
    Starting at $q_1$ and stepping geometrically by $\Delta q$ to $q_n$
    in $n$ points gives a geometric average of:

    .. math::

         \log \Delta q = (\log q_n - \log q_1) / (n - 1)

    From this we can compute the number of steps required to extend $q$
    from $q_n$ to $q_\text{max}$ by $\Delta q$ as:

    .. math::

         n_\text{extend} = (\log q_\text{max} - \log q_n) / \log \Delta q

    Substituting:

    .. math::

         n_\text{extend} = (n-1) (\log q_\text{max} - \log q_n)
            / (\log q_n - \log q_1)
    """
    DEFAULT_POINTS_PER_DECADE = 10
    q = np.sort(q)
    data_min, data_max = q[0], q[-1]
    if points_per_decade is None:
        if data_max > data_min:
            log_delta_q = (len(q) - 1) / (np.log(data_max) - np.log(data_min))
        else:
            log_delta_q = np.log(10.0) / DEFAULT_POINTS_PER_DECADE
    else:
        log_delta_q = np.log(10.0) / points_per_decade
    if q_min <= 0:
        q_min = data_min * MINIMUM_ABSOLUTE_Q
    if q_min < data_min:
        n_low = int(np.ceil(log_delta_q * (np.log(data_min) - np.log(q_min))))
        q_low = np.logspace(np.log10(q_min), np.log10(data_min), n_low + 1)[:-1]
    else:
        q_low = []
    if q_max > data_max:
        n_high = int(np.ceil(log_delta_q * (np.log(q_max) - np.log(data_max))))
        q_high = np.logspace(np.log10(data_max), np.log10(q_max), n_high + 1)[1:]
    else:
        q_high = []
    return np.concatenate([q_low, q, q_high])
