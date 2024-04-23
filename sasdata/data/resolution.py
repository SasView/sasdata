import numpy as np

from sasmodels.resolution import Slit1D, Pinhole1D
from sasmodels.sesans import SesansTransform
from sasmodels.resolution2d import Pinhole2D


class PySmear:
    """Wrapper for pure python sasmodels resolution functions."""
    def __init__(self, resolution, model, offset=None):
        self.model = model
        self.resolution = resolution
        if offset is None:
            offset = np.searchsorted(self.resolution.q_calc, self.resolution.q[0])
        self.offset = offset

    def apply(self, iq_in, first_bin=0, last_bin=None):
        """
        Apply the resolution function to the data.
        Note that this is called with iq_in matching data.x, but with
        iq_in[first_bin:last_bin] set to theory values for these bins,
        and the remainder left undefined.  The first_bin, last_bin values
        should be those returned from get_bin_range.
        The returned value is of the same length as iq_in, with the range
        first_bin:last_bin set to the resolution smeared values.
        """
        if last_bin is None: last_bin = len(iq_in)
        start, end = first_bin + self.offset, last_bin + self.offset
        q_calc = self.resolution.q_calc
        iq_calc = np.empty_like(q_calc)
        if start > 0:
            iq_calc[:start] = self.model.evalDistribution(q_calc[:start])
        if end+1 < len(q_calc):
            iq_calc[end+1:] = self.model.evalDistribution(q_calc[end+1:])
        iq_calc[start:end+1] = iq_in[first_bin:last_bin+1]
        smeared = self.resolution.apply(iq_calc)
        return smeared

    __call__ = apply

    def get_bin_range(self, q_min=None, q_max=None):
        """
        For a given q_min, q_max, find the corresponding indices in the data.
        Returns first, last.
        Note that these are indexes into q from the data, not the q_calc
        needed by the resolution function.  Note also that these are the
        indices, not the range limits.  That is, the complete range will be
        q[first:last+1].
        """
        q = self.resolution.q
        first = np.searchsorted(q, q_min)
        last = np.searchsorted(q, q_max)
        return first, min(last,len(q)-1)