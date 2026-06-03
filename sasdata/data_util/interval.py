from enum import Enum, auto

import numpy as np


class IntervalType(Enum):
    HALF_OPEN = auto()
    CLOSED = auto()

    def weights_for_interval(self, array, l_bound, u_bound):
        """
        Weight coordinate data by position relative to a specified interval.

        :param array: the array for which the weights are calculated
        :param l_bound: value defining the lower limit of the region of interest
        :param u_bound: value defining the upper limit of the region of interest

        If and when fractional binning is implemented (ask Lucas), this function
        will be changed so that instead of outputting zeros and ones, it gives
        fractional values instead. These will depend on how close the array value
        is to being within the interval defined.
        """

        # Whether the endpoint should be included depends on circumstance.
        # Half-open is used when binning the major axis (except for the final bin)
        # and closed used for the minor axis and the final bin of the major axis.
        if self.name.lower() == 'half_open':
            in_range = np.logical_and(l_bound <= array, array < u_bound)
        elif self.name.lower() == 'closed':
            in_range = np.logical_and(l_bound <= array, array <= u_bound)
        else:
            msg = f"Unrecognised interval_type: {self.name}"
            raise ValueError(msg)

        return np.asarray(in_range, dtype=int)
