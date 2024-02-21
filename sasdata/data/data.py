from dataclasses import dataclass

import numpy as np


@dataclass
class DataMixin:
    """Class for storing as-loaded data from any source."""
    # An array of dependent values (n-dimensional)
    dependent: np.ndarray
    # An array of signal values
    signal: np.ndarray
    # An array of uncertainties for the signal values where np.shape(noise) == np.shape(signal)
    noise: np.ndarray
    # An array of uncertainties for the dependent variables
    resolution: np.ndarray
    # Plottable dependent values
