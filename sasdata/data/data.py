from dataclasses import dataclass
import numpy as np
from sasdata.condition.condition_meta import ConditionVector


@dataclass
class DataMixin:
    """Class for storing as-loaded data from any source."""
    # An array of dependent values (n-dimensional)
    dependent: np.ndarray
    # An array of resolutions (known uncertainties) for the dependent variables
    resolution: np.ndarray
    # An array of (systematic) uncertainties for the dependent variables
    uncertainties: np.ndarray
    # An array of signal values (m datasets, each n-dimensional)
    signal: np.ndarray
    # An array of uncertainties for the signal values where np.shape(noise) == np.shape(signal)
    noise: np.ndarray
    # A series of external conditions applied during the measurement that resulted in this data set.
    conditions: ConditionVector
