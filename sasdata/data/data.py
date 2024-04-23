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


class Data(DataMixin):
    # TODO: Define more for this class as more functionality is required.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SASData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__()


class PinholeData(SASData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution_matrix = None


class SlitData(SASData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SESANSData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

