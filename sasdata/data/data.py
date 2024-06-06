from dataclasses import dataclass
import numpy as np
from sasdata.condition.condition_meta import ConditionVector


# TODO: The following items are still needed:
#   - DataMixin.id: A unique id that can be used to map Data into Trends and Views to Data
#   - DataMixin.views: An object that holds the direct child Views associated with the Data
#   - Differentiate as-loaded data from plottable data (this would not be a datclass, just a separate class)
#   - Create the 'View' class - This could be a subclass of the plottable data - that holds transforms of the Data
#       - should subclass DataMixin and have an attribute that holds the DataMixin.id value the View was generated from
#   - Incorporate the resolution functions (maybe post-1.0?)
#   - Create a data manager that lives in this package that holds all Data, Views, and Trends (DataModel?)
#   - Convert Reader and Writer classes to Import and Export classes
#       - Convert them all to use the newest Data class
#   - Define specific Condition subclasses for often-used sample environments and sample conditions (maybe post-1.0?)
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

