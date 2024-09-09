from dataclasses import dataclass
from sasdata.quantities.quantity import BaseQuantity, NamedQuantity
from sasdata.metadata import Metadata

import numpy as np

from sasdata.model_requirements import ModellingRequirements


class SasData:
    def __init__(self, name: str,
                 data_contents: list[Quantity],
                 raw_metadata: Group,
                 instrument: Instrument,
                 verbose: bool=False):

        self.name = name
        self._data_contents = data_contents
        self._raw_metadata = raw_metadata
        self._verbose = verbose

@dataclass
class DataSet:
    abscissae: list[NamedQuantity[np.ndarray]]
    ordinate: NamedQuantity[np.ndarray]
    other: list[NamedQuantity[np.ndarray]]

    metadata: Metadata
    model_requirements: ModellingRequirements
