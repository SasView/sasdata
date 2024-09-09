from dataclasses import dataclass
from sasdata.quantities.quantity import BaseQuantity, NamedQuantity
from sasdata.metadata import Metadata

import numpy as np

from sasdata.model_requirements import ModellingRequirements




@dataclass
class DataSet:
    abscissae: list[NamedQuantity[np.ndarray]]
    ordinate: NamedQuantity[np.ndarray]
    other: list[NamedQuantity[np.ndarray]]

    metadata: Metadata
    model_requirements: ModellingRequirements
