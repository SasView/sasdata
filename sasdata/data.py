from dataclasses import dataclass
from quantities.quantities import Quantity, NamedQuantity
from sasdata.metadata import MetaData

import numpy as np

from sasdata.model_requirements import ModellingRequirements




@dataclass
class SASData:
    abscissae: list[NamedQuantity[np.ndarray]]
    ordinate: NamedQuantity[np.ndarray]
    other: list[NamedQuantity[np.ndarray]]

    metadata: MetaData
    model_requirements: ModellingRequirements
