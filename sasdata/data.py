from dataclasses import dataclass
from units_temp import Quantity, NamedQuantity

import numpy as np

from sasdata.model_requirements import ModellingRequirements




@dataclass
class SASData:
    abscissae: list[NamedQuantity[np.ndarray]]
    ordinate: NamedQuantity[np.ndarray]
    other: list[NamedQuantity[np.ndarray]]

    metadata: MetaData
    model_requirements: ModellingRequirements
