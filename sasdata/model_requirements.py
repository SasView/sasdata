from dataclasses import dataclass

import numpy as np

from transforms.operation import Operation


@dataclass
class ModellingRequirements:
    """ Requirements that need to be passed to any modelling step """
    dimensionality: int
    operation: Operation


    def from_qi_transformation(self, data: np.ndarray) -> np.ndaarray:
        pass




def guess_requirements(abscissae, ordinate) -> ModellingRequirements:
    """ Use names of axes and units to guess what kind of processing needs to be done """