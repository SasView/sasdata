import numpy as np
from sasdata.quantities.quantity import BaseQuantity

class Operation:
    """ Sketch of what model post-processing classes might look like """

    children: list["Operation"]
    named_children: dict[str, "Operation"]

    @property
    def name(self) -> str:
        raise NotImplementedError("No name for transform")

    def evaluate(self) -> BaseQuantity[np.ndarray]:
        pass

    def __call__(self, *children, **named_children):
        self.children = children
        self.named_children = named_children