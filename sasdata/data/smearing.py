from abc import ABC, abstractmethod
from sasdata.data.string_representations import format_parameters

class SmearingSpecification(ABC):
    """ Base class for Smearing"""

    @abstractmethod
    def _data_string(self) -> str:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data_string()})"

class PinholeSmearing(SmearingSpecification):
    def __init__(self, diameter):
        self.diameter = diameter

    def _data_string(self) -> str:
        return format_parameters({"diameter": self.diameter})


class SlitSmearing(SmearingSpecification):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def _data_string(self) -> str:
        return format_parameters({"width": self.width, "height": self.height})