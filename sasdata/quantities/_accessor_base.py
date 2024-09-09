from typing import TypeVar, Sequence

from sasdata.quantities.quantity import BaseQuantity
import sasdata.quantities.units as units
from sasdata.quantities.units import Dimensions, Unit


DataType = TypeVar("DataType")
OutputType = TypeVar("OutputType")

class Accessor[DataType, OutputType]:
    """ Base class """
    def __init__(self, target_object, value_target: str):
        self.target_object = target_object
        self.value_target = value_target

    @property
    def value(self) -> OutputType | None:
        pass

class StringAccessor(Accessor[str, str]):
    """ String based fields """
    @property
    def value(self) -> str | None:
        pass

class FloatAccessor(Accessor[float, float]):
    """ Float based fields """
    @property
    def value(self) -> float | None:
        pass




class QuantityAccessor[DataType](Accessor[DataType, BaseQuantity[DataType]]):
    """ Base class for accessors that work with quantities that have units """
    def __init__(self, target_object, value_target: str, unit_target: str, default_unit=None):
        super().__init__(target_object, value_target)
        self._unit_target = unit_target
        self.default_unit = default_unit

    def _numerical_part(self) -> DataType | None:
        """ Numerical part of the data """

    def _unit_part(self) -> str | None:
        """ String form of units for the data """

    @property
    def unit(self) -> Unit:
        if self._unit_part() is None:
            return self.default_unit
        else:
            return Unit.parse(self._unit_part())

    @property
    def value(self) -> BaseQuantity[DataType] | None:
        if self._unit_part() is not None and self._numerical_part() is not None:
            return BaseQuantity(self._numerical_part(), self.unit)


