from typing import TypeVar

from sasdata.quantities.quantity import Quantity
import sasdata.quantities.units as units


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

class QuantityAccessor[DataType](Accessor[DataType, Quantity[DataType]]):
    """ Base class for accessors that work with quantities that have units """
    def __init__(self, target_object, value_target: str, unit_target: str, default_unit=None):
        super().__init__(target_object, value_target)
        self._unit_target = unit_target
        self.default_unit = default_unit

    def _lookup_unit(self) -> units.Unit | None:
        # TODO: Implement
        return None

    def data_unit(self):
        unit = self._lookup_unit
        if unit is None:
            return self.default_unit
        else:
            return unit


    @property
    def quantity(self) -> Quantity[DataType]:
        raise NotImplementedError("Not implemented yet")

