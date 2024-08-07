from typing import TypeVar

from numpy._typing import ArrayLike

from sasdata.quantities.quantity import Unit, Quantity


class RawMetaData:
    pass


FieldDataType = TypeVar("FieldDataType")
OutputDataType = TypeVar("OutputDataType")

class Accessor[FieldDataType, OutputDataType]:
    def __init__(self, target_field: str):
        self._target_field = target_field

    def _raw_values(self) -> FieldDataType:
        raise NotImplementedError("not implemented in base class")

    @property
    def value(self) -> OutputDataType:
        raise NotImplementedError("value not implemented in base class")



class QuantityAccessor(Accessor[ArrayLike, Quantity[ArrayLike]]):
    def __init__(self, target_field: str, units_field: str | None = None):
        super().__init__(target_field)
        self._units_field = units_field

    def _units(self) -> Unit:
        pass

    def _raw_values(self) -> ArrayLike:
        pass

    @property
    def value(self) -> Quantity[ArrayLike]:
        return Quantity(self._raw_values(), self._units())


class StringAccessor(Accessor[str, str]):

    def _raw_values(self) -> str:
        pass

    @property
    def value(self) -> str:
        return self._raw_values()

#
# Quantity specific accessors, provides helper methods for quantities with known dimensionality
#

class LengthAccessor(QuantityAccessor):
    @property
    def m(self):
        return self.value.in_units_of("m")


class TimeAccessor(QuantityAccessor):
    pass


class TemperatureAccessor(QuantityAccessor):
    pass


class AbsoluteTemperatureAccessor(QuantityAccessor):
    pass


#
# Main metadata object
#


class MetaData:
    def __init__(self, raw: RawMetaData):
        self._raw = raw

    # Put the structure of the metadata that should be exposed to a power-user / developer in here
