from typing import Generic, TypeVar

from numpy._typing import ArrayLike

from sasdata.quantities.quantities import Unit, Quantity


class RawMetaData:
    pass

class MetaData:
    pass


FieldDataType = TypeVar("FieldDataType")
OutputDataType = TypeVar("OutputDataType")

class Accessor(Generic[FieldDataType, OutputDataType]):
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

    def _get_units(self) -> Unit:
        pass

    def _raw_values(self) -> ArrayLike:
        pass


class StringAccessor(Accessor[str]):
    @property
    def value(self) -> str:
        return self._raw_values()


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