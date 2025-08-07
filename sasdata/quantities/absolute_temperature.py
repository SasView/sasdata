from typing import TypeVar

from sasdata.quantities.accessors import TemperatureAccessor
from sasdata.quantities.quantity import Quantity

DataType = TypeVar("DataType")
class AbsoluteTemperatureAccessor(TemperatureAccessor[DataType]):
    """ Parsing for absolute temperatures """
    @property
    def value(self) -> Quantity[DataType] | None:
        if self._numerical_part() is None:
            return None
        else:
            return Quantity.parse(self._numerical_part(), self._unit_part(), absolute_temperature=True)
