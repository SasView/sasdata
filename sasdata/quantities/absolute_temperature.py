from typing import TypeVar

from quantities.quantity import Quantity
from sasdata.quantities.accessors import TemperatureAccessor


DataType = TypeVar("DataType")
class AbsoluteTemperatureAccessor(TemperatureAccessor[DataType]):
    """ Parsing for absolute temperatures """
    @property
    def value(self) -> Quantity[DataType] | None:
        if self._numerical_part() is None:
            return None
        else:
            return Quantity.parse(self._numerical_part(), self._unit_part(), absolute_temperature=True)