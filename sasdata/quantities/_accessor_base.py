from typing import TypeVar, Sequence

from sasdata.quantities.quantity import Quantity
import sasdata.quantities.units as units
from sasdata.quantities.units import Dimensions, Unit

from sasdata.data_backing import Group, Dataset

DataType = TypeVar("DataType")
OutputType = TypeVar("OutputType")

class AccessorTarget:
    def __init__(self, data: Group):
        self._data = data

    def get_value(self, path: str):

        tokens = path.split(".")

        # Navigate the tree from the entry we need

        current_tree_position: Group | Dataset = self._data

        for token in tokens:
            if isinstance(current_tree_position, Group):
                current_tree_position = current_tree_position.children[token]
            elif isinstance(current_tree_position, Dataset):
                current_tree_position = current_tree_position.attributes[token]

        return current_tree_position



class Accessor[DataType, OutputType]:
    """ Base class """
    def __init__(self, target_object: AccessorTarget, value_target: str):
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




class QuantityAccessor[DataType](Accessor[DataType, Quantity[DataType]]):
    """ Base class for accessors that work with quantities that have units """
    def __init__(self, target_object: AccessorTarget, value_target: str, unit_target: str, default_unit=units.none):
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
    def value(self) -> Quantity[DataType] | None:
        if self._unit_part() is not None and self._numerical_part() is not None:
            return Quantity(self._numerical_part(), self.unit)


