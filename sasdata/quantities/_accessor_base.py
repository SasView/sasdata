from typing import TypeVar

import sasdata.quantities.units as units
from sasdata.data_backing import Dataset, Group
from sasdata.quantities.quantity import Quantity
from sasdata.quantities.unit_parser import parse_unit
from sasdata.quantities.units import Unit


# logger = logging.getLogger("Accessors")
class LoggerDummy:
    def info(self, data):
        print(data)
logger = LoggerDummy()

DataType = TypeVar("DataType")
OutputType = TypeVar("OutputType")


class AccessorTarget:
    def __init__(self, data: Group, verbose=False, prefix_tokens: tuple=()):
        self._data = data
        self.verbose = verbose

        self.prefix_tokens = list(prefix_tokens)

    def with_path_prefix(self, path_prexix: str):
        """ Get an accessor that looks at a subtree of the metadata with the supplied prefix

        For example, accessors aiming at a.b, when the target it c.d will look at c.d.a.b
        """
        return AccessorTarget(self._data,
                              verbose=self.verbose,
                              prefix_tokens=tuple(self.prefix_tokens + [path_prexix]))

    def get_value(self, path: str):

        tokens = self.prefix_tokens + path.split(".")

        if self.verbose:
            logger.info(f"Finding: {path}")
            logger.info(f"Full path: {tokens}")

        # Navigate the tree from the entry we need

        current_tree_position: Group | Dataset = self._data

        for token in tokens:

            options = token.split("|")

            if isinstance(current_tree_position, Group):

                found = False
                for option in options:
                    if option in current_tree_position.children:
                        current_tree_position = current_tree_position.children[option]
                        found = True

                        if self.verbose:
                            logger.info(f"Found option: {option}")

                if not found:
                    if self.verbose:
                        logger.info(f"Failed to find any of {options} on group {current_tree_position.name}. Options: " +
                                    ",".join([key for key in current_tree_position.children]))
                    return None

            elif isinstance(current_tree_position, Dataset):

                found = False
                for option in options:
                    if option in current_tree_position.attributes:
                        current_tree_position = current_tree_position.attributes[option]
                        found = True

                        if self.verbose:
                            logger.info(f"Found option: {option}")

                if not found:
                    if self.verbose:
                        logger.info(f"Failed to find any of {options} on attribute {current_tree_position.name}. Options: " +
                                    ",".join([key for key in current_tree_position.attributes]))
                    return None

        if self.verbose:
            logger.info(f"Found value: {current_tree_position}")

        return current_tree_position.data



class Accessor[DataType, OutputType]:
    """ Base class """
    def __init__(self, target_object: AccessorTarget, value_target: str):
        self.target_object = target_object
        self.value_target = value_target

    @property
    def value(self) -> OutputType | None:
        return self.target_object.get_value(self.value_target)

class StringAccessor(Accessor[str, str]):
    """ String based fields """
    @property
    def value(self) -> str | None:
        return self.target_object.get_value(self.value_target)

class FloatAccessor(Accessor[float, float]):
    """ Float based fields """
    @property
    def value(self) -> float | None:
        return self.target_object.get_value(self.value_target)




class QuantityAccessor[DataType](Accessor[DataType, Quantity[DataType]]):
    """ Base class for accessors that work with quantities that have units """
    def __init__(self, target_object: AccessorTarget, value_target: str, unit_target: str, default_unit=units.none):
        super().__init__(target_object, value_target)
        self._unit_target = unit_target
        self.default_unit = default_unit

    def _numerical_part(self) -> DataType | None:
        """ Numerical part of the data """
        return self.target_object.get_value(self.value_target)

    def _unit_part(self) -> str | None:
        """ String form of units for the data """
        return self.target_object.get_value(self._unit_target)

    @property
    def unit(self) -> Unit:
        u = self._unit_part()
        if u is None:
            return self.default_unit
        else:
            return parse_unit(u)

    @property
    def value(self) -> Quantity[DataType] | None:
        if self._unit_part() is not None and self._numerical_part() is not None:
            return Quantity(self._numerical_part(), self.unit)
        return None

    @property
    def quantity(self):
        if self._unit_part() is not None and self._numerical_part() is not None:
            return Quantity(self._numerical_part(), self.unit)
        return None

