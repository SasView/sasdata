from typing import Collection, Sequence, TypeVar, Generic, Self
from dataclasses import dataclass

import numpy as np
from numpy._typing import ArrayLike

from quantities.operations import Operation, Variable
from sasdata.quantities.units import Unit

import hashlib


class UnitError(Exception):
    """ Errors caused by unit specification not being correct """

def hash_data_via_numpy(*data: ArrayLike):

    md5_hash = hashlib.md5()

    for datum in data:
        data_bytes = np.array(datum).tobytes()
        md5_hash.update(data_bytes)

    # Hash function returns a hex string, we want an int
    return int(md5_hash.hexdigest(), 16)


QuantityType = TypeVar("QuantityType")


class QuantityHistory:
    def __init__(self, operation_tree: Operation, references: dict[int, "Quantity"]):
        self.operation_tree = operation_tree
        self.references = references

    def jacobian(self) -> list[Operation]:
        """ Derivative of this quantity's operation history with respect to each of the references """

        # Use the hash value to specify the variable of differentiation
        return [self.operation_tree.derivative(hash_value) for hash_value in self.references]

    def standard_error_propagate(self, covariances: dict[tuple[int, int]: "Quantity"] = {}):
        """ Do standard error propagation to calculate the uncertainties associated with this quantity

        @param: covariances, off diagonal entries for the covariance matrix
        """

        jacobian = self.jacobian()

        # Evaluate the jacobian
        evaluated_jacobian = [entry.evaluate(self.references) for entry in jacobian] # should we use quantities here?

        output = 0

        for hash_value in self.references:
            output += evaluated_jacobian * (self.references[hash_value].variance * evaluated_jacobian)

        for (cov1, cov2) in covariances:
            pass



    @staticmethod
    def variable(quantity: "Quantity"):
        """ Create a history that starts with the provided data """
        return QuantityHistory(Variable(quantity.hash_value), {quantity.hash_value: quantity})

    @staticmethod
    def _apply_operation(operation: type[Operation], *histories: "QuantityHistory") -> "QuantityHistory":
        """ Apply an operation to the history

        This is slightly unsafe as it is possible to attempt to apply an n-ary operation to a number of trees other
        than n, but it is relatively concise. Because it is concise we'll go with this for now and see if it causes
        any problems down the line. It is a private static method to discourage misuse.

        """

        # Copy references over, even though it overrides on collision,
        # this should behave because only data based variables should be represented.
        # Should not be a problem any more than losing histories
        references = {}
        for history in histories:
            references.update(history.references)

        return QuantityHistory(
            operation(*[history.operation_tree for history in histories]),
            references)



class Quantity[QuantityType]:


    def __init__(self,
                 value: QuantityType,
                 units: Unit,
                 variance: QuantityType | None = None):

        self.value = value
        """ Numerical value of this data, in the specified units"""

        self.units = units
        """ Units of this data """

        self.hash_value = -1
        """ Hash based on value and uncertainty for data, -1 if it is a derived hash value """

        self._variance = variance
        """ Contains the variance if it is data driven, else it is """

        if variance is None:
            self.hash_value = hash_data_via_numpy(value)
        else:
            self.hash_value = hash_data_via_numpy(value, variance.value)

        self.history = QuantityHistory.variable(self)

    @property
    def variance(self) -> "Quantity":
        pass

    def standard_deviation(self) -> "Quantity":
        return self.variance ** (1/2)

    def in_units_of(self, units: Unit) -> QuantityType:
        """ Get this quantity in other units """
        if self.units.equivalent(units):
            return (self.units.scale / units.scale) * self.value
        else:
            raise UnitError(f"Target units ({units}) not compatible with existing units ({self.units}).")

    def __mul__(self: Self, other: ArrayLike | Self ) -> Self:
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.units * other.units)

        else:
            return Quantity(self.value * other, self.units)

    def __rmul__(self: Self, other: ArrayLike | Self):
        if isinstance(other, Quantity):
            return Quantity(other.value * self.value, other.units * self.units)

        else:
            return Quantity(other * self.value, self.units)

    def __truediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.units / other.units)

        else:
            return Quantity(self.value / other, self.units)

    def __rtruediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.units / other.units)

        else:
            return Quantity(self.value / other, self.units)

    def __add__(self: Self, other: Self | ArrayLike) -> Self:
        if isinstance(other, Quantity):
            if self.units.equivalent(other.units):
                return Quantity(self.value + (other.value * other.units.scale) / self.units.scale, self.units)
            else:
                raise UnitError(f"Units do not have the same dimensionality: {self.units} vs {other.units}")

        else:
            raise UnitError(f"Cannot perform addition/subtraction non-quantity {type(other)} with quantity")

    # Don't need __radd__ because only quantity/quantity operations should be allowed

    def __neg__(self):
        return Quantity(-self.value, self.units)

    def __sub__(self: Self, other: Self | ArrayLike) -> Self:
        return self + (-other)

    def __rsub__(self: Self, other: Self | ArrayLike) -> Self:
        return (-self) + other

    def __pow__(self: Self, other: int):
        return Quantity(self.value ** other, self.units ** other)

    @staticmethod
    def parse(number_or_string: str | ArrayLike, unit: str, absolute_temperature: False):
        pass


class NamedQuantity[QuantityType](Quantity[QuantityType]):
    def __init__(self,
                 value: QuantityType,
                 units: Unit,
                 name: str,
                 variance: QuantityType | None = None):

        super().__init__(value, units, variance=variance)
        self.name = name

class DerivedQuantity[QuantityType](Quantity[QuantityType]):
    def __init__(self, value, units, variance, history):

        self._variance_cache = None
    @property
    def variance(self):
        pass