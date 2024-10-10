from typing import Collection, Sequence, TypeVar, Generic, Self
from dataclasses import dataclass

import numpy as np
from numpy._typing import ArrayLike

from quantities.operations import Operation, Variable
from quantities import operations, units
from sasdata.quantities.units import Unit, NamedUnit

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

        self.reference_key_list = [key for key in self.references]
        self.si_reference_values = {key: self.references[key].in_si() for key in self.references}

    def jacobian(self) -> list[Operation]:
        """ Derivative of this quantity's operation history with respect to each of the references """

        # Use the hash value to specify the variable of differentiation
        return [self.operation_tree.derivative(key) for key in self.reference_key_list]

    def variance_propagate(self, quantity_units: Unit, covariances: dict[tuple[int, int]: "Quantity"] = {}):
        """ Do standard error propagation to calculate the uncertainties associated with this quantity

        :param quantity_units: units in which the output should be calculated
        :param covariances: off diagonal entries for the covariance matrix
        """

        if covariances:
            raise NotImplementedError("User specified covariances not currently implemented")

        jacobian = self.jacobian()
        # jacobian_units = [quantity_units / self.references[key].units for key in self.reference_key_list]
        #
        # # Evaluate the jacobian
        # # TODO: should we use quantities here, does that work automatically?
        # evaluated_jacobian = [Quantity(
        #                         value=entry.evaluate(self.si_reference_values),
        #                         units=unit.si_equivalent())
        #                       for entry, unit in zip(jacobian, jacobian_units)]

        evaluated_jacobian = [entry.evaluate(self.references) for entry in jacobian]

        hash_values = [key for key in self.references]
        output = None

        for hash_value, jac_component in zip(hash_values, evaluated_jacobian):
            if output is None:
                output = jac_component * (self.references[hash_value].variance * jac_component)
            else:
                output += jac_component * (self.references[hash_value].variance * jac_component)

        return output


    @staticmethod
    def variable(quantity: "Quantity"):
        """ Create a history that starts with the provided data """
        return QuantityHistory(Variable(quantity.hash_value), {quantity.hash_value: quantity})

    @staticmethod
    def apply_operation(operation: type[Operation], *histories: "QuantityHistory") -> "QuantityHistory":
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

    def has_variance(self):
        for key in self.references:
            if self.references[key].has_variance:
                return True

        return False


class Quantity[QuantityType]:


    def __init__(self,
                 value: QuantityType,
                 units: Unit,
                 standard_error: QuantityType | None = None,
                 hash_seed = ""):

        self.value = value
        """ Numerical value of this data, in the specified units"""

        self.units = units
        """ Units of this data """

        self._hash_seed = hash_seed
        """ Retain this for copying operations"""

        self.hash_value = -1
        """ Hash based on value and uncertainty for data, -1 if it is a derived hash value """

        """ Contains the variance if it is data driven, else it is """

        if standard_error is None:
            self._variance = None
            self.hash_value = hash_data_via_numpy(hash_seed, value)
        else:
            self._variance = standard_error ** 2
            self.hash_value = hash_data_via_numpy(hash_seed, value, standard_error)

        self.history = QuantityHistory.variable(self)

    @property
    def has_variance(self):
        return self._variance is not None

    @property
    def variance(self) -> "Quantity":
        """ Get the variance of this object"""
        if self._variance is None:
            return Quantity(np.zeros_like(self.value), self.units**2)
        else:
            return Quantity(self._variance, self.units**2)

    def standard_deviation(self) -> "Quantity":
        return self.variance ** 0.5

    def in_units_of(self, units: Unit) -> QuantityType:
        """ Get this quantity in other units """
        if self.units.equivalent(units):
            return (self.units.scale / units.scale) * self.value
        else:
            raise UnitError(f"Target units ({units}) not compatible with existing units ({self.units}).")

    def to_units_of(self, new_units: Unit) -> "Quantity[QuantityType]":
        new_value, new_error = self.in_units_of_with_standard_error(new_units)
        return Quantity(value=new_value,
                        units=new_units,
                        standard_error=new_error,
                        hash_seed=self._hash_seed)

    def variance_in_units_of(self, units: Unit) -> QuantityType:
        """ Get the variance of quantity in other units """
        variance = self.variance
        if variance.units.equivalent(units):
            return (variance.units.scale / units.scale) * variance
        else:
            raise UnitError(f"Target units ({units}) not compatible with existing units ({variance.units}).")

    def in_si(self):
        si_units = self.units.si_equivalent()
        return self.in_units_of(si_units)

    def in_units_of_with_standard_error(self, units):
        variance = self.variance
        units_squared = units**2

        if variance.units.equivalent(units_squared):

            return self.in_units_of(units), np.sqrt(self.variance.in_units_of(units_squared))
        else:
            raise UnitError(f"Target units ({units}) not compatible with existing units ({variance.units}).")

    def in_si_with_standard_error(self):
        if self.has_variance:
            return self.in_units_of_with_standard_error(self.units.si_equivalent())
        else:
            return self.in_si(), None

    def __mul__(self: Self, other: ArrayLike | Self ) -> Self:
        if isinstance(other, Quantity):
            return DerivedQuantity(
                self.value * other.value,
                self.units * other.units,
                history=QuantityHistory.apply_operation(operations.Mul, self.history, other.history))

        else:
            return DerivedQuantity(self.value * other, self.units,
                                   QuantityHistory(
                                       operations.Mul(
                                           self.history.operation_tree,
                                           operations.Constant(other)),
                                       self.history.references))

    def __rmul__(self: Self, other: ArrayLike | Self):
        if isinstance(other, Quantity):
            return DerivedQuantity(
                    other.value * self.value,
                    other.units * self.units,
                    history=QuantityHistory.apply_operation(
                        operations.Mul,
                        other.history,
                        self.history))

        else:
            return DerivedQuantity(other * self.value, self.units,
                                   QuantityHistory(
                                       operations.Mul(
                                           operations.Constant(other),
                                           self.history.operation_tree),
                                       self.history.references))




    def __truediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, Quantity):
            return DerivedQuantity(
                    self.value / other.value,
                    self.units / other.units,
                    history=QuantityHistory.apply_operation(
                        operations.Div,
                        self.history,
                        other.history))

        else:
            return DerivedQuantity(self.value / other, self.units,
                                   QuantityHistory(
                                       operations.Div(
                                           operations.Constant(other),
                                           self.history.operation_tree),
                                       self.history.references))

    def __rtruediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, Quantity):
            return DerivedQuantity(
                    other.value / self.value,
                    other.units / self.units,
                    history=QuantityHistory.apply_operation(
                        operations.Div,
                        other.history,
                        self.history
                    ))

        else:
            return DerivedQuantity(
                    other / self.value,
                    self.units ** -1,
                               QuantityHistory(
                                   operations.Div(
                                       operations.Constant(other),
                                       self.history.operation_tree),
                                   self.history.references))

    def __add__(self: Self, other: Self | ArrayLike) -> Self:
        if isinstance(other, Quantity):
            if self.units.equivalent(other.units):
                return DerivedQuantity(
                            self.value + (other.value * other.units.scale) / self.units.scale,
                            self.units,
                            QuantityHistory.apply_operation(
                                operations.Add,
                                self.history,
                                other.history))
            else:
                raise UnitError(f"Units do not have the same dimensionality: {self.units} vs {other.units}")

        else:
            raise UnitError(f"Cannot perform addition/subtraction non-quantity {type(other)} with quantity")

    # Don't need __radd__ because only quantity/quantity operations should be allowed

    def __neg__(self):
        return DerivedQuantity(-self.value, self.units,
                               QuantityHistory.apply_operation(
                                   operations.Neg,
                                   self.history
                               ))

    def __sub__(self: Self, other: Self | ArrayLike) -> Self:
        return self + (-other)

    def __rsub__(self: Self, other: Self | ArrayLike) -> Self:
        return (-self) + other

    def __pow__(self: Self, other: int | float):
        return DerivedQuantity(self.value ** other,
                               self.units ** other,
                               QuantityHistory(
                                   operations.Pow(
                                       self.history.operation_tree,
                                       other),
                                   self.history.references))

    @staticmethod
    def _array_repr_format(arr: np.ndarray):
        """ Format the array """
        order = len(arr.shape)
        reshaped = arr.reshape(-1)
        if len(reshaped) <= 2:
            numbers = ",".join([f"{n}" for n in reshaped])
        else:
            numbers = f"{reshaped[0]} ... {reshaped[-1]}"

        # if len(reshaped) <= 4:
        #     numbers = ",".join([f"{n}" for n in reshaped])
        # else:
        #     numbers = f"{reshaped[0]}, {reshaped[1]} ... {reshaped[-2]}, {reshaped[-1]}"

        return "["*order + numbers + "]"*order

    def __repr__(self):

        if isinstance(self.units, NamedUnit):

            value = self.value
            error = self.standard_deviation().in_units_of(self.units)
            unit_string = self.units.symbol

        else:
            value, error = self.in_si_with_standard_error()
            unit_string = self.units.dimensions.si_repr()

        if isinstance(self.value, np.ndarray):
            # Get the array in short form
            numeric_string = self._array_repr_format(value)

            if self.has_variance:
                numeric_string += " ± " + self._array_repr_format(error)

        else:
            numeric_string = f"{value}"
            if self.has_variance:
                numeric_string += f" ± {error}"

        return numeric_string + " " + unit_string

    @staticmethod
    def parse(number_or_string: str | ArrayLike, unit: str, absolute_temperature: False):
        pass


class NamedQuantity[QuantityType](Quantity[QuantityType]):
    def __init__(self,
                 name: str,
                 value: QuantityType,
                 units: Unit,
                 standard_error: QuantityType | None = None):

        super().__init__(value, units, standard_error=standard_error, hash_seed=name)
        self.name = name

    def __repr__(self):
        return f"[{self.name}] " + super().__repr__()

    def to_units_of(self, new_units: Unit) -> "NamedQuantity[QuantityType]":
        new_value, new_error = self.in_units_of_with_standard_error(new_units)
        return NamedQuantity(value=new_value,
                             units=new_units,
                             standard_error=new_error,
                             name=self.name)

    def with_standard_error(self, standard_error: Quantity):
        if standard_error.units.equivalent(self.units):
            return NamedQuantity(
                value=self.value,
                units=self.units,
                standard_error=standard_error.in_units_of(self.units),
                name=self.name)

        else:
            raise UnitError(f"Standard error units ({standard_error.units}) "
                            f"are not compatible with value units ({self.units})")


class DerivedQuantity[QuantityType](Quantity[QuantityType]):
    def __init__(self, value: QuantityType, units: Unit, history: QuantityHistory):
        super().__init__(value, units, standard_error=None)

        self.history = history
        self._variance_cache = None
        self._has_variance = history.has_variance()


    def to_units_of(self, new_units: Unit) -> "Quantity[QuantityType]":
        # TODO: Lots of tests needed for this
        return DerivedQuantity(
                value=self.in_units_of(new_units),
                units=new_units,
                history=self.history)

    @property
    def has_variance(self):
        return self._has_variance

    @property
    def variance(self) -> Quantity:
        if self._variance_cache is None:
            self._variance_cache = self.history.variance_propagate(self.units)

        return self._variance_cache