import hashlib
import json
from math import e, log
from typing import Any, Self, TypeVar, Union

import h5py
import numpy as np
from numpy._typing import ArrayLike

from sasdata.quantities import units
from sasdata.quantities.numerical_encoding import numerical_decode, numerical_encode
from sasdata.quantities.unit_parser import parse_unit
from sasdata.quantities.units import NamedUnit, Unit

T = TypeVar("T")


################### Quantity based operations, need to be here to avoid cyclic dependencies #####################


def transpose(a: Union["Quantity[ArrayLike]", ArrayLike], axes: tuple | None = None):
    """Transpose an array or an array based quantity, can also do reordering of axes"""
    if isinstance(a, Quantity):
        if axes is None:
            return DerivedQuantity(
                value=np.transpose(a.value, axes=axes),
                units=a.units,
                history=QuantityHistory.apply_operation(Transpose, a.history),
            )

        else:
            return DerivedQuantity(
                value=np.transpose(a.value, axes=axes),
                units=a.units,
                history=QuantityHistory.apply_operation(Transpose, a.history, axes=axes),
            )

    else:
        return np.transpose(a, axes=axes)


def dot(a: Union["Quantity[ArrayLike]", ArrayLike], b: Union["Quantity[ArrayLike]", ArrayLike]):
    """Dot product of two arrays or two array based quantities"""
    a_is_quantity = isinstance(a, Quantity)
    b_is_quantity = isinstance(b, Quantity)

    if a_is_quantity or b_is_quantity:
        # If its only one of them that is a quantity, convert the other one

        if not a_is_quantity:
            a = Quantity(a, units.none)

        if not b_is_quantity:
            b = Quantity(b, units.none)

        return DerivedQuantity(
            value=np.dot(a.value, b.value),
            units=a.units * b.units,
            history=QuantityHistory.apply_operation(Dot, a.history, b.history),
        )

    else:
        return np.dot(a, b)


def tensordot(
    a: Union["Quantity[ArrayLike]", ArrayLike] | ArrayLike,
    b: Union["Quantity[ArrayLike]", ArrayLike],
    a_index: int,
    b_index: int,
):
    """Tensor dot product - equivalent to contracting two tensors, such as

    A_{i0, i1, i2, i3...} and B_{j0, j1, j2...}

    e.g. if a_index is 1 and b_index is zero, it will be the sum

    C_{i0, i2, i3 ..., j1, j2 ...} = sum_k A_{i0, k, i2, i3 ...} B_{k, j1, j2 ...}

    (I think, have to check what happens with indices TODO!)

    """

    a_is_quantity = isinstance(a, Quantity)
    b_is_quantity = isinstance(b, Quantity)

    if a_is_quantity or b_is_quantity:
        # If its only one of them that is a quantity, convert the other one

        if not a_is_quantity:
            a = Quantity(a, units.none)

        if not b_is_quantity:
            b = Quantity(b, units.none)

        return DerivedQuantity(
            value=np.tensordot(a.value, b.value, axes=(a_index, b_index)),
            units=a.units * b.units,
            history=QuantityHistory.apply_operation(TensorDot, a.history, b.history, a_index=a_index, b_index=b_index),
        )

    else:
        return np.tensordot(a, b, axes=(a_index, b_index))


################### Operation Definitions #######################################


def hash_and_name(hash_or_name: int | str):
    """Infer the name of a variable from a hash, or the hash from the name

    Note: hash_and_name(hash_and_name(number)[1]) is not the identity
          however: hash_and_name(hash_and_name(number)) is
    """

    if isinstance(hash_or_name, str):
        hash_value = hash(hash_or_name)
        name = hash_or_name

        return hash_value, name

    elif isinstance(hash_or_name, int):
        hash_value = hash_or_name
        name = f"#{hash_or_name}"

        return hash_value, name

    elif isinstance(hash_or_name, tuple):
        return hash_or_name

    else:
        raise TypeError("Variable name_or_hash_value must be either str or int")


class Operation:
    serialisation_name = "unknown"

    def summary(self, indent_amount: int = 0, indent: str = "  "):
        """Summary of the operation tree"""

        s = f"{indent_amount * indent}{self.__class__.__name__}(\n"

        for chunk in self._summary_components():
            s += chunk.summary(indent_amount + 1, indent) + "\n"

        s += f"{indent_amount * indent})"

        return s

    def _summary_components(self) -> list["Operation"]:
        return []

    def evaluate(self, variables: dict[int, T]) -> T:
        """Evaluate this operation"""
        pass

    def _derivative(self, hash_value: int) -> "Operation":
        """Get the derivative of this operation"""
        pass

    def _clean(self):
        """Clean up this operation - i.e. remove silly things like 1*x"""
        return self

    def derivative(self, variable: Union[str, int, "Variable"], simplify=True):
        if isinstance(variable, Variable):
            hash_value = variable.hash_value
        else:
            hash_value, _ = hash_and_name(variable)

        derivative = self._derivative(hash_value)

        if not simplify:
            return derivative

        derivative_string = derivative.serialise()

        # print("---------------")
        # print("Base")
        # print("---------------")
        # print(derivative.summary())

        # Inefficient way of doing repeated simplification, but it will work
        for i in range(100):  # set max iterations
            derivative = derivative._clean()
            #
            # print("-------------------")
            # print("Iteration", i+1)
            # print("-------------------")
            # print(derivative.summary())
            # print("-------------------")

            new_derivative_string = derivative.serialise()

            if derivative_string == new_derivative_string:
                break

            derivative_string = new_derivative_string

        return derivative

    @staticmethod
    def deserialise(data: str) -> "Operation":
        json_data = json.loads(data)
        return Operation.deserialise_json(json_data)

    @staticmethod
    def deserialise_json(json_data: dict) -> "Operation":
        operation = json_data["operation"]
        parameters = json_data["parameters"]
        class_ = _serialisation_lookup[operation]

        try:
            return class_._deserialise(parameters)
        except NotImplementedError:
            raise NotImplementedError(f"No method to deserialise {operation} with {parameters} (class={class_})")

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        raise NotImplementedError("Deserialise not implemented for this class")

    def serialise(self) -> str:
        return json.dumps(self._serialise_json())

    def _serialise_json(self) -> dict[str, Any]:
        return {"operation": self.serialisation_name, "parameters": self._serialise_parameters()}

    def _serialise_parameters(self) -> dict[str, Any]:
        raise NotImplementedError("_serialise_parameters not implemented for this class")

    def __eq__(self, other: "Operation"):
        return NotImplemented


class ConstantBase(Operation):
    pass


class AdditiveIdentity(ConstantBase):
    serialisation_name = "zero"

    def evaluate(self, variables: dict[int, T]) -> T:
        return 0

    def _derivative(self, hash_value: int) -> "Operation":
        return AdditiveIdentity()

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return AdditiveIdentity()

    def _serialise_parameters(self) -> dict[str, Any]:
        return {}

    def summary(self, indent_amount: int = 0, indent="  "):
        return f"{indent_amount * indent}0 [Add.Id.]"

    def __eq__(self, other):
        if isinstance(other, AdditiveIdentity):
            return True
        elif isinstance(other, Constant):
            if other.value == 0:
                return True

        return False


class MultiplicativeIdentity(ConstantBase):
    serialisation_name = "one"

    def evaluate(self, variables: dict[int, T]) -> T:
        return 1

    def _derivative(self, hash_value: int):
        return AdditiveIdentity()

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return MultiplicativeIdentity()

    def _serialise_parameters(self) -> dict[str, Any]:
        return {}

    def summary(self, indent_amount: int = 0, indent="  "):
        return f"{indent_amount * indent}1 [Mul.Id.]"

    def __eq__(self, other):
        if isinstance(other, MultiplicativeIdentity):
            return True
        elif isinstance(other, Constant):
            if other.value == 1:
                return True

        return False


class Constant(ConstantBase):
    serialisation_name = "constant"

    def __init__(self, value):
        self.value = value

    def evaluate(self, variables: dict[int, T]) -> T:
        return self.value

    def _derivative(self, hash_value: int):
        return AdditiveIdentity()

    def _clean(self):
        if self.value == 0:
            return AdditiveIdentity()

        elif self.value == 1:
            return MultiplicativeIdentity()

        else:
            return self

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        value = numerical_decode(parameters["value"])
        return Constant(value)

    def _serialise_parameters(self) -> dict[str, Any]:
        return {"value": numerical_encode(self.value)}

    def summary(self, indent_amount: int = 0, indent="  "):
        return f"{indent_amount * indent}{self.value}"

    def __eq__(self, other):
        if isinstance(other, AdditiveIdentity):
            return self.value == 0

        elif isinstance(other, MultiplicativeIdentity):
            return self.value == 1

        elif isinstance(other, Constant):
            return other.value == self.value

        return False


class Variable(Operation):
    serialisation_name = "variable"

    def __init__(self, name_or_hash_value: int | str | tuple[int, str]):
        self.hash_value, self.name = hash_and_name(name_or_hash_value)

    def evaluate(self, variables: dict[int, T]) -> T:
        try:
            return variables[self.hash_value]
        except KeyError:
            raise ValueError(f"Variable dictionary didn't have an entry for {self.name} (hash={self.hash_value})")

    def _derivative(self, hash_value: int) -> Operation:
        if hash_value == self.hash_value:
            return MultiplicativeIdentity()
        else:
            return AdditiveIdentity()

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        hash_value = parameters["hash_value"]
        name = parameters["name"]

        return Variable((hash_value, name))

    def _serialise_parameters(self) -> dict[str, Any]:
        return {"hash_value": self.hash_value, "name": self.name}

    def summary(self, indent_amount: int = 0, indent: str = "  "):
        return f"{indent_amount * indent}{self.name}"

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.hash_value == other.hash_value
        return False


class UnaryOperation(Operation):
    def __init__(self, a: Operation):
        self.a = a

    def _serialise_parameters(self) -> dict[str, Any]:
        return {"a": self.a._serialise_json()}

    @classmethod
    def _deserialise(cls, parameters: dict) -> "UnaryOperation":
        return cls(Operation.deserialise_json(parameters["a"]))

    def _summary_components(self) -> list["Operation"]:
        return [self.a]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.a == other.a
        return False


class Neg(UnaryOperation):
    serialisation_name = "neg"

    def evaluate(self, variables: dict[int, T]) -> T:
        return -self.a.evaluate(variables)

    def _derivative(self, hash_value: int):
        return Neg(self.a._derivative(hash_value))

    def _clean(self):
        clean_a = self.a._clean()

        if isinstance(clean_a, Neg):
            # Removes double negations
            return clean_a.a

        elif isinstance(clean_a, Constant):
            return Constant(-clean_a.value)._clean()

        else:
            return Neg(clean_a)


class Inv(UnaryOperation):
    serialisation_name = "reciprocal"

    def evaluate(self, variables: dict[int, T]) -> T:
        return 1.0 / self.a.evaluate(variables)

    def _derivative(self, hash_value: int) -> Operation:
        return Neg(Div(self.a._derivative(hash_value), Mul(self.a, self.a)))

    def _clean(self):
        clean_a = self.a._clean()

        if isinstance(clean_a, Inv):
            # Removes double inversions
            return clean_a.a

        elif isinstance(clean_a, Neg):
            # cannonicalise 1/-a to -(1/a)
            # over multiple iterations this should have the effect of ordering and gathering Neg and Inv
            return Neg(Inv(clean_a.a))

        elif isinstance(clean_a, Constant):
            return Constant(1.0 / clean_a.value)._clean()

        else:
            return Inv(clean_a)


class Ln(UnaryOperation):
    serialisation_name = "ln"

    def evaluate(self, variables: dict[int, T]) -> Operation:
        return log(self.a.evaluate(variables))

    def _derivative(self, hash_value: int) -> Operation:
        return Inv(self.a)

    def _clean(self, a):
        clean_a = self.a._clean()

        if isinstance(a, MultiplicativeIdentity):
            # Convert ln(1) to 0
            return AdditiveIdentity()

        elif a == e:
            # Convert ln(e) to 1
            return MultiplicativeIdentity()

        else:
            return Log(clean_a)


class BinaryOperation(Operation):
    def __init__(self, a: Operation, b: Operation):
        self.a = a
        self.b = b

    def _clean(self):
        return self._clean_ab(self.a._clean(), self.b._clean())

    def _clean_ab(self, a, b):
        raise NotImplementedError("_clean_ab not implemented")

    def _serialise_parameters(self) -> dict[str, Any]:
        return {"a": self.a._serialise_json(), "b": self.b._serialise_json()}

    @classmethod
    def _deserialise(cls, parameters: dict) -> "BinaryOperation":
        return cls(*BinaryOperation._deserialise_ab(parameters))

    @staticmethod
    def _deserialise_ab(parameters) -> tuple[Operation, Operation]:
        return (Operation.deserialise_json(parameters["a"]), Operation.deserialise_json(parameters["b"]))

    def _summary_components(self) -> list["Operation"]:
        return [self.a, self.b]

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.a == other.a and self.b == other.b
        return False


class Add(BinaryOperation):
    serialisation_name = "add"

    def evaluate(self, variables: dict[int, T]) -> T:
        return self.a.evaluate(variables) + self.b.evaluate(variables)

    def _derivative(self, hash_value: int) -> Operation:
        return Add(self.a._derivative(hash_value), self.b._derivative(hash_value))

    def _clean_ab(self, a, b):
        if isinstance(a, AdditiveIdentity):
            # Convert 0 + b to b
            return b

        elif isinstance(b, AdditiveIdentity):
            # Convert a + 0 to a
            return a

        elif isinstance(a, ConstantBase) and isinstance(b, ConstantBase):
            # Convert constant "a"+"b" to "a+b"
            return Constant(a.evaluate({}) + b.evaluate({}))._clean()

        elif isinstance(a, Neg):
            if isinstance(b, Neg):
                # Convert (-a)+(-b) to -(a+b)
                return Neg(Add(a.a, b.a))
            else:
                # Convert (-a) + b to b-a
                return Sub(b, a.a)

        elif isinstance(b, Neg):
            # Convert a+(-b) to a-b
            return Sub(a, b.a)

        elif a == b:
            return Mul(Constant(2), a)

        else:
            return Add(a, b)


class Sub(BinaryOperation):
    serialisation_name = "sub"

    def evaluate(self, variables: dict[int, T]) -> T:
        return self.a.evaluate(variables) - self.b.evaluate(variables)

    def _derivative(self, hash_value: int) -> Operation:
        return Sub(self.a._derivative(hash_value), self.b._derivative(hash_value))

    def _clean_ab(self, a, b):
        if isinstance(a, AdditiveIdentity):
            # Convert 0 - b to -b
            return Neg(b)

        elif isinstance(b, AdditiveIdentity):
            # Convert a - 0 to a
            return a

        elif isinstance(a, ConstantBase) and isinstance(b, ConstantBase):
            # Convert constant pair "a" - "b" to "a-b"
            return Constant(a.evaluate({}) - b.evaluate({}))._clean()

        elif isinstance(a, Neg):
            if isinstance(b, Neg):
                # Convert (-a)-(-b) to b-a
                return Sub(b.a, a.a)
            else:
                # Convert (-a)-b to -(a+b)
                return Neg(Add(a.a, b))

        elif isinstance(b, Neg):
            # Convert a-(-b) to a+b
            return Add(a, b.a)

        elif a == b:
            return AdditiveIdentity()

        else:
            return Sub(a, b)


class Mul(BinaryOperation):
    serialisation_name = "mul"

    def evaluate(self, variables: dict[int, T]) -> T:
        return self.a.evaluate(variables) * self.b.evaluate(variables)

    def _derivative(self, hash_value: int) -> Operation:
        return Add(Mul(self.a, self.b._derivative(hash_value)), Mul(self.a._derivative(hash_value), self.b))

    def _clean_ab(self, a, b):
        if isinstance(a, AdditiveIdentity) or isinstance(b, AdditiveIdentity):
            # Convert 0*b or a*0 to 0
            return AdditiveIdentity()

        elif isinstance(a, MultiplicativeIdentity):
            # Convert 1*b to b
            return b

        elif isinstance(b, MultiplicativeIdentity):
            # Convert a*1 to a
            return a

        elif isinstance(a, ConstantBase) and isinstance(b, ConstantBase):
            # Convert constant "a"*"b" to "a*b"
            return Constant(a.evaluate({}) * b.evaluate({}))._clean()

        elif isinstance(a, Inv) and isinstance(b, Inv):
            return Inv(Mul(a.a, b.a))

        elif isinstance(a, Inv) and not isinstance(b, Inv):
            return Div(b, a.a)

        elif not isinstance(a, Inv) and isinstance(b, Inv):
            return Div(a, b.a)

        elif isinstance(a, Neg):
            return Neg(Mul(a.a, b))

        elif isinstance(b, Neg):
            return Neg(Mul(a, b.a))

        elif a == b:
            return Pow(a, 2)

        elif isinstance(a, Pow) and a.a == b:
            return Pow(b, a.power + 1)

        elif isinstance(b, Pow) and b.a == a:
            return Pow(a, b.power + 1)

        elif isinstance(a, Pow) and isinstance(b, Pow) and a.a == b.a:
            return Pow(a.a, a.power + b.power)

        else:
            return Mul(a, b)


class Div(BinaryOperation):
    serialisation_name = "div"

    def evaluate(self, variables: dict[int, T]) -> T:
        return self.a.evaluate(variables) / self.b.evaluate(variables)

    def _derivative(self, hash_value: int) -> Operation:
        return Div(
            Sub(Mul(self.a.derivative(hash_value), self.b), Mul(self.a, self.b.derivative(hash_value))),
            Mul(self.b, self.b),
        )

    def _clean_ab(self, a, b):
        if isinstance(a, AdditiveIdentity):
            # Convert 0/b to 0
            return AdditiveIdentity()

        elif isinstance(a, MultiplicativeIdentity):
            # Convert 1/b to inverse of b
            return Inv(b)

        elif isinstance(b, MultiplicativeIdentity):
            # Convert a/1 to a
            return a

        elif isinstance(a, ConstantBase) and isinstance(b, ConstantBase):
            # Convert constants "a"/"b" to "a/b"
            return Constant(self.a.evaluate({}) / self.b.evaluate({}))._clean()

        elif isinstance(a, Inv) and isinstance(b, Inv):
            return Div(b.a, a.a)

        elif isinstance(a, Inv) and not isinstance(b, Inv):
            return Inv(Mul(a.a, b))

        elif not isinstance(a, Inv) and isinstance(b, Inv):
            return Mul(a, b.a)

        elif a == b:
            return MultiplicativeIdentity()

        elif isinstance(a, Pow) and a.a == b:
            return Pow(b, a.power - 1)

        elif isinstance(b, Pow) and b.a == a:
            return Pow(a, 1 - b.power)

        elif isinstance(a, Pow) and isinstance(b, Pow) and a.a == b.a:
            return Pow(a.a, a.power - b.power)

        else:
            return Div(a, b)


class Log(Operation):
    serialisation_name = "log"

    def __init__(self, a: Operation, base: float):
        self.a = a
        self.base = base

    def evaluate(self, variables: dict[int, T]) -> Operation:
        return log(self.a.evaluate(variables), self.base)

    def _derivative(self, hash_value: int) -> Operation:
        return Inv(Mul(self.a, Ln(Constant(self.base))))

    def _clean_ab(self) -> Operation:
        a = self.a._clean()

        if isinstance(a, MultiplicativeIdentity):
            # Convert log(1) to 0
            return AdditiveIdentity()

        elif a == self.base:
            # Convert log(base) to 1
            return MultiplicativeIdentity()

        else:
            return Log(a, self.base)

    def _serialise_parameters(self) -> dict[str, Any]:
        return {"a": Operation._serialise_json(self.a), "base": self.base}

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return Log(Operation.deserialise_json(parameters["a"]), parameters["base"])

    def summary(self, indent_amount: int = 0, indent="  "):
        return (
            f"{indent_amount * indent}Log(\n"
            + self.a.summary(indent_amount + 1, indent)
            + "\n"
            + f"{(indent_amount + 1) * indent}{self.base}\n"
            + f"{indent_amount * indent})"
        )

    def __eq__(self, other):
        if isinstance(other, Log):
            return self.a == other.a and self.base == other.base
        return False


class Pow(Operation):
    serialisation_name = "pow"

    def __init__(self, a: Operation, power: float):
        self.a = a
        self.power = power

    def evaluate(self, variables: dict[int, T]) -> T:
        return self.a.evaluate(variables) ** self.power

    def _derivative(self, hash_value: int) -> Operation:
        if self.power == 0:
            return AdditiveIdentity()

        elif self.power == 1:
            return self.a._derivative(hash_value)

        else:
            return Mul(Constant(self.power), Mul(Pow(self.a, self.power - 1), self.a._derivative(hash_value)))

    def _clean(self) -> Operation:
        a = self.a._clean()

        if self.power == 1:
            return a

        elif self.power == 0:
            return MultiplicativeIdentity()

        elif self.power == -1:
            return Inv(a)

        else:
            return Pow(a, self.power)

    def _serialise_parameters(self) -> dict[str, Any]:
        return {"a": Operation._serialise_json(self.a), "power": self.power}

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return Pow(Operation.deserialise_json(parameters["a"]), parameters["power"])

    def summary(self, indent_amount: int = 0, indent="  "):
        return (
            f"{indent_amount * indent}Pow(\n"
            + self.a.summary(indent_amount + 1, indent)
            + "\n"
            + f"{(indent_amount + 1) * indent}{self.power}\n"
            + f"{indent_amount * indent})"
        )

    def __eq__(self, other):
        if isinstance(other, Pow):
            return self.a == other.a and self.power == other.power
        return False


#
# Matrix operations
#


class Transpose(Operation):
    """Transpose operation - as per numpy"""

    serialisation_name = "transpose"

    def __init__(self, a: Operation, axes: tuple[int] | None = None):
        self.a = a
        self.axes = axes

    def evaluate(self, variables: dict[int, T]) -> T:
        return np.transpose(self.a.evaluate(variables))

    def _derivative(self, hash_value: int) -> Operation:
        return Transpose(self.a.derivative(hash_value))  # TODO: Check!

    def _clean(self):
        clean_a = self.a._clean()
        return Transpose(clean_a)

    def _serialise_parameters(self) -> dict[str, Any]:
        if self.axes is None:
            return {"a": self.a._serialise_json()}
        else:
            return {"a": self.a._serialise_json(), "axes": list(self.axes)}

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        if "axes" in parameters:
            return Transpose(a=Operation.deserialise_json(parameters["a"]), axes=tuple(parameters["axes"]))
        else:
            return Transpose(a=Operation.deserialise_json(parameters["a"]))

    def summary(self, indent_amount: int = 0, indent="  "):
        if self.axes is None:
            return (
                f"{indent_amount * indent}Transpose(\n"
                + self.a.summary(indent_amount + 1, indent)
                + "\n"
                + f"{indent_amount * indent})"
            )
        else:
            return (
                f"{indent_amount * indent}Transpose(\n"
                + self.a.summary(indent_amount + 1, indent)
                + "\n"
                + f"{(indent_amount + 1) * indent}{self.axes}\n"
                + f"{indent_amount * indent})"
            )

    def __eq__(self, other):
        if isinstance(other, Transpose):
            return other.a == self.a
        return False


class Dot(BinaryOperation):
    """Dot product - backed by numpy's dot method"""

    serialisation_name = "dot"

    def evaluate(self, variables: dict[int, T]) -> T:
        return dot(self.a.evaluate(variables), self.b.evaluate(variables))

    def _derivative(self, hash_value: int) -> Operation:
        return Add(Dot(self.a, self.b._derivative(hash_value)), Dot(self.a._derivative(hash_value), self.b))

    def _clean_ab(self, a, b):
        return Dot(a, b)  # Do nothing for now


# TODO: Add to base operation class, and to quantities
class MatMul(BinaryOperation):
    """Matrix multiplication, using __matmul__ dunder"""

    serialisation_name = "matmul"

    def evaluate(self, variables: dict[int, T]) -> T:
        return self.a.evaluate(variables) @ self.b.evaluate(variables)

    def _derivative(self, hash_value: int) -> Operation:
        return Add(MatMul(self.a, self.b._derivative(hash_value)), MatMul(self.a._derivative(hash_value), self.b))

    def _clean_ab(self, a, b):
        if isinstance(a, AdditiveIdentity) or isinstance(b, AdditiveIdentity):
            # Convert 0*b or a*0 to 0
            return AdditiveIdentity()

        elif isinstance(a, ConstantBase) and isinstance(b, ConstantBase):
            # Convert constant "a"@"b" to "a@b"
            return Constant(a.evaluate({}) @ b.evaluate({}))._clean()

        elif isinstance(a, Neg):
            return Neg(Mul(a.a, b))

        elif isinstance(b, Neg):
            return Neg(Mul(a, b.a))

        return MatMul(a, b)


class TensorDot(Operation):
    serialisation_name = "tensor_product"

    def __init__(self, a: Operation, b: Operation, a_index: int, b_index: int):
        self.a = a
        self.b = b
        self.a_index = a_index
        self.b_index = b_index

    def evaluate(self, variables: dict[int, T]) -> T:
        return tensordot(self.a, self.b, self.a_index, self.b_index)

    def _serialise_parameters(self) -> dict[str, Any]:
        return {
            "a": self.a._serialise_json(),
            "b": self.b._serialise_json(),
            "a_index": self.a_index,
            "b_index": self.b_index,
        }

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return TensorDot(
            a=Operation.deserialise_json(parameters["a"]),
            b=Operation.deserialise_json(parameters["b"]),
            a_index=int(parameters["a_index"]),
            b_index=int(parameters["b_index"]),
        )


_serialisable_classes = [
    AdditiveIdentity,
    MultiplicativeIdentity,
    Constant,
    Variable,
    Neg,
    Inv,
    Ln,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Log,
    Transpose,
    Dot,
    MatMul,
    TensorDot,
]

_serialisation_lookup = {class_.serialisation_name: class_ for class_ in _serialisable_classes}


class UnitError(Exception):
    """Errors caused by unit specification not being correct"""


def hash_data_via_numpy(*data: ArrayLike):
    md5_hash = hashlib.md5()

    for datum in data:
        data_bytes = np.array(datum).tobytes()
        md5_hash.update(data_bytes)

    # Hash function returns a hex string, we want an int
    return int(md5_hash.hexdigest(), 16)


#####################################
#                                   #
#                                   #
#                                   #
#       Quantities begin here       #
#                                   #
#                                   #
#                                   #
#####################################


QuantityType = TypeVar("QuantityType")


class QuantityHistory:
    """Class that holds the information for keeping track of operations done on quantities"""

    def __init__(self, operation_tree: Operation, references: dict[int, "Quantity"]):
        self.operation_tree = operation_tree
        self.references = references

        self.reference_key_list = [key for key in self.references]
        self.si_reference_values = {key: self.references[key].in_si() for key in self.references}

    def jacobian(self) -> list[Operation]:
        """Derivative of this quantity's operation history with respect to each of the references"""

        # Use the hash value to specify the variable of differentiation
        return [self.operation_tree.derivative(key) for key in self.reference_key_list]

    def _recalculate(self):
        """Recalculate the value of this object - primary use case is for testing"""
        return self.operation_tree.evaluate(self.references)

    def variance_propagate(self, quantity_units: Unit, covariances: dict[tuple[int, int] : "Quantity"] = {}):
        """Do standard error propagation to calculate the uncertainties associated with this quantity

        :param quantity_units: units in which the output should be calculated
        :param covariances: off diagonal entries for the covariance matrix
        """

        if covariances:
            raise NotImplementedError("User specified covariances not currently implemented")

        jacobian = self.jacobian()

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
        """Create a history that starts with the provided data"""
        return QuantityHistory(Variable(quantity.hash_value), {quantity.hash_value: quantity})

    @staticmethod
    def apply_operation(
        operation: type[Operation], *histories: "QuantityHistory", **extra_parameters
    ) -> "QuantityHistory":
        """Apply an operation to the history

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
            operation(*[history.operation_tree for history in histories], **extra_parameters), references
        )

    def has_variance(self):
        for key in self.references:
            if self.references[key].has_variance:
                return True

        return False

    def summary(self):
        variable_strings = [self.references[key].string_repr for key in self.references]

        s = "Variables: " + ",".join(variable_strings)
        s += "\n"
        s += self.operation_tree.summary()

        return s


class Quantity[QuantityType]:
    def __init__(self,
                 value: QuantityType,
                 units: Unit,
                 standard_error: QuantityType | None = None,
                 hash_seed="",
                 name="",
                 id_header=""):
        self.value = value
        """ Numerical value of this data, in the specified units"""

        self.units = units
        """ Units of this data """

        self._hash_seed = hash_seed
        """ Retain this for copying operations"""

        self.hash_value = -1
        """ Hash based on value and uncertainty for data, -1 if it is a derived hash value """

        self._variance = None
        """ Contains the variance if it is data driven """

        if standard_error is None:
            self.hash_value = hash_data_via_numpy(hash_seed, value)
        else:
            self._variance = standard_error**2
            self.hash_value = hash_data_via_numpy(hash_seed, value, standard_error)

        self.history = QuantityHistory.variable(self)

        self._id_header = id_header
        self.name = name

    # TODO: Adding this method as a temporary measure but we need a single
    # method that does this.
    def with_standard_error(self, standard_error: "Quantity"):
        if standard_error.units.equivalent(self.units):
            return Quantity(
                value=self.value,
                units=self.units,
                standard_error=standard_error.in_units_of(self.units),
                name=self.name,
                id_header=self._id_header,
            )
        else:
            raise UnitError(
                f"Standard error units ({standard_error.units}) are not compatible with value units ({self.units})"
            )

    @property
    def has_variance(self):
        return self._variance is not None

    @property
    def variance(self) -> "Quantity":
        """Get the variance of this object"""
        if self._variance is None:
            return Quantity(np.zeros_like(self.value), self.units**2, name=self.name, id_header=self._id_header)
        else:
            return Quantity(self._variance, self.units**2)

    def _base62_hash(self) -> str:
        """Encode the hash_value in base62 for better readability"""
        hashed = ""
        current_hash = self.hash_value
        while current_hash:
            digit = current_hash % 62
            if digit < 10:
                hashed = f"{digit}{hashed}"
            elif digit < 36:
                hashed = f"{chr(55 + digit)}{hashed}"
            else:
                hashed = f"{chr(61 + digit)}{hashed}"
            current_hash = (current_hash - digit) // 62
        return hashed

    @property
    def unique_id(self) -> str:
        """Get a human readable unique id for a data set"""
        return f"{self._id_header}:{self.name}:{self._base62_hash()}"

    def standard_deviation(self) -> "Quantity":
        return self.variance**0.5

    def in_units_of(self, units: Unit) -> QuantityType:
        """Get this quantity in other units"""
        if self.units.equivalent(units):
            return (self.units.scale / units.scale) * self.value
        else:
            raise UnitError(f"Target units ({units}) not compatible with existing units ({self.units}).")

    def to_units_of(self, new_units: Unit) -> "Quantity[QuantityType]":
        new_value, new_error = self.in_units_of_with_standard_error(new_units)
        return Quantity(value=new_value,
                        units=new_units,
                        standard_error=new_error,
                        hash_seed=self._hash_seed,
                        id_header=self._id_header)

    def variance_in_units_of(self, units: Unit) -> QuantityType:
        """Get the variance of quantity in other units"""
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

    def explicitly_formatted(self, unit_string: str) -> str:
        """Returns quantity as a string with specific unit formatting

        Performs any necessary unit conversions, but maintains the exact unit
        formatting provided by the user.  This can be useful if you have a
        power expressed in horsepower and you want it expressed as "745.7 N m/s" and not as "745.7 W"."""
        unit = parse_unit(unit_string)
        quantity = self.in_units_of(unit)
        return f"{quantity} {unit_string}"

    def __eq__(self: Self, other: Self) -> bool | np.ndarray:
        return self.value == other.in_units_of(self.units)

    def __mul__(self: Self, other: ArrayLike | Self) -> Self:
        if isinstance(other, Quantity):
            return DerivedQuantity(
                self.value * other.value,
                self.units * other.units,
                history=QuantityHistory.apply_operation(Mul, self.history, other.history),
            )

        else:
            return DerivedQuantity(
                self.value * other,
                self.units,
                QuantityHistory(Mul(self.history.operation_tree, Constant(other)), self.history.references),
            )

    def __rmul__(self: Self, other: ArrayLike | Self):
        if isinstance(other, Quantity):
            return DerivedQuantity(
                other.value * self.value,
                other.units * self.units,
                history=QuantityHistory.apply_operation(Mul, other.history, self.history),
            )

        else:
            return DerivedQuantity(
                other * self.value,
                self.units,
                QuantityHistory(Mul(Constant(other), self.history.operation_tree), self.history.references),
            )

    def __matmul__(self, other: ArrayLike | Self):
        if isinstance(other, Quantity):
            return DerivedQuantity(
                self.value @ other.value,
                self.units * other.units,
                history=QuantityHistory.apply_operation(MatMul, self.history, other.history),
            )
        else:
            return DerivedQuantity(
                self.value @ other,
                self.units,
                QuantityHistory(MatMul(self.history.operation_tree, Constant(other)), self.history.references),
            )

    def __rmatmul__(self, other: ArrayLike | Self):
        if isinstance(other, Quantity):
            return DerivedQuantity(
                other.value @ self.value,
                other.units * self.units,
                history=QuantityHistory.apply_operation(MatMul, other.history, self.history),
            )

        else:
            return DerivedQuantity(
                other @ self.value,
                self.units,
                QuantityHistory(MatMul(Constant(other), self.history.operation_tree), self.history.references),
            )

    def __truediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, Quantity):
            return DerivedQuantity(
                self.value / other.value,
                self.units / other.units,
                history=QuantityHistory.apply_operation(Div, self.history, other.history),
            )

        else:
            return DerivedQuantity(
                self.value / other,
                self.units,
                QuantityHistory(Div(Constant(other), self.history.operation_tree), self.history.references),
            )

    def __rtruediv__(self: Self, other: float | Self) -> Self:
        if isinstance(other, Quantity):
            return DerivedQuantity(
                other.value / self.value,
                other.units / self.units,
                history=QuantityHistory.apply_operation(Div, other.history, self.history),
            )

        else:
            return DerivedQuantity(
                other / self.value,
                self.units**-1,
                QuantityHistory(Div(Constant(other), self.history.operation_tree), self.history.references),
            )

    def __add__(self: Self, other: Self | ArrayLike) -> Self:
        if isinstance(other, Quantity):
            if self.units.equivalent(other.units):
                return DerivedQuantity(
                    self.value + (other.value * other.units.scale) / self.units.scale,
                    self.units,
                    QuantityHistory.apply_operation(Add, self.history, other.history),
                )
            else:
                raise UnitError(f"Units do not have the same dimensionality: {self.units} vs {other.units}")

        else:
            raise UnitError(f"Cannot perform addition/subtraction non-quantity {type(other)} with quantity")

    # Don't need __radd__ because only quantity/quantity operations should be allowed

    def __neg__(self):
        return DerivedQuantity(-self.value, self.units, QuantityHistory.apply_operation(Neg, self.history))

    def __sub__(self: Self, other: Self | ArrayLike) -> Self:
        return self + (-other)

    def __rsub__(self: Self, other: Self | ArrayLike) -> Self:
        return (-self) + other

    def __pow__(self: Self, other: int | float):
        return DerivedQuantity(
            self.value**other,
            self.units**other,
            QuantityHistory(Pow(self.history.operation_tree, other), self.history.references),
        )

    @staticmethod
    def _array_repr_format(arr: np.ndarray):
        """Format the array"""
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

        return "[" * order + numbers + "]" * order

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

    @property
    def string_repr(self):
        return str(self.hash_value)

    def as_h5(self, group: h5py.Group, name: str):
        """Add this data onto a group as a dataset under the given name"""
        boxed = self.value if type(self.value) is np.ndarray else [self.value]
        data = group.create_dataset(name, data=boxed)
        data.attrs["units"] = self.units.ascii_symbol


class NamedQuantity[QuantityType](Quantity[QuantityType]):
    def __init__(self,
                 name: str,
                 value: QuantityType,
                 units: Unit,
                 standard_error: QuantityType | None = None,
                 id_header=""):
        super().__init__(value, units, standard_error=standard_error, hash_seed=name, name=name, id_header=id_header)

    def __repr__(self):
        return f"[{self.name}] " + super().__repr__()

    def to_units_of(self, new_units: Unit) -> "NamedQuantity[QuantityType]":
        new_value, new_error = self.in_units_of_with_standard_error(new_units)
        return NamedQuantity(value=new_value, units=new_units, standard_error=new_error, name=self.name)

    def with_standard_error(self, standard_error: Quantity):
        if standard_error.units.equivalent(self.units):
            return NamedQuantity(
                value=self.value,
                units=self.units,
                standard_error=standard_error.in_units_of(self.units),
                name=self.name,
                id_header=self._id_header,
            )

        else:
            raise UnitError(
                f"Standard error units ({standard_error.units}) are not compatible with value units ({self.units})"
            )

    @property
    def string_repr(self):
        return self.name


class DerivedQuantity[QuantityType](Quantity[QuantityType]):
    def __init__(self, value: QuantityType, units: Unit, history: QuantityHistory):
        super().__init__(value, units, standard_error=None)

        self.history = history
        self._variance_cache = None
        self._has_variance = history.has_variance()

    def to_units_of(self, new_units: Unit) -> "Quantity[QuantityType]":
        # TODO: Lots of tests needed for this
        return DerivedQuantity(value=self.in_units_of(new_units), units=new_units, history=self.history)

    @property
    def has_variance(self):
        return self._has_variance

    @property
    def variance(self) -> Quantity:
        if self._variance_cache is None:
            self._variance_cache = self.history.variance_propagate(self.units)

        return self._variance_cache
