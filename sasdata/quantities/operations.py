from typing import Any, TypeVar, Union

import json

T = TypeVar("T")

def hash_and_name(hash_or_name: int | str):
    """ Infer the name of a variable from a hash, or the hash from the name

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
    def summary(self, indent_amount: int = 0, indent: str="  "):
        """ Summary of the operation tree"""

        s = f"{indent_amount*indent}{self._summary_open()}(\n"

        for chunk in self._summary_components():
            s += chunk.summary(indent_amount+1, indent) + "\n"

        s += f"{indent_amount*indent})"

        return s
    def _summary_open(self):
        """ First line of summary """

    def _summary_components(self) -> list["Operation"]:
        return []
    def evaluate(self, variables: dict[int, T]) -> T:

        """ Evaluate this operation """

    def _derivative(self, hash_value: int) -> "Operation":
        """ Get the derivative of this operation """

    def _clean(self):
        """ Clean up this operation - i.e. remove silly things like 1*x """
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
        for i in range(100): # set max iterations

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
        cls = _serialisation_lookup[operation]

        try:
            return cls._deserialise(parameters)

        except NotImplementedError:
            raise NotImplementedError(f"No method to deserialise {operation} with {parameters} (cls={cls})")

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        raise NotImplementedError(f"Deserialise not implemented for this class")

    def serialise(self) -> str:
        return json.dumps(self._serialise_json())

    def _serialise_json(self) -> dict[str, Any]:
        return {"operation": self.serialisation_name,
                "parameters": self._serialise_parameters()}

    def _serialise_parameters(self) -> dict[str, Any]:
        raise NotImplementedError("_serialise_parameters not implemented")

    def __eq__(self, other: "Operation"):
        return NotImplemented

class ConstantBase(Operation):
    pass

class AdditiveIdentity(ConstantBase):

    serialisation_name = "zero"
    def evaluate(self, variables: dict[int, T]) -> T:
        return 0

    def _derivative(self, hash_value: int) -> Operation:
        return AdditiveIdentity()

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return AdditiveIdentity()

    def _serialise_parameters(self) -> dict[str, Any]:
        return {}

    def summary(self, indent_amount: int=0, indent="  "):
        return f"{indent_amount*indent}0 [Add.Id.]"

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


    def summary(self, indent_amount: int=0, indent="  "):
        return f"{indent_amount*indent}1 [Mul.Id.]"

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

    def summary(self, indent_amount: int = 0, indent: str="  "):
        pass

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
        value = parameters["value"]
        return Constant(value)


    def _serialise_parameters(self) -> dict[str, Any]:
        return {"value": self.value}


    def summary(self, indent_amount: int=0, indent="  "):
        return f"{indent_amount*indent}{self.value}"

    def __eq__(self, other):
        if isinstance(other, AdditiveIdentity):
            return self.value == 0

        elif isinstance(other, MultiplicativeIdentity):
            return self.value == 1

        elif isinstance(other, Constant):
            if other.value == self.value:
                return True

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
        return {"hash_value": self.hash_value,
                "name": self.name}

    def summary(self, indent_amount: int = 0, indent: str="  "):
        return f"{indent_amount*indent}{self.name}"

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.hash_value == other.hash_value

        return False

class UnaryOperation(Operation):

    def __init__(self, a: Operation):
        self.a = a

    def _serialise_parameters(self) -> dict[str, Any]:
        return {"a": self.a._serialise_json()}

    def _summary_components(self) -> list["Operation"]:
        return [self.a]




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

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return Neg(Operation.deserialise_json(parameters["a"]))


    def _summary_open(self):
        return "Neg"

    def __eq__(self, other):
        if isinstance(other, Neg):
            return other.a == self.a


class Inv(UnaryOperation):

    serialisation_name = "reciprocal"

    def evaluate(self, variables: dict[int, T]) -> T:
        return 1/self.a.evaluate(variables)

    def _derivative(self, hash_value: int) -> Operation:
        return Neg(Div(self.a._derivative(hash_value), Mul(self.a, self.a)))

    def _clean(self):
        clean_a = self.a._clean()

        if isinstance(clean_a, Inv):
            # Removes double negations
            return clean_a.a

        elif isinstance(clean_a, Neg):
            # cannonicalise 1/-a to -(1/a)
            # over multiple iterations this should have the effect of ordering and gathering Neg and Inv
            return Neg(Inv(clean_a.a))

        elif isinstance(clean_a, Constant):
            return Constant(1/clean_a.value)._clean()

        else:
            return Inv(clean_a)


    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return Inv(Operation.deserialise_json(parameters["a"]))

    def _summary_open(self):
        return "Inv"


    def __eq__(self, other):
        if isinstance(other, Inv):
            return other.a == self.a

class BinaryOperation(Operation):
    def __init__(self, a: Operation, b: Operation):
        self.a = a
        self.b = b

    def _clean(self):
        return self._clean_ab(self.a._clean(), self.b._clean())

    def _clean_ab(self, a, b):
        raise NotImplementedError("_clean_ab not implemented")

    def _serialise_parameters(self) -> dict[str, Any]:
        return {"a": self.a._serialise_json(),
                "b": self.b._serialise_json()}

    @staticmethod
    def _deserialise_ab(parameters) -> tuple[Operation, Operation]:
        return (Operation.deserialise_json(parameters["a"]),
                Operation.deserialise_json(parameters["b"]))


    def _summary_components(self) -> list["Operation"]:
        return [self.a, self.b]

    def _self_cls(self) -> type:
        """ Own class"""
    def __eq__(self, other):
        if isinstance(other, self._self_cls()):
            return other.a == self.a and self.b == other.b

class Add(BinaryOperation):

    serialisation_name = "add"

    def _self_cls(self) -> type:
        return Add
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

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return Add(*BinaryOperation._deserialise_ab(parameters))

    def _summary_open(self):
        return "Add"

class Sub(BinaryOperation):

    serialisation_name = "sub"


    def _self_cls(self) -> type:
        return Sub
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

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return Sub(*BinaryOperation._deserialise_ab(parameters))


    def _summary_open(self):
        return "Sub"

class Mul(BinaryOperation):

    serialisation_name = "mul"


    def _self_cls(self) -> type:
        return Mul
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


    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return Mul(*BinaryOperation._deserialise_ab(parameters))


    def _summary_open(self):
        return "Mul"

class Div(BinaryOperation):

    serialisation_name = "div"


    def _self_cls(self) -> type:
        return Div

    def evaluate(self, variables: dict[int, T]) -> T:
        return self.a.evaluate(variables) / self.b.evaluate(variables)

    def _derivative(self, hash_value: int) -> Operation:
        return Sub(Div(self.a.derivative(hash_value), self.b),
                   Div(Mul(self.a, self.b.derivative(hash_value)), Mul(self.b, self.b)))

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


    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return Div(*BinaryOperation._deserialise_ab(parameters))

    def _summary_open(self):
        return "Div"

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
            return Mul(Constant(self.power), Mul(Pow(self.a, self.power-1), self.a._derivative(hash_value)))

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
        return {"a": Operation._serialise_json(self.a),
                "power": self.power}

    @staticmethod
    def _deserialise(parameters: dict) -> "Operation":
        return Pow(Operation.deserialise_json(parameters["a"]), parameters["power"])

    def summary(self, indent_amount: int=0, indent="  "):
        return (f"{indent_amount*indent}Pow\n" +
                self.a.summary(indent_amount+1, indent) + "\n" +
                f"{(indent_amount+1)*indent}{self.power}\n" +
                f"{indent_amount*indent})")

    def __eq__(self, other):
        if isinstance(other, Pow):
            return self.a == other.a and self.power == other.power

_serialisable_classes = [AdditiveIdentity, MultiplicativeIdentity, Constant,
                        Variable,
                        Neg, Inv,
                        Add, Sub, Mul, Div, Pow]

_serialisation_lookup = {cls.serialisation_name: cls for cls in _serialisable_classes}
