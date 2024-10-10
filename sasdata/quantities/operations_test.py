import pytest

from sasdata.quantities.operations import Operation, \
    Neg, Inv, \
    Add, Sub, Mul, Div, Pow, Sqrt, Sin, Cos, \
    Variable, Constant, AdditiveIdentity, MultiplicativeIdentity

from math import pi

operation_with_everything = \
    Div(
        Pow(
            Mul(
                Sub(
                    Add(
                        Neg(Inv(MultiplicativeIdentity())),
                        Variable("x")),
                    Constant(7)),
                AdditiveIdentity()),
            2),
        Variable("y"))

def test_serialise_deserialise():
    print(operation_with_everything._serialise_json())

    serialised = operation_with_everything.serialise()
    deserialised = Operation.deserialise(serialised)
    reserialised = deserialised.serialise()

    assert serialised == reserialised


@pytest.mark.parametrize("op, a, b, result", [
    (Add, 1, 1, 2),
    (Add, 7, 8, 15),
    (Sub, 1, 1, 0),
    (Sub, 7, 8, -1),
    (Mul, 1, 1, 1),
    (Mul, 7, 8, 56),
    (Div, 1, 1, 1),
    (Div, 7, 8, 7/8),
    (Pow, 1, 1, 1),
    (Pow, 7, 2, 49)])
def test_binary_evaluation(op, a, b, result):
    f = op(Constant(a), b if op == Pow else Constant(b))
    assert f.evaluate({}) == result

@pytest.mark.parametrize("op, a, result", [
    (Inv, 1, 1.0),
    (Sqrt, 25, 5.0),
    (Sin, pi/2, 1.0),
    (Cos, pi, -1.0)
])
def test_unary_operation(op: Operation, a: int, result: int | float):
    f = op(Constant(a))
    assert f.evaluate({}) == result

x = Variable("x")
y = Variable("y")
z = Variable("z")
@pytest.mark.parametrize("x_over_x", [
                         Div(x,x),
                         Mul(Inv(x), x),
                         Mul(x, Inv(x)),
])
def test_dx_over_x_by_dx_should_be_zero(x_over_x):


    dfdx = x_over_x.derivative(x)

    print(dfdx.summary())

    assert dfdx == AdditiveIdentity()


def test_d_xyz_by_components_should_be_1():
    f = Mul(Mul(x, y), z)
    assert f.derivative(x).derivative(y).derivative(z) == MultiplicativeIdentity()

@pytest.mark.parametrize("f, expected_derivative", [
    (Mul(x, x), Mul(Constant(2), x)),
    (Add(Mul(Constant(2), Pow(x, 2)), Add(Mul(Constant(5), x), Constant(3))),
     Add(Mul(Constant(4), x), Constant(5)))
])
def test_expected_derivative(f: Operation, expected_derivative: Operation):
    assert f.derivative(x) == expected_derivative

