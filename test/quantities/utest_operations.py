import pytest

from sasdata.quantities.quantity import Operation, \
    Neg, Inv, \
    Add, Sub, Mul, Div, Pow, \
    Variable, Constant, AdditiveIdentity, MultiplicativeIdentity

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


