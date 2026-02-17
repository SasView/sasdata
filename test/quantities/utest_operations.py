import math

import numpy as np
import pytest

from sasdata.quantities.quantity import (
    Add,
    AdditiveIdentity,
    ArcCos,
    ArcSin,
    ArcTan,
    Constant,
    Cos,
    Div,
    Dot,
    Exp,
    Inv,
    Ln,
    Log,
    MatMul,
    Mul,
    MultiplicativeIdentity,
    Neg,
    Operation,
    Pow,
    Sin,
    Sub,
    Tan,
    Transpose,
    Variable,
)

x = Variable("x")
y = Variable("y")
z = Variable("z")

operation_with_everything = Div(
    Pow(
        Mul(
            Sub(Add(Neg(Inv(MultiplicativeIdentity())), Ln(Transpose(x))), Log(Constant(7), 2)),
            AdditiveIdentity(),
        ),
        2,
    ),
    y,
)


@pytest.fixture(params=[Inv, Exp, Ln, Neg, Sin, ArcSin, Cos, ArcCos, Tan, ArcTan, Transpose])
def unary_operation(request):
    return request.param(x)


@pytest.fixture(params=[Add, Div, Dot, MatMul, Mul, Sub])
def binary_operation(request):
    return request.param(x, y)


@pytest.fixture(params=[Log, Pow])
def log_pow_operation(request):
    return request.param(x, 2)


def test_serialise_deserialise():
    serialised = operation_with_everything.serialise()
    deserialised = Operation.deserialise(serialised)
    reserialised = deserialised.serialise()

    assert serialised == reserialised


def test_unary_serialise_deserialise(unary_operation):
    serialised = unary_operation.serialise()
    deserialised = Operation.deserialise(serialised)
    reserialised = deserialised.serialise()

    assert serialised == reserialised


def test_binary_serialise_deserialise(binary_operation):
    serialised = binary_operation.serialise()
    deserialised = Operation.deserialise(serialised)
    reserialised = deserialised.serialise()

    assert serialised == reserialised


def test_log_pow_serialise_deserialise(log_pow_operation):
    serialised = log_pow_operation.serialise()
    deserialised = Operation.deserialise(serialised)
    reserialised = deserialised.serialise()

    assert serialised == reserialised


@pytest.mark.parametrize(
    "op, summary",
    [(AdditiveIdentity, "0 [Add.Id.]"), (MultiplicativeIdentity, "1 [Mul.Id.]"), (Operation, "Operation(\n)")],
)
def test_summary(op, summary):
    f = op()
    assert f.summary() == summary


def test_variable_summary():
    assert x.summary() == "x"


def test_unary_summary(unary_operation):
    assert unary_operation.summary() == f"{unary_operation.__class__.__name__}(\n  x\n)"


def test_binary_summary(binary_operation):
    assert binary_operation.summary() == f"{binary_operation.__class__.__name__}(\n  x\n  y\n)"


def test_log_pow_summary(log_pow_operation):
    assert log_pow_operation.summary() == f"{log_pow_operation.__class__.__name__}(\n  x\n  2\n)"


@pytest.mark.parametrize("op, result", [(AdditiveIdentity, 0), (MultiplicativeIdentity, 1), (Operation, None)])
def test_evaluation(op, result):
    f = op()
    assert f.evaluate({}) == result


@pytest.mark.parametrize(
    "op, a, result",
    [
        (Neg, 1, -1),
        (Neg, -7, 7),
        (Inv, 2, 0.5),
        (Inv, 0.125, 8),
        (Exp, 1, math.e),
        (Exp, math.log(5.0), pytest.approx(5.0)),
        (Ln, np.sqrt(math.e), 0.5),
        (Ln, math.e**5, pytest.approx(5.0)),
        (Sin, math.pi / 6.0, pytest.approx(0.5)),
        (Sin, 0.5 * math.pi, pytest.approx(1.0)),
        (Cos, 0.0, pytest.approx(1.0)),
        (Cos, math.pi / 3.0, pytest.approx(0.5)),
        (Tan, 0.0, pytest.approx(0.0)),
        (Tan, 0.25 * math.pi, pytest.approx(1.0)),
        (ArcSin, 1.0, 0.5 * math.pi),
        (ArcSin, -1.0, -0.5 * math.pi),
        (ArcCos, 1.0, 0.0),
        (ArcCos, -1.0, math.pi),
        (ArcTan, 0.0, 0.0),
        (ArcTan, -1.0, -0.25 * math.pi),
    ],
)
def test_unary_evaluation(op, a, result):
    f = op(Constant(a))
    assert f.evaluate({}) == result


@pytest.mark.parametrize(
    "op, a, b, result",
    [
        (Add, 1, 1, 2),
        (Add, 7, 8, 15),
        (Sub, 1, 1, 0),
        (Sub, 7, 8, -1),
        (Mul, 1, 1, 1),
        (Mul, 7, 8, 56),
        (Div, 1, 1, 1),
        (Div, 7, 8, 7.0 / 8.0),
        (Dot, [1, 2], [2, 1], 4),
        (Dot, [7, 8], [8, 7], 112),
        (Pow, 1, 1, 1),
        (Pow, 7, 2, 49),
        (Log, 100, 10, 2),
        (Log, 256, 2, 8),
    ],
)
def test_binary_evaluation(op, a, b, result):
    f = op(Constant(a), b if op == Log or op == Pow else Constant(b))
    assert f.evaluate({}) == result


@pytest.mark.parametrize(
    "op, a, b, result",
    [
        (MatMul, np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]), np.array([[2, 2], [2, 2]])),
        (MatMul, np.array([[7, 7], [7, 7]]), np.array([[8, 8], [8, 8]]), np.array([[112, 112], [112, 112]])),
    ],
)
def test_matmul_evaluation(op, a, b, result):
    f = op(Constant(a), Constant(b))
    assert (f.evaluate({}) == result).all()


@pytest.mark.parametrize(
    "op, a, result",
    [(Transpose, np.array([[1, 2]]), np.array([[1], [2]])), (Transpose, [[1, 2], [3, 4]], [[1, 3], [2, 4]])],
)
def test_transpose_evaluation(op, a, result):
    f = op(Constant(a))
    assert (f.evaluate({}) == result).all()


@pytest.mark.parametrize(
    "op, result",
    [(AdditiveIdentity, AdditiveIdentity()), (MultiplicativeIdentity, AdditiveIdentity()), (Operation, None)],
)
def test_derivative(op, result):
    f = op()
    assert f.derivative(x, simplify=False) == result


@pytest.mark.parametrize(
    "op",
    [
        (Neg(Neg(x))),
        (Inv(Inv(x))),
    ],
)
def test_clean_double_applications(op):
    assert op._clean() == x


@pytest.mark.parametrize(
    "op",
    [
        (Exp(Ln(x))),
        (Ln(Exp(x))),
    ],
)
def test_clean_exp_ln_functions(op):
    assert op._clean() == x


@pytest.mark.parametrize(
    "op",
    [
        (Sin(ArcSin(x))),
        (Cos(ArcCos(x))),
        (Tan(ArcTan(x))),
        (ArcSin(Sin(x))),
        (ArcCos(Cos(x))),
        (ArcTan(Tan(x))),
    ],
)
def test_clean_trig_functions(op):
    assert op._clean() == x


@pytest.mark.parametrize(
    "x_over_x",
    [
        Div(x, x),
        Mul(Inv(x), x),
        Mul(x, Inv(x)),
    ],
)
def test_dx_over_x_by_dx_should_be_zero(x_over_x):
    dfdx = x_over_x.derivative(x)
    assert dfdx == AdditiveIdentity()


def test_d_xyz_by_components_should_be_1():
    f = Mul(Mul(x, y), z)
    assert f.derivative(x).derivative(y).derivative(z) == MultiplicativeIdentity()
