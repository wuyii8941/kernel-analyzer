"""Independent automatic differentiation of the ACT forward formulas.

The reference's closed-form backward functions are not called to derive the
expected gradients.  Forward expressions are differentiated with dual numbers
and evaluated with high-precision Arb ball arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import flint
import pytest

from kernel_spec import registry
from kernel_spec.num import ends


ACT = registry.load("ACT").adapter


def Q(value):
    return value if isinstance(value, Fraction) else Fraction(value)


def A(value):
    value = Q(value)
    return flint.arb(int(value.numerator)) / flint.arb(int(value.denominator))


def arb_bounds(value):
    def endpoint(bound):
        mantissa, exponent = bound.man_exp()
        return Fraction(int(mantissa)) * Fraction(2) ** int(exponent)
    return endpoint(value.lower()), endpoint(value.upper())


@dataclass(frozen=True)
class Dual:
    value: object
    derivative: object

    @classmethod
    def constant(cls, value):
        return cls(value if isinstance(value, flint.arb) else A(value), A(0))

    @classmethod
    def variable(cls, value):
        return cls(A(value), A(1))

    def _coerce(self, other):
        return other if isinstance(other, Dual) else Dual.constant(other)

    def __add__(self, other):
        other = self._coerce(other)
        return Dual(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.derivative)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        return Dual(self.value * other.value,
                    self.derivative * other.value + self.value * other.derivative)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._coerce(other)
        return Dual(self.value / other.value,
                    (self.derivative * other.value - self.value * other.derivative) /
                    (other.value * other.value))

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __pow__(self, power):
        assert type(power) is int and power >= 0
        out = Dual.constant(1)
        for _ in range(power):
            out = out * self
        return out


def exp(value):
    out = value.value.exp()
    return Dual(out, value.derivative * out)


def log(value):
    return Dual(value.value.log(), value.derivative / value.value)


def tanh(value):
    out = value.value.tanh()
    return Dual(out, value.derivative * (1 - out * out))


def erf(value):
    out = value.value.erf()
    derivative = 2 / flint.arb.pi().sqrt() * (-value.value * value.value).exp()
    return Dual(out, value.derivative * derivative)


def sigmoid(value):
    return 1 / (1 + exp(-value))


def gelu_erf(x):
    return x * (1 + erf(x / flint.arb(2).sqrt())) / 2


def gelu_tanh(x):
    coefficient = (flint.arb(2) / flint.arb.pi()).sqrt()
    return x * (1 + tanh(coefficient * (x + Fraction(44715, 1000000) * x ** 3))) / 2


def silu(x):
    return x * sigmoid(x)


def softplus_formula(x, beta):
    return log(1 + exp(beta * x)) / beta


def T(value):
    return {"t": "f64", "v": [value]}


def reference(profile, op, args, kwargs=None):
    query = {"op": op, "args": args}
    if kwargs is not None:
        query["kwargs"] = kwargs
    result = ACT.reference(profile, query)
    assert result.outcome == "value"
    return result.fields["out"]


def assert_contains(reference_value, independent_ball):
    lower, upper = ends(reference_value)
    expected_lower, expected_upper = arb_bounds(independent_ball)
    assert lower <= expected_lower <= expected_upper <= upper


@pytest.mark.parametrize("forward,op,approximate", [
    (gelu_erf, "aten.gelu", "none"),
    (gelu_tanh, "aten.gelu", "tanh"),
    (silu, "aten.silu", None),
])
@pytest.mark.parametrize("x", [Fraction(-2), Fraction(-1, 2), Fraction(0), Fraction(5, 4)])
def test_forward_and_backward_follow_independent_dual_differentiation(forward, op, approximate, x):
    dual = forward(Dual.variable(x))
    kwargs = None if approximate is None else {"approximate": approximate}
    forward_value = reference("torch", op, [T(x)], kwargs)[0]
    backward_op = op + "_backward"
    backward_value = reference("torch", backward_op, [T(3), T(x)], kwargs)[0]
    assert_contains(forward_value, dual.value)
    assert_contains(backward_value, 3 * dual.derivative)


def test_formula_softplus_and_both_glu_gradients_are_independently_differentiated():
    x, beta = Fraction(1, 3), Fraction(2)
    dual = softplus_formula(Dual.variable(x), beta)
    kwargs = {"beta": beta, "threshold": -100}
    assert_contains(reference("softplus_formula", "aten.softplus", [T(x)], kwargs)[0], dual.value)
    assert_contains(reference("softplus_formula", "aten.softplus_backward", [T(3), T(x)], kwargs)[0],
                    3 * dual.derivative)

    a, b, upstream = Fraction(1, 2), Fraction(-3, 2), Fraction(2)
    for op, forward in (("act.swiglu", silu), ("act.geglu", gelu_tanh)):
        kwargs = {"approximate": "tanh"} if op == "act.geglu" else None
        da = forward(Dual.variable(a)) * Dual.constant(b)
        db = forward(Dual.constant(a)) * Dual.variable(b)
        value = reference("torch", op, [T(a), T(b)], kwargs)[0]
        gradients = reference("torch", op + "_backward", [T(upstream), T(a), T(b)], kwargs)
        assert_contains(value, da.value)
        assert_contains(gradients[0][0], A(upstream) * da.derivative)
        assert_contains(gradients[1][0], A(upstream) * db.derivative)


@pytest.mark.parametrize("forward,op", [
    (tanh, "vjp.aten.tanh"),
    (sigmoid, "vjp.aten.sigmoid"),
])
@pytest.mark.parametrize("x", [Fraction(-2), Fraction(-1, 2), Fraction(0), Fraction(5, 4)])
def test_public_tanh_and_sigmoid_vjps_start_from_the_original_input(forward, op, x):
    dual = forward(Dual.variable(x))
    value = reference("torch", op, [T(3), T(x)])[0]
    assert_contains(value, 3 * dual.derivative)


@pytest.fixture(autouse=True)
def high_precision_arb():
    saved = flint.ctx.prec
    flint.ctx.prec = 768
    try:
        yield
    finally:
        flint.ctx.prec = saved
