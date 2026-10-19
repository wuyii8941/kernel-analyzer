"""Independent exact differentiation of the NORM forward definitions.

The code below does not call c_norm's VJP formulas.  It differentiates the
documented forward expression one input at a time with rational dual numbers.
The selected cases have rational square roots, so every comparison is exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

import pytest

from kernel_spec import registry
from kernel_spec.real import R


def Q(value):
    return value if isinstance(value, Fraction) else Fraction(value)


@dataclass(frozen=True)
class Dual:
    value: Fraction
    derivative: Fraction = Fraction(0)

    @staticmethod
    def of(value):
        return value if isinstance(value, Dual) else Dual(Q(value))

    def __add__(self, other):
        other = self.of(other)
        return Dual(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.derivative)

    def __sub__(self, other):
        return self + (-self.of(other))

    def __rsub__(self, other):
        return self.of(other) - self

    def __mul__(self, other):
        other = self.of(other)
        return Dual(self.value * other.value,
                    self.derivative * other.value + self.value * other.derivative)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.of(other)
        return Dual(self.value / other.value,
                    (self.derivative * other.value - self.value * other.derivative) /
                    (other.value * other.value))

    def __rtruediv__(self, other):
        return self.of(other) / self

    def sqrt(self):
        numerator, denominator = isqrt(self.value.numerator), isqrt(self.value.denominator)
        assert numerator * numerator == self.value.numerator
        assert denominator * denominator == self.value.denominator
        root = Fraction(numerator, denominator)
        return Dual(root, self.derivative / (2 * root))


def _normalise(values, eps, centred):
    n = len(values)
    mean = sum(values, Dual(Fraction(0))) / n if centred else Dual(Fraction(0))
    deviations = [value - mean for value in values]
    moment = sum((value * value for value in deviations), Dual(Fraction(0))) / n
    rstd = 1 / (moment + eps).sqrt()
    return [value * rstd for value in deviations], mean, rstd


def _differentiate(values, forward, upstream):
    outputs, saved = forward([Dual(Q(value)) for value in values])
    gradients = []
    for active in range(len(values)):
        variables = [Dual(Q(value), Fraction(index == active)) for index, value in enumerate(values)]
        outputs, _ = forward(variables)
        loss = sum((Dual.of(g) * output for g, output in zip(upstream, outputs)), Dual(Fraction(0)))
        gradients.append(loss.derivative)
    return [value.value for value in outputs], [value.value for value in saved], gradients


def _plain(value):
    if isinstance(value, R):
        return value.v
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return Q(value)


def _flat(value):
    if isinstance(value, list):
        return [item for child in value for item in _flat(child)]
    return [value]


def _reference(profile, inp):
    return _plain(registry.load("NORM").adapter.reference(profile, inp).fields)


def test_layer_norm_vjp_from_forward_dual_numbers():
    x = [1, 3, 5, 7]
    weight = [2, -1]
    bias = [Fraction(1, 2), 0]
    upstream = [1, 2, 3, -1]

    def forward(v):
        xs, ws, bs = v[:4], v[4:6], v[6:8]
        out, saved = [], []
        for start in (0, 2):
            normal, mean, rstd = _normalise(xs[start:start + 2], Fraction(9, 16), centred=True)
            out.extend([normal[j] * ws[j] + bs[j] for j in range(2)])
            saved.extend([mean, rstd])
        return out, saved

    y, saved, grad = _differentiate(x + weight + bias, forward, upstream)
    inp = {"op": "layer_norm", "x": [[1, 3], [5, 7]], "normalized_shape": [2],
           "weight": weight, "bias": bias, "eps": Fraction(9, 16), "upstream": [[1, 2], [3, -1]]}
    got = _reference("torch", inp)
    assert _flat(got["y"]) == y
    assert _flat(got["saved"]["mean"]) == saved[::2]
    assert _flat(got["saved"]["rstd"]) == saved[1::2]
    assert _flat(got["grad"]["x"]) == grad[:4]
    assert _flat(got["grad"]["weight"]) == grad[4:6]
    assert _flat(got["grad"]["bias"]) == grad[6:8]


@pytest.mark.parametrize("profile,offset", [("torch", None), ("rms_offset", Fraction(1))])
def test_rms_norm_vjp_from_forward_dual_numbers(profile, offset):
    x = [2, 4, 4, 2]
    weight = [1, 3]
    upstream = [2, -1, -2, 3]

    def forward(v):
        xs, ws = v[:4], v[4:]
        out, saved = [], []
        effective = ws if offset is None else [value + offset for value in ws]
        for start in (0, 2):
            normal, _, rstd = _normalise(xs[start:start + 2], Fraction(6), centred=False)
            out.extend([normal[j] * effective[j] for j in range(2)])
            saved.append(rstd)
        return out, saved

    y, saved, grad = _differentiate(x + weight, forward, upstream)
    inp = {"op": "rms_norm", "x": [[2, 4], [4, 2]], "normalized_shape": [2],
           "weight": weight, "eps": 6, "upstream": [[2, -1], [-2, 3]]}
    if offset is not None:
        inp["offset"] = offset
    got = _reference(profile, inp)
    assert _flat(got["y"]) == y
    assert _flat(got["saved"]["rstd"]) == saved
    assert _flat(got["grad"]["x"]) == grad[:4]
    assert _flat(got["grad"]["weight"]) == grad[4:]


def test_group_norm_vjp_from_forward_dual_numbers():
    x = [0, 2, 0, 2, 4, 6, 4, 6]
    weight = [1, 2, -1, Fraction(1, 2)]
    bias = [0, 1, 0, -1]
    upstream = [1, 0, 0, 1, 2, 0, 0, 0]

    def forward(v):
        xs, ws, bs = v[:8], v[8:12], v[12:16]
        out = [None] * 8
        saved = []
        for group, positions in enumerate(((0, 1, 2, 3), (4, 5, 6, 7))):
            normal, mean, rstd = _normalise([xs[index] for index in positions], Fraction(9, 16), centred=True)
            saved.extend([mean, rstd])
            for local, index in enumerate(positions):
                channel = group * 2 + local // 2
                out[index] = normal[local] * ws[channel] + bs[channel]
        return out, saved

    y, saved, grad = _differentiate(x + weight + bias, forward, upstream)
    inp = {"op": "group_norm", "x": [[[0, 2], [0, 2], [4, 6], [4, 6]]], "num_groups": 2,
           "weight": weight, "bias": bias, "eps": Fraction(9, 16),
           "upstream": [[[1, 0], [0, 1], [2, 0], [0, 0]]]}
    got = _reference("torch", inp)
    assert _flat(got["y"]) == y
    assert _flat(got["saved"]["mean"]) == saved[::2]
    assert _flat(got["saved"]["rstd"]) == saved[1::2]
    assert _flat(got["grad"]["x"]) == grad[:8]
    assert _flat(got["grad"]["weight"]) == grad[8:12]
    assert _flat(got["grad"]["bias"]) == grad[12:16]
