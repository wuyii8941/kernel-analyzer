"""Adapter for contracts/ACT.yaml: activation functions and their VJPs, ATen-style inputs ({op, args, kwargs}).

The reference is kernel_spec.aten (real-number semantics of the ATen activation operations) behind a profile-aware
wrapper: profile `torch` uses the documented softplus reversion to the linear function, profile `softplus_formula`
the displayed formula only. The GLU composites (act.swiglu / act.geglu and their VJPs, as Liger and the GLU-variants
paper define them) are composed from the ATen operations here. Every mutant is an independent element-wise
implementation of one misreading listed in the contract.
"""
from __future__ import annotations

from fractions import Fraction

import yaml

from kernel_spec import aten as A
from kernel_spec import real as rl
from kernel_spec.aten_adapter import make_reference
from kernel_spec.num import Invalid, Undefined, Unsupported
from kernel_spec.real import R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "ACT.yaml").read_text())["covers"]
PROFILES = ("torch", "softplus_formula")
APPROX = ("none", "tanh")
K = Fraction(44715, 1000000)


def _same_shape(*ts):
    shapes = {A.as_ten(t).shape for t in ts}
    if len(shapes) != 1:
        raise Invalid(f"the GLU operands must have one shape, got {sorted(shapes)}")


def _approx(a):
    if a not in APPROX:
        raise Invalid(f"approximate={a!r} (documented: 'none' | 'tanh')")
    return a


# ------------------------------------------------------------------ GLU composites (reference)
def swiglu(a, b):
    _same_shape(a, b)
    return A.mul(A.silu(a), b)


def swiglu_backward(g, a, b):
    _same_shape(g, a, b)
    return [A.silu_backward(A.mul(g, b), a), A.mul(g, A.silu(a))]


def geglu(a, b, approximate="none"):
    _same_shape(a, b)
    return A.mul(A.gelu(a, _approx(approximate)), b)


def geglu_backward(g, a, b, approximate="none"):
    _same_shape(g, a, b)
    ap = _approx(approximate)
    return [A.gelu_backward(A.mul(g, b), a, ap), A.mul(g, A.gelu(a, ap))]


def gelu_backward(grad, a, approximate="none"):
    return A.gelu_backward(grad, a, _approx(approximate))


GLU = {"act.swiglu": swiglu, "act.swiglu_backward": swiglu_backward, "act.geglu": geglu,
       "act.geglu_backward": geglu_backward, "aten.gelu_backward": gelu_backward}


# ------------------------------------------------------------------ softplus without the reversion (profile softplus_formula)
def softplus_formula(a, beta=1, threshold=20):
    bt = R(beta)
    return A.unary(lambda x: rl.log1p(rl.exp(bt * x)) / bt, A._floatify(a))


def softplus_backward_formula(grad, a, beta=1, threshold=20):
    bt = R(beta)
    return A.mul(grad, A.unary(lambda x: rl.sigmoid(bt * x), A._floatify(a)))


_REF = {"torch": make_reference(COVERS, extra=GLU),
        "softplus_formula": make_reference(COVERS, extra={**GLU, "aten.softplus": softplus_formula,
                                                          "aten.softplus_backward": softplus_backward_formula})}


def _scan(x):
    """Infinite values are outside every profile (ACT.FINITE); NaN is rejected by the decoder as invalid."""
    if isinstance(x, dict):
        for v in x.values():
            _scan(v)
    elif isinstance(x, list):
        for v in x:
            _scan(v)
    elif isinstance(x, str) and x.strip() in ("inf", "+inf", "-inf"):
        raise Unsupported("infinite input values are outside the ACT profiles (decision ACT.FINITE)")


def _checked(table):
    def ref(profile, inp):
        if profile not in PROFILES:
            raise Unsupported(f"profile {profile!r}")
        _scan(inp.get("args", [])), _scan(inp.get("kwargs") or {})
        return table(profile, inp)
    return ref


def reference(profile, inp):
    return _checked(_REF.get(profile, _REF["torch"]))(profile, inp)


# ------------------------------------------------------------------ independent element-wise implementations (mutants)
def _ew(fn, *ts):
    """Element-wise map over same-shape (or scalar-broadcast) tensors with plain R arithmetic."""
    ts = [A._floatify(A.as_ten(t)) for t in ts]
    return A.Ten(__import__("numpy").vectorize(lambda *v: fn(*[R(e) for e in v]), otypes=[object])(*[t.a for t in ts]),
                 ts[0].dtype)


def _sig(x):
    return R(1) / (1 + rl.exp(-x))


def _phi_cdf(x):
    return (1 + rl.erf(x / rl.sqrt(R(2)))) / 2


def _pdf(x):
    return rl.exp(-(x * x) / 2) / rl.sqrt(2 * R(A._pi()))


def _gelu_tanh(x):
    c = rl.sqrt(R(2) / R(A._pi()))
    return x * (1 + rl.tanh(c * (x + K * x * x * x))) / 2


def _gelu_tanh_d(x):
    c = rl.sqrt(R(2) / R(A._pi()))
    t = rl.tanh(c * (x + K * x * x * x))
    return (1 + t) / 2 + x * (1 - t * t) * c * (1 + 3 * K * x * x) / 2


def m_relu0_one(grad, a, threshold):                          # relu'(0) = 1: pass the gradient where x >= threshold
    return _ew(lambda g, x: g if not (x < R(threshold)) else R(0), grad, a)


def m_relu0_v1_set(grad, a, threshold):                       # v1: at x == threshold the set {0, upstream}
    ga, xa = A._floatify(A.as_ten(grad)).a, A._floatify(A.as_ten(a)).a
    out = []
    for g, x in zip(ga.reshape(-1), xa.reshape(-1)):
        if x.same(R(threshold)):
            out.append(A.AllowedSet([R(0), g]))
        else:
            out.append(g if x > R(threshold) else R(0))
    return A.Ten(__import__("numpy").array(out, dtype=object).reshape(ga.shape), "f32")


def m_gelu_always_erf(a, approximate="none"):                 # approximate ignored
    return _ew(lambda x: x * _phi_cdf(x), a)


def m_gelu_always_tanh(a, approximate="none"):                # tanh form used for approximate='none'
    return _ew(_gelu_tanh, a)


def m_gelu_backward_cdf_only(grad, a, approximate="none"):    # x * pdf term dropped
    return _ew(lambda g, x: g * (_phi_cdf(x) if approximate == "none" else (1 + rl.tanh(
        rl.sqrt(R(2) / R(A._pi())) * (x + K * x * x * x))) / 2), grad, a)


def m_silu_backward_sigmoid(grad, a):                         # silu' = sigmoid
    return _ew(lambda g, x: g * _sig(x), grad, a)


def m_softplus_no_beta(a, beta=1, threshold=20):              # beta ignored
    return _ew(lambda x: x if x > R(threshold) else rl.log(1 + rl.exp(x)), a)


def m_softplus_threshold_on_x(a, beta=1, threshold=20):       # reversion tested on x instead of beta * x
    bt = R(beta)
    return _ew(lambda x: x if x > R(threshold) else rl.log(1 + rl.exp(bt * x)) / bt, a)


def m_softplus_no_reversion(a, beta=1, threshold=20):         # torch profile without the documented reversion
    bt = R(beta)
    return _ew(lambda x: rl.log(1 + rl.exp(bt * x)) / bt, a)


def m_softplus_reversion_in_formula(a, beta=1, threshold=20):  # reversion applied in the softplus_formula profile
    bt = R(beta)
    return _ew(lambda x: x if bt * x > R(threshold) else rl.log(1 + rl.exp(bt * x)) / bt, a)


def m_tanh_backward_from_input(grad, y):                      # 1 - tanh(y)^2 instead of 1 - y^2
    return _ew(lambda g, v: g * (1 - rl.tanh(v) * rl.tanh(v)), grad, y)


def m_sigmoid_backward_from_input(grad, y):                   # sigma(y)(1 - sigma(y)) instead of y(1 - y)
    return _ew(lambda g, v: g * _sig(v) * (1 - _sig(v)), grad, y)


def m_swiglu_backward_no_b(g, a, b):                          # da misses the factor b
    return [_ew(lambda u, x: u * _sig(x) * (1 + x * (1 - _sig(x))), g, a), _ew(lambda u, x: u * x * _sig(x), g, a)]


def m_swiglu_backward_silu_not_derivative(g, a, b):           # da uses silu(a) instead of silu'(a)
    return [_ew(lambda u, x, v: u * v * x * _sig(x), g, a, b), _ew(lambda u, x: u * x * _sig(x), g, a)]


def m_geglu_wrong_form(a, b, approximate="none"):              # the other GELU form
    if approximate == "none":
        return _ew(lambda x, v: _gelu_tanh(x) * v, a, b)
    return _ew(lambda x, v: x * _phi_cdf(x) * v, a, b)


def m_gelu_backward_overflow(grad, a, approximate="none"):    # B023: x*x overflows float32, 0 * inf taken as undefined
    def d(g, x):
        if approximate == "tanh" and x * x >= R(2 ** 128):
            raise Undefined("x*x overflow: 0 * inf")
        return g * (_gelu_tanh_d(x) if approximate == "tanh" else _phi_cdf(x) + x * _pdf(x))
    return _ew(d, grad, a)


def _mut(**ops):
    torch_tab = make_reference(COVERS, extra={**GLU, **ops})
    form_tab = make_reference(COVERS, extra={**GLU, "aten.softplus": softplus_formula,
                                             "aten.softplus_backward": softplus_backward_formula, **ops})

    def ref(profile, inp):
        return _checked(torch_tab if profile == "torch" else form_tab)(profile, inp)
    return ref


MUTANTS = {
    "ACT.M1": _mut(**{"aten.threshold_backward": m_relu0_one}),
    "ACT.M2": _mut(**{"aten.threshold_backward": m_relu0_v1_set}),
    "ACT.M3": _mut(**{"aten.gelu": m_gelu_always_erf}),
    "ACT.M4": _mut(**{"aten.gelu": m_gelu_always_tanh}),
    "ACT.M5": _mut(**{"aten.gelu_backward": m_gelu_backward_cdf_only}),
    "ACT.M6": _mut(**{"aten.silu_backward": m_silu_backward_sigmoid}),
    "ACT.M7": _mut(**{"aten.softplus": m_softplus_no_beta}),
    "ACT.M8": _mut(**{"aten.softplus": m_softplus_threshold_on_x}),
    "ACT.M9": _mut(**{"aten.softplus": m_softplus_no_reversion}),
    "ACT.M10": _mut(**{"aten.softplus": m_softplus_reversion_in_formula}),
    "ACT.M11": _mut(**{"aten.tanh_backward": m_tanh_backward_from_input}),
    "ACT.M12": _mut(**{"aten.sigmoid_backward": m_sigmoid_backward_from_input}),
    "ACT.M13": _mut(**{"act.swiglu_backward": m_swiglu_backward_no_b}),
    "ACT.M14": _mut(**{"act.swiglu_backward": m_swiglu_backward_silu_not_derivative}),
    "ACT.M15": _mut(**{"act.geglu": m_geglu_wrong_form}),
    "ACT.M16": _mut(**{"aten.gelu_backward": m_gelu_backward_overflow}),
}
READINGS = {}
