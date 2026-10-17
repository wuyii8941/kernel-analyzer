"""Reference for contracts/CLIP.yaml: get_total_norm, clip_grad_norm_ (= get_total_norm then clip_grads_with_norm_)
and clip_grad_value_, in real arithmetic (profile real) or with the IEEE 754 special values of the documented formula
for non-finite gradients (profile ieee_nonfinite)."""
from __future__ import annotations

from fractions import Fraction

from .e2_common import NAN, is_nan, nested, plain, tensor
from .num import Invalid, Undefined, Unsupported, rational
from .real import NEG, POS, R, abs_, max_, min_, sqrt

EPS = Fraction(1, 10 ** 6)                      # the documented 1e-6, read as the decimal it denotes (CLIP.EPS)
OPS = ("clip_grad_norm", "clip_grad_value", "get_total_norm")
PROFILES = ("real", "ieee_nonfinite")


def _grads(inp, ieee):
    out = []
    for i, g in enumerate(inp["grads"]):
        if g is None:
            out.append(None)
            continue
        t = tensor(g if isinstance(g, list) else [g], f"grads[{i}]", allow_nan=True)
        out.append((t, isinstance(g, list)))
    return out


def _norm_type(v):
    if isinstance(v, str) and v.strip() in ("inf", "+inf"):
        return "inf"
    p = rational(v, "norm_type")
    if p <= 0:
        raise Unsupported("norm_type <= 0 (0-norm, -inf-norm, negative p) is outside this contract")
    return p


def total_norm(vals, p, ieee):
    if any(is_nan(v) for v in vals):
        if not ieee:
            raise Undefined("NaN gradient: outside the real profile")
        if p == "inf":
            raise Undefined("inf-norm of a NaN: no documented or IEEE-specified result (max is not an IEEE operation)")
        return NAN                                   # IEEE 754 6.2: a sum with a NaN operand is NaN
    if not vals:
        return R(0)
    if p == "inf":
        m = abs_(vals[0])
        for v in vals[1:]:
            m = max_(m, abs_(v))
        return m
    if p == 1:
        s = R(0)
        for v in vals:
            s = s + abs_(v)
        return s
    s = R(0)
    for v in vals:
        s = s + abs_(v) ** p
    if p == 2:
        return sqrt(s)
    return s ** (Fraction(1) / p)


def _mul(g, c, ieee):
    if is_nan(g) or is_nan(c):
        return NAN
    if ieee and ((not g.finite and c.sign() == 0) or (not c.finite and g.sign() == 0)):
        return NAN                                   # IEEE 754 7.2: multiplication(0, inf) is invalid -> NaN
    return g * c


def evaluate(profile, inp, min_reading="minimum"):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    ieee = profile == "ieee_nonfinite"
    op = inp.get("op")
    if op not in OPS:
        raise Unsupported(f"operation {op!r}")
    grads = _grads(inp, ieee)
    vals = [v for g in grads if g is not None for v in g[0].reshape(-1)]
    if any(is_nan(v) for v in vals) and not ieee:
        raise Undefined("NaN gradient: outside the real profile")

    def emit(fn):
        return [[] if g is None else (nested(g[0], lambda v: plain(fn(v))) if g[1] else plain(fn(g[0].reshape(-1)[0])))
                for g in grads]

    if op == "clip_grad_value":
        c = R(rational(inp["clip_value"], "clip_value"))
        if c.sign() < 0:
            raise Unsupported("negative clip_value (empty range) is outside this contract")
        return {"total_norm": [], "grads": emit(lambda v: NAN if is_nan(v) else min_(max_(v, -c), c))}
    p = _norm_type(inp.get("norm_type", 2))
    N = total_norm(vals, p, ieee)
    if inp.get("error_if_nonfinite", False) is True and (is_nan(N) or not N.finite):
        raise Invalid("error_if_nonfinite=True and the total norm is not finite: documented error")
    if op == "get_total_norm":
        return {"total_norm": plain(N), "grads": emit(lambda v: v)}
    mx = R(rational(inp["max_norm"], "max_norm"))
    if is_nan(N):
        coef = R(1) if min_reading == "minimum_number" else NAN
    else:
        coef = min_(mx / (N + EPS), R(1))
    return {"total_norm": plain(N), "grads": emit(lambda v: _mul(v, coef, ieee))}
