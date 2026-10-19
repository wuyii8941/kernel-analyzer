"""Adapter for contracts/CE.yaml: complete inputs in, loss and the VJP against the explicit upstream out.

The reference is the v2.4 phase1 cross-entropy (Decimal intervals), called only after the contract's input checks;
its intervals are converted exactly. The mutants are an independent mpmath implementation of the misreadings listed
in the contract's mutation requirements.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import v24
from kernel_spec.num import Invalid, NotEstablished, Result, Undefined, Unsupported, integer, matrix, rational, \
    to_iv, vector

ce = v24.module("spec_cross_entropy")
PROFILES = ("index_unsmoothed", "index_smoothing_unweighted", "index_smoothing_R_A", "probability_unsmoothed")


def _rectangular_shape(value):
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    shapes = [_rectangular_shape(item) for item in value]
    if shapes[0] is None or any(shape != shapes[0] for shape in shapes):
        return None
    return (len(value),) + shapes[0]


def _inputs(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    required = {"logits", "target", "eps", "reduction", "upstream"}
    if profile != "probability_unsmoothed":
        required.add("ignore_index")
    if not isinstance(inp, dict) or not required.issubset(inp):
        raise Invalid("cross entropy requires all declared inputs for the selected profile")
    if set(inp) - required - {"weights", "ignore_index"}:
        raise Unsupported("cross entropy input contains options outside the selected profile")
    shape = _rectangular_shape(inp["logits"])
    if shape is not None and (len(shape) > 2 or len(shape) == 1 and shape[0] > 0):
        raise Unsupported("unbatched and higher-rank logits require a separate flattening binding")
    x = matrix(inp["logits"], "logits")
    N, C = len(x), len(x[0])
    if C == 0:
        raise Invalid("logits must contain at least one class")
    if C == 1:
        raise Unsupported("one class is outside this contract")
    red = inp["reduction"]
    if red not in ("sum", "mean", "none"):
        raise Invalid(f"reduction {red!r}")
    eps = rational(inp["eps"], "eps")
    if not 0 <= eps <= 1:
        raise Invalid("eps outside [0, 1]")
    w = None if inp.get("weights") is None else vector(inp["weights"], "weights", C)
    if w is not None and any(v < 0 for v in w):
        raise Unsupported("negative class weights")
    prob = profile == "probability_unsmoothed"
    if prob:
        y = matrix(inp["target"], "target", N, C)
        if eps != 0:
            raise Unsupported("probability-target smoothing")
        if any(v < 0 for r in y for v in r) or any(sum(r) != 1 for r in y):
            raise Invalid("probability rows must be nonnegative and sum to one")
        ii = None
    else:
        ii = integer(inp["ignore_index"], "ignore_index")
        y = vector(inp["target"], "target", N, item=integer)
        for t in y:
            if t != ii and not 0 <= t < C:
                raise Invalid(f"target {t} outside [0, {C}) and not the ignore_index")
        if profile == "index_unsmoothed" and eps != 0:
            raise Invalid("index_unsmoothed requires eps = 0")
        if profile == "index_smoothing_unweighted" and w is not None:
            raise Invalid("index_smoothing_unweighted takes no weights")
    u = inp["upstream"]
    if red == "none":
        u = vector(u, "upstream", N)
    else:
        if isinstance(u, (list, tuple)):
            raise Invalid("upstream must be a scalar for sum / mean")
        u = rational(u, "upstream")
    return x, y, w, eps, ii, red, u, prob


def _require_defined_mean(target, weights, ignore_index, reduction, probability):
    if reduction == "mean" and not probability:
        mass = sum((1 if weights is None else weights[t]) for t in target if t != ignore_index)
        if mass == 0:
            raise Undefined("mean normalization mass is zero")


def reference(profile, inp):
    x, y, w, eps, ii, red, u, prob = _inputs(profile, inp)
    _require_defined_mean(y, w, ii, red, prob)
    z = ce.cross_entropy(x, y, weights=w, ignore_index=-10 ** 9 if prob else ii, reduction=red, label_smoothing=eps,
                         reading="R_A", prob_target=prob)
    if not z.get("established", False):
        raise NotEstablished(z.get("reason", "v2.4 cross entropy did not establish an enclosure"))
    if z["nan"] or not z["grad_defined"]:
        raise Undefined("mean normalization mass is zero")
    grad = v24.tree(z["grad"])
    rows = u if red == "none" else [u] * len(x)
    grad = [[g * rows[n] if isinstance(g, Fraction) else g * to_iv(rows[n]) for g in row] for n, row in enumerate(grad)]
    return Result("value", {"loss": v24.tree(z["loss"]), "gradient": grad})


def check_output(profile, inp, observed):
    """Complete loss/VJP; interval containment alone never certifies equality."""
    from kernel_spec.compare import check_enclosed_output
    result = reference(profile, inp)
    return check_enclosed_output(result.fields, observed)


# ------------------------------------------------------------------ independent implementation for the mutants
def _independent(profile, inp, misreading=None):
    if misreading == "hard_coded_ignore":
        inp = dict(inp, ignore_index=-100)
    x, y, w, eps, ii, red, u, prob = _inputs(profile, inp)
    N, C = len(x), len(x[0])
    w = w or [Fraction(1)] * C
    rows, grads, mass = [], [], []
    for n in range(N):
        if not prob and y[n] == ii:
            rows.append(to_iv(0)), grads.append([to_iv(0)] * C), mass.append(Fraction(0))
            continue
        e = [iv.exp(to_iv(v)) for v in x[n]]
        s = sum(e[1:], e[0])
        p = [v / s for v in e]
        if prob:
            a = [w[c] * y[n][c] for c in range(C)]
        else:
            alt = C - 1 if misreading == "smoothing_c_minus_1" else C
            a = [w[c] * ((1 - eps) * (c == y[n]) + (eps / alt if (misreading != "smoothing_c_minus_1" or c != y[n])
                                                      else 0)) for c in range(C)]
        A = sum(a)
        rows.append(-sum((to_iv(a[c]) * iv.log(p[c]) for c in range(C)), to_iv(0)))
        grads.append([p[c] * to_iv(A) - to_iv(a[c]) for c in range(C)])
        mass.append(Fraction(1) if prob else w[y[n]])
    if misreading == "gradient_zero":
        grads = [[to_iv(0)] * C for _ in range(N)]
    grads = [[to_iv(g) for g in r] for r in grads]
    if red == "none":
        up = [Fraction(1)] * N if misreading == "upstream_ignored" else u
        return Result("value", {"loss": rows, "gradient": [[g * to_iv(up[n]) for g in r] for n, r in enumerate(grads)]})
    total = sum(rows[1:], rows[0])
    if red == "sum":
        return Result("value", {"loss": total, "gradient": [[g * to_iv(u) for g in r] for r in grads]})
    D = sum(mass)
    if misreading == "mean_by_N":
        D = Fraction(N)
    elif misreading == "mean_by_rows":
        D = Fraction(sum(1 for n in range(N) if prob or y[n] != ii))
    elif misreading == "ignored_in_denominator":
        D = D + sum(1 for n in range(N) if not prob and y[n] == ii)
    elif misreading == "prob_mean_by_target_weight":
        D = sum(w[c] * y[n][c] for n in range(N) for c in range(C))
    if D == 0:
        raise Undefined("zero mass")
    return Result("value", {"loss": total / to_iv(D), "gradient": [[g * to_iv(u) / to_iv(D) for g in r] for r in grads]})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {
    "CE.M1": _m("mean_by_N"),
    "CE.M2": _m("mean_by_rows"),
    "CE.M3": _m("smoothing_c_minus_1"),
    "CE.M4": _m("ignored_in_denominator"),
    "CE.M5": _m("gradient_zero"),
    "CE.M6": _m("hard_coded_ignore"),
    "CE.M7": _m("upstream_ignored"),
    "CE.M8": _m("prob_mean_by_target_weight"),
}


def reading_R_B(profile, inp):
    """Diagnostic reading R_B of weighted label smoothing (target-class weight on every class term); not a task
    definition, used only where a contract lists it as a reading."""
    x, y, w, eps, ii, red, u, prob = _inputs(profile, inp)
    _require_defined_mean(y, w, ii, red, prob)
    z = ce.cross_entropy(x, y, weights=w, ignore_index=ii, reduction=red, label_smoothing=eps, reading="R_B",
                         prob_target=prob)
    if not z.get("established", False):
        raise NotEstablished(z.get("reason", ""))
    if z["nan"] or not z["grad_defined"]:
        raise Undefined("mean normalization mass is zero")
    rows = u if red == "none" else [u] * len(x)
    grad = [[g * to_iv(rows[n]) for g in row] for n, row in enumerate(v24.tree(z["grad"]))]
    return Result("value", {"loss": v24.tree(z["loss"]), "gradient": grad})


READINGS = {"R_B": reading_R_B}
