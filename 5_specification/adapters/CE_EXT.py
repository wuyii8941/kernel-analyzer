"""Adapter for contracts/CE_EXT.yaml: the cross-entropy variants outside the reviewed pilot (contracts/CE.yaml) that
were tested - C = 1, probability targets (with label smoothing, rows as received), K-dimensional inputs.

Input as the pilot: {logits, target, weights, eps, ignore_index, reduction, upstream}; logits (N, C, d_1..d_K) as a
nested list, target (N, d_1..d_K) class indices or (N, C, d_1..d_K) probabilities, upstream a scalar for sum / mean
and of the loss's shape for none. Outputs: loss and gradient (the VJP, shape of the logits).

The reference is the v2.4 phase1 cross entropy (the pilot's numerics) on the flattened positions (kernel_spec.c_ce).
The mutants are an independent mpmath implementation with one misreading switched on each.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import c_ce as K
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, rational, to_iv, vector

PROFILES = ("index_single_class", "index_kdim", "probability", "liger_inplace_index_unsmoothed")
KEYS = {"logits", "target", "weights", "eps", "ignore_index", "reduction", "upstream"}


def _inputs(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    need = KEYS - ({"ignore_index"} if profile == "probability" else set())
    if not need <= set(inp) <= KEYS:
        raise Invalid(f"inputs {sorted(inp)}; expected {sorted(need)}")
    shape, flat = K.tensor(inp["logits"], "logits")
    if len(shape) < 2:
        raise Unsupported("unbatched (C) input is outside these profiles")
    N, C = shape[0], shape[1]
    dims = shape[2:]
    D = K.prod(dims)
    red = K.check_reduction(inp["reduction"])
    eps = K.check_eps(inp["eps"])
    w = None if inp.get("weights") is None else vector(inp["weights"], "weights", C)
    if w is not None and any(v < 0 for v in w):
        raise Unsupported("negative class weights")
    prob = profile == "probability"
    ii = None
    if prob:
        tshape, tflat = K.tensor(inp["target"], "target")
        if tshape != shape:
            raise Invalid(f"probability target shape {tshape}, expected {shape}")
        if any(not 0 <= v <= 1 for v in tflat):
            raise Invalid("probability target entries must lie in [0, 1] (CEX.PROB_ROWS)")
        targets = K.rows_of(tflat, N, C, D)
    else:
        ii = integer(inp["ignore_index"], "ignore_index")
        if profile == "liger_inplace_index_unsmoothed":
            if len(shape) != 2 or C < 2:
                raise Unsupported("Liger in-place profile requires two-dimensional logits with at least two classes")
            if eps != 0:
                raise Unsupported("Liger in-place pilot covers unsmoothed class-index targets")
        if profile == "index_single_class" and C != 1:
            raise Unsupported("index_single_class covers C = 1 only (C >= 2: the CE pilot or index_kdim)")
        if profile == "index_kdim":
            if C < 2:
                raise Unsupported("index_kdim covers C >= 2")
            if not dims:
                raise Unsupported("2-D inputs are the CE pilot's profiles")
            if eps != 0 and w is not None:
                raise Unsupported("weighted label smoothing with K-dimensional input is not in this contract")
        targets = K.index_targets(inp["target"], (N,) + dims, C, ii)
    lshape = (N,) + dims
    u = inp["upstream"]
    if red == "none":
        us, uflat = K.tensor(u, "upstream")
        if us != lshape:
            raise Invalid(f"upstream shape {us}, expected the loss shape {lshape}")
    else:
        if isinstance(u, (list, tuple)):
            raise Invalid("upstream must be a scalar for sum / mean")
        uflat = None
        u = rational(u, "upstream")
    return dict(shape=shape, flat=flat, N=N, C=C, D=D, dims=dims, red=red, eps=eps, w=w, prob=prob, ii=ii,
                targets=targets, u=u, uflat=uflat, lshape=lshape)


def reference(profile, inp):
    a = _inputs(profile, inp)
    rows = K.rows_of(a["flat"], a["N"], a["C"], a["D"])
    loss, grad = K.ce_rows(rows, a["targets"], a["w"], a["eps"], a["ii"], a["red"], a["prob"])
    stored_grad = [list(row) for row in grad]
    if a["red"] == "none":
        grad = [[K.scale(g, a["uflat"][r]) for g in row] for r, row in enumerate(grad)]
        loss = K.nest(loss, a["lshape"])
    else:
        grad = [[K.scale(g, a["u"]) for g in row] for row in grad]
    gflat = K.unrows(grad, a["N"], a["C"], a["D"])
    fields = {"loss": loss, "gradient": K.nest(gflat, a["shape"])}
    if profile == "liger_inplace_index_unsmoothed":
        # Forward stores the derivative in the logits buffer. Scalar backward
        # scales that buffer in place; reduction='none' forms a new scaled
        # tensor and leaves the public logits buffer at the unscaled derivative.
        post = grad if a["red"] != "none" else stored_grad
        post_flat = K.unrows(post, a["N"], a["C"], a["D"])
        fields["post_logits"] = K.nest(post_flat, a["shape"])
    return Result("value", fields)


def check_output(profile, inp, observed):
    """Compare every result and post-state field selected by the profile."""
    from kernel_spec.compare import check_enclosed_output
    return check_enclosed_output(reference(profile, inp).fields, observed)


# ------------------------------------------------------------------ independent implementation for the mutants
def _independent(profile, inp, mis=None):
    a = _inputs(profile, inp)
    N, C, D = a["N"], a["C"], a["D"]
    shape = a["shape"]
    if mis == "class_axis_last" and a["dims"]:
        # misreading: (N, d.., C) layout - the last axis holds the classes
        C2 = shape[-1]
        rows = [a["flat"][i:i + C2] for i in range(0, len(a["flat"]), C2)]
        if a["prob"]:
            tg = [list(map(Fraction, K.tensor(inp["target"], "t")[1][i:i + C2])) for i in range(0, len(a["flat"]), C2)]
        else:
            raise Undefined("class_axis_last mutant needs probability targets")
        C = C2
    else:
        rows = K.rows_of(a["flat"], N, C, D)
        tg = a["targets"]
    w = a["w"] or [Fraction(1)] * C
    eps = a["eps"]
    if mis == "single_class_undefined" and C == 1:
        raise Undefined("log of a one-class softmax taken as undefined")
    losses, grads, mass = [], [], []
    for r, x in enumerate(rows):
        if not a["prob"] and tg[r] == a["ii"]:
            losses.append(to_iv(0)), grads.append([to_iv(0)] * C), mass.append(Fraction(0))
            continue
        e = [iv.exp(to_iv(v)) for v in x]
        s = sum(e[1:], e[0])
        p = [v / s for v in e]
        if a["prob"]:
            y = [Fraction(v) for v in tg[r]]
            if mis == "renormalise":
                t = sum(y)
                y = [v / t for v in y]
            if mis == "smooth_c_minus_1":
                q = [(1 - eps) * v + eps / (C - 1) for v in y]
            elif mis == "smooth_no_uniform":
                q = [(1 - eps) * v for v in y]
            else:
                q = [(1 - eps) * v + eps / C for v in y]
            coef = [w[c] * q[c] for c in range(C)]
            if mis == "uniform_unweighted":
                coef = [w[c] * (1 - eps) * y[c] + eps / C for c in range(C)]
            mass.append(sum(w[c] * y[c] for c in range(C)) if mis == "prob_mean_by_mass" else Fraction(1))
        else:
            coef = [w[c] * ((1 - eps) * (c == tg[r]) + eps / C) for c in range(C)]
            mass.append(w[tg[r]])
        A = sum(coef)
        losses.append(-sum((to_iv(coef[c]) * iv.log(p[c]) for c in range(C)), to_iv(0)))
        grads.append([p[c] * to_iv(A) - to_iv(coef[c]) for c in range(C)])
    red = a["red"]
    if red == "none":
        up = [Fraction(1)] * len(rows) if mis == "upstream_ignored" else a["uflat"]
        g = [[v * to_iv(up[r]) for v in row] for r, row in enumerate(grads)]
        lo = K.nest(losses, a["lshape"])
    else:
        tot = sum(losses[1:], losses[0])
        den = Fraction(1)
        if red == "mean":
            den = sum(mass)
            if mis == "kdim_mean_by_N":
                den = den / D
            if den == 0:
                raise Undefined("zero mass")
        lo = tot / to_iv(den)
        g = [[v * to_iv(a["u"]) / to_iv(den) for v in row] for row in grads]
    if mis == "class_axis_last" and a["dims"]:
        return Result("value", {"loss": lo, "gradient": K.nest([v for row in g for v in row], shape)})
    return Result("value", {"loss": lo, "gradient": K.nest(K.unrows(g, N, C, D), shape)})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {
    "CEX.M1": _m("smooth_c_minus_1"),
    "CEX.M2": _m("smooth_no_uniform"),
    "CEX.M3": _m("prob_mean_by_mass"),
    "CEX.M4": _m("kdim_mean_by_N"),
    "CEX.M5": _m("single_class_undefined"),
    "CEX.M6": _m("renormalise"),
    "CEX.M7": _m("uniform_unweighted"),
    "CEX.M8": _m("class_axis_last"),
    "CEX.M9": _m("upstream_ignored"),
}
READINGS = {}
