"""References for the training-program contracts SCALER (GradScaler iteration), ACCUM (accumulation-window
normalisation) and LABELS (causal label shift and valid-label counting)."""
from __future__ import annotations

from fractions import Fraction

from .e2_common import NAN, flag, is_nan, nested, plain, tensor
from .num import Invalid, Undefined, Unsupported, integer, rational, vector
from .real import R


# ------------------------------------------------------------------ SCALER
def scaler_iteration(profile, inp):
    if profile != "real":
        raise Unsupported(f"profile {profile!r}")
    S = R(rational(inp.get("scale", 65536), "scale"))
    gf = R(rational(inp.get("growth_factor", 2), "growth_factor"))
    bf = R(rational(inp.get("backoff_factor", Fraction(1, 2)), "backoff_factor"))
    gi = integer(inp.get("growth_interval", 2000), "growth_interval", 1)
    tr = integer(inp.get("growth_tracker", 0), "growth_tracker", 0)
    enabled = flag(inp.get("enabled", True), "enabled")
    if tr >= gi:
        raise Invalid("growth_tracker must be below growth_interval (it is reset on reaching it)")
    if S.sign() == 0:
        raise Undefined("scale 0: unscaling divides by zero")
    grads = [tensor(g, f"grads[{i}]", allow_nan=True) for i, g in enumerate(inp["grads"])]
    if not enabled:
        return {"unscaled": [nested(g) for g in grads], "found_inf": False, "stepped": True, "scale": S.v,
                "growth_tracker": tr}
    found = any(is_nan(v) or not v.finite for g in grads for v in g.reshape(-1))
    unscaled = [nested(g, lambda v: plain(NAN if is_nan(v) else v / S)) for g in grads]
    if found:
        S2, tr2 = S * bf, 0
    elif tr + 1 == gi:
        S2, tr2 = S * gf, 0
    else:
        S2, tr2 = S, tr + 1
    return {"unscaled": unscaled, "found_inf": found, "stepped": not found, "scale": S2.v, "growth_tracker": tr2}


# ------------------------------------------------------------------ ACCUM
def _window(inp):
    mbs = inp["microbatches"]
    if not isinstance(mbs, list) or not mbs:
        raise Invalid("microbatches: a non-empty list of micro-batches")
    losses, grads = [], []
    dim = None
    for b, mb in enumerate(mbs):
        ls = vector(mb.get("losses", []), f"microbatches[{b}].losses")
        gs = mb.get("grads", [])
        if not isinstance(gs, list) or len(gs) != len(ls):
            raise Invalid(f"microbatch {b}: one gradient per valid token is required ({len(ls)} losses, "
                          f"{len(gs) if isinstance(gs, list) else 'no'} gradients)")
        for t, g in enumerate(gs):
            g = vector(g, f"microbatches[{b}].grads[{t}]")
            if dim is None:
                dim = len(g)
            elif len(g) != dim:
                raise Invalid("all per-token gradients must have the same length")
            grads.append(g)
        losses.extend(ls)
    return losses, grads, dim


def accum_window(profile, inp):
    if profile != "token_mean":
        raise Unsupported(f"profile {profile!r}")
    losses, grads, dim = _window(inp)
    n = len(losses)
    if n == 0:
        raise Undefined("no valid token in the whole accumulation window (ACC-A1)")
    loss = sum(losses, Fraction(0)) / n
    grad = [sum((g[k] for g in grads), Fraction(0)) / n for k in range(dim)]
    return {"loss": loss, "grad": grad}


# ------------------------------------------------------------------ LABELS
def shift_and_count(profile, inp):
    if profile != "causal_shift":
        raise Unsupported(f"profile {profile!r}")
    ii = integer(inp.get("ignore_index", -100), "ignore_index")
    rows = inp["labels"]
    if not isinstance(rows, list):
        raise Invalid("labels: list of sequences (one list of label ids per sequence of the window)")
    vocab = inp.get("vocab_size")
    vocab = None if vocab is None else integer(vocab, "vocab_size", 1)
    shifted = []
    for r, row in enumerate(rows):
        lab = vector(row, f"labels[{r}]", item=integer)
        for t in lab:
            if t != ii and (t < 0 or (vocab is not None and t >= vocab)):
                raise Invalid(f"label {t} is neither ignore_index nor a class index")
        shifted.append(lab[1:] + [ii] if lab else [])
    count = sum(1 for row in shifted for t in row if t != ii)
    return {"shifted": shifted, "count": count}
