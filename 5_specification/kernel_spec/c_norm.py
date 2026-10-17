"""Real-number references for the NORM and BN contracts (group C_act_norm_loss).

Tensors are nested lists of exact rationals on input. Statistics are exact rationals; 1/sqrt(var + eps) is exact when
var + eps is the square of a rational and a rigorous enclosure otherwise (kernel_spec.real.R). The VJPs are the
closed forms derived in the contracts (NM.VJP, BN.VJP); the adapters' mutants are an independent implementation.
Nothing here rounds.
"""
from __future__ import annotations

from fractions import Fraction

from .num import Invalid, Undefined, integer, rational
from .real import R
from . import real as rl


# ------------------------------------------------------------------ nested tensors
def shape_of(x, what):
    """Shape of a rectangular nested list (a scalar has shape ()); ragged input is invalid."""
    if not isinstance(x, (list, tuple)):
        return ()
    if len(x) == 0:
        return (0,)
    subs = [shape_of(e, what) for e in x]
    if any(s != subs[0] for s in subs):
        raise Invalid(f"{what}: ragged nested list")
    return (len(x),) + subs[0]


def flatten(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    return [v for e in x for v in flatten(e)]


def tensor(x, what, shape=None):
    """(flat list of Fractions, shape) of a nested list of exact rationals."""
    if not isinstance(x, (list, tuple)):
        raise Invalid(f"{what}: not a nested list")
    shp = shape_of(x, what)
    if shape is not None and tuple(shape) != shp:
        raise Invalid(f"{what}: shape {shp}, expected {tuple(shape)}")
    return [rational(v, f"{what} element") for v in flatten(x)], shp


def nest(flat, shape):
    """Nested lists of `shape` from a flat row-major list (a () shape gives the scalar)."""
    if len(shape) == 0:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    step = len(flat) // shape[0] if shape[0] else 0
    return [nest(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def prod(xs):
    p = 1
    for v in xs:
        p *= v
    return p


def plain(v):
    return v.v if isinstance(v, R) else v


# ------------------------------------------------------------------ one normalisation set
def rstd_of(second_moment, eps):
    """1 / sqrt(second_moment + eps); zero is undefined (NM.ZERO_VAR / BN.ZERO_VAR)."""
    d = second_moment + eps
    if d == 0:
        raise Undefined("variance + eps = 0: 1/sqrt(0) is undefined")
    return R(1) / rl.sqrt(R(d))


def moments(xs):
    n = len(xs)
    if n == 0:
        raise Undefined("mean of an empty normalisation set")
    mu = sum(xs, Fraction(0)) / n
    var = sum(((v - mu) ** 2 for v in xs), Fraction(0)) / n
    return mu, var


def centred_vjp(r, xhat, h):
    """d/dx of x_hat = (x - mean) * r against the upstream h on x_hat: r * (h - mean(h) - x_hat * mean(h x_hat))."""
    n = len(h)
    mh = sum((R(v) for v in h), R(0)) / n
    mhx = sum((R(a) * b for a, b in zip(h, xhat)), R(0)) / n
    return [r * (R(a) - mh - b * mhx) for a, b in zip(h, xhat)]


def rms_vjp(r, xhat, h):
    """d/dx of x_hat = x * r with r = (mean(x^2) + eps)^(-1/2): r * (h - x_hat * mean(h x_hat))."""
    n = len(h)
    mhx = sum((R(a) * b for a, b in zip(h, xhat)), R(0)) / n
    return [r * (R(a) - b * mhx) for a, b in zip(h, xhat)]


# ------------------------------------------------------------------ NORM: layer_norm, rms_norm, group_norm
def layer_norm(x, normalized_shape, weight, bias, eps, upstream):
    xs, shp = tensor(x, "x")
    ns = [integer(v, "normalized_shape", lo=0) for v in normalized_shape]
    D = len(ns)
    if D < 1 or len(shp) < D or tuple(shp[len(shp) - D:]) != tuple(ns):
        raise Invalid(f"normalized_shape {ns} is not the trailing shape of x {shp}")
    inner = prod(ns)
    outer_shape = shp[:len(shp) - D]
    M = prod(outer_shape)
    ws = None if weight is None else tensor(weight, "weight", ns)[0]
    bs = None if bias is None else tensor(bias, "bias", ns)[0]
    gs = tensor(upstream, "upstream", shp)[0]
    eps = _eps(eps)
    y, dx = [None] * len(xs), [None] * len(xs)
    dw = None if ws is None else [R(0)] * inner
    db = None if bs is None else [R(0)] * inner
    means, rstds = [], []
    for m in range(M):
        idx = range(m * inner, (m + 1) * inner)
        vals = [xs[i] for i in idx]
        mu, var = moments(vals)
        r = rstd_of(var, eps)
        xhat = [R(v - mu) * r for v in vals]
        for j, i in enumerate(idx):
            yi = xhat[j] * ws[j] if ws is not None else xhat[j]
            y[i] = yi + bs[j] if bs is not None else yi
        h = [gs[i] * (ws[j] if ws is not None else 1) for j, i in enumerate(idx)]
        for j, v in zip(idx, centred_vjp(r, xhat, h)):
            dx[j] = v
        for j, i in enumerate(idx):
            if dw is not None:
                dw[j] = dw[j] + xhat[j] * gs[i]
            if db is not None:
                db[j] = db[j] + gs[i]
        means.append(mu), rstds.append(r)
    grad = {"x": nest([plain(v) for v in dx], shp)}
    if dw is not None:
        grad["weight"] = nest([plain(v) for v in dw], tuple(ns))
    if db is not None:
        grad["bias"] = nest([plain(v) for v in db], tuple(ns))
    return {"y": nest([plain(v) for v in y], shp),
            "saved": {"mean": nest(means, outer_shape), "rstd": nest([plain(v) for v in rstds], outer_shape)},
            "grad": grad}


def rms_norm(x, normalized_shape, weight, eps, upstream, offset=None):
    """torch.nn.RMSNorm (offset None) or the offset variant y = x * rstd * (offset + w)."""
    xs, shp = tensor(x, "x")
    ns = [integer(v, "normalized_shape", lo=0) for v in normalized_shape]
    D = len(ns)
    if D < 1 or len(shp) < D or tuple(shp[len(shp) - D:]) != tuple(ns):
        raise Invalid(f"normalized_shape {ns} is not the trailing shape of x {shp}")
    inner = prod(ns)
    outer_shape = shp[:len(shp) - D]
    M = prod(outer_shape)
    ws = None if weight is None else tensor(weight, "weight", ns)[0]
    if offset is not None:
        if ws is None:
            raise Invalid("the offset variant needs a weight")
        weff = [offset + v for v in ws]
    else:
        weff = ws
    gs = tensor(upstream, "upstream", shp)[0]
    eps = _eps(eps)
    if inner == 0:
        raise Undefined("mean of an empty normalisation set")
    y, dx = [None] * len(xs), [None] * len(xs)
    dw = None if ws is None else [R(0)] * inner
    rstds = []
    for m in range(M):
        idx = range(m * inner, (m + 1) * inner)
        vals = [xs[i] for i in idx]
        ms = sum((v * v for v in vals), Fraction(0)) / inner
        r = rstd_of(ms, eps)
        xhat = [R(v) * r for v in vals]
        for j, i in enumerate(idx):
            y[i] = xhat[j] * weff[j] if weff is not None else xhat[j]
        h = [gs[i] * (weff[j] if weff is not None else 1) for j, i in enumerate(idx)]
        for j, v in zip(idx, rms_vjp(r, xhat, h)):
            dx[j] = v
        if dw is not None:
            for j, i in enumerate(idx):
                dw[j] = dw[j] + xhat[j] * gs[i]
        rstds.append(r)
    grad = {"x": nest([plain(v) for v in dx], shp)}
    if dw is not None:
        grad["weight"] = nest([plain(v) for v in dw], tuple(ns))
    return {"y": nest([plain(v) for v in y], shp), "saved": {"rstd": nest([plain(v) for v in rstds], outer_shape)},
            "grad": grad}


def group_norm(x, num_groups, weight, bias, eps, upstream):
    xs, shp = tensor(x, "x")
    if len(shp) < 2:
        raise Invalid("group_norm needs an input of shape (N, C, *)")
    N, C = shp[0], shp[1]
    G = integer(num_groups, "num_groups", lo=1)
    if C % G:
        raise Invalid(f"num_channels {C} is not divisible by num_groups {G}")
    HW = prod(shp[2:])
    cpg = C // G
    ws = None if weight is None else tensor(weight, "weight", (C,))[0]
    bs = None if bias is None else tensor(bias, "bias", (C,))[0]
    gs = tensor(upstream, "upstream", shp)[0]
    eps = _eps(eps)
    y, dx = [None] * len(xs), [None] * len(xs)
    dw = None if ws is None else [R(0)] * C
    db = None if bs is None else [R(0)] * C
    means, rstds = [], []
    for n in range(N):
        mrow, rrow = [], []
        for gi in range(G):
            chans = range(gi * cpg, (gi + 1) * cpg)                      # contiguous channel block (NM.GN_GROUPS)
            idx = [(n * C + c) * HW + s for c in chans for s in range(HW)]
            chan = [c for c in chans for _ in range(HW)]
            vals = [xs[i] for i in idx]
            mu, var = moments(vals)
            r = rstd_of(var, eps)
            xhat = [R(v - mu) * r for v in vals]
            for j, i in enumerate(idx):
                c = chan[j]
                yi = xhat[j] * ws[c] if ws is not None else xhat[j]
                y[i] = yi + bs[c] if bs is not None else yi
                if dw is not None:
                    dw[c] = dw[c] + xhat[j] * gs[i]
                if db is not None:
                    db[c] = db[c] + gs[i]
            h = [gs[i] * (ws[chan[j]] if ws is not None else 1) for j, i in enumerate(idx)]
            for i, v in zip(idx, centred_vjp(r, xhat, h)):
                dx[i] = v
            mrow.append(mu), rrow.append(plain(r))
        means.append(mrow), rstds.append(rrow)
    grad = {"x": nest([plain(v) for v in dx], shp)}
    if dw is not None:
        grad["weight"] = [plain(v) for v in dw]
    if db is not None:
        grad["bias"] = [plain(v) for v in db]
    return {"y": nest([plain(v) for v in y], shp), "saved": {"mean": means, "rstd": rstds}, "grad": grad}


def _eps(eps):
    e = rational(eps, "eps")
    if e < 0:
        raise Invalid("eps < 0")
    return e


# ------------------------------------------------------------------ BN
def batch_norm(x, weight, bias, training, momentum, eps, state, upstream):
    """One BatchNorm call (BN contract). state: None (track_running_stats=False) or a dict with running_mean,
    running_var and optionally num_batches_tracked."""
    xs, shp = tensor(x, "x")
    if len(shp) < 2:
        raise Invalid("batch_norm needs an input of shape (N, C, *)")
    N, C = shp[0], shp[1]
    HW = prod(shp[2:])
    n = N * HW
    if not isinstance(training, bool):
        raise Invalid("training must be a boolean")
    ws = None if weight is None else tensor(weight, "weight", (C,))[0]
    bs = None if bias is None else tensor(bias, "bias", (C,))[0]
    gs = tensor(upstream, "upstream", shp)[0]
    eps = _eps(eps)
    if momentum is not None:
        momentum = rational(momentum, "momentum")
        if not 0 <= momentum <= 1:
            raise Invalid("momentum outside [0, 1]")
    rm = rv = nbt = None
    if state is not None:
        if not isinstance(state, dict) or not {"running_mean", "running_var"} <= set(state) or \
                not set(state) <= {"running_mean", "running_var", "num_batches_tracked"}:
            raise Invalid("state must have running_mean, running_var and optionally num_batches_tracked")
        rm = [rational(v, "running_mean") for v in _vec(state["running_mean"], C, "running_mean")]
        rv = [rational(v, "running_var") for v in _vec(state["running_var"], C, "running_var")]
        if any(v < 0 for v in rv):
            raise Invalid("negative running_var")
        if "num_batches_tracked" in state:
            nbt = integer(state["num_batches_tracked"], "num_batches_tracked", lo=0)
    if momentum is None and training and state is not None and nbt is None:
        raise Invalid("momentum=None (cumulative average) needs num_batches_tracked")
    if eps <= 0:
        raise Invalid("batch_norm eps must be positive (torch.nn.functional.batch_norm raises; BN.N1)")
    if training and n == 1:
        raise Invalid("expected more than 1 value per channel when training (torch.nn.functional.batch_norm raises; "
                      "BN.N1)")
    batch_stats = training or state is None
    idx_of = [[(b * C + c) * HW + s for b in range(N) for s in range(HW)] for c in range(C)]
    if batch_stats and n == 0:
        raise Undefined("batch statistics of an empty channel set")
    y, dx = [None] * len(xs), [None] * len(xs)
    dw = None if ws is None else [R(0)] * C
    db = None if bs is None else [R(0)] * C
    smean, srstd = [], []
    new_rm, new_rv = (list(rm), list(rv)) if state is not None else (None, None)
    new_nbt = nbt
    if training and state is not None:
        if nbt is not None:
            new_nbt = nbt + 1
        f = momentum if momentum is not None else Fraction(1, new_nbt)
    for c in range(C):
        idx = idx_of[c]
        vals = [xs[i] for i in idx]
        if batch_stats:
            mu, var = moments(vals)
        else:
            mu, var = rm[c], rv[c]
        r = rstd_of(var, eps)
        if training and state is not None:
            new_rm[c] = (1 - f) * rm[c] + f * mu
            new_rv[c] = (1 - f) * rv[c] + f * var * Fraction(n, n - 1)
        xhat = [R(v - mu) * r for v in vals]
        wc = ws[c] if ws is not None else 1
        for j, i in enumerate(idx):
            yi = xhat[j] * ws[c] if ws is not None else xhat[j]
            y[i] = yi + bs[c] if bs is not None else yi
            if dw is not None:
                dw[c] = dw[c] + xhat[j] * gs[i]
            if db is not None:
                db[c] = db[c] + gs[i]
        h = [gs[i] * wc for i in idx]
        dxs = centred_vjp(r, xhat, h) if batch_stats else [r * a for a in h]
        for i, v in zip(idx, dxs):
            dx[i] = v
        smean.append(mu), srstd.append(plain(r))
    out_state = {}
    if state is not None:
        out_state = {"running_mean": new_rm, "running_var": new_rv}
        if nbt is not None:
            out_state["num_batches_tracked"] = new_nbt
    grad = {"x": nest([plain(v) for v in dx], shp)}
    if dw is not None:
        grad["weight"] = [plain(v) for v in dw]
    if db is not None:
        grad["bias"] = [plain(v) for v in db]
    return {"y": nest([plain(v) for v in y], shp), "state": out_state, "saved": {"mean": smean, "rstd": srstd},
            "grad": grad}


def _vec(v, C, what):
    if not isinstance(v, (list, tuple)) or len(v) != C:
        raise Invalid(f"{what}: expected a list of length {C}")
    return v
