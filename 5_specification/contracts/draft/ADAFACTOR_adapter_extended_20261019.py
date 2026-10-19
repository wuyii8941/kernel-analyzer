"""Adapter for contracts/ADAFACTOR.yaml: one Adafactor step from an explicit pre-state to the full post-state, for
torch.optim.Adafactor (PyTorch 2.10 box, profile torch210) and transformers.optimization.Adafactor 4.57.3 with a manual
learning rate (profile hf_manual_lr).

Tensors are nested lists of any rank >= 1. Rank >= 2 is factored over the last two dimensions (leading dimensions
batched): row has the shape of the parameter without its last dimension, col the shape without its penultimate
dimension, var = []. Rank 1: row = col = [], var has the parameter's shape. Mutants are separate code."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec.e1_optim import flag, hyper, state_dict
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, rational
from kernel_spec.real import R, max_, min_, sqrt


# ---------------------------------------------------------------- nested tensors
def shape_of(x, what):
    if not isinstance(x, list):
        return ()
    if not x:
        raise Invalid(f"{what}: empty dimension")
    subs = [shape_of(e, what) for e in x]
    if any(s != subs[0] for s in subs):
        raise Invalid(f"{what}: ragged")
    return (len(x),) + subs[0]


def read(x, shape, what):
    if shape_of(x, what) != tuple(shape):
        raise Invalid(f"{what}: shape {shape_of(x, what)}, expected {tuple(shape)}")
    return _map(lambda e: rational(e, what), x)


def _map(f, x):
    return [_map(f, e) for e in x] if isinstance(x, list) else f(x)


def flat(x):
    return [z for e in x for z in flat(e)] if isinstance(x, list) else [x]


def mats(x):
    """The list of trailing 2-D matrices of a rank >= 2 nested list (row-major over the leading dimensions)."""
    return [x] if not isinstance(x[0][0], list) else [m for e in x for m in mats(e)]


def vecs(x):
    """The list of trailing 1-D vectors of a rank >= 1 nested list."""
    return [x] if not isinstance(x[0], list) else [v for e in x for v in vecs(e)]


def rebuild(values, shape):
    it = iter(values)

    def go(s):
        return [go(s[1:]) for _ in range(s[0])] if len(s) > 1 else [next(it) for _ in range(s[0])]
    return go(tuple(shape))


def mean(xs):
    return sum(xs, Fraction(0)) / len(xs) if all(isinstance(x, Fraction) for x in xs) else \
        sum((R(x) for x in xs), R(0)) / R(len(xs))


def rms(xs):
    return sqrt(R(mean([x * x for x in xs]))) if all(isinstance(x, Fraction) for x in xs) else \
        sqrt(sum((R(x) * R(x) for x in xs), R(0)) / R(len(xs)))


def one_minus_beta(t1, tau):
    """t^tau (torch: one_minus_beta2_t; transformers: 1 - beta2t)."""
    if t1 == 1 or tau == 0:
        return Fraction(1)
    if tau.denominator == 1:
        return Fraction(t1) ** int(tau)
    return R(t1) ** R(tau)


def lerp(old, new, c):
    if isinstance(c, Fraction):
        return old + c * (new - old)
    return R(old) + R(c) * (R(new) - R(old))


# ---------------------------------------------------------------- inputs
def _inputs(profile, inp):
    if profile not in ("torch210", "hf_manual_lr"):
        raise Unsupported(f"profile {profile!r}")
    maximize = flag(inp, "maximize", default=False)
    if profile == "hf_manual_lr" and maximize:
        raise Invalid("transformers Adafactor has no maximize option")
    st = state_dict(inp, {"theta", "row", "col", "var", "t"})
    shape = shape_of(st["theta"], "theta")
    if len(shape) < 1:
        raise Invalid("a scalar parameter is outside this contract's shapes (rank >= 1)")
    theta = read(st["theta"], shape, "theta")
    if len(shape) >= 2:
        row = read(st["row"], shape[:-1], "row")
        col = read(st["col"], shape[:-2] + shape[-1:], "col")
        if st["var"] != []:
            raise Invalid("var must be [] for a factored parameter")
        var = []
        if any(z < 0 for z in flat(row) + flat(col)):
            raise Invalid("negative factor")
    else:
        if st["row"] != [] or st["col"] != []:
            raise Invalid("row and col must be [] for a vector parameter")
        row, col = [], []
        var = read(st["var"], shape, "var")
        if any(z < 0 for z in var):
            raise Invalid("negative variance")
    t = integer(st["t"], "t", lo=0)
    g = inp.get("gradient", "missing")
    if g == "missing":
        raise Invalid("missing gradient")
    g = None if g is None else read(g, shape, "gradient")
    if profile == "torch210":
        hp = hyper(inp, ("lr", "beta2_decay", "eps1", "eps2", "d", "weight_decay"),
                   nonneg=("lr", "eps1", "eps2", "weight_decay"))
        if hp["beta2_decay"] > 0 or hp["d"] < 1:
            raise Invalid("beta2_decay must be <= 0 and d >= 1")
    else:
        hp = hyper(inp, ("lr", "decay_rate", "eps1", "eps2", "clip_threshold", "weight_decay"),
                   nonneg=("lr", "eps1", "eps2", "weight_decay"))
        if hp["clip_threshold"] <= 0:
            raise Invalid("clip_threshold must be > 0")
        if hp["decay_rate"] > 0:
            raise Invalid("decay_rate must be <= 0 (beta2t in [0, 1))")
    return shape, theta, row, col, var, t, g, hp, maximize


def _out(shape, theta, row, col, var, t):
    return Result("value", {"theta": theta, "row": row, "col": col, "var": var, "t": t})


# ---------------------------------------------------------------- reference
def reference(profile, inp):
    shape, theta, row, col, var, t, g, hp, maximize = _inputs(profile, inp)
    if g is None:
        return _out(shape, theta, row, col, var, t)
    if maximize:
        g = _map(lambda z: -z, g)
    return (_torch if profile == "torch210" else _hf)(shape, theta, row, col, var, t, g, hp)


def _torch(shape, theta, row, col, var, t, g, hp):
    t1 = t + 1
    c = one_minus_beta(t1, hp["beta2_decay"])
    rho = min_(R(hp["lr"]), R(1) / sqrt(R(t1)))
    alpha = max_(R(hp["eps2"]), rms(flat(theta))) * rho
    eps1 = hp["eps1"]
    if len(shape) >= 2:
        rows_new, cols_new, U = [], [], []
        for G, Rv, Cv in zip(mats(g), vecs(row) if len(shape) > 2 else [row], _cols(col, shape)):
            n, m = len(G), len(G[0])
            Rn = [lerp(Rv[i], mean([G[i][j] ** 2 for j in range(m)]), c) for i in range(n)]
            Cn = [lerp(Cv[j], mean([G[i][j] ** 2 for i in range(n)]), c) for j in range(m)]
            mR = mean(Rn)
            mR_r = R(mR)
            if mR_r < R(eps1) and any((R(Rn[i]) * R(Cn[j])).sign() != 0 for i in range(n) for j in range(m)):
                raise Undefined("eps1 clamp of the row-factor mean is active (decision AF.CLAMP)")
            den = max_(mR_r, R(eps1))
            if den.sign() == 0:
                raise Undefined("zero row factors with eps1 = 0: V = 0/0")
            for i in range(n):
                for j in range(m):
                    V = R(Rn[i]) * R(Cn[j]) / den
                    U.append(R(G[i][j]) / _denom(V, eps1, G[i][j]))
            rows_new.append(Rn), cols_new.append(Cn)
        row_out = rebuild([z for r in rows_new for z in r], shape[:-1])
        col_out = rebuild([z for cc in cols_new for z in cc], shape[:-2] + shape[-1:])
        var_out = []
    else:
        var_out = [lerp(v, gi * gi, c) for v, gi in zip(var, g)]
        U = [R(gi) / _denom(R(v), eps1, gi) for v, gi in zip(var_out, g)]
        row_out, col_out = [], []
    scale = max_(R(1), rms(U) / R(hp["d"]))
    th = [R(x * (1 - hp["lr"] * hp["weight_decay"])) - alpha * u / scale for x, u in zip(flat(theta), U)]
    return _out(shape, rebuild(th, shape), row_out, col_out, var_out, t1)


def _cols(col, shape):
    return vecs(col) if len(shape) > 2 else [col]


def _denom(V, eps1, gij):
    if V.sign() == 0 and eps1 == 0:
        raise Undefined("max(sqrt(V), eps1) = 0")
    return max_(sqrt(V), R(eps1))


def _hf(shape, theta, row, col, var, t, g, hp):
    t1 = t + 1
    c = one_minus_beta(t1, hp["decay_rate"])        # 1 - beta2t
    e1 = hp["eps1"]
    if len(shape) >= 2:
        rows_new, cols_new, U = [], [], []
        for G, Rv, Cv in zip(mats(g), vecs(row) if len(shape) > 2 else [row], _cols(col, shape)):
            n, m = len(G), len(G[0])
            Rn = [lerp(Rv[i], mean([G[i][j] ** 2 + e1 for j in range(m)]), c) for i in range(n)]
            Cn = [lerp(Cv[j], mean([G[i][j] ** 2 + e1 for i in range(n)]), c) for j in range(m)]
            mR = R(mean(Rn))
            for i in range(n):
                for j in range(m):
                    if R(Rn[i]).sign() == 0 or R(Cn[j]).sign() == 0:
                        raise Undefined("zero factor (rsqrt of 0)")
                    U.append(R(G[i][j]) / sqrt(R(Rn[i]) / mR * R(Cn[j])))
            rows_new.append(Rn), cols_new.append(Cn)
        row_out = rebuild([z for r in rows_new for z in r], shape[:-1])
        col_out = rebuild([z for cc in cols_new for z in cc], shape[:-2] + shape[-1:])
        var_out = []
    else:
        var_out = [lerp(v, gi * gi + e1, c) for v, gi in zip(var, g)]
        if any(R(v).sign() == 0 for v in var_out):
            raise Undefined("zero second moment (rsqrt of 0)")
        U = [R(gi) / sqrt(R(v)) for v, gi in zip(var_out, g)]
        row_out, col_out = [], []
    scale = max_(rms(U) / R(hp["clip_threshold"]), R(1))
    lr = hp["lr"]
    th = [R(x - hp["weight_decay"] * lr * x) - R(lr) * u / scale for x, u in zip(flat(theta), U)]
    return _out(shape, rebuild(th, shape), row_out, col_out, var_out, t1)


# ---------------------------------------------------------------- independent mutants (torch210 unless stated)
def _mut(profile, inp, mis):
    shape, theta, row, col, var, t, g, hp, maximize = _inputs(profile, inp)
    if g is None:
        if mis != "none_as_zero":
            return _out(shape, theta, row, col, var, t)
        g = _map(lambda z: Fraction(0), theta)
    if maximize and mis != "maximize_ignored":
        g = _map(lambda z: -z, g)
    if profile == "hf_manual_lr":
        return _mut_hf(shape, theta, row, col, var, t, g, hp, mis)
    t1 = t + 1
    tb = t if mis == "old_step_in_beta" else t1
    tau = hp["beta2_decay"]
    c = R(1) if tb == 1 else (R(tb) ** R(tau) if tb > 0 else R(1))
    cap = Fraction(1, 100) if mis == "paper_rho_cap" else hp["lr"]
    rho = min_(R(cap), R(1) / sqrt(R(t1)))
    th_flat = flat(theta)
    dec = [x * (1 - hp["lr"] * hp["weight_decay"]) for x in th_flat]
    alpha = max_(R(hp["eps2"]), rms(dec if mis == "alpha_after_decay" else th_flat)) * rho
    e1 = R(hp["eps1"])
    if len(shape) >= 2:
        Gs = mats(g)
        if mis == "flatten_leading":                  # factor a (prod(leading)*n, m) matrix instead of batching
            Gs = [[r for M in Gs for r in M]]
            Rs, Cs = [flat(row)], [[Fraction(0)] * shape[-1]]
        else:
            Rs = vecs(row) if len(shape) > 2 else [row]
            Cs = _cols(col, shape)
        U, rows_new, cols_new = [], [], []
        for G, Rv, Cv in zip(Gs, Rs, Cs):
            n, m = len(G), len(G[0])
            red = (lambda xs: sum(xs, Fraction(0))) if mis == "sums_in_state" else (lambda xs: mean(xs))
            Rn = [R(Rv[i]) * (R(1) - c) + c * R(red([G[i][j] ** 2 for j in range(m)])) for i in range(n)]
            Cn = [R(Cv[j]) * (R(1) - c) + c * R(red([G[i][j] ** 2 for i in range(n)])) for j in range(m)]
            if mis == "sums_in_state":
                den = max_(sum(Rn, R(0)), e1)
            else:
                den = max_(sum(Rn, R(0)) / R(n), e1)
            for i in range(n):
                for j in range(m):
                    V = Rn[i] * Cn[j] / den
                    U.append(R(G[i][j]) / max_(sqrt(V), e1))
            rows_new.append(Rn), cols_new.append(Cn)
        if mis == "flatten_leading":
            row_out, col_out = rebuild(rows_new[0], shape[:-1]), rebuild(cols_new[0] * (len(flat(col)) // shape[-1]),
                                                                        shape[:-2] + shape[-1:])
        else:
            row_out = rebuild([z for r in rows_new for z in r], shape[:-1])
            col_out = rebuild([z for cc in cols_new for z in cc], shape[:-2] + shape[-1:])
        var_out = []
    else:
        if mis == "eps1_added":
            var_out = [R(v) * (R(1) - c) + c * R(gi * gi + hp["eps1"]) for v, gi in zip(var, g)]
            U = [R(gi) / sqrt(v) for v, gi in zip(var_out, g)]
        else:
            var_out = [R(v) * (R(1) - c) + c * R(gi * gi) for v, gi in zip(var, g)]
            U = [R(gi) / max_(sqrt(v), e1) for v, gi in zip(var_out, g)]
        row_out, col_out = [], []
    scale = R(1) if mis == "no_clipping" else max_(R(1), rms(U) / R(hp["d"]))
    base = th_flat if mis == "decay_lost" else dec
    th = [R(x) - alpha * u / scale for x, u in zip(base, U)]
    return _out(shape, rebuild(th, shape), row_out, col_out, var_out, t1)


def _mut_hf(shape, theta, row, col, var, t, g, hp, mis):
    t1 = t + 1
    tau = hp["decay_rate"]
    c = R(1) if t1 == 1 else R(t1) ** R(tau)
    e1 = hp["eps1"]
    lr = R(hp["lr"])
    if len(shape) >= 2:
        U, rows_new, cols_new = [], [], []
        for G, Rv, Cv in zip(mats(g), vecs(row) if len(shape) > 2 else [row], _cols(col, shape)):
            n, m = len(G), len(G[0])
            Rn = [R(Rv[i]) * (R(1) - c) + c * R(mean([G[i][j] ** 2 + e1 for j in range(m)])) for i in range(n)]
            Cn = [R(Cv[j]) * (R(1) - c) + c * R(mean([G[i][j] ** 2 + e1 for i in range(n)])) for j in range(m)]
            mR = sum(Rn, R(0)) / R(n)
            for i in range(n):
                for j in range(m):
                    V = Rn[i] * Cn[j] if mis == "hf_row_not_normalized" else Rn[i] / mR * Cn[j]
                    U.append(R(G[i][j]) / sqrt(V))
            rows_new.append(Rn), cols_new.append(Cn)
        row_out = rebuild([z for r in rows_new for z in r], shape[:-1])
        col_out = rebuild([z for cc in cols_new for z in cc], shape[:-2] + shape[-1:])
        var_out = []
    else:
        var_out = [R(v) * (R(1) - c) + c * R(gi * gi + e1) for v, gi in zip(var, g)]
        U = [R(gi) / sqrt(v) for v, gi in zip(var_out, g)]
        row_out, col_out = [], []
    scale = max_(rms(U) / R(hp["clip_threshold"]), R(1))
    wd = R(hp["weight_decay"])
    if mis == "hf_decay_after_update":
        th = [(R(x) - lr * u / scale) * (R(1) - wd * lr) for x, u in zip(flat(theta), U)]
    else:
        th = [R(x) - wd * lr * R(x) - lr * u / scale for x, u in zip(flat(theta), U)]
    return _out(shape, rebuild(th, shape), row_out, col_out, var_out, t1)


def _m(kind):
    return lambda profile, inp: _mut(profile, inp, kind)


MUTANTS = {
    "AF.M1": _m("sums_in_state"),
    "AF.M2": _m("alpha_after_decay"),
    "AF.M3": _m("decay_lost"),
    "AF.M4": _m("eps1_added"),
    "AF.M5": _m("no_clipping"),
    "AF.M6": _m("paper_rho_cap"),
    "AF.M7": _m("old_step_in_beta"),
    "AF.M8": _m("maximize_ignored"),
    "AF.M9": _m("none_as_zero"),
    "AF.M10": _m("flatten_leading"),
    "AF.M11": _m("hf_row_not_normalized"),
    "AF.M12": _m("hf_decay_after_update"),
}
READINGS = {}
