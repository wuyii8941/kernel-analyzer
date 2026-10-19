"""Adapter for contracts/OPT4BIT.yaml: torchao 0.16 AdamW4bit, one step from an explicit pre-state whose moments are
given as their dequantised values, to the full post-state (theta, stored dequantised m and v, t).

The quantisation maps are the binary32 constants of the torchao source (create_dynamic_map(True, 3, 4) and
linspace(0, 1, 17)[1:]); block-wise absmax scaling and nearest-code rounding with ties up are the source's rules,
evaluated in exact arithmetic. Mutants are separate code."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import v24
from kernel_spec.e1_optim import flag, grad_or_none, hyper, state_dict, vec
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer
from kernel_spec.real import R, sqrt

opt = v24.module("spec_optimizers")

QMAP_SIGNED = [Fraction(float.fromhex(h)) for h in (
    "-0x1.c666660000000p-1", "-0x1.5333340000000p-1", "-0x1.c000000000000p-2", "-0x1.b333320000000p-3",
    "-0x1.3d70a40000000p-4", "-0x1.0a3d720000000p-5", "-0x1.6872b00000000p-8", "0x0.0p+0", "0x1.6872b00000000p-8",
    "0x1.0a3d720000000p-5", "0x1.3d70a40000000p-4", "0x1.b333320000000p-3", "0x1.c000000000000p-2",
    "0x1.5333340000000p-1", "0x1.c666660000000p-1", "0x1.0000000000000p+0")]
QMAP_UNSIGNED = [Fraction(k, 16) for k in range(1, 17)]
SCALE_MIN = Fraction(2305843, 2305843009213693952)   # binary32 nearest to 1e-12 (clip of a float32 tensor)


def _inputs(profile, inp):
    if profile != "torchao016":
        raise Unsupported(f"profile {profile!r}")
    if flag(inp, "amsgrad", default=False):
        raise Unsupported("amsgrad=True was not tested")
    st = state_dict(inp, {"theta", "m", "v", "t"})
    theta = vec(st["theta"], "theta")
    d = len(theta)
    m, v = vec(st["m"], "m", d), vec(st["v"], "v", d)
    if any(x < 0 for x in v):
        raise Invalid("negative second moment")
    t = integer(st["t"], "t", lo=0)
    g = grad_or_none(inp, d)
    bs = integer(inp.get("block_size", 128), "block_size", lo=1)
    nmin = integer(inp.get("min_quant_numel", 4096), "min_quant_numel", lo=1)
    hp = hyper(inp, ("lr", "beta1", "beta2", "eps", "weight_decay"), nonneg=("lr", "eps", "weight_decay"))
    if not (0 <= hp["beta1"] < 1 and 0 <= hp["beta2"] < 1):
        raise Invalid("betas outside [0, 1)")
    return theta, m, v, t, g, bs, nmin, hp


def quantize(xs, block, qmap):
    """Stored (dequantised) values: per block scale = max(absmax, SCALE_MIN), code = largest index with
    qmap[code] <= x/scale (0 if none), moved up iff the residual reaches half the gap to the next code."""
    out = []
    for k in range(0, len(xs), block):
        blk = xs[k:k + block]
        scale = max(max(abs(x) for x in blk), SCALE_MIN)
        for x in blk:
            y = x / scale
            c = max([i for i, q in enumerate(qmap) if q <= y], default=0)
            up = min(c + 1, len(qmap) - 1)
            if y - qmap[c] >= (qmap[up] - qmap[c]) / 2:
                c = up
            out.append(qmap[c] * scale)
    return out


def reference(profile, inp):
    theta, m, v, t, g, bs, nmin, hp = _inputs(profile, inp)
    if g is None:
        return Result("value", {"theta": theta, "m": m, "v": v, "t": t})
    b2, eps = hp["beta2"], hp["eps"]
    if eps == 0 and any(b2 * vi + (1 - b2) * gi * gi == 0 for vi, gi in zip(v, g)):
        raise Undefined("sqrt(v_hat) + eps = 0")
    state = opt.AdamState(len(theta))
    state.m, state.v, state.t = list(m), list(v), t
    out = opt.adam_step(theta, g, state, hp["lr"], hp["beta1"], b2, eps, hp["weight_decay"], amsgrad=False,
                        maximize=False, decoupled=True)
    m1, v1 = list(state.m), list(state.v)
    if len(theta) >= nmin and len(theta) % bs == 0:  # quantised states (torchao _new_buffer gate)
        m1, v1 = quantize(m1, bs, QMAP_SIGNED), quantize(v1, bs, QMAP_UNSIGNED)
    return Result("value", {"theta": v24.tree(out), "m": m1, "v": v1, "t": state.t})


# ------------------------------------------------------------------ independent mutants
def _q(xs, block, qmap, mis):
    out = []
    blocks = [xs] if mis == "per_tensor_scale" else [xs[k:k + block] for k in range(0, len(xs), block)]
    for blk in blocks:
        s = max(max(abs(x) for x in blk), SCALE_MIN)
        for x in blk:
            y = x / s
            best = min(range(len(qmap)), key=lambda i: (abs(qmap[i] - y), -qmap[i]))   # nearest, ties up
            if mis == "round_down":
                best = max([i for i, q in enumerate(qmap) if q <= y], default=0)
            out.append(qmap[best] * s)
    return out


def _mut(profile, inp, mis):
    theta, m, v, t, g, bs, nmin, hp = _inputs(profile, inp)
    lr, b1, b2, eps, wd = (hp[k] for k in ("lr", "beta1", "beta2", "eps", "weight_decay"))
    if g is None:
        return Result("value", {"theta": theta, "m": m, "v": v, "t": t})
    t1 = t + 1
    mf = [b1 * a + (1 - b1) * x for a, x in zip(m, g)]
    vf = [b2 * a + (1 - b2) * x * x for a, x in zip(v, g)]
    quant = mis == "gate_ignored" or (len(theta) >= nmin and len(theta) % bs == 0)
    if mis == "no_quantization":
        quant = False
    signed = QMAP_SIGNED
    unsigned = [Fraction(k, 15) for k in range(16)] if mis == "unsigned_with_zero" else QMAP_UNSIGNED
    mq = _q(mf, bs, signed, mis) if quant else mf
    vq = _q(vf, bs, unsigned, mis) if quant else vf
    mu, vu = (mq, vq) if mis == "update_from_stored" else (mf, vf)
    th = []
    for x, a, b in zip(theta, mu, vu):
        if eps == 0 and b == 0:
            raise Undefined("zero denominator")
        th.append(R(x * (1 - lr * wd)) - R(lr * a / (1 - b1 ** t1)) / (sqrt(R(b / (1 - b2 ** t1))) + R(eps)))
    return Result("value", {"theta": th, "m": mq, "v": vq, "t": t1})


def _m(kind):
    return lambda profile, inp: _mut(profile, inp, kind)


MUTANTS = {
    "Q4.M1": _m("update_from_stored"),
    "Q4.M2": _m("per_tensor_scale"),
    "Q4.M3": _m("round_down"),
    "Q4.M4": _m("unsigned_with_zero"),
    "Q4.M5": _m("no_quantization"),
    "Q4.M6": _m("gate_ignored"),
}
READINGS = {}
