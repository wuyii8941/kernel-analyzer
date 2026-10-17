"""Adapter for contracts/DIVERG.yaml: KL divergence (torch.nn.KLDivLoss, and Liger's eps-clamped variant), Liger's
generalised Jensen-Shannon divergence and Liger's total variation distance, each with the VJP w.r.t. the first
argument against an explicit upstream.

Inputs: {op: kl_div, input, target, reduction, log_target, upstream[, eps (profile liger_kl)]};
{op: jsd, input (log Q), target (log P), beta, labels, ignore_index, upstream};
{op: tvd, input (p), target (q), reduction, labels, ignore_index, upstream}. Outputs: loss, grad (shape of input).

The reference uses kernel_spec.real (exact rationals, enclosures); the mutants are an independent mpmath-interval
implementation with one misreading switched on each.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import c_ce as T
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, rational, to_iv
from kernel_spec.real import R

OPS = {"torch_kl": "kl_div", "liger_kl": "kl_div", "liger_jsd": "jsd", "liger_tvd": "tvd", "tvd_min_norm": "tvd"}
KEYS = {"kl_div": {"op", "input", "target", "reduction", "log_target", "upstream"},
        "jsd": {"op", "input", "target", "beta", "labels", "ignore_index", "upstream"},
        "tvd": {"op", "input", "target", "reduction", "labels", "ignore_index", "upstream"}}
REDS = {"kl_div": ("none", "sum", "mean", "batchmean"), "tvd": ("none", "sum", "mean", "batchmean")}


def _v(x):
    return x.v if isinstance(x, R) else x


def _inputs(profile, inp):
    if profile not in OPS:
        raise Unsupported(f"profile {profile!r}")
    op = inp.get("op")
    if op != OPS[profile]:
        raise Unsupported(f"op {op!r} is not in profile {profile}") if op in KEYS else Invalid(f"unknown op {op!r}")
    want = KEYS[op] | ({"eps"} if profile == "liger_kl" else set())
    if set(inp) != want:
        raise Invalid(f"{op} in {profile} takes exactly {sorted(want)}, got {sorted(inp)}")
    shape, x = T.tensor(inp["input"], "input")
    tshape, t = T.tensor(inp["target"], "target")
    if tshape != shape or not shape:
        raise Invalid(f"input / target shapes {shape} / {tshape}")
    if profile != "torch_kl" and len(shape) != 2:
        raise Invalid("Liger's operations take (BT, V) inputs")
    a = dict(op=op, shape=shape, x=x, t=t)
    if op in REDS:
        if inp["reduction"] not in REDS[op]:
            raise Invalid(f"reduction {inp['reduction']!r}")
        a["red"] = inp["reduction"]
    if op == "kl_div":
        if not isinstance(inp["log_target"], bool):
            raise Invalid("log_target must be a boolean")
        a["log_target"] = inp["log_target"]
        if profile == "liger_kl":
            a["eps"] = rational(inp["eps"], "eps")
            if a["eps"] <= 0:
                raise Invalid("eps must be positive")
    if op == "jsd":
        a["beta"] = rational(inp["beta"], "beta")
        if not 0 <= a["beta"] <= 1:
            raise Invalid("beta outside [0, 1]")
    if op in ("jsd", "tvd"):
        BT = shape[0]
        ii = integer(inp["ignore_index"], "ignore_index")
        lab = inp["labels"]
        if lab is None:
            keep = [True] * BT
        else:
            if not isinstance(lab, list) or len(lab) != BT:
                raise Invalid("labels must be a list of length BT")
            keep = [integer(v, "label") != ii for v in lab]
        a["keep"] = keep
    u = inp["upstream"]
    if a.get("red") == "none":
        us, uf = T.tensor(u, "upstream")
        if us != shape:
            raise Invalid("upstream must have the input's shape for reduction none")
        a["u"] = uf
    else:
        if isinstance(u, (list, tuple)):
            raise Invalid("upstream must be a scalar")
        a["u"] = rational(u, "upstream")
    return a


def _reduce(a, pw, gr, divisor):
    """pw / gr: flat pointwise losses and d(pointwise)/d(input) (R values); divisor for sum-type reductions."""
    if a.get("red") == "none":
        return T.nest([_v(v) for v in pw], a["shape"]), T.nest([_v(g * u) for g, u in zip(gr, a["u"])], a["shape"])
    if divisor == 0:
        raise Undefined("the normalising count is zero")
    tot = R(0)
    for v in pw:
        tot = tot + v
    return _v(tot / divisor), T.nest([_v(g * a["u"] / divisor) for g in gr], a["shape"])


def reference(profile, inp):
    a = _inputs(profile, inp)
    x, t = [R(v) for v in a["x"]], [R(v) for v in a["t"]]
    if a["op"] == "kl_div":
        pw, gr = [], []
        for xv, tv in zip(x, t):
            if a["log_target"]:
                et = rl.exp(tv)
                pw.append(et * (tv - xv)), gr.append(-et)
            elif profile == "liger_kl":
                pw.append(tv * (rl.log(rl.max_(tv, R(a["eps"]))) - xv)), gr.append(-tv)
            else:
                s = tv.sign()
                if s < 0:
                    raise Undefined("log of a negative target (log_target=False)")
                pw.append(R(0) if s == 0 else tv * (rl.log(tv) - xv)), gr.append(-tv)
        n = len(x)
        div = {"sum": 1, "mean": n, "batchmean": a["shape"][0]}.get(a.get("red"), 1)
        loss, grad = _reduce(a, pw, gr, div)
        return Result("value", {"loss": loss, "grad": grad})
    BT, V = a["shape"]
    keep = a["keep"]
    nk = sum(keep)
    pw, gr = [], []
    for i, (xv, tv) in enumerate(zip(x, t)):
        if not keep[i // V]:
            pw.append(R(0)), gr.append(R(0))
            continue
        if a["op"] == "jsd":
            b = R(a["beta"])
            P, Q = rl.exp(tv), rl.exp(xv)
            if a["beta"] == 0:
                pw.append(P * (tv - xv)), gr.append(-P)
            elif a["beta"] == 1:
                pw.append(Q * (xv - tv)), gr.append(Q * (xv - tv) + Q)
            else:
                M = b * P + (1 - b) * Q
                lm = rl.log(M)
                pw.append(b * P * tv + (1 - b) * Q * xv - M * lm), gr.append((1 - b) * Q * (xv - lm))
        else:
            d = xv - tv
            s = d.sign()
            pw.append(rl.abs_(d) / 2)
            gr.append(R(Fraction(1, 2)) if s > 0 else (R(0) if s == 0 and profile == "tvd_min_norm" else R(Fraction(-1, 2))))
    if a["op"] == "jsd":
        div = nk
        a = dict(a, red="sum")
    else:
        div = {"sum": 1, "mean": nk * V, "batchmean": nk}.get(a["red"], 1)
    loss, grad = _reduce(a, pw, gr, div)
    return Result("value", {"loss": loss, "grad": grad})


# ------------------------------------------------------------------ independent implementation for the mutants
def _independent(profile, inp, mis=None):
    a = _inputs(profile, inp)
    shape = a["shape"]
    x, t = [to_iv(v) for v in a["x"]], [Fraction(v) for v in a["t"]]
    n = len(x)
    if a["op"] == "kl_div":
        pw, gr = [], []
        for xv, tv in zip(x, t):
            lt = a["log_target"] and mis != "log_target_ignored"
            if mis == "args_swapped":
                q = iv.exp(xv)
                pw.append(q * (xv - (to_iv(tv) if lt else iv.log(to_iv(tv))))), gr.append(q * (xv - (to_iv(tv) if lt else iv.log(to_iv(tv)))) + q)
                continue
            if lt:
                e = iv.exp(to_iv(tv))
                pw.append(e * (to_iv(tv) - xv)), gr.append(-e)
            else:
                if profile == "liger_kl" and mis != "liger_no_clamp":
                    m = tv if tv > a["eps"] else a["eps"]
                    pw.append(to_iv(tv) * (iv.log(to_iv(m)) - xv))
                else:
                    if tv < 0 or (tv == 0 and mis == "zero_target_undefined"):
                        raise Undefined("log of a non-positive target")
                    pw.append(to_iv(0) if tv == 0 else to_iv(tv) * (iv.log(to_iv(tv)) - xv))
                gr.append(to_iv(tv) if mis == "grad_sign" else -to_iv(tv))
        red = a["red"]
        div = {"sum": 1, "mean": n, "batchmean": n if mis == "batchmean_as_mean" else shape[0]}.get(red, 1)
    else:
        BT, V = shape
        keep = a["keep"]
        nk = sum(keep)
        pw, gr = [], []
        for i, (xv, tv) in enumerate(zip(x, t)):
            if not keep[i // V]:
                pw.append(to_iv(0)), gr.append(to_iv(0))
                continue
            if a["op"] == "jsd":
                b = a["beta"]
                P, Q = iv.exp(to_iv(tv)), iv.exp(xv)
                if b == 0 and mis != "beta0_general":
                    pw.append(P * (to_iv(tv) - xv)), gr.append(-P)
                elif b == 1 and mis != "beta0_general":
                    pw.append(Q * (xv - to_iv(tv))), gr.append(Q * (xv - to_iv(tv)) + Q)
                else:
                    if mis == "beta_swapped":          # beta weights Q, 1 - beta weights P
                        M = to_iv(1 - b) * P + to_iv(b) * Q
                        lm = iv.log(M)
                        pw.append(to_iv(1 - b) * P * to_iv(tv) + to_iv(b) * Q * xv - M * lm)
                        gr.append(to_iv(b) * Q * (xv - lm))
                        continue
                    M = to_iv(b) * P + to_iv(1 - b) * Q
                    lm = to_iv(b) * to_iv(tv) + to_iv(1 - b) * xv if mis == "geometric_mixture" else iv.log(M)
                    pw.append(to_iv(b) * P * to_iv(tv) + to_iv(1 - b) * Q * xv - M * lm)
                    gr.append(to_iv(1 - b) * Q * (xv - lm))
            else:
                d = a["x"][i] - tv
                half = Fraction(1) if mis == "tvd_no_half" else Fraction(1, 2)
                pw.append(to_iv(abs(d) * half))
                if d > 0:
                    gr.append(to_iv(half))
                elif d < 0:
                    gr.append(to_iv(-half))
                else:
                    tie = Fraction(0) if profile == "tvd_min_norm" else Fraction(-1, 2)
                    if mis == "tie_swapped":
                        tie = Fraction(-1, 2) if profile == "tvd_min_norm" else Fraction(0)
                    gr.append(to_iv(tie))
        if a["op"] == "jsd":
            red = "sum"
            div = BT if mis == "jsd_by_rows" else nk
        else:
            red = a["red"]
            div = {"sum": 1, "mean": nk * V, "batchmean": nk * V if mis == "tvd_batchmean_elements" else nk}.get(red, 1)
    if red == "none":
        return Result("value", {"loss": T.nest(pw, shape), "grad": T.nest([g * to_iv(u) for g, u in zip(gr, a["u"])],
                                                                           shape)})
    if div == 0:
        raise Undefined("zero count")
    tot = sum(pw[1:], pw[0])
    return Result("value", {"loss": tot / to_iv(div), "grad": T.nest([g * to_iv(a["u"]) / to_iv(div) for g in gr],
                                                                     shape)})


def _m(kind):
    return lambda profile, inp: _independent(profile, inp, kind)


MUTANTS = {
    "DV.M1": _m("batchmean_as_mean"),
    "DV.M2": _m("log_target_ignored"),
    "DV.M3": _m("args_swapped"),
    "DV.M4": _m("grad_sign"),
    "DV.M5": _m("beta_swapped"),
    "DV.M6": _m("geometric_mixture"),
    "DV.M7": _m("jsd_by_rows"),
    "DV.M8": _m("tvd_no_half"),
    "DV.M9": _m("tvd_batchmean_elements"),
    "DV.M10": _m("zero_target_undefined"),
    "DV.M11": _m("liger_no_clamp"),
    "DV.M12": _m("tie_swapped"),
    "DV.M13": _m("beta0_general"),
}
READINGS = {}
