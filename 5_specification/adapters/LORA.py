"""Adapter for contracts/LORA.yaml: vLLM multi-LoRA shrink (_lora_shrink), exact rational semantics.

y[s][t][r] = scaling * sum_k A_s[m_t][r][k] x[t][k] for tokens with a LoRA (m_t >= 0), 0 for tokens without one, and the
buffer content before the call when no token has a LoRA (the entry point returns before zeroing). MUTANTS are
independent loops with one misreading each.
"""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import d2_common as U
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational

KEYS = {"x", "weights", "mapping", "scaling", "out_before"}


def _mat(x, rank, what):
    t, s = U.tensor(x, rank, what)
    return U.plain(t), s


def parse(profile, inp):
    if profile != "vllm_shrink":
        raise Unsupported(f"profile {profile!r}")
    U.check_keys(inp, KEYS)
    x, (T, K) = _mat(inp["x"], 2, "x")
    ws = inp["weights"]
    if not isinstance(ws, list) or not ws:
        raise Invalid("weights: a non-empty list of per-slice tensors [num_loras][rank][hidden]")
    W, shapes = [], []
    for s, w in enumerate(ws):
        a, sh = _mat(w, 3, f"weights[{s}]")
        W.append(a)
        shapes.append(sh)
    if len(set(shapes)) != 1:
        raise Invalid("all slice weights must have the same shape (same strides)")
    L, Rk, K2 = shapes[0]
    if K2 != K:
        raise Invalid(f"hidden size of the weights {K2} differs from the input's {K}")
    m = [integer(v, "mapping") for v in inp["mapping"]]
    if len(m) != T:
        raise Invalid("one mapping entry per token")
    if any(v < -1 or v >= L for v in m):
        raise Invalid("mapping entries must be -1 or a LoRA index below num_loras")
    sc = rational(inp["scaling"], "scaling")
    before = None
    if inp.get("out_before") is not None:
        before, bs = _mat(inp["out_before"], 3, "out_before")
        if bs != (len(W), T, Rk):
            raise Invalid("out_before must have shape [num_slices][num_tokens][rank]")
    return dict(x=x, W=W, m=m, sc=sc, T=T, K=K, L=L, R=Rk, S=len(W), before=before)


def reference(profile, inp):
    a = parse(profile, inp)
    if all(v == -1 for v in a["m"]):
        if a["before"] is None:
            raise Unsupported("no token has a LoRA: the call leaves the output buffer as it was; its prior content is needed")
        return Result("value", {"y": a["before"]})
    y = []
    for s in range(a["S"]):
        rows = []
        for t in range(a["T"]):
            if a["m"][t] < 0:
                rows.append([Fraction(0)] * a["R"])
                continue
            A = a["W"][s][a["m"][t]]
            rows.append([a["sc"] * sum((A[r][k] * a["x"][t][k] for k in range(a["K"])), Fraction(0))
                         for r in range(a["R"])])
        y.append(rows)
    return Result("value", {"y": y})


def _mut(kind):
    def run(profile, inp):
        a = parse(profile, inp)
        if all(v == -1 for v in a["m"]) and kind != "zero_always":
            if a["before"] is None:
                raise Unsupported("prior buffer needed")
            return Result("value", {"y": a["before"]})
        out = [[[Fraction(0)] * a["R"] for _ in range(a["T"])] for _ in range(a["S"])]
        for t, lid in enumerate(a["m"]):
            if lid == -1 and kind != "negative_index_wraps":
                continue
            for s in range(a["S"]):
                A = a["W"][0 if kind == "first_slice_only" else s][lid]
                for r in range(a["R"]):
                    acc = Fraction(0)
                    for k in range(a["K"]):
                        w = A[k][r] if kind == "transposed" else A[r][k]
                        acc += w * a["x"][t][k]
                    out[s][t][r] = acc if kind == "scaling_ignored" else acc * a["sc"]
        return Result("value", {"y": out})
    return run


MUTANTS = {
    "LR.M1": _mut("scaling_ignored"),
    "LR.M2": _mut("negative_index_wraps"),
    "LR.M3": _mut("first_slice_only"),
    "LR.M4": _mut("transposed"),
    "LR.M5": _mut("zero_always"),
}
READINGS = {}
