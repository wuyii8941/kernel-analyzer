#!/usr/bin/env python3
"""Strict enclosure verification of K_R on the external corpus (next-phase plan, stage A item 4).

The two cross-checks already on record carry a declared slack (NumPy float64 declared semantics: 2^-40 max|.|;
FPCore with titanfp: 2^-200 |v|, and the FPCore exporter shares the TTIR parser).  This check has neither: the
declared real-number program of each kernel (written from the kernel source, the same programs as
``external_corpus_semantics.declared``) is evaluated in rigorous interval arithmetic (mpmath ``iv``, 160 bits,
outward rounding; tanh as 1 - 2/(exp(2t) + 1) in interval operations), and every element must satisfy
    T_lo >= K_R_lo  and  T_hi <= K_R_hi
with no slack.  An element whose rigorous enclosure T is not inside K_R is a violation if T and K_R are disjoint,
and "undecided" if they overlap (T wider than K_R would be a weakness of this check, not evidence about K_R).

Elements: the 28 representative conditions x seeds 0-1 of the FPCore cross-check (``results/fpcore/fragments.jsonl``
holds K_R and the element offsets).

    python scripts/strict_enclosure_check.py     # -> results/fpcore/strict_enclosure.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from mpmath import iv, mpf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import external_corpus as ec  # noqa: E402

iv.prec = 160
OUT = ROOT / "results/fpcore/strict_enclosure.json"


def I(x):
    return iv.mpf(float(x))  # exact: a binary64 value is representable at 160 bits


def c32(v):
    return I(np.float32(v))


EXP = np.frompyfunc(iv.exp, 1, 1)
EXPM1 = np.frompyfunc(iv.expm1, 1, 1)
SQRT = np.frompyfunc(iv.sqrt, 1, 1)


def TANH(t):
    return np.frompyfunc(lambda z: 1 - 2 / (iv.exp(2 * z) + 1), 1, 1)(t)


def imax(a, b):
    return iv.mpf([max(a.a, b.a), max(a.b, b.b)])


def obj(a):
    return np.frompyfunc(I, 1, 1)(np.asarray(a, dtype=np.float64))


def rowmax(r):
    out = r[:, 0].copy()
    for j in range(1, r.shape[1]):
        out = np.frompyfunc(imax, 2, 1)(out, r[:, j])
    return out[:, None]


def softmax_rows(r, pad=0):
    m = rowmax(r)
    if pad:
        m = np.frompyfunc(imax, 2, 1)(m, I(0.0))
    e = EXP(r - m)
    den = e.sum(axis=1, keepdims=True) + (pad * EXP(-m) if pad else 0)
    return e / den


def declared_iv(entry, inputs):
    a = {k: obj(v) for k, v in inputs.items()}
    if entry.startswith(("attention", "flash_attention")):
        q, k, v = a["q"], a["k"], a["v"]
        scale = c32(1.0 / math.sqrt(q.shape[1]))
        s = q.dot(k.T)
        if entry == "flash_attention_triton_buggy":
            n = k.shape[0]
            block_n = max(8, 1 << (min(n, 32) - 1).bit_length())
            m = None
            l = None
            acc = None
            for start in range(0, n, block_n):
                st = (q.dot(k[start:start + block_n].T)) * scale
                tile_max = rowmax(st)[:, 0]
                m_new = tile_max if m is None else np.frompyfunc(imax, 2, 1)(m, tile_max)
                p = EXP(st - m_new[:, None])
                if m is None:
                    l, acc = p.sum(axis=1), p.dot(v[start:start + block_n])
                else:
                    l = l * EXP(m - m_new) + p.sum(axis=1)
                    acc = acc + p.dot(v[start:start + block_n])
                m = m_new
            return (acc / l[:, None]).reshape(-1)
        s = s if entry == "attention_triton_buggy" else s * scale
        return softmax_rows(s).dot(v).reshape(-1)
    if entry.startswith("matmul_triton"):
        x, y = a["a"], a["b"]
        return (np.outer(x[:, -1], y[-1, :]) if entry.endswith("buggy") else x.dot(y)).reshape(-1)
    x = a["input"]
    if entry.startswith("softmax_triton"):
        r = x.reshape(-1, x.shape[-1])
        pad = (1 << (r.shape[1] - 1).bit_length()) - r.shape[1] if entry.endswith("buggy") else 0
        return softmax_rows(r, pad).reshape(-1)
    if entry.startswith(("rmsnorm_triton", "l2norm_triton")):
        r = x.reshape(-1, x.shape[-1])
        ss = (r * r).sum(axis=1, keepdims=True)
        d = (ss / r.shape[1] + c32(1e-5)) if entry.startswith("rmsnorm") else (ss + c32(1e-12))
        return (r / (d if entry.endswith("buggy") else SQRT(d))).reshape(-1)
    x = x.reshape(-1)
    xf = np.asarray(inputs["input"], dtype=np.float64).reshape(-1)
    if entry == "relu_triton":
        return np.where(xf > 0, x, I(0.0))
    if entry.startswith("leaky_relu_triton"):
        return np.where(xf > 0, x, x * c32(0.1 if entry.endswith("buggy") else 0.01))
    if entry == "elu_triton":
        return np.where(xf >= 0, x, EXPM1(x))
    if entry.startswith("gelu_triton"):
        inner = (x + x * x * x * c32(0.044715)) * c32(0.7978845608028654)  # array first: iv * ndarray is unsupported
        y = x * (1 + TANH(inner))
        return y if entry.endswith("buggy") else y * I(0.5)
    if entry == "sigmoid_triton":
        return 1 / (1 + EXP(-x))
    if entry.startswith("silu_triton"):
        return x / (1 + EXP(x * (I(-2.0) if entry.endswith("buggy") else I(-1.0))))
    if entry == "tanh_triton":
        return TANH(x)
    raise KeyError(entry)


def main():
    entries = {e["name"]: e for e in ec.entries()}
    rows, tot = [], {"elements": 0, "inside": 0, "violations": 0, "undecided": 0, "kr_not_complete": 0}
    for line in (ROOT / "results/fpcore/fragments.jsonl").read_text().splitlines():
        fr = json.loads(line)
        e = entries[fr["entry"]]
        cond = next(c for c in ec.conditions(e["meta"]) if c["name"] == fr["condition"])
        t = declared_iv(fr["entry"], ec.make_inputs(e["meta"], cond, fr["seed"]))
        res = {"entry": fr["entry"], "condition": fr["condition"], "seed": fr["seed"], "elements": 0, "inside": 0,
               "violations": 0, "undecided": 0, "max_T_width_over_KR_width": 0.0, "examples": []}
        for el in fr.get("elements", []):
            if not el["kr_ok"]:
                tot["kr_not_complete"] += 1
                continue
            T = t[el["offset"]]
            lo, hi = mpf(el["kr_lo"]), mpf(el["kr_hi"])
            res["elements"] += 1
            if T.a >= lo and T.b <= hi:
                res["inside"] += 1
                w = hi - lo
                if w > 0:
                    res["max_T_width_over_KR_width"] = max(res["max_T_width_over_KR_width"], float(((T.b - T.a) / w).b))
            elif T.b < lo or T.a > hi:
                res["violations"] += 1
                if len(res["examples"]) < 3:
                    res["examples"].append({"offset": el["offset"], "T": [float(T.a), float(T.b)], "K_R": [el["kr_lo"], el["kr_hi"]]})
            else:
                res["undecided"] += 1
        for k in ("elements", "inside", "violations", "undecided"):
            tot[k] += res[k]
        rows.append(res)
        print(f"{fr['entry']:30s} {fr['condition']:12s} seed {fr['seed']}  {res['inside']}/{res['elements']} inside, "
              f"{res['violations']} violations, {res['undecided']} undecided", flush=True)
    OUT.write_text(json.dumps({"method": "mpmath iv, 160 bits, outward rounding; strict containment T within K_R, no slack",
                               "totals": tot, "fragments": rows}, indent=1) + "\n")
    print(tot)


if __name__ == "__main__":
    main()
