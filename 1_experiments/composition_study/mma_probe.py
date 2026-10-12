"""RQ3: the composition rule of a tensor-core dot on sm_86 (Triton 3.6.0 lowers tl.dot to mma.sync m16n8k8 for tf32
and m16n8k16 for bf16 / f16).

A dot is a composition of three steps the TTIR does not spell out: the conversion of each operand to the mma input
format, the products, and the accumulation of the products with the incoming accumulator inside one mma (and then
across the mma instructions of the k loop).  Each step is measured with constructed inputs:

1. conversion (tf32): A = [x, 0, ...], B = e_0, C0 = 0, so C = conv(x) exactly; compared with RZ / RNE / RNA to tf32.
   (tl.dot needs K >= 16, so every probe has K = 16: two m16n8k8 instructions for tf32, one m16n8k16 otherwise; the
   group size of the fitted model shows how the products are split.)
2. accumulation inside one mma: operands exactly representable in the input format (the conversion is the identity),
   B = ones in the probed column, so the products equal the A entries exactly; C = acc(c, p_1 .. p_k).  A family of
   models is fitted on constructed inputs and must then reproduce held-out random inputs bit for bit:
   - sequential: c + p_1 + ... + p_k (or reversed, or pairwise) with RN in fp32 after every addition;
   - aligned: the terms of one group are aligned to the largest exponent among them, every term is cut toward zero
     at 2^(E - 23 - g) (g extra bits), the cut terms are summed exactly and the sum is normalised to fp32 with mode
     ``final``; groups of size G are chained through the accumulator.
   With B = ones the products are the A entries; a second stage with random B (full-significand products) decides
   the alignment exponent of a product (its own exponent, or the sum of the operands' exponent fields), and for f16 a
   third stage with subnormal operands decides how they enter.
3. across mma instructions: K > k_mma chains the accumulator through consecutive mma instructions (checked with the
   fitted model on K = 64).

Output: results/mma_rules.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exact as X  # noqa: E402

M = N = 16


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def dot_k(A, B, C0, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, IP: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rk = tl.arange(0, K)
        a = tl.load(A + rm[:, None] * K + rk[None, :])
        b = tl.load(B + rk[:, None] * N + rn[None, :])
        c = tl.load(C0 + rm[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, c, input_precision=IP)
        tl.store(C + rm[:, None] * N + rn[None, :], acc)
    return dot_k


_K = None


def run_dot(A, B, C0, ip, dtype):
    """A (M x K), B (K x N) float64 arrays of values representable in ``dtype``; C0 fp32 values. Returns C (float64)."""
    import torch
    global _K
    if _K is None:
        _K = kernel()
    td = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    a = torch.from_numpy(np.ascontiguousarray(A)).to(td).cuda()
    b = torch.from_numpy(np.ascontiguousarray(B)).to(td).cuda()
    c0 = torch.from_numpy(np.ascontiguousarray(C0)).float().cuda()
    c = torch.empty(M, N, dtype=torch.float32, device="cuda")
    _K[(1,)](a, b, c0, c, M=M, N=N, K=A.shape[1], IP=ip)
    torch.cuda.synchronize()
    return c.double().cpu().numpy()


# ----------------------------------------------------------------------------- 1. conversion
def tf32_round(x: float, mode: str) -> float:
    r = X.round_frac(F(x), "tf32", mode)
    return float(r)


def probe_conversion(rng, n=4096):
    out = {}
    for side in ("A", "B"):
        xs = rng.standard_normal(n) * 2.0 ** rng.integers(-20, 20, n)
        xs = xs.astype(np.float32).astype(np.float64)
        dev = []
        for i in range(0, n, M):
            chunk = xs[i:i + M]
            A = np.zeros((M, 16))
            B = np.zeros((16, N))
            C0 = np.zeros((M, N))
            if side == "A":
                A[:, 0] = chunk
                B[0, 0] = 1.0
                dev.extend(run_dot(A, B, C0, "tf32", "f32")[:, 0])
            else:
                B[0, :] = chunk
                A[0, 0] = 1.0
                dev.extend(run_dot(A, B, C0, "tf32", "f32")[0, :])
        dev = np.array(dev)
        out[side] = {mode: int(sum(tf32_round(x, mode) == d for x, d in zip(xs, dev))) for mode in ("rz", "rne", "rna")}
        out[side]["n"] = n
        rel = (dev - xs) / xs
        out[side]["mean_relative_change"] = float(rel.mean())
    return out


# ----------------------------------------------------------------------------- 2. accumulation models
def exp_of(x: F) -> int:
    return X.floor_log2(abs(x))


def cut(x: F, q: int) -> F:
    """x cut toward zero at a multiple of 2^q."""
    u = F(2) ** q
    n = abs(x) // u
    return (n * u) if x >= 0 else -(n * u)


def norm(x: F, mode: str) -> F:
    r = X.round_frac(x, "f32", mode)
    return r if not isinstance(r, float) else F(0)


def model_aligned(c: F, ps: list, g: int, final: str, G: int, c_in_group: bool = True, pexps=None) -> F:
    """``pexps``: the alignment exponent of each product (default: the product's own exponent)."""
    acc = c
    for s in range(0, len(ps), G):
        grp = ps[s:s + G]
        terms = ([acc] if c_in_group else []) + grp
        ex = ([exp_of(acc) if acc != 0 else None] if c_in_group else []) + \
            [(pexps[s + i] if pexps is not None else exp_of(t)) if t != 0 else None for i, t in enumerate(grp)]
        nz = [e for e in ex if e is not None]
        if not nz:
            acc = F(0) if c_in_group else acc
            continue
        E = max(nz)
        q = E - 23 - g
        tot = sum((cut(t, q) for t in terms), F(0))
        if not c_in_group:
            s_ = norm(tot, final)
            acc = norm(acc + s_, "rne")
        else:
            acc = norm(tot, final)
    return acc


def model_sequential(c: F, ps: list, order: str) -> F:
    seq = list(ps)
    if order == "reverse":
        seq = seq[::-1]
    if order == "pairwise":
        vals = list(seq)
        while len(vals) > 1:
            vals = [norm(vals[i] + vals[i + 1], "rne") if i + 1 < len(vals) else vals[i] for i in range(0, len(vals), 2)]
        return norm(c + vals[0], "rne")
    acc = c
    for p in seq:
        acc = norm(acc + p, "rne")
    return acc


def candidate_models(k):
    ms = [("sequential", o) for o in ("forward", "reverse", "pairwise")]
    for g in range(0, 7):
        for final in ("rz", "rne"):
            for G in sorted({k, 8, 4, 2} & set(range(1, k + 1))):
                for cin in (True, False):
                    ms.append(("aligned", g, final, G, cin))
    return ms


def eval_model(m, c, ps):
    if m[0] == "sequential":
        return model_sequential(c, ps, m[1])
    return model_aligned(c, ps, m[1], m[2], m[3], m[4])


def representable(rng, fmt, n, spread):
    x = rng.standard_normal(n) * 2.0 ** rng.integers(-spread, spread + 1, n)
    import catalog as C
    fm = {"tf32": "f32", "bf16": "bf16", "f16": "f16"}[fmt]
    v = C.fmt_round_array(x, fm)
    if fmt == "tf32":
        v = np.array([float(X.round_frac(F(t), "tf32", "rne")) for t in v])
    return v


def accumulation_data(rng, fmt, k, trials, spread):
    """Rows of (c, products) with products exactly the A entries (B column of ones)."""
    rows = []
    dtype = {"tf32": "f32", "bf16": "bf16", "f16": "f16"}[fmt]
    ip = "tf32"
    for _ in range(trials):
        A = representable(rng, fmt, M * k, spread).reshape(M, k)
        if fmt == "f16":
            A = np.clip(A, -6e4, 6e4)
        B = np.zeros((k, N))
        B[:, 0] = 1.0
        C0 = np.zeros((M, N))
        C0[:, 0] = (rng.standard_normal(M) * 2.0 ** rng.integers(-spread, spread + 1, M)).astype(np.float32)
        Cd = run_dot(A, B, C0, ip, dtype)
        for i in range(M):
            rows.append((F(float(C0[i, 0])), [F(float(a)) for a in A[i]], F(float(Cd[i, 0]))))
    return rows


def constructed_rows(rng, fmt, k, n=64):
    """Inputs built to separate the models: one large term and many terms below its ulp, mixed signs."""
    rows = []
    dtype = {"tf32": "f32", "bf16": "bf16", "f16": "f16"}[fmt]
    ip = "tf32"
    p = {"tf32": 11, "bf16": 8, "f16": 11}[fmt]
    for t in range(n):
        A = np.zeros((M, k))
        C0 = np.zeros((M, N))
        for i in range(M):
            big = float(rng.choice([1.0, -1.0, 1.5, -1.75]))
            C0[i, 0] = big if (t + i) % 2 == 0 else 0.0
            for j in range(k):
                e = -int(rng.integers(20, 30)) if (t + i) % 2 == 0 else -int(rng.integers(0, 30))
                m = 1 + int(rng.integers(0, 2 ** (p - 1))) / 2 ** (p - 1)
                A[i, j] = float(rng.choice([1.0, -1.0])) * m * 2.0 ** e
            if (t + i) % 2 == 1:
                A[i, 0] = big
        if fmt == "f16":
            A = np.clip(A, -6e4, 6e4)
            A = np.where(np.abs(A) < 2.0 ** -14, 0.0, A)
        B = np.zeros((k, N))
        B[:, 0] = 1.0
        import catalog as C
        A = C.fmt_round_array(A.ravel(), dtype if fmt != "tf32" else "f32").reshape(M, k)
        if fmt == "tf32":
            A = np.array([float(X.round_frac(F(t_), "tf32", "rne")) for t_ in A.ravel()]).reshape(M, k)
        Cd = run_dot(A, B, C0, ip, dtype)
        for i in range(M):
            rows.append((F(float(C0[i, 0])), [F(float(a)) for a in A[i]], F(float(Cd[i, 0]))))
    return rows


def fit(rows, k):
    scores = {}
    for m in candidate_models(k):
        ok = 0
        for c, ps, d in rows:
            if eval_model(m, c, ps) == d:
                ok += 1
        scores[m] = ok
    return scores


def exponent_field(x: F, fmt: str) -> int:
    emin = {"tf32": -126, "bf16": -126, "f16": -14}[fmt]
    return max(exp_of(x), emin)


def general_products(rng, fmt, trials, sub_spread=0):
    """Random A and B (both representable), so the products have full significands; with ``sub_spread`` the operands
    reach far below the normal range (f16 subnormals)."""
    import torch
    td = {"tf32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[fmt]
    rows = []
    for _ in range(trials):
        sa = 2.0 ** rng.integers(-sub_spread, 1, (M, 16)) if sub_spread else 1.0
        sb = 2.0 ** rng.integers(-sub_spread, 1, (16, N)) if sub_spread else 0.05
        A = torch.from_numpy(rng.standard_normal((M, 16)) * sa).to(td).double().numpy()
        B = torch.from_numpy(rng.standard_normal((16, N)) * sb).to(td).double().numpy()
        if fmt == "tf32":
            A = np.array([float(X.round_frac(F(t), "tf32", "rne")) for t in A.ravel()]).reshape(M, 16)
            B = np.array([float(X.round_frac(F(t), "tf32", "rne")) for t in B.ravel()]).reshape(16, N)
        C0 = (rng.standard_normal((M, N)) * 0.1).astype(np.float32).astype(np.float64)
        Cd = run_dot(A, B, C0, "tf32", {"tf32": "f32", "bf16": "bf16", "f16": "f16"}[fmt])
        for i in range(M):
            for j in range(0, N, 4):
                rows.append((F(float(C0[i, j])), [F(float(A[i, k])) * F(float(B[k, j])) for k in range(16)],
                             [exponent_field(F(float(A[i, k])), fmt) + exponent_field(F(float(B[k, j])), fmt)
                              if A[i, k] != 0 and B[k, j] != 0 else 0 for k in range(16)],
                             [exp_of(F(float(A[i, k])) * F(float(B[k, j]))) if A[i, k] * B[k, j] != 0 else 0
                              for k in range(16)], F(float(Cd[i, j]))))
    return rows


def score_general(rows, G):
    """Product alignment exponent: the sum of the operand exponent fields versus the product's own exponent."""
    out = {}
    for name, idx in (("operand exponent sum", 2), ("product exponent", 3)):
        out[name] = sum(model_aligned(c, ps, 1, "rz", G, True, pexps=r[idx]) == d
                        for r in rows for (c, ps, d) in [(r[0], r[1], r[4])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/mma_rules.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(20261013)
    res = {"device": "RTX A6000 (sm_86)", "triton": "3.6.0"}
    res["conversion_tf32"] = probe_conversion(rng)
    for fmt, k in (("tf32", 16), ("bf16", 16), ("f16", 16)):   # tl.dot needs K >= 16
        cons = constructed_rows(rng, fmt, k)
        sc = fit(cons, k)
        best = max(sc.values())
        winners = [m for m, v in sc.items() if v == best]
        held = accumulation_data(rng, fmt, k, trials=32, spread=12)
        held_sc = {m: sum(eval_model(m, c, ps) == d for c, ps, d in held) for m in winners}
        # the chain across mma instructions: K = 64 with the winning models
        chain = accumulation_data(rng, fmt, 64, trials=8, spread=8)
        chain_sc = {m: sum(eval_model(m, c, ps) == d for c, ps, d in chain) for m in winners}
        seq_scores = {str(m): v for m, v in sc.items() if m[0] == "sequential"}
        G = winners[0][3] if winners and winners[0][0] == "aligned" else k
        gen = general_products(rng, fmt, trials=8)
        gen_sc = score_general(gen, G)
        sub = general_products(rng, fmt, trials=8, sub_spread=20) if fmt == "f16" else []
        sub_sc = score_general(sub, G) if sub else None
        res[fmt] = {"k_per_mma": k, "constructed_rows": len(cons), "best_constructed": best,
                    "winners": [str(m) for m in winners], "sequential_models": seq_scores,
                    "held_out_rows": len(held), "held_out_matches": {str(m): v for m, v in held_sc.items()},
                    "chain_K64_rows": len(chain), "chain_K64_matches": {str(m): v for m, v in chain_sc.items()},
                    "general_products_rows": len(gen), "general_products_matches": gen_sc,
                    "f16_subnormal_operand_rows": len(sub), "f16_subnormal_operand_matches": sub_sc}
        print(fmt, best, len(cons), winners[:6], held_sc, chain_sc, gen_sc, sub_sc, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["conversion_tf32"]))


if __name__ == "__main__":
    main()
