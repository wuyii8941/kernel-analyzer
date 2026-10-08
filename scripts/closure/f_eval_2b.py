#!/usr/bin/env python3
"""F group and precision invariance for the remaining 2b families (continuation of scripts/closure/f_eval.py; same contract).

Families: embedding, matmul_linear, reductions, activations, gather_layout, attention, rope, packing, optimizers,
schedulers, clip_amp, moe.  The spec is evaluated on the values each candidate received: the 2b harnesses give every
candidate float32-representable inputs except attention / rope / packing / optimizers (inputs rounded to the candidate dtype
at tensor creation) -- there the spec is evaluated on the float32-rounded values for float32 candidates and on the unrounded
values for float64 candidates; bf16 candidates are recorded only.  Outputs the spec does not define (most gradients;
contract-outside conditions) are listed as "no f".

    python scripts/closure/f_eval_2b.py embedding matmul_linear ...
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "specs" / "phase2"))
sys.path.insert(0, str(ROOT / "specs" / "phase1"))
sys.path.insert(0, str(ROOT / "scripts" / "essential"))
sys.path.insert(0, str(ROOT / "scripts" / "closure"))
import contract_v3 as CV  # noqa: E402
from f_eval import bounds, compare, load, spec_arrays  # noqa: E402

OUT = ROOT / "results/closure/f_eval"


def F32(a):
    return np.asarray(a, dtype=np.float64).astype(np.float32).astype(np.float64)


def frs(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return Fr(float(a))
    return [frs(x) for x in a]


class Collector:
    def __init__(self, family, spec):
        self.res = {"family": family, "spec": spec, "candidates": defaultdict(dict), "precision_invariance": {},
                    "spec_status": {}, "no_f": defaultdict(int)}

    def compare_all(self, key, cands, rec_of, fa, dtype_of=None):
        for name, d in cands.items():
            rec = rec_of(d)
            if rec is None:
                continue
            dtype = (dtype_of or (lambda d_: d_["meta"]["candidate"]["dtype"]))(d)
            row = {}
            for k, v in rec.items():
                if k in fa:
                    lo, hi = fa[k]
                    defined = ~(np.isnan(lo) | np.isnan(hi))
                    kv = np.asarray(v, dtype=np.float64).ravel()
                    if defined.all() or kv.size != lo.size:
                        row[k] = compare(kv, lo, hi, dtype)
                    else:
                        row[k] = compare(kv[defined], lo[defined], hi[defined], dtype)
                        row[k]["declined_elements_excluded"] = int((~defined).sum())
                else:
                    self.res["no_f"][k] += 1
            self.res["candidates"][name][key] = row

    def pinv(self, key, k32, k64, lo, hi, lo64=None, hi64=None):
        self.res["precision_invariance"][key] = CV.precision_invariance(
            k32, k64, 0.5 * (lo + hi), None if lo64 is None else 0.5 * (lo64 + hi64))


def basic_recs(fam):
    cache = ROOT / ".cache/essential/p2b/basic"
    return {p.stem.split("__", 1)[1]: load(p) for p in sorted(cache.glob(f"{fam}__*.pkl"))}


def base_out(d, cid, seed):
    r = d["res"].get((cid, seed), {}).get("base")
    return r["outputs"] if r and r["status"] == "ok" else None


def pinv_pairs(col, cands, key, cid, seed, fa, outputs):
    for dev in ("cpu", "cuda"):
        a, b = cands.get(f"eager_{dev}_float32"), cands.get(f"eager_{dev}_float64")
        if not a or not b:
            continue
        oa, ob = base_out(a, cid, seed), base_out(b, cid, seed)
        if not oa or not ob:
            continue
        for o in outputs:
            if o in fa and o in oa and o in ob:
                lo, hi = fa[o]
                m = ~(np.isnan(lo) | np.isnan(hi))
                ka, kb = np.asarray(oa[o], np.float64).ravel(), np.asarray(ob[o], np.float64).ravel()
                if ka.size == lo.size == kb.size:
                    col.pinv(f"{dev}/{key}/{o}", ka[m], kb[m], lo[m], hi[m])


def pinv_named(col, key, pairs, fa32, fa64, outputs):
    """pairs: [(label, outputs32, outputs64)] of one algorithm on one device; fa32 / fa64: spec on each candidate's inputs."""
    for label, oa, ob in pairs:
        if not oa or not ob:
            continue
        for o in outputs:
            if o not in fa32 or o not in oa or o not in ob:
                continue
            lo, hi = (np.asarray(a, np.float64).ravel() for a in fa32[o])
            lo64, hi64 = (np.asarray(a, np.float64).ravel() for a in fa64[o])
            m = ~(np.isnan(lo) | np.isnan(hi) | np.isnan(lo64) | np.isnan(hi64))
            ka, kb = np.asarray(oa[o], np.float64).ravel(), np.asarray(ob[o], np.float64).ravel()
            if ka.size == lo.size == kb.size:
                col.pinv(f"{label}/{key}/{o}", ka[m], kb[m], lo[m], hi[m], lo64[m], hi64[m])


# ------------------------------------------------------------------------------------------------ matmul_linear

def matmul_job(args):
    cid, seed = args
    import p2b_basic as BA
    import spec_base_ops as SB
    fam = BA.FAMILIES["matmul_linear"]
    c = {x["id"]: x for x in fam.conditions()}[cid]
    if c["shape"] == "batch-broadcast":
        return cid, seed, None, "refused: BASE-A4 (broadcast outside the spec)"
    inp = fam.inputs(c, seed, "base")
    A, B, g = inp["a"], inp["b"], inp["g"]
    out = SB.matmul2(frs(A), frs(B)) if A.shape[1] else [[Fr(0)] * B.shape[1] for _ in range(A.shape[0])]
    if inp["bias"] is not None:
        bias = np.asarray(inp["bias"]).reshape(-1)
        out = [[v + Fr(float(bias[j])) for j, v in enumerate(r)] for r in out]
    fa = {"out": spec_arrays(out)}
    gF = frs(g)
    if A.shape[1]:
        dA = SB.matmul2(gF, SB.transpose(frs(B)))
        dB = SB.matmul2(SB.transpose(frs(A)), gF)
    else:
        dA = [[] for _ in range(A.shape[0])]
        dB = [[Fr(0)] * B.shape[1] for _ in range(0)]
    fa["da"] = spec_arrays(dA) if A.shape[1] else (np.zeros(0), np.zeros(0))
    fa["db"] = spec_arrays(dB) if A.shape[1] else (np.zeros(0), np.zeros(0))
    if inp["bias"] is not None:
        db = [sum((r[j] for r in gF), Fr(0)) for j in range(B.shape[1])]
        fa["dbias"] = spec_arrays(db)
    return cid, seed, fa, "ok"


def reduction_job(args):
    cid, seed = args
    import p2b_basic as BA
    import spec_base_ops as SB
    from contract_v2 import classify_condition
    fam = BA.FAMILIES["reductions"]
    c = {x["id"]: x for x in fam.conditions()}[cid]
    cls = classify_condition("reductions", c)["classes"]
    if not any(x.startswith(("A", "C")) for x in cls):
        return cid, seed, None, f"contract v2: {cls[0]} (not judged against f)"
    inp = fam.inputs(c, seed, "base")
    x = np.asarray(inp["x"], dtype=np.float64)
    op, dims = c["op"], c["dims"]
    if dims in ("single", "empty-extent"):
        rows = [x[i] for i in range(x.shape[0])]
    elif dims == "multi":
        rows = [x[i].ravel() for i in range(x.shape[0])]
    else:
        rows = [x.ravel()]
    fn = {"sum": SB.rsum, "mean": SB.rmean, "prod": SB.rprod, "amax": SB.ramax, "amin": SB.ramin,
          "var": SB.rvar, "std": SB.rstd, "softmax": SB.softmax, "log_softmax": SB.log_softmax,
          "logsumexp": SB.logsumexp, "cumsum": SB.cumsum}[op]
    per_row = op in ("softmax", "log_softmax", "cumsum")
    lo, hi, declined = [], [], []
    for r in rows:                                       # rows the spec declines (undefined / not established): NaN, excluded
        arg = [float("-inf") if v == -np.inf else Fr(float(v)) for v in r]
        try:
            v = fn(arg)
        except Exception as exc:  # noqa: BLE001
            declined.append(f"{type(exc).__name__}: {str(exc)[:80]}")
            n = len(r) if per_row else 1
            lo.extend([np.nan] * n); hi.extend([np.nan] * n)
            continue
        for e in (v if isinstance(v, list) else [v]):
            a, b = (e, e) if isinstance(e, float) else bounds(e)
            lo.append(a); hi.append(b)
    status = "ok" if not declined else f"ok; rows declined by the spec: {len(declined)} ({declined[0]})"
    return cid, seed, {"out": (np.array(lo), np.array(hi))}, status


def activation_job(args):
    cid, seed = args
    import p2b_basic as BA
    import spec_base_ops as SB
    fam = BA.FAMILIES["activations"]
    c = {x["id"]: x for x in fam.conditions()}[cid]
    inp = fam.inputs(c, seed, "base")
    a, b, g = (np.asarray(inp[k], dtype=np.float64).ravel() for k in ("a", "b", "g"))
    op = c["op"]
    try:
        if op == "relu":
            out = [SB.relu(Fr(float(v))) for v in a]
        elif op == "silu":
            out = [SB.silu(Fr(float(v))) for v in a]
        elif op == "gelu_erf":
            out = [SB.gelu_erf(Fr(float(v))) for v in a]
        elif op == "gelu_tanh":
            out = [SB.gelu_tanh(Fr(float(v))) for v in a]
        elif op == "swiglu":
            out = [SB.swiglu(Fr(float(u)), Fr(float(w))) for u, w in zip(a, b)]
        else:
            out = [SB.geglu(Fr(float(u)), Fr(float(w))) for u, w in zip(a, b)]
    except Exception as exc:  # noqa: BLE001
        return cid, seed, None, f"spec: {type(exc).__name__}: {str(exc)[:120]}"
    fa = {"out": spec_arrays(out)}
    if op == "relu":                                         # subgradient set at 0: report membership separately
        sets = [SB.relu_grad(Fr(float(v)), Fr(float(u))) for v, u in zip(a, g)]
        fa["da_set"] = sets
    return cid, seed, fa, "ok"


def gather_job(args):
    cid, seed = args
    import p2b_basic as BA
    import spec_base_ops as SB
    fam = BA.FAMILIES["gather_layout"]
    c = {x["id"]: x for x in fam.conditions()}[cid]
    if c["op"] not in ("gather", "index_select", "take_along_dim"):
        return cid, seed, None, "outside the contract"
    inp = fam.inputs(c, seed, "base")
    x, idx, g = inp["x"], np.asarray(inp["idx"]), inp["g"]
    if c["op"] == "index_select":
        sel = [int(v) for v in idx[0]]
        out = [r[:5] for r in SB.index_select(frs(x), 1, sel)]
        index = [sel[:5]] * 6
    else:
        index = idx.tolist()
        out = SB.gather(frs(x), 1, index)
    dx = SB.gather_backward((6, 9), 1, index, frs(g))
    return cid, seed, {"out": spec_arrays(out), "dx": spec_arrays(dx)}, "ok"


def run_basic(fam, job):
    import p2b_basic as BA
    cands = basic_recs(fam)
    conds = BA.FAMILIES[fam].conditions()
    col = Collector(fam, "specs/phase2/spec_base_ops.py v0.1")
    jobs = [(c["id"], s) for c in conds for s in (0, 1, 2)]
    with ProcessPoolExecutor(max_workers=24) as ex:
        for cid, seed, fa, status in ex.map(job, jobs):
            key = f"{cid}/{seed}"
            col.res["spec_status"][key] = status
            if fa is None:
                continue
            sets = fa.pop("da_set", None)
            col.compare_all(key, cands, lambda d: base_out(d, cid, seed), fa)
            if sets is not None:                             # ReLU at 0: K must be one of the documented-set values
                from f_eval import TAU
                for name, d in cands.items():
                    o = base_out(d, cid, seed)
                    if o and "da" in o:
                        dt = d["meta"]["candidate"]["dtype"]
                        tau = TAU.get(dt, 2.0 ** -8)
                        k = np.asarray(o["da"], dtype=np.float64).ravel()
                        bad = sum(1 for kv, sv in zip(k, sets) if not any(abs(kv - float(s)) <= tau * (1 + abs(float(s))) for s in sv))
                        col.res["candidates"][name][key]["da_relu_set"] = {"elements": int(k.size), "beyond": int(bad),
                                                                           "judged": dt in TAU}
            pinv_pairs(col, cands, key, cid, seed, fa, list(fa))
    return col.res


# ------------------------------------------------------------------------------------------------ embedding

def embedding_job(args):
    cid, seed = args
    import p2b_embedding as EM
    import spec_embedding as SE
    c = {x["id"]: x for x in EM.conditions()}[cid]
    inp = EM.make_inputs(c, seed)
    W = frs(inp["w"])
    idx = np.asarray(inp["idx"])
    fa = {}
    try:
        if c["max_norm"]:
            ren = SE.embedding_renorm(W, [int(v) for v in idx.ravel()], Fr(c["max_norm"]))
            W2 = [ren.get(j, W[j]) for j in range(len(W))]
            fa["weight_after"] = spec_arrays(W2)
        else:
            W2 = W
        if c["op"] == "embedding":
            out = [[W2[int(j)] for j in row] for row in idx]
            fa["out"] = spec_arrays(out)
            if not c["max_norm"]:
                gw = SE.embedding_grad_weight((len(W), len(W[0])), [int(v) for v in idx.ravel()],
                                              frs(np.asarray(inp["g"]).reshape(-1, len(W[0]))),
                                              padding_idx=c["padding_idx"], scale_grad_by_freq=c["scale_grad_by_freq"])
                fa["dweight"] = spec_arrays(gw)
        else:
            bags = SE.embedding_bag(W2, [int(v) for v in idx], list(inp["offsets"]), mode=c["mode"],
                                    per_sample_weights=frs(inp["psw"]) if c["per_sample_weights"] else None,
                                    padding_idx=c["padding_idx"])
            fa["out"] = spec_arrays(bags)
    except Exception as exc:  # noqa: BLE001
        return cid, seed, None, f"spec: {type(exc).__name__}: {str(exc)[:160]}"
    return cid, seed, fa, "ok"


def run_embedding():
    import p2b_embedding as EM
    cands = {p.stem: load(p) for p in sorted(EM.CACHE.glob("*.pkl"))}
    col = Collector("embedding", "specs/phase2/spec_embedding.py v0.1")
    jobs = [(c["id"], s) for c in EM.conditions() for s in EM.SEEDS]
    from contract_v2 import PENDING
    conds = {c["id"]: c for c in EM.conditions()}
    with ProcessPoolExecutor(max_workers=24) as ex:
        for cid, seed, fa, status in ex.map(embedding_job, jobs):
            key = f"{cid}/{seed}"
            col.res["spec_status"][key] = status
            if fa is None:
                continue
            col.compare_all(key, cands, lambda d: base_out(d, cid, seed), fa)
            pend = PENDING["embedding"](conds[cid])
            if pend:                                         # E-D2 scope stopped: weight_after (all rows) not adjudicated
                for name in cands:
                    if "weight_after" in col.res["candidates"][name].get(key, {}):
                        col.res["candidates"][name][key]["weight_after"]["clause_pending"] = pend
            pinv_pairs(col, cands, key, cid, seed, fa, list(fa))
    return col.res


# ------------------------------------------------------------------------------------------------ attention / rope / packing

def attention_job(args):
    cid, seed, rounding = args
    import p2b_attention as AT
    import spec_attention as SA
    c = {x["id"]: x for x in AT.conditions()}[cid]
    inp = AT.make_inputs(c, seed, "base")
    rnd = F32 if rounding == "float32" else (lambda a: np.asarray(a, dtype=np.float64))
    q, k, v = rnd(inp["q"]), rnd(inp["k"]), rnd(inp["v"])
    allowed = inp["allowed"]
    Bn, H, Lq, E = q.shape
    Hk = k.shape[1]
    scale = Fr(c["scale"]) if c["scale"] is not None else None
    lo = np.full(q.shape[:3] + (v.shape[-1],), np.nan)
    hi = lo.copy()
    undefined = 0
    for b in range(Bn):
        for h in range(H):
            hk = SA.gqa_kv_head(h, H, Hk)
            kk, vv = frs(k[b, hk]), frs(v[b, hk])
            for i in range(Lq):
                mask = [[bool(x) for x in allowed[b, i]]]
                try:
                    o, _ = SA.attention([frs(q[b, h, i])], kk, vv, attn_mask=mask, scale=scale)
                except SA.SpecNotEstablished:
                    undefined += 1
                    continue
                for d_ in range(v.shape[-1]):
                    lo[b, h, i, d_], hi[b, h, i, d_] = bounds(o[0][d_])
    return cid, seed, rounding, {"out": (lo, hi)}, undefined


def run_attention():
    import p2b_attention as AT
    cands = {p.stem: load(p) for p in sorted(AT.CACHE.glob("*.pkl"))}
    col = Collector("attention", "specs/phase2/spec_attention.py v0.1")
    jobs = [(c["id"], s, r) for c in AT.conditions() for s in AT.SEEDS for r in ("float32", "float64")]
    specs = {}
    with ProcessPoolExecutor(max_workers=32) as ex:
        for cid, seed, rounding, fa, und in ex.map(attention_job, jobs):
            specs[(cid, seed, rounding)] = fa
            col.res["spec_status"][f"{cid}/{seed}/{rounding}"] = f"ok; rows without an allowed key (ATT-A1, undefined): {und}"
    for c in AT.conditions():
        for seed in AT.SEEDS:
            key = f"{c['id']}/{seed}"
            for name, d in cands.items():
                o = base_out(d, c["id"], seed)
                if not o:
                    continue
                dtype = d["meta"]["candidate"]["dtype"]
                fa = specs[(c["id"], seed, "float64" if dtype == "float64" else "float32")]["out"]
                defined = np.isfinite(fa[0])
                k = np.asarray(o["out"], dtype=np.float64)
                row = {"out": compare(k[defined], fa[0][defined], fa[1][defined], dtype)}
                row["out"]["undefined_elements_excluded"] = int((~defined).sum())
                col.res["candidates"][name][key] = row
            pairs = [(lab, base_out(cands[a], c["id"], seed) if a in cands else None,
                      base_out(cands[b], c["id"], seed) if b in cands else None)
                     for lab, a, b in (("cuda_sdpa_math", "sdpa_math_float32", "sdpa_math_cuda_float64"),
                                       ("cuda_manual", "manual_eager_float32", "manual_eager_float64"),
                                       ("cpu_sdpa_math", "sdpa_math_cpu_float32", "sdpa_math_cpu_float64"))]
            pinv_named(col, key, pairs, specs[(c["id"], seed, "float32")], specs[(c["id"], seed, "float64")], ["out"])
    col.res["precision_invariance_note"] = ("pairs: CUDA SDPA math, CUDA manual eager, CPU SDPA math (float64 / float32 "
                                            "counterparts run for the closure, env ka_main_f64); fused / bf16 kernels: not applicable")
    return col.res


def rope_job(args):
    cid, seed, rounding = args
    import p2b_small_families as SM
    import spec_attention as SA
    import torch
    c = {x["id"]: x for x in SM.plan("rope", SM.RO_KEYS)}[cid]
    rng = SM._rng(f"ro/{c['id']}/{seed}")
    L, H, d = 11, 4, c["head_dim"]
    q = rng.normal(0, 1, (2, H, L, d))
    k = rng.normal(0, 1, (2, H, L, d))
    rnd = F32 if rounding == "float32" else (lambda a: np.asarray(a, dtype=np.float64))
    q, k = rnd(q), rnd(k)
    pos = np.arange(L) + c["offset"]
    out = {}
    for name, x in (("q", q), ("k", k)):
        lo = np.zeros(x.shape)
        hi = np.zeros(x.shape)
        for b in range(2):
            for h in range(H):
                for i in range(L):
                    p_ = SA.I(Fr(int(pos[i]), 2)) if c["scaling"] == "linear2" else int(pos[i])
                    y = SA.rope_apply([Fr(float(v)) for v in x[b, h, i]], p_, d, layout=c["convention"])
                    for j, yy in enumerate(y):
                        lo[b, h, i, j], hi[b, h, i, j] = bounds(yy)
        out[name] = (lo, hi)
    return cid, seed, rounding, out


def run_rope():
    import p2b_small_families as SM
    cands = {p.stem.split("__", 1)[1]: load(p) for p in sorted(SM.CACHE.glob("rope__*.pkl"))}
    col = Collector("rope", "specs/phase2/spec_attention.py v0.1 (rope_apply)")
    conds = SM.plan("rope", SM.RO_KEYS)
    jobs = [(c["id"], s, r) for c in conds for s in SM.SEEDS for r in ("float32", "float64")]
    specs = {}
    with ProcessPoolExecutor(max_workers=24) as ex:
        for cid, seed, rounding, fa in ex.map(rope_job, jobs):
            specs[(cid, seed, rounding)] = fa
    for c in conds:
        for seed in SM.SEEDS:
            key = f"{c['id']}/{seed}"
            col.res["spec_status"][key] = "ok"
            for name, d in cands.items():
                r = d["res"].get((c["id"], seed), {}).get("base")
                if not r or r["status"] != "ok":
                    continue
                dtype = d["meta"]["candidate"]["dtype"]
                fa = specs[(c["id"], seed, "float64" if dtype == "float64" else "float32")]
                col.res["candidates"][name][key] = {o: compare(r["outputs"][o], *fa[o], dtype) for o in ("q", "k")}

            def ro(n):
                r = cands.get(n, {}).get("res", {}).get((c["id"], seed), {}).get("base")
                return r["outputs"] if r and r["status"] == "ok" else None
            pinv_named(col, key, [("cuda_ref_torch", ro("ref_torch"), ro("ref_torch_cuda_float64"))],
                       specs[(c["id"], seed, "float32")], specs[(c["id"], seed, "float64")], ["q", "k"])
    return col.res


def packing_job(args):
    cid, seed, rounding = args
    import p2b_packing as PK
    import spec_attention as SA
    c = {x["id"]: x for x in PK.conditions()}[cid]
    rng = PK._rng(f"pack/{cid}/{seed}")
    t = sum(c["doc_lengths"])
    rnd = F32 if rounding == "float32" else (lambda a: np.asarray(a, dtype=np.float64))
    q, k, v = (rnd(rng.normal(0, 1, (1, PK.H, t, PK.D))) for _ in range(3))
    mask = SA.packed_mask(c["doc_lengths"], causal=True)
    lo = np.zeros(q.shape)
    hi = np.zeros(q.shape)
    for h in range(PK.H):
        o, _ = SA.attention(frs(q[0, h]), frs(k[0, h]), frs(v[0, h]), attn_mask=[[bool(x) for x in r] for r in mask])
        for i in range(t):
            for d_ in range(PK.D):
                lo[0, h, i, d_], hi[0, h, i, d_] = bounds(o[i][d_])
    return cid, seed, rounding, {"out": (lo, hi)}


def run_packing():
    import p2b_packing as PK
    cands = {p.stem: load(p) for p in sorted(PK.CACHE.glob("*.pkl")) if not p.stem.startswith("hf_")}
    col = Collector("packing", "specs/phase2/spec_attention.py v0.1 (packed_mask)")
    jobs = [(c["id"], s, r) for c in PK.conditions() for s in PK.SEEDS for r in ("float32", "float64")]
    specs = {}
    with ProcessPoolExecutor(max_workers=24) as ex:
        for cid, seed, rounding, fa in ex.map(packing_job, jobs):
            specs[(cid, seed, rounding)] = fa
    for c in PK.conditions():
        for seed in PK.SEEDS:
            cid, key = c["id"], f"{c['id']}/{seed}"
            col.res["spec_status"][key] = "ok"
            for name, d in cands.items():
                r = d["res"].get((cid, seed))
                if not r or r["status"] != "ok":
                    continue
                dt = d["meta"]["candidate"].get("dtype", "float32")
                fa = specs[(cid, seed, dt if dt == "float64" else "float32")]
                col.res["candidates"][name][key] = {"out_packed": compare(r["packed"]["out"], *fa["out"], dt)}

            def po(n):
                r = cands.get(n, {}).get("res", {}).get((cid, seed))
                return {"out": r["packed"]["out"]} if r and r["status"] == "ok" else None
            pinv_named(col, key, [("cuda_sdpa_math", po("sdpa_math_block_mask"), po("sdpa_math_block_mask_float64")),
                                  ("cuda_flex_eager", po("flex_eager_block_mask"), po("flex_eager_block_mask_float64"))],
                       specs[(cid, seed, "float32")], specs[(cid, seed, "float64")], ["out"])
    col.res["hf_packed_note"] = "model-level candidate (2-layer Llama): no spec of the whole model, F not applicable"
    return col.res


# ------------------------------------------------------------------------------------------------ optimizers

def optimizer_job(args):
    cid, seed, maximize_variant, rounding = args
    import p2b_optimizers as OP
    import spec_optimizers as SO
    import spec_training_program as ST
    c = {x["id"]: x for x in OP.conditions()}[cid]
    if c["opt"] == "adafactor":
        return cid, seed, rounding, None, "outside the contract (Adafactor)"
    cls, kw = OP.OPTS[c["opt"]]
    grads = OP.grad_sequence(c, seed, maximize_variant)
    rnd = F32 if rounding == "float32" else (lambda a: np.asarray(a, dtype=np.float64))
    theta0 = [rnd(a) for a in OP.initial_params(seed)]
    maximize = c["maximize"] != (maximize_variant == "mirror")
    traj = [[None] * 3 for _ in range(len(grads) + 1)]
    states = [[None] * 3 for _ in range(len(grads) + 1)]
    for pi in range(3):
        lr = Fr(kw["lr"]) * (2 if pi == 2 else 1)
        wd = Fr(c["weight_decay"]) if pi < 2 else Fr(0)
        n = theta0[pi].size
        cur = [Fr(float(v)) for v in theta0[pi].ravel()]
        traj[0][pi] = cur
        if cls in ("Adam", "AdamW"):
            st = SO.AdamState(n)
            hp = dict(lr=lr, beta1=Fr(kw["betas"][0]), beta2=Fr(kw["betas"][1]), eps=Fr(kw["eps"]), weight_decay=wd,
                      amsgrad=kw.get("amsgrad", False), maximize=maximize, decoupled=cls == "AdamW")
            fn = SO.adam_step
        elif cls == "SGD":
            st = SO.SGDState(n)
            hp = dict(lr=lr, momentum=Fr(kw["momentum"]), dampening=Fr(kw.get("dampening", 0)), weight_decay=wd,
                      nesterov=kw.get("nesterov", False), maximize=maximize)
            fn = SO.sgd_step
        else:
            st = SO.RMSpropState(n)
            hp = dict(lr=lr, alpha=Fr(kw["alpha"]), eps=Fr(kw["eps"]), weight_decay=wd, momentum=Fr(kw["momentum"]),
                      centered=kw.get("centered", False), maximize=maximize)
            fn = SO.rmsprop_step
        scaler = ST.ScalerState() if c["state"] == "nonfinite_skip" else None
        try:
            for t, gs in enumerate(grads):
                g = gs[pi]
                if scaler is not None:
                    allg = [x for x in gs if x is not None]
                    nonfin = any(not np.isfinite(x).all() for x in allg)
                    if nonfin:
                        scaler.scale = scaler.scale * scaler.backoff
                        scaler.tracker = 0
                        traj[t + 1][pi] = cur
                        states[t + 1][pi] = _state_snapshot(st)
                        continue
                    scaler.tracker += 1
                gl = None if g is None else [Fr(float(v)) for v in rnd(g).ravel()]
                cur = fn(cur, gl, st, **hp)
                traj[t + 1][pi] = cur
                states[t + 1][pi] = _state_snapshot(st)
        except Exception as exc:  # noqa: BLE001
            return cid, seed, rounding, None, f"spec: {type(exc).__name__}: {str(exc)[:160]}"
    out = {}
    for t in range(len(grads) + 1):
        for pi in range(3):
            out[f"param{pi}@{t}"] = spec_arrays(traj[t][pi])
            if states[t][pi]:
                for k_, v_ in states[t][pi].items():
                    out[f"state{pi}.{k_}@{t}"] = spec_arrays(v_)
    return cid, seed, rounding, out, "ok"


def _state_snapshot(st):
    if hasattr(st, "m"):
        d = {"exp_avg": list(st.m), "exp_avg_sq": list(st.v)}
        if any(x != 0 for x in st.vmax):
            d["max_exp_avg_sq"] = list(st.vmax)
        return d
    if hasattr(st, "gave"):
        d = {"square_avg": list(st.v), "grad_avg": list(st.gave)}
        if any(not (isinstance(x, Fr) and x == 0) for x in st.b):
            d["momentum_buffer"] = list(st.b)
        return d
    return {"momentum_buffer": list(st.b)} if all(x is not None for x in st.b) else {}


def opt_groups(r, fa):
    """one candidate run -> {group: (K, lo, hi)} over all steps and parameters, in the spec's key order."""
    acc = defaultdict(lambda: ([], [], []))
    for t, snap in enumerate(r["traj"]):
        items = [(f"param{pi}@{t}", snap["params"][pi], "param") for pi in range(3)]
        for pi, st in snap["state"].items():
            for k_, v_ in st.items():
                if k_ != "step":
                    items.append((f"state{pi}.{k_}@{t}", v_, f"state.{k_}"))
        for fk, kv, group in items:
            if fk in fa:
                kv = np.asarray(kv, np.float64).ravel()
                if kv.size != fa[fk][0].size:
                    continue
                a = acc[group]
                a[0].append(kv); a[1].append(fa[fk][0]); a[2].append(fa[fk][1])
    return {g: tuple(np.concatenate(x) for x in v) for g, v in acc.items()}


def run_optimizers():
    import p2b_optimizers as OP
    from contract_v2 import PENDING
    cands = {p.stem: load(p) for p in sorted(OP.CACHE.glob("*.pkl")) if not p.stem.endswith("_pilot")}
    col = Collector("optimizers", "specs/phase2/spec_optimizers.py v0.1 (+ spec_training_program GradScaler)")
    conds = {c["id"]: c for c in OP.conditions()}
    jobs = [(cid, s, "base", r) for cid in conds for s in OP.SEEDS for r in ("float32", "float64")]
    specs = {}
    with ProcessPoolExecutor(max_workers=24) as ex:
        for cid, seed, rounding, fa, status in ex.map(optimizer_job, jobs):
            specs[(cid, seed, rounding)] = fa
            if rounding == "float32":
                col.res["spec_status"][f"{cid}/{seed}"] = status
    for cid, c in conds.items():
        pending = PENDING.get("optimizers", lambda c_: [])(c)
        for seed in OP.SEEDS:
            key = f"{cid}/{seed}"
            if specs[(cid, seed, "float32")] is None:
                continue
            groups_of = {}
            for name, d in cands.items():
                r = d["res"].get((cid, seed), {}).get("base")
                if not r or r["status"] != "ok":
                    continue
                dt = d["meta"]["candidate"].get("dtype", "float32")
                fa = specs[(cid, seed, "float64" if dt == "float64" else "float32")]
                groups_of[name] = opt_groups(r, fa)
                row = {}
                for g, (kv, lo, hi) in groups_of[name].items():
                    row[g] = compare(kv, lo, hi, dt)
                    if pending and g == "state.momentum_buffer":
                        row[g]["clause_pending"] = pending
                col.res["candidates"][name][key] = row
            for impl in ("for_loop", "foreach"):
                for dev in ("cpu", "cuda"):
                    a, b = groups_of.get(f"torch_{impl}_{dev}"), groups_of.get(f"torch_{impl}_{dev}_float64")
                    if not a or not b:
                        continue
                    for g in a:
                        if g in b and a[g][0].size == b[g][0].size:
                            col.pinv(f"{dev}_{impl}/{key}/{g}", a[g][0], b[g][0], a[g][1], a[g][2], b[g][1], b[g][2])
    col.res["precision_invariance_note"] = "pairs: torch for_loop / foreach on CPU and CUDA, float32 vs float64 (env ka_main_f64)"
    return col.res


# ------------------------------------------------------------------------------------------------ schedulers / clip_amp / moe

def run_schedulers():
    import p2b_schedulers as SC
    import spec_training_program as ST
    from contract_v2 import scheduler_entry
    cands = {p.stem: load(p) for p in sorted(SC.CACHE.glob("*.pkl"))}
    col = Collector("schedulers", "specs/phase2/spec_training_program.py v0.1")
    for c in SC.conditions():
        cls = scheduler_entry(c)["classes"]
        for i, base in enumerate(SC.BASES):
            key = f"{c['id']}/{i}"
            if any(x.startswith(("D", "X")) for x in cls):
                col.res["spec_status"][key] = "refused / outside the contract: " + cls[0]
                continue
            n = c["steps"]
            total = 2 * n + 2
            vals = []
            for t in range(total + 1):
                if c["sched"] == "cosine_annealing":
                    vals.append(ST.cosine_annealing(Fr(base), Fr(base) / 10, t, n))
                elif c["sched"] == "linear_warmup_torch":
                    vals.append(ST.linear_lr(Fr(base), Fr(1, 10), Fr(1), n, t))
                else:
                    vals.append(ST.hf_cosine_with_warmup(Fr(base), t, SC.warmup_of(n), n))
            lo, hi = spec_arrays(vals)
            col.res["spec_status"][key] = "ok" + ("; SCH-A1 pending for T_cur >= T_max" if c["sched"] == "cosine_annealing" else "")
            for name, d in cands.items():
                r = d["res"].get((c["id"], i), {}).get("base")
                if not r or r["status"] != "ok":
                    continue
                m = min(len(r["lrs"]), lo.size)
                lim = n if c["sched"] == "cosine_annealing" else m      # SCH-A1 stopped at T_cur >= T_max
                row = {"lr_before_Tmax": compare(r["lrs"][:min(lim, m)], lo[:min(lim, m)], hi[:min(lim, m)], "float64")}
                if c["sched"] == "cosine_annealing":
                    row["lr_from_Tmax_pending"] = compare(r["lrs"][lim:m], lo[lim:m], hi[lim:m], "float64")
                col.res["candidates"][name][key] = row
    return col.res


def run_clip():
    import p2b_clip_amp as CL
    import spec_training_program as ST
    cands = {p.stem: load(p) for p in sorted(CL.CACHE.glob("*.pkl"))}
    col = Collector("clip_amp", "specs/phase2/spec_training_program.py v0.1")
    for c in CL.conditions():
        for seed in CL.SEEDS:
            key = f"{c['id']}/{seed}"
            if c["op"] == "clip_value":
                col.res["spec_status"][key] = "outside the contract (clip_grad_value_)"
                continue
            gs = CL.grads_for(c, seed)
            if not all(np.isfinite(g).all() for g in gs):
                col.res["spec_status"][key] = "non-finite gradients: documented scaling by the non-finite coefficient -> column 4 (recorded)"
                continue
            nt = math.inf if c["norm_type"] == "inf" else float(c["norm_type"])
            G = [[Fr(float(v)) for v in g.ravel()] for g in gs]
            fa = {}
            if c["op"] == "clip_norm":
                clipped, tn = ST.clip_grad_norm(G, Fr(CL.MAX_NORM), nt if nt != math.inf else float("inf"))
                fa["clipped"] = spec_arrays(clipped)
                fa["total_norm"] = spec_arrays([tn])
            else:
                st = ST.ScalerState()
                un, stepped = ST.scaler_step(st, [[v * st.scale for v in g] for g in G], False)
                clipped, tn = ST.clip_grad_norm(un, Fr(CL.MAX_NORM), nt if nt != math.inf else float("inf"))
                fa["unscaled"] = spec_arrays(un)
                fa["clipped"] = spec_arrays(clipped)
                fa["total_norm"] = spec_arrays([tn])
            col.res["spec_status"][key] = "ok"
            for name, d in cands.items():
                r = d["res"].get((c["id"], seed), {}).get("base")
                if not r or r["status"] != "ok":
                    continue
                o = r["outputs"]
                row = {}
                for k_ in fa:
                    if k_ in o:
                        kv = np.concatenate([np.asarray(a, float).ravel() for a in o[k_]]) if isinstance(o[k_], list) else np.asarray([o[k_]])
                        row[k_] = compare(kv, *fa[k_], d["meta"]["candidate"]["dtype"])
                col.res["candidates"][name][key] = row

            def co(n):
                r = cands.get(n, {}).get("res", {}).get((c["id"], seed), {}).get("base")
                if not r or r["status"] != "ok":
                    return None
                return {k_: (np.concatenate([np.asarray(a, float).ravel() for a in v]) if isinstance(v, list) else np.asarray([v]))
                        for k_, v in r["outputs"].items() if k_ in fa}
            pinv_named(col, key, [("cpu_torch_clip", co("torch_clip_cpu"), co("torch_clip_cpu_float64"))], fa, fa, list(fa))
    return col.res


def moe_job(args):
    cid, seed = args
    import p2b_small_families as SM
    import spec_moe as SMO
    c = {x["id"]: x for x in SM.plan("moe", SM.MO_KEYS)}[cid]
    xn, ln, Wn, cap = SM.moe_inputs(c, seed)
    W = frs(Wn)
    X = frs(xn)

    def expert(e):
        return lambda xv: [sum((xv[d_] * W[e][d_][j] for d_ in range(len(xv))), Fr(0)) for j in range(len(xv))]
    try:
        outs, rec = SMO.moe_forward(X, frs(ln), [expert(e) for e in range(c["experts"])], c["topk"], True, cap)
    except Exception as exc:  # noqa: BLE001
        return cid, seed, None, None, f"spec: {type(exc).__name__}: {str(exc)[:160]}"
    sets = [[sorted(s) for s in r["sets"]] for r in rec]
    return cid, seed, {"out": spec_arrays(outs)}, sets, "ok"


def run_moe():
    import p2b_small_families as SM
    cands = {p.stem.split("__", 1)[1]: load(p) for p in sorted(SM.CACHE.glob("moe__*.pkl"))}
    col = Collector("moe", "specs/phase2/spec_moe.py v0.1 (MOE-C1 renormalised, MOE-C3 drop in order of appearance)")
    jobs = [(c["id"], s) for c in SM.plan("moe", SM.MO_KEYS) for s in SM.SEEDS]
    with ProcessPoolExecutor(max_workers=24) as ex:
        for cid, seed, fa, sets, status in ex.map(moe_job, jobs):
            key = f"{cid}/{seed}"
            col.res["spec_status"][key] = status
            if fa is None:
                continue
            for name, d in cands.items():
                r = d["res"].get((cid, seed), {}).get("base")
                if not r or r["status"] != "ok":
                    continue
                o = r["outputs"]
                chosen = [sorted(int(e) for e in row) for row in np.asarray(o["order"])]
                set_ok = sum(1 for ch, valid in zip(chosen, sets) if ch in valid)
                dtype = d["meta"]["candidate"]["dtype"]
                row = {"topk_set_valid": {"tokens": len(chosen), "valid": set_ok}}
                if all(ch == valid[0] for ch, valid in zip(chosen, sets)):
                    row["out"] = compare(o["out"], *fa["out"], dtype)
                else:
                    row["out"] = {"note": "candidate chose another valid tie set (MOE-C2): output compared only through the set check"}
                row["aux"] = {"note": "MOE-C4 declared difference: the candidate's aux counts top-1 assignments; spec counts dispatched experts"}
                col.res["candidates"][name][key] = row

            def mo(n):
                r = cands.get(n, {}).get("res", {}).get((cid, seed), {}).get("base")
                if not r or r["status"] != "ok":
                    return None
                chosen = [sorted(int(e) for e in row) for row in np.asarray(r["outputs"]["order"])]
                return {"out": r["outputs"]["out"]} if all(ch == v[0] for ch, v in zip(chosen, sets)) else None
            pinv_named(col, key, [("cpu_ref_loop", mo("ref_loop_cpu_float32"), mo("ref_loop_cpu")),
                                  ("cuda_vectorised", mo("vectorised_scatter_cuda"), mo("vectorised_scatter_cuda_float64"))],
                       fa, fa, ["out"])
    return col.res


FAMILIES = {"embedding": run_embedding, "matmul_linear": lambda: run_basic("matmul_linear", matmul_job),
            "reductions": lambda: run_basic("reductions", reduction_job),
            "activations": lambda: run_basic("activations", activation_job),
            "gather_layout": lambda: run_basic("gather_layout", gather_job), "attention": run_attention,
            "rope": run_rope, "packing": run_packing, "optimizers": run_optimizers, "schedulers": run_schedulers,
            "clip_amp": run_clip, "moe": run_moe}


def summarize(res):
    per = {}
    for name, rows in res["candidates"].items():
        t = defaultdict(lambda: [0, 0, 0])
        conds = set()
        for key, row in rows.items():
            for k, c in row.items():
                if not isinstance(c, dict) or "elements" not in c:
                    continue
                t[k][0] += c["elements"]
                t[k][1] += c.get("beyond", 0)
                t[k][2] += c.get("nonfinite_column4", 0)
                if c.get("beyond"):
                    conds.add(key.rsplit("/", 1)[0])
        per[name] = {"by_output": {k: {"elements": v[0], "beyond": v[1], "nonfinite": v[2]} for k, v in t.items()},
                     "conditions_with_beyond": sorted(conds)}
    pi = defaultdict(int)
    for r in res["precision_invariance"].values():
        pi[r["condition"]] += 1
    st = defaultdict(int)
    for v in res["spec_status"].values():
        st[v.split(";")[0][:70]] += 1
    return {"per_candidate": per, "precision_invariance_conditions": dict(pi), "spec_status": dict(st), "no_f": dict(res["no_f"])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fam in sys.argv[1:]:
        t0 = time.time()
        try:
            res = FAMILIES[fam]()
        except Exception:  # noqa: BLE001
            print(fam, "FAILED", traceback.format_exc()[-2000:], flush=True)
            continue
        res["summary"] = summarize(res)
        res["seconds"] = round(time.time() - t0, 1)
        (OUT / f"{fam}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
        s = res["summary"]
        print(f"== {fam} ({res['seconds']} s) spec {s['spec_status']} pinv {s['precision_invariance_conditions']} no_f {s['no_f']}", flush=True)
        for n, v in s["per_candidate"].items():
            print(f"   {n:30s} " + " ".join(f"{k}:{x['beyond']}/{x['elements']}" + (f"(nf {x['nonfinite']})" if x['nonfinite'] else "")
                                         for k, x in v["by_output"].items()) + f"  conds {v['conditions_with_beyond'][:4]}", flush=True)


if __name__ == "__main__":
    main()
