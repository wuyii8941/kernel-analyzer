#!/usr/bin/env python3
"""P group additions of the closure (protocol v3 section 7): gradcheck, torch.func.jvp against the autograd VJP (dot
product), and the spec properties not already run as 2b pre-registered P (prop_bag_sum_linearity).

- gradcheck: float64 eager candidates (CPU, CUDA), fast mode (random projections), default tolerances; float32 kernels are
  not gradchecked (finite differences in float32 are not a test).
- jvp vs VJP: <J u, g> against <u, J^T g> with random u, g; |lhs - rhs| <= τ * (Σ|J u · g| + Σ|u · J^T g|),
  τ64 = 1e-9, τ32 = 2^-12; every candidate whose forward-mode AD runs (an op without a forward-mode formula -> unsupported).
- Points where the function is not differentiable (ReLU at 0, ties of amax / amin, small-integer and constant inputs into
  kinked ops) are recorded but not judged: one-sided choices of JVP and VJP may differ there (documented convention, D1).
- prop_bag_sum_linearity (spec_embedding): bag(w1 + w2) = bag(w1) + bag(w2) for mode = sum with per-sample weights.

Output results/closure/p_extra/<family>.json.

    python scripts/closure/p_extra.py matmul_linear reductions activations gather_layout normalization attention embedding rope moe
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in ("specs/phase2", "scripts/essential", "scripts/closure"):
    sys.path.insert(0, str(ROOT / p))

OUT = ROOT / "results/closure/p_extra"
TAU = {torch.float64: 1e-9, torch.float32: 2.0 ** -12}
KINK_VALUES = {"small_ints", "all_equal", "zeros"}


def cands_basic():
    return [("eager_cpu_float64", "cpu", torch.float64, False), ("eager_cuda_float64", "cuda", torch.float64, False),
            ("eager_cpu_float32", "cpu", torch.float32, False), ("eager_cuda_float32", "cuda", torch.float32, False),
            ("inductor_cuda_float32", "cuda", torch.float32, True)]


def jvp_vjp(fn, primals, seed):
    g_ = torch.Generator().manual_seed(seed)
    tangents = [torch.randn(p.shape, generator=g_, dtype=torch.float64).to(p.dtype).to(p.device) for p in primals]
    out, ju = torch.func.jvp(fn, tuple(primals), tuple(tangents))
    gout = torch.randn(out.shape, generator=g_, dtype=torch.float64).to(out.dtype).to(out.device)
    ps = [p.detach().clone().requires_grad_(True) for p in primals]
    o2 = fn(*ps)
    grads = torch.autograd.grad(o2, ps, gout, allow_unused=True)
    lhs_t = (ju.double() * gout.double())
    rhs_t = [(u.double() * (gr.double() if gr is not None else torch.zeros_like(u, dtype=torch.float64))) for u, gr in zip(tangents, grads)]
    lhs = float(lhs_t.sum())
    rhs = float(sum(r.sum() for r in rhs_t))
    scale = float(lhs_t.abs().sum()) + float(sum(r.abs().sum() for r in rhs_t))
    return lhs, rhs, scale


def check(fn, primals, dtype, do_gradcheck, kink, seed=0):
    rec = {}
    t0 = time.time()
    try:
        lhs, rhs, scale = jvp_vjp(fn, primals, seed)
        if not np.isfinite([lhs, rhs, scale]).all():
            rec["jvp_vjp"] = "non-finite (column 4 if f finite)"
        else:
            ok = abs(lhs - rhs) <= TAU[dtype] * max(scale, 1e-300)
            rec["jvp_vjp"] = ("pass" if ok else "fail") + (" (kink: not judged)" if kink else "")
            rec["jvp_vjp_rel"] = abs(lhs - rhs) / max(scale, 1e-300)
    except Exception as exc:  # noqa: BLE001
        rec["jvp_vjp"] = f"unsupported: {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
    rec["jvp_seconds"] = round(time.time() - t0, 3)
    if do_gradcheck and dtype == torch.float64:
        t0 = time.time()
        ps = [p.detach().clone().requires_grad_(True) for p in primals]
        try:
            ok = torch.autograd.gradcheck(fn, tuple(ps), fast_mode=True, raise_exception=False)
            rec["gradcheck"] = ("pass" if ok else "fail") + (" (kink: not judged)" if kink else "")
        except Exception as exc:  # noqa: BLE001
            rec["gradcheck"] = f"unsupported: {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
        rec["gradcheck_seconds"] = round(time.time() - t0, 3)
    return rec


# ------------------------------------------------------------------------------------------------ families

def basic(family):
    import p2b_basic as BA
    from contract_v2 import classify_condition
    fam = BA.FAMILIES[family]
    rows = {}
    for c in fam.conditions():
        cls = classify_condition(family, c)["classes"]
        if not any(x.startswith(("A", "C")) for x in cls):
            rows[c["id"]] = {"status": f"contract v2 {cls[0]}: not run"}
            continue
        d = fam.inputs(c, 0, "base")
        gi = [k for k in fam.grad_inputs(c) if d.get(k) is not None]
        kink = (c.get("values") in KINK_VALUES and (family == "reductions" and c["op"] in ("amax", "amin")
                                                   or family == "activations" and c["op"] == "relu")) or \
               (family == "activations" and c["op"] == "relu") or (family == "reductions" and c["op"] in ("amax", "amin"))
        if any(np.asarray(d[k]).size == 0 for k in gi):
            rows[c["id"]] = {"status": "empty input: no derivative to check"}
            continue
        per = {}
        for cid, dev, dt, compiled in cands_basic():
            lay = c.get("layout", "contiguous")
            fixed = {k: (None if v is None else (torch.as_tensor(v, dtype=torch.long, device=dev) if k == "idx"
                                                 else BA.layout(v, "contiguous", dt, dev))) for k, v in d.items() if k not in gi}

            def fn(*xs, fixed=fixed, lay=lay):
                t = dict(fixed)
                for k, x in zip(gi, xs):
                    if lay == "transposed" and x.dim() >= 2:
                        x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
                    t[k] = x
                return fam.run(c, t, "base")["out"]
            f = torch.compile(fn, dynamic=False) if compiled else fn
            primals = [BA.layout(d[k], "contiguous", dt, dev) for k in gi]
            if compiled:
                torch._dynamo.reset()
            per[cid] = check(f, primals, dt, not compiled, kink)
        rows[c["id"]] = {"status": "run", "kink": bool(kink), "candidates": per}
    return rows


def normalization():
    import p2b_normalization as NM
    rows = {}
    for c in NM.conditions():
        d = NM.make_inputs(c, 0, "base")
        per = {}
        for cid, dev, dt, compiled in cands_basic():
            rm, rv = (torch.tensor(d[k], dtype=dt, device=dev) for k in ("rm", "rv"))
            base = NM.norm_fn(c)

            def fn(x, w, b, rm=rm, rv=rv, base=base):
                return base(x, w, b, rm.clone(), rv.clone())
            f = torch.compile(fn, dynamic=False) if compiled else fn
            if compiled:
                torch._dynamo.reset()
            primals = [torch.tensor(d[k], dtype=dt, device=dev) for k in ("x", "w", "b")]
            per[cid] = check(f, primals, dt, not compiled, False)
        rows[c["id"]] = {"status": "run", "kink": False, "candidates": per}
    return rows


def attention():
    import p2b_attention as A
    cands = [("sdpa_math_cpu_float64", {"kind": "sdpa", "backend": "MATH"}, "cpu", torch.float64, True),
             ("sdpa_math_cuda_float64", {"kind": "sdpa", "backend": "MATH"}, "cuda", torch.float64, True),
             ("manual_eager_float64", {"kind": "manual"}, "cuda", torch.float64, True),
             ("sdpa_math_float32", {"kind": "sdpa", "backend": "MATH"}, "cuda", torch.float32, False),
             ("sdpa_efficient_float32", {"kind": "sdpa", "backend": "EFFICIENT_ATTENTION"}, "cuda", torch.float32, False),
             ("manual_eager_float32", {"kind": "manual"}, "cuda", torch.float32, False),
             ("inductor_attention_float32", {"kind": "manual", "compiled": True}, "cuda", torch.float32, False),
             ("flex_eager_float32", {"kind": "flex"}, "cuda", torch.float32, False)]
    rows = {}
    for c in A.conditions():
        d = A.make_inputs(c, 0, "base")
        a = d["allowed"]
        if not a.any(-1).all():
            rows[c["id"]] = {"status": "rows without an allowed key (ATT-A1 undefined): derivatives not checked"}
            continue
        per = {}
        for cid, cand, dev, dt, gc in cands:
            if cand["kind"] == "flex" and c["head_dim"] == 72:
                per[cid] = {"jvp_vjp": "not run (flex head_dim 72, #164931)"}
                continue
            impl = A.Impl(cand, c)
            if cand.get("compiled"):
                torch._dynamo.reset()

            def fn(q, k, v, impl=impl):
                return impl(q, k, v, a, c["gqa"])
            primals = [torch.tensor(d[x], dtype=dt, device=dev) for x in ("q", "k", "v")]
            per[cid] = check(fn, primals, dt, gc, False)
        rows[c["id"]] = {"status": "run", "kink": False, "candidates": per}
    return rows


def embedding():
    import p2b_embedding as EM
    rows = {}
    for c in EM.conditions():
        if c["max_norm"]:
            rows[c["id"]] = {"status": "max_norm renormalises the weight in place: derivative checks not applicable"}
            continue
        d = EM.make_inputs(c, 0)
        psw_on = c["op"] == "embedding_bag" and c["per_sample_weights"]
        per = {}
        for cid, dev, dt, compiled in cands_basic():
            base = EM.emb_fn(c, "base")
            idx = torch.tensor(d["idx"], dtype=torch.long, device=dev)
            off = None if d["offsets"] is None else torch.tensor(d["offsets"], dtype=torch.long, device=dev)
            psw0 = torch.tensor(d["psw"], dtype=dt, device=dev)

            def fn(w, *rest, base=base, idx=idx, off=off, psw0=psw0):
                return base(w, idx, off, rest[0] if rest else psw0)
            f = torch.compile(fn, dynamic=False) if compiled else fn
            if compiled:
                torch._dynamo.reset()
            primals = [torch.tensor(d["w"], dtype=dt, device=dev)] + ([psw0.clone()] if psw_on else [])
            per[cid] = check(f, primals, dt, not compiled, False)
        rec = {"status": "run", "kink": False, "candidates": per}
        if psw_on and c["mode"] == "sum":
            rec["prop_bag_sum_linearity"] = bag_linearity(c, d)
        rows[c["id"]] = rec
    return rows


def bag_linearity(c, d):
    import p2b_embedding as EM
    out = {}
    rng = np.random.default_rng(7)
    w1 = rng.normal(0, 1, np.asarray(d["psw"]).shape).astype(np.float32).astype(float)
    w2 = rng.normal(0, 1, w1.shape).astype(np.float32).astype(float)
    for cid, dev, dt, compiled in cands_basic():
        fn = EM.emb_fn(c, "base")
        if compiled:
            torch._dynamo.reset()
            fn = torch.compile(fn, dynamic=False)
        W = torch.tensor(d["w"], dtype=dt, device=dev)
        idx = torch.tensor(d["idx"], dtype=torch.long, device=dev)
        off = torch.tensor(d["offsets"], dtype=torch.long, device=dev)
        t = lambda a: torch.tensor(a, dtype=dt, device=dev)   # noqa: E731
        with torch.no_grad():
            a = fn(W, idx, off, t(w1 + w2)).double()          # w1 + w2 exact in float64; rounded once into dt
            b = fn(W, idx, off, t(w1)).double() + fn(W, idx, off, t(w2)).double()
        tau = TAU[dt]
        rel = float(((a - b).abs() / (1 + b.abs())).max())
        out[cid] = {"max_rel": rel, "verdict": "pass" if rel <= 4 * tau else "beyond 4τ"}
    return out


def rope():
    import p2b_small_families as SM
    cands = [("ref_torch_float64", {"lib": "ref"}, "cpu", torch.float64, True),
             ("ref_torch_cuda_float64", {"lib": "ref"}, "cuda", torch.float64, True),
             ("ref_torch", {"lib": "ref"}, "cuda", torch.float32, False),
             ("hf_apply_rotary", {"lib": "hf"}, "cuda", torch.float32, False)]
    rows = {}
    for c in SM.plan("rope", SM.RO_KEYS):
        rng = SM._rng(f"ro/{c['id']}/0")
        L, H, dd = 11, 4, c["head_dim"]
        q = rng.normal(0, 1, (2, H, L, dd))
        k = rng.normal(0, 1, (2, H, L, dd))
        pos = np.arange(L) + c["offset"]
        per = {}
        for cid, cand, dev, dt, gc in cands:
            def fn(qq, kk, cand=cand):
                a, b = SM.rope_call(cand, c, qq, kk, pos)
                return torch.cat([a.reshape(-1), b.reshape(-1)])
            try:
                per[cid] = check(fn, [torch.tensor(q, dtype=dt, device=dev), torch.tensor(k, dtype=dt, device=dev)], dt, gc, False)
            except Exception as exc:  # noqa: BLE001
                per[cid] = {"jvp_vjp": f"not run: {type(exc).__name__}: {exc}"[:200]}
        rows[c["id"]] = {"status": "run", "kink": False, "candidates": per}
    return rows


def moe():
    import p2b_small_families as SM
    cands = [("ref_loop_cpu", "loop", "cpu", torch.float64, True), ("ref_loop_cpu_float32", "loop", "cpu", torch.float32, False),
             ("vectorised_scatter_cuda_float64", "vectorised", "cuda", torch.float64, True),
             ("vectorised_scatter_cuda", "vectorised", "cuda", torch.float32, False),
             ("vectorised_compiled", "compiled", "cuda", torch.float32, False)]
    rows = {}
    for c in SM.plan("moe", SM.MO_KEYS):
        xn, ln, Wn, cap = SM.moe_inputs(c, 0)
        per = {}
        for cid, impl, dev, dt, gc in cands:
            logits = torch.tensor(ln, dtype=dt, device=dev)
            if impl == "loop":
                base = SM.moe_loop
            elif impl == "compiled":
                torch._dynamo.reset()
                base = torch.compile(SM.moe_vectorised)
            else:
                base = SM.moe_vectorised

            def fn(x, W, base=base, logits=logits):
                return base(c, x, logits, W, cap)[0]
            primals = [torch.tensor(xn, dtype=dt, device=dev), torch.tensor(Wn, dtype=dt, device=dev)]
            per[cid] = check(fn, primals, dt, gc, False)
        rows[c["id"]] = {"status": "run", "kink": False, "candidates": per,
                         "note": "derivatives in x and the expert weights; routing logits fixed (top-k selection is piecewise constant)"}
    return rows


FAMILIES = {"matmul_linear": lambda: basic("matmul_linear"), "reductions": lambda: basic("reductions"),
            "activations": lambda: basic("activations"), "gather_layout": lambda: basic("gather_layout"),
            "normalization": normalization, "attention": attention, "embedding": embedding, "rope": rope, "moe": moe}


def tally(rows):
    t = defaultdict(lambda: defaultdict(int))
    for r in rows.values():
        for cid, rec in (r.get("candidates") or {}).items():
            for k in ("gradcheck", "jvp_vjp"):
                if k in rec:
                    t[f"{cid}/{k}"][rec[k].split(":")[0]] += 1
    return {k: dict(v) for k, v in t.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fam in sys.argv[1:]:
        t0 = time.time()
        try:
            rows = FAMILIES[fam]()
        except Exception:  # noqa: BLE001
            print(fam, "FAILED", traceback.format_exc()[-2000:], flush=True)
            continue
        out = {"family": fam, "seconds": round(time.time() - t0, 1), "tally": tally(rows), "conditions": rows}
        (OUT / f"{fam}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
        print(f"== {fam} ({out['seconds']} s)", flush=True)
        for k, v in out["tally"].items():
            print(f"   {k:48s} {v}", flush=True)


if __name__ == "__main__":
    main()
