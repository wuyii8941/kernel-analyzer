#!/usr/bin/env python3
"""Phase-1 supplement S1 and phase-2 item C (docs/protocol_essential_bugs_phase2_20261007.md).

S1: the implementation's own forward/backward consistency, <v, J u> from torch.func.jvp against <B(v), u> from autograd,
    random integer directions, on every phase-1 condition and seed, eager CPU float64 and eager CUDA float32.  Ops without
    forward-mode AD (index_reduce, scatter_reduce prod) are "not applicable".
C:  lightweight methods on the known errors - gradcheck, the JVP/VJP dot product, the pre-registered P, F, FR - with whether
    each detects, the manual inputs it needs (a written f, TTIR), and its run time.

    python scripts/essential/methods_compare.py s1
    python scripts/essential/methods_compare.py c
"""
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidates as K  # noqa: E402
import common  # noqa: E402
import run_phase1 as R  # noqa: E402

ROOT = common.ROOT
OUT = ROOT / "results/essential/phase2a"
TOL = {torch.float64: 1e-9, torch.float32: 2.0 ** -12}


def _fn_and_inputs(family, cond, inp, device, dtype, seed):
    """(f, primal tensors) for the base run of one condition, f a function of the differentiable inputs only."""
    if family == "ce":
        logits = K._t(inp["logits"], device, dtype)
        target = (K._t(inp["target"], device, dtype) if cond["target"] == "prob"
                  else torch.as_tensor(inp["target"], dtype=torch.long, device=device))
        weight = None if inp["weights"] is None else K._t(inp["weights"], device, dtype)
        kw = dict(ignore_index=cond["ignore_index"], reduction=cond["reduction"], label_smoothing=float(cond["eps"]))
        return (lambda x: F.cross_entropy(x, target, weight=weight, **kw)), (logits,)
    if family == "pool":
        x = K._t(inp["x"][None], device, dtype, cond.get("layout", "contiguous"))
        nd = cond["nd"]
        if cond["op"] == "avg_pool":
            if nd == 1 and cond["divisor_override"] is not None:
                return None, None
            pool = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[nd]
            kw = dict(kernel_size=cond["kernel"], stride=cond["stride"], padding=cond["padding"],
                      ceil_mode=cond["ceil_mode"], count_include_pad=cond["count_include_pad"])
            if nd > 1:
                kw["divisor_override"] = cond["divisor_override"]
        else:
            pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[nd]
            kw = dict(kernel_size=cond["kernel"], stride=cond["stride"], padding=cond["padding"],
                      dilation=cond["dilation"], ceil_mode=cond["ceil_mode"])
        return (lambda x: pool(x, **kw)), (x,)
    s = K._t(inp["self"], device, dtype, cond.get("layout", "contiguous"))
    src = K._t(inp["source"], device, dtype, cond.get("layout", "contiguous"))
    idx = torch.as_tensor(inp["index"], dtype=torch.long, device=device)
    dim = inp["dim"]
    if cond["op"] == "index_add":
        return (lambda a, b: torch.index_add(a, dim, idx, b, alpha=cond["alpha"])), (s, src)
    if cond["op"] == "index_reduce":
        return (lambda a, b: torch.index_reduce(a, dim, idx, b, cond["reduce"], include_self=cond["include_self"])), (s, src)
    return (lambda a, b: torch.scatter_reduce(a, dim, idx, b, cond["reduce"], include_self=cond["include_self"])), (s, src)


def jvp_vjp(family, cond, inp, device, dtype, seed):
    """{'status', 'jvp_dot', 'vjp_dot', 'consistent'} for one input."""
    f, xs = _fn_and_inputs(family, cond, inp, device, dtype, seed)
    if f is None:
        return {"status": "unsupported"}
    if any(not torch.isfinite(x).all() for x in xs):
        return {"status": "non-finite inputs"}
    g = torch.Generator().manual_seed(1000 + seed)
    us = tuple(torch.randint(-3, 4, x.shape, generator=g).to(device=device, dtype=dtype) for x in xs)
    try:
        out, jv = torch.func.jvp(f, xs, us)
    except Exception as exc:  # noqa: BLE001
        return {"status": "forward-mode not supported", "reason": f"{type(exc).__name__}: {str(exc)[:120]}"}
    v = torch.randint(-3, 4, out.shape, generator=g).to(device=device, dtype=dtype)
    xg = tuple(x.detach().requires_grad_(True) for x in xs)
    o2 = f(*xg)
    grads = torch.autograd.grad(o2, xg, v, allow_unused=True)
    jd = float((v.double() * jv.double()).sum())
    vd = float(sum((0.0 if gr is None else (gr.double() * u.double()).sum()) for gr, u in zip(grads, us)))
    scale = 1.0 + float((v.double().abs() * jv.double().abs()).sum())
    ok = abs(jd - vd) <= TOL[dtype] * scale
    return {"status": "ok", "jvp_dot": jd, "vjp_dot": vd, "consistent": bool(ok)}


def s1():
    """incremental (.cache/essential/s1_rows.pkl); a CUDA error is recorded for its input and the process exits 3 so a
    fresh process continues (a device-side assert loses the CUDA context)."""
    import pickle
    cache = R.CACHE / "s1_rows.pkl"
    rows = pickle.loads(cache.read_bytes()) if cache.exists() else []
    done = {(r["family"], r["condition"], r["seed"], r["device"]) for r in rows}
    t0 = time.time()
    for fam in ("ce", "pool", "index"):
        conds, make = R.FAMILIES[fam]
        for cond in conds():
            for seed in R.C.SEEDS:
                inp = make(cond, seed)
                for dev, dt in (("cpu", torch.float64), ("cuda", torch.float32)):
                    if (fam, cond["id"], seed, dev) in done:
                        continue
                    try:
                        r = jvp_vjp(fam, cond, inp, dev, dt, seed)
                        if dev == "cuda":
                            torch.cuda.synchronize()
                    except Exception as exc:  # noqa: BLE001
                        r = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}
                    spec = R.load_spec(fam, cond["id"], seed, "base", "float64") if fam != "ce" else None
                    ties = bool(spec and (spec.get("ties") or any(
                        len(st) > 1 for rd in spec.get("readings", {}).values() if isinstance(rd, dict) for st in rd.get("sets", []))))
                    rows.append({"family": fam, "condition": cond["id"], "op": cond["op"], "reduce": cond.get("reduce"),
                                 "seed": seed, "device": dev, "ties": ties, **r})
                    if r["status"] == "error" and ("CUDA" in r["reason"] or "Accelerator" in r["reason"]):
                        cache.write_bytes(pickle.dumps(rows))
                        print("CUDA error at", fam, cond["id"], seed, "- restarting", flush=True)
                        sys.exit(3)
            if len(rows) % 300 < 6:
                cache.write_bytes(pickle.dumps(rows))
        cache.write_bytes(pickle.dumps(rows))
        print(fam, "done", round(time.time() - t0), "s", flush=True)
    s1_summary(rows)


def s1_summary(rows=None):
    """non-finite dot products (CE-A3: undefined loss) are 'undefined', not inconsistent; known upstream classes noted."""
    import pickle
    if rows is None:
        rows = pickle.loads((R.CACHE / "s1_rows.pkl").read_bytes())
    for r in rows:
        if r["status"] == "ok" and not (np.isfinite(r["jvp_dot"]) and np.isfinite(r["vjp_dot"])):
            r["status"], r["consistent"] = "undefined (non-finite)", None
    summ = {}
    for r in rows:
        k = (r["family"], r["device"], "ties" if r["ties"] else "no_ties")
        d = summ.setdefault("|".join(k), {})
        key = r["status"] if r["status"] != "ok" else ("consistent" if r["consistent"] else "INCONSISTENT")
        d[key] = d.get(key, 0) + 1
    incons = [r for r in rows if r.get("consistent") is False]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s1_jvp_vjp.json").write_text(json.dumps({"summary": summ, "inconsistent": incons[:300],
                                                       "inconsistent_count": len(incons)}, indent=1) + "\n")
    for k, v in sorted(summ.items()):
        print(k, v)
    by = {}
    for r in incons:
        key = (r["family"], r["op"], r["reduce"], r["device"], r["ties"])
        by[key] = by.get(key, 0) + 1
    print("inconsistent by op:", by)




# ------------------------------------------------------------------------------------------------ item C

def _timed(fn):
    t = time.time()
    r = fn()
    return r, round(time.time() - t, 2)


def _gradcheck(f, xs):
    try:
        return bool(torch.autograd.gradcheck(f, xs, raise_exception=False)), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:160]}"


def c():
    import pickle
    from classify import load_raw  # noqa: F401
    res = {"note": "post-hoc comparison (phase-2 item C); the pre-registered phase-1 P result (28/50 for B020) is unchanged",
           "methods": {"gradcheck": {"needs_f": False, "needs_ttir": False},
                       "jvp_vjp_dot": {"needs_f": False, "needs_ttir": False, "limit": "eager only; ops with forward-mode AD"},
                       "P_preregistered": {"needs_f": False, "needs_ttir": False},
                       "F": {"needs_f": True, "needs_ttir": False},
                       "FR": {"needs_f": True, "needs_ttir": True, "limit": "Triton candidates only"}},
                "targets": {}}
    cls = {fam: json.load(gzip.open(ROOT / f"results/essential/phase1/classification_{fam}.json.gz", "rt")) for fam in ("pool", "index")}
    s1rows = pickle.loads((R.CACHE / "s1_rows.pkl").read_bytes())
    w5 = json.loads((OUT / "w5_review.json").read_text())["gradcheck_on_B020"]["rows"]

    # B020: the 50 conditions where eager CPU float64's backward deviates
    b020 = [r for r in cls["index"]["records"] if r["candidates"]["eager_cpu64"]["class"]["bwd"]["class"].startswith("deviates")]
    ids = {r["condition"]["id"] for r in b020}
    gc = {cid: any(not x["gradcheck_passes"] for x in w5 if x["condition"] == cid) for cid in ids}
    scat = {r["condition"]["id"] for r in b020 if r["condition"]["op"] == "scatter_reduce"}
    s1_hit = {cid: any(x.get("consistent") is False for x in s1rows if x["condition"] == cid and x["device"] == "cpu") for cid in scat}
    p_hit = {r["condition"]["id"]: r["candidates"]["eager_cpu64"]["P"]["per_property"].get("tie_subgradient_set", {}).get("violated", 0) > 0
             for r in b020}
    fr_hit = {}
    for r in b020:
        fr = (r["candidates"].get("inductor_cuda32") or {}).get("FR") or {}
        if fr.get("status") == "ok":
            fr_hit[r["condition"]["id"]] = any(o.get("semantic_vs_main") for o in fr["outputs"].values() if o.get("phase") == "bwd")
    res["targets"]["B020"] = {"conditions": len(ids),
                              "gradcheck": f"{sum(gc.values())}/{len(gc)}",
                              "jvp_vjp_dot": f"{sum(s1_hit.values())}/{len(s1_hit)} (scatter_reduce only; index_reduce has no forward-mode AD: "
                                             f"{len(ids) - len(scat)} conditions not applicable)",
                              "P_preregistered": f"{sum(p_hit.values())}/{len(p_hit)}",
                              "F": f"{len(ids)}/{len(ids)}",
                              "FR": f"{sum(fr_hit.values())}/{len(fr_hit)} (Inductor backward kernels)"}

    # B021: pool654 / pool655 (padding-only window) on eager CPU float64, batch of 2 channels
    b021 = {}
    for cid in ("pool654", "pool655"):
        cond = {c["id"]: c for c in R.FAMILIES["pool"][0]()}[cid]
        x = torch.tensor(R.FAMILIES["pool"][1](cond, 0)["x"][None], dtype=torch.float64, requires_grad=True)
        f = lambda t: F.max_pool1d(t, 2, 1, 1, dilation=2)  # noqa: E731
        (ok, err), sec = _timed(lambda: _gradcheck(f, (x,)))
        jv = [x for x in s1rows if x["condition"] == cid]
        b021[cid] = {"gradcheck_passes": ok, "gradcheck_error": err, "seconds": sec,
                     "jvp_vjp_status": sorted({f"{r['device']}: {r['status']}" for r in jv})}
    rec = {r["condition"]["id"]: r for r in cls["pool"]["records"]}
    res["targets"]["B021"] = {"cases": b021,
                              "gradcheck": f"{sum(v['gradcheck_passes'] is False for v in b021.values())}/2 detect",
                              "jvp_vjp_dot": "forward-mode AD raises on CPU and hits a device-side assert on CUDA (the out-of-range index): "
                                             "seen as an error, not as an inconsistency",
                              "P_preregistered": "/".join(str(rec[c]["candidates"]["eager_cpu64"]["P"]["per_property"].get("grad_mass", {}).get("violated", 0) > 0)
                                                        for c in ("pool654", "pool655")),
                              "F": "2/2", "FR": "not applicable (eager CPU is not Triton)"}

    # Inductor avg_pool backward (overhang divisor): gradcheck through the compiled function, CUDA float64
    avg = [r for r in cls["pool"]["records"] if r["candidates"].get("inductor_cuda32", {}).get("class", {}).get("bwd", {})
           .get("category", "").startswith("reading_difference")]
    hits, secs = 0, 0.0
    for r in avg:
        cond = r["condition"]
        nd = cond["nd"]
        pool = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[nd]
        kw = dict(kernel_size=cond["kernel"], stride=cond["stride"], padding=cond["padding"], ceil_mode=cond["ceil_mode"],
                  count_include_pad=cond["count_include_pad"])
        if nd > 1:
            kw["divisor_override"] = cond["divisor_override"]
        torch._dynamo.reset()
        cf = torch.compile(lambda t: pool(t, **kw), dynamic=False)
        x = torch.tensor(R.FAMILIES["pool"][1](cond, 0)["x"][None], dtype=torch.float64, device="cuda", requires_grad=True)
        (ok, err), sec = _timed(lambda: _gradcheck(cf, (x,)))
        hits += int(ok is False)
        secs += sec
    p_avg = sum(r["candidates"]["inductor_cuda32"]["P"]["per_property"].get("adjoint", {}).get("violated", 0) > 0 for r in avg)
    fr_avg = sum(any(o.get("semantic_vs_main") for o in (r["candidates"]["inductor_cuda32"].get("FR") or {}).get("outputs", {}).values()
                     if o.get("phase") == "bwd") for r in avg)
    res["targets"]["inductor_avg_pool_backward"] = {"conditions": len(avg), "gradcheck_compiled": f"{hits}/{len(avg)}",
                                                    "gradcheck_seconds_total": round(secs, 1),
                                                    "jvp_vjp_dot": "not applicable (no forward-mode AD through compiled graphs)",
                                                    "P_preregistered": f"{p_avg}/{len(avg)} (adjoint)", "F": f"{len(avg)}/{len(avg)}",
                                                    "FR": f"{fr_avg}/{len(avg)}"}

    # B016: the three writings of the W7 recall, gradcheck through the compiled function (CUDA float64)
    b016 = {}
    dev = "cuda"
    for name, fn, mk in (
            ("count", lambda x: torch.zeros(1, device=dev, dtype=x.dtype).scatter_add(0, torch.zeros(5, dtype=torch.long, device=dev), x + 1),
             lambda: (torch.zeros(5, device=dev, dtype=torch.float64, requires_grad=True),)),
            ("const_index", lambda x: torch.zeros(4, device=dev, dtype=x.dtype).index_add(0, torch.arange(30, device=dev) // 1000, x.cos()),
             lambda: (torch.randn(30, device=dev, dtype=torch.float64, generator=torch.Generator(device=dev).manual_seed(0), requires_grad=True),)),
            ("mean_bwd", lambda t, s: t.scatter_reduce(0, torch.zeros(3, dtype=torch.long, device=dev), s, "mean", include_self=True),
             lambda: (torch.randn(1, device=dev, dtype=torch.float64, requires_grad=True), torch.randn(3, device=dev, dtype=torch.float64, requires_grad=True)))):
        torch._dynamo.reset()
        cf = torch.compile(fn, dynamic=False)
        (ok, err), sec = _timed(lambda: _gradcheck(cf, mk()))
        b016[name] = {"gradcheck_passes": ok, "error": err, "seconds": sec}
    res["targets"]["B016"] = {"cases": b016,
                              "gradcheck_compiled": f"{sum(v['gradcheck_passes'] is False for v in b016.values())}/3 detect "
                                                    "(a wrong forward whose derivative is right - 'count' - passes)",
                              "jvp_vjp_dot": "not applicable (compiled)", "P_preregistered": "3/3 (W7)", "F": "3/3 (W7)", "FR": "3/3 (W7)"}
    res["targets"]["HF_grad_accum"] = {
        "gradcheck": "cannot see it: the per-micro-batch normalisation is a forward-level contract; backward is the exact derivative "
                     "of the (wrong) objective", "jvp_vjp_dot": "cannot see it (same reason)",
        "E_accum_vs_not": "4.45.2: yes for unequal token counts; 4.57.3: no (shared)", "P_split_invariance": "4.45.2: yes; 4.57.3: no",
        "F": "yes in both", "FR": "not applicable (not a Triton kernel)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c_methods.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cases"} for k, v in res["targets"].items()}, indent=1))


if __name__ == "__main__":
    {"s1": s1, "s1-summary": s1_summary, "c": c}[sys.argv[1]]()
