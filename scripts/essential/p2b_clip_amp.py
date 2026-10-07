#!/usr/bin/env python3
"""2b family "clip_amp" (protocol v2 section 3; registry tier1.clip_amp): E and P for gradient clipping and GradScaler.
F stays closed until the spec is delivered.

Conditions: the coverage plan with the vacuous combinations merged (``norm_type`` has no effect on clip_value and on the
scaler step's own logic: (clip_value, inf / 1) and (scaler_step, inf / 1) coincide with their norm_type = 2 entries) -- 11
combinations -- plus two post-hoc conditions that the registered GradScaler property needs and the plan does not contain:
(scaler_step, with_inf), (scaler_step, with_nan).  x 3 seeds.  Parameters (5, 3), (7,), (2, 4) float32, gradients N(0, 1)
scaled to a total 2-norm of 2.5 (max_norm 1.0, clip_value 0.5); with_nan / with_inf put one NaN / +inf into the first
gradient; zero = all zero.  shards = 2: the parameters are split into two groups whose norms are computed separately.
scaler_step: SGD lr 1, GradScaler(init_scale 2^16): grads = scaler.scale(g); unscale_; clip_grad_norm_(max_norm, norm_type);
scaler.step; scaler.update.

P (pre-registered): clipping leaves gradients bitwise unchanged when the total norm <= c (variant ``loose``: max_norm 1e3)
and otherwise scales them to total norm c (relative 1e-6 + τ32: the documented coefficient is c / (norm + 1e-6));
clip_value clamps to [-c, c] (exact); shard merge -- the per-shard norms combined (sqrt of the sum of squares / max / sum)
equal the global norm; a non-finite step is skipped (parameters unchanged) and the scale halves.

    python scripts/essential/p2b_clip_amp.py run --env ka_main
    /data1/tzh/envs/liger/bin/python scripts/essential/p2b_clip_amp.py run --env liger
    python scripts/essential/p2b_clip_amp.py analyse
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/essential/p2b/clip_amp"
OUT = ROOT / "results/essential/phase2b/clip_amp"
SEEDS = (0, 1, 2)
SHAPES = [(5, 3), (7,), (2, 4)]
MAX_NORM, CLIP_VALUE, TAU32 = 1.0, 0.5, 2.0 ** -12


def conditions():
    plan = json.loads((ROOT / "results/essential/phase2b/coverage_plan.json").read_text())["tier1"]["clip_amp"]["conditions"]
    out, seen = [], {}
    for c in plan:
        c = dict(c)
        if c["op"] != "clip_norm" and c["norm_type"] != 2.0:
            c["merged_from_norm_type"] = c["norm_type"]
            c["norm_type"] = 2.0
        key = (c["op"], c["norm_type"], c["grads"], c["foreach"], c["shards"])
        if key in seen:
            seen[key]["high_risk"] |= bool(c.get("high_risk"))
            continue
        c["high_risk"] = bool(c.get("high_risk"))
        c["source"] = "coverage_plan"
        seen[key] = c
        out.append(c)
    for g in ("with_inf", "with_nan"):
        out.append({"op": "scaler_step", "norm_type": 2.0, "grads": g, "foreach": False, "shards": 1, "high_risk": False,
                    "source": "post-hoc (needed by the registered GradScaler property)"})
    for c in out:
        c["id"] = f"clip_{c['op']}_n{c['norm_type']}_{c['grads']}_{'fe' if c['foreach'] else 'loop'}_s{c['shards']}"
    return out


def _rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def grads_for(cond, seed):
    rng = _rng(f"clip/{cond['id']}/{seed}")
    gs = [rng.normal(0, 1, s) for s in SHAPES]
    tot = math.sqrt(sum(float((g ** 2).sum()) for g in gs))
    gs = [(g * 2.5 / tot).astype(np.float32).astype(np.float64) for g in gs]
    if cond["grads"] == "zero":
        gs = [np.zeros(s) for s in SHAPES]
    elif cond["grads"] == "with_nan":
        gs[0][0, 0] = np.nan
    elif cond["grads"] == "with_inf":
        gs[0][0, 0] = np.inf
    return gs


def nt(cond):
    return math.inf if cond["norm_type"] == "inf" else float(cond["norm_type"])


def candidates_for(env):
    if env == "ka_main":
        return [{"id": "torch_clip_cpu", "device": "cpu", "dtype": "float32"},
                {"id": "torch_clip_cuda", "device": "cuda", "dtype": "float32"},
                {"id": "torch_clip_cpu_float64", "device": "cpu", "dtype": "float64", "reference_only": True}]
    if env == "liger":
        return [{"id": "accelerate_clip", "device": "cuda", "dtype": "float32", "accelerate": True}]
    raise KeyError(env)


def run_one(cand, cond, seed, variant):
    dt, dev = getattr(torch, cand["dtype"]), cand["device"]
    params = [torch.zeros(s, dtype=dt, device=dev, requires_grad=True) for s in SHAPES]
    gs = grads_for(cond, seed)
    max_norm = 1e3 if variant == "loose" else MAX_NORM
    f = lambda t: t.detach().double().cpu().numpy().copy()     # noqa: E731  (copy: float64 CPU tensors would alias)
    res = {}
    if cond["op"] == "scaler_step":
        opt = torch.optim.SGD(params, lr=1.0)
        scaler = torch.amp.GradScaler(dev, init_scale=2.0 ** 16)
        for p, g in zip(params, gs):
            p.grad = scaler.scale(torch.tensor(g, dtype=dt, device=dev))
        scaler.unscale_(opt)
        res["unscaled"] = [f(p.grad) for p in params]
        tot = torch.nn.utils.clip_grad_norm_(params, max_norm, nt(cond), foreach=cond["foreach"])
        res["total_norm"] = float(tot)
        res["clipped"] = [f(p.grad) for p in params]
        scaler.step(opt)
        scaler.update()
        res["params_after"] = [f(p) for p in params]
        res["scale_after"] = float(scaler.get_scale())
        return res
    for p, g in zip(params, gs):
        p.grad = torch.tensor(g, dtype=dt, device=dev)
    if cond["op"] == "clip_value":
        torch.nn.utils.clip_grad_value_(params, CLIP_VALUE, foreach=cond["foreach"])
        res["clipped"] = [f(p.grad) for p in params]
        return res
    if cand.get("accelerate"):
        from accelerate import Accelerator
        acc = Accelerator(cpu=False)
        tot = acc.clip_grad_norm_(params, max_norm, norm_type=nt(cond))
    else:
        tot = torch.nn.utils.clip_grad_norm_(params, max_norm, nt(cond), foreach=cond["foreach"])
    res["total_norm"] = float(tot)
    res["clipped"] = [f(p.grad) for p in params]
    if cond["shards"] == 2:
        shard_norms = []
        for part in (gs[:2], gs[2:]):                        # the two shards' (unclipped) gradients
            qs = [torch.tensor(g, dtype=dt, device=dev) for g in part]
            shard_norms.append(float(torch.nn.utils.get_total_norm(qs, nt(cond), foreach=cond["foreach"])))
        res["shard_norms"] = shard_norms
    return res


def variants_for(cond):
    v = ["base"]
    if cond["op"] == "clip_norm" and cond["grads"] == "normal":
        v.append("loose")
    return v


def run(env):
    CACHE.mkdir(parents=True, exist_ok=True)
    for cand in candidates_for(env):
        res, t0 = {}, time.time()
        for cond in conditions():
            if cand.get("accelerate") and cond["op"] != "clip_norm":
                continue
            for seed in SEEDS:
                rec = {}
                for var in variants_for(cond):
                    try:
                        rec[var] = {"status": "ok", "outputs": run_one(cand, cond, seed, var)}
                    except Exception as exc:  # noqa: BLE001
                        rec[var] = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
                res[(cond["id"], seed)] = rec
        meta = {"candidate": cand, "torch": torch.__version__, "seconds": round(time.time() - t0, 2), "env": env}
        (CACHE / f"{cand['id']}.pkl").write_bytes(pickle.dumps({"meta": meta, "res": res}))
        st = defaultdict(int)
        for rec in res.values():
            for r in rec.values():
                st[r["status"]] += 1
        print(cand["id"], dict(st), meta["seconds"], "s")


REFERENCE = {"accelerate_clip": "torch_clip_cuda"}


def _flat(lst):
    return np.concatenate([np.asarray(a, float).ravel() for a in lst])


def norm_of(v, p):
    return float(np.max(np.abs(v))) if p == math.inf else float(np.sum(np.abs(v) ** p) ** (1 / p))


def analyse():
    data = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(CACHE.glob("*.pkl"))}
    conds = {c["id"]: c for c in conditions()}
    out = {"conditions": len(conds), "candidates": {}, "examples": []}
    for name, d in data.items():
        P, E, E64, conv = defaultdict(lambda: [0, 0]), [0, 0], [0, 0], defaultdict(int)
        for (cid, seed), rec in d["res"].items():
            cond = conds[cid]
            b = rec.get("base")
            if not b or b["status"] != "ok":
                out["examples"].append({"candidate": name, "condition": cid, "seed": seed, "error": b and b.get("reason")})
                continue
            o = b["outputs"]
            gs = grads_for(cond, seed)
            g = _flat(gs)
            p = nt(cond)
            finite = bool(np.isfinite(g).all())
            for tgt, acc in ((REFERENCE.get(name), E), ("torch_clip_cpu_float64", E64)):
                if tgt not in data or tgt == name:
                    continue
                rb = data[tgt]["res"].get((cid, seed), {}).get("base")
                if not rb or rb["status"] != "ok":
                    continue
                k, r = _flat(o["clipped"]), _flat(rb["outputs"]["clipped"])
                same = bool(np.all((np.abs(k - r) <= TAU32 * (1 + np.abs(r))) | (np.isnan(k) & np.isnan(r)) | (k == r)))
                acc[0] += 1
                acc[1] += int(not same)
                if not same:
                    out["examples"].append({"candidate": name, "vs": tgt, "condition": cid, "seed": seed})
            if cond["op"] == "clip_value":
                ok = bool(np.array_equal(_flat(o["clipped"]), np.clip(g, -CLIP_VALUE, CLIP_VALUE).astype(np.float32).astype(float)
                                         if name != "torch_clip_cpu_float64" else np.clip(g, -CLIP_VALUE, CLIP_VALUE), equal_nan=True))
                P["P_clip_value_clamps"][0] += 1
                P["P_clip_value_clamps"][1] += int(not ok)
            elif cond["op"] == "clip_norm":
                if not finite:
                    conv[f"{cond['grads']} (norm {cond['norm_type']}): total_norm={o['total_norm']}, clipped grads finite: "
                         f"{bool(np.isfinite(_flat(o['clipped'])).all())}"] += 1
                else:
                    total = norm_of(g, p)
                    after = norm_of(_flat(o["clipped"]), p)
                    ok_total = abs(o["total_norm"] - total) <= TAU32 * (1 + total)
                    if total > MAX_NORM:
                        ok = abs(after - MAX_NORM) <= 1e-6 * MAX_NORM + TAU32 * (1 + MAX_NORM)
                    else:
                        ok = bool(np.array_equal(_flat(o["clipped"]), g))
                    P["P_clip_norm_scales_to_c"][0] += 1
                    P["P_clip_norm_scales_to_c"][1] += int(not (ok and ok_total))
                    if not (ok and ok_total):
                        out["examples"].append({"candidate": name, "condition": cid, "seed": seed, "property": "P_clip_norm_scales_to_c",
                                                "total": total, "reported": o["total_norm"], "after": after})
                    lo = rec.get("loose")
                    if lo and lo["status"] == "ok":
                        same = bool(np.array_equal(_flat(lo["outputs"]["clipped"]), g))
                        P["P_clip_norm_leaves_small_unchanged"][0] += 1
                        P["P_clip_norm_leaves_small_unchanged"][1] += int(not same)
                    if "shard_norms" in o:
                        s1, s2 = o["shard_norms"]
                        merged = max(s1, s2) if p == math.inf else (s1 ** p + s2 ** p) ** (1 / p)
                        ok = abs(merged - o["total_norm"]) <= TAU32 * (1 + o["total_norm"])
                        P["P_shard_merge"][0] += 1
                        P["P_shard_merge"][1] += int(not ok)
            else:                                                      # scaler_step
                unscaled = _flat(o["unscaled"])
                if finite:
                    ok = bool(np.array_equal(unscaled, g)) and o["scale_after"] == 2.0 ** 16
                    stepped = not np.array_equal(_flat(o["params_after"]), np.zeros_like(g))
                    P["H_scaler_finite_step_unscaled_exactly"][0] += 1
                    P["H_scaler_finite_step_unscaled_exactly"][1] += int(not (ok and stepped))
                else:
                    skipped = bool(np.array_equal(_flat(o["params_after"]), np.zeros_like(g)))
                    halved = o["scale_after"] == 2.0 ** 15
                    P["P_scaler_nonfinite_skipped_scale_halves"][0] += 1
                    P["P_scaler_nonfinite_skipped_scale_halves"][1] += int(not (skipped and halved))
                    if not (skipped and halved):
                        out["examples"].append({"candidate": name, "condition": cid, "seed": seed,
                                                "property": "P_scaler_nonfinite_skipped_scale_halves", "skipped": skipped,
                                                "scale_after": o["scale_after"]})
        out["candidates"][name] = {"meta": {k: v for k, v in d["meta"].items() if k != "candidate"},
                                   "P": {k: f"{v[1]}/{v[0]}" for k, v in P.items()},
                                   "E": f"{E[1]}/{E[0]}" if E[0] else None, "E64": f"{E64[1]}/{E64[0]}" if E64[0] else None,
                                   "non_finite_conventions": dict(conv)}
        c = out["candidates"][name]
        print(f"{name:24s} P {c['P']}  E {c['E']}  E64 {c['E64']}")
        for k, v in conv.items():
            print(f"{'':24s} convention: {k} x{v}")
    for e in out["examples"][:20]:
        print("  ", e)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse"])
    ap.add_argument("--env", default="ka_main")
    a = ap.parse_args()
    run(a.env) if a.stage == "run" else analyse()
