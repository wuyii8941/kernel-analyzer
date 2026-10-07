#!/usr/bin/env python3
"""2b family "embedding" (protocol v2 section 3; registry tier1.embedding): E and P, forward and backward.  F and mode-B FR
stay closed until the spec is delivered.

Conditions: the coverage plan's 13 combinations.  The plan generator took ``op = embedding`` as the first level for the
single-factor boundaries of ``mode``, ``bags`` and ``per_sample_weights``, which exist only for embedding_bag; those three
boundaries are run on embedding_bag (protocol v2 deviation 9).  Table 20 x 8; embedding indices (3, 7); embedding_bag 15
indices with offsets: normal [0, 4, 9, 12], empty [0, 4, 4, 9] (bag 1 empty), single [0, 1, ..., 14].  Indices always contain
duplicates and, when padding_idx = 0, the index 0 (for embedding_bag with padding_idx, bag 2 consists of padding only).
Outputs: out, dweight (random upstream), dpsw (per_sample_weights), weight_after (max_norm renormalises in place).

P (pre-registered): the padding_idx row gets zero gradient; scale_grad_by_freq divides each row's gradient by the row's
frequency in the batch (variant ``unscaled``); embedding_bag(mode=sum) equals the per-bag sum of F.embedding rows (variant
``via_embedding``; mean and max likewise, post-hoc); empty bags give zero (sum / mean).  Post-hoc: max_norm -- output rows
have norm <= max_norm, referenced rows above max_norm are rescaled to it, unreferenced rows untouched.

    python scripts/essential/p2b_embedding.py run --env ka_main
    PYTHONPATH=.cache/pylibs/nightly_np /data1/tzh/envs/pt_nightly_cu126/bin/python scripts/essential/p2b_embedding.py run --env nightly
    python scripts/essential/p2b_embedding.py analyse
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/essential/p2b/embedding"
OUT = ROOT / "results/essential/phase2b/embedding"
SEEDS = (0, 1, 2)
NUM, DIM = 20, 8
TAU = {"float32": 2.0 ** -12, "float64": 1e-9}
BAG_ONLY = ("mode", "bags", "per_sample_weights")
OFFSETS = {"normal": [0, 4, 9, 12], "empty": [0, 4, 4, 9], "single": list(range(15))}


def conditions():
    plan = json.loads((ROOT / "results/essential/phase2b/coverage_plan.json").read_text())["tier1"]["embedding"]["conditions"]
    base = plan[0]
    out, seen = [], {}
    for c in plan:
        c = dict(c)
        moved = [k for k in BAG_ONLY if c[k] != base[k]]
        if c["op"] == "embedding" and moved:
            c["op"] = "embedding_bag"
            c["moved_to_embedding_bag"] = moved
        key = tuple(c[k] for k in ("op", "padding_idx", "max_norm", "scale_grad_by_freq", "mode", "bags", "per_sample_weights"))
        if key in seen:
            seen[key]["high_risk"] |= bool(c.get("high_risk"))
            continue
        c["high_risk"] = bool(c.get("high_risk"))
        parts = [c["op"], f"pad{c['padding_idx']}", f"mn{c['max_norm']}", "freq" if c["scale_grad_by_freq"] else "nofreq"]
        if c["op"] == "embedding_bag":
            parts += [c["mode"], c["bags"], "psw" if c["per_sample_weights"] else "nopsw"]
        c["id"] = "emb_" + "_".join(parts)
        seen[key] = c
        out.append(c)
    return out


def _rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def make_inputs(cond, seed):
    rng = _rng(f"emb/{cond['id']}/{seed}")
    w = rng.normal(0, 1, (NUM, DIM)) * (2.0 if cond["max_norm"] else 1.0)     # max_norm: most rows above 1
    w = w.astype(np.float32).astype(np.float64)
    if cond["op"] == "embedding":
        idx = rng.integers(1, 9, (3, 7))                    # small range: duplicates
        if cond["padding_idx"] is not None:
            idx[0, 2] = idx[1, 5] = idx[2, 0] = cond["padding_idx"]
        offsets = None
    else:
        idx = rng.integers(1, 9, 15)
        offsets = OFFSETS[cond["bags"]]
        if cond["padding_idx"] is not None:
            idx[offsets[2]:offsets[3]] = cond["padding_idx"]   # bag 2 is padding only
            idx[1] = cond["padding_idx"]                       # and one padding entry inside bag 0
    psw = rng.normal(1, 0.5, idx.shape).astype(np.float32).astype(np.float64)
    nbags = None if offsets is None else len(offsets)
    up_shape = idx.shape + (DIM,) if cond["op"] == "embedding" else (nbags, DIM)
    g = rng.normal(0, 1, up_shape).astype(np.float32).astype(np.float64)
    return {"w": w, "idx": idx, "offsets": offsets, "psw": psw, "g": g}


def variants_for(cond):
    v = ["base"]
    if cond["scale_grad_by_freq"]:
        v.append("unscaled")
    if cond["op"] == "embedding_bag" and cond["max_norm"] is None:
        v.append("via_embedding")
    return v


# ------------------------------------------------------------------------------------------------ candidates

def candidates_for(env):
    if env == "ka_main":
        out = [{"id": f"eager_{d}_{t}", "device": d, "dtype": t} for d, t in
               (("cpu", "float64"), ("cuda", "float64"), ("cpu", "float32"), ("cuda", "float32"))]
        return out + [{"id": "inductor_cuda_float32", "device": "cuda", "dtype": "float32", "compiled": True}]
    if env == "nightly":
        return [{"id": "nightly_eager_cuda_float32", "device": "cuda", "dtype": "float32"},
                {"id": "nightly_inductor_cuda_float32", "device": "cuda", "dtype": "float32", "compiled": True}]
    raise KeyError(env)


REFERENCE = {"inductor_cuda_float32": "eager_cuda_float32", "nightly_inductor_cuda_float32": "nightly_eager_cuda_float32",
             "nightly_eager_cuda_float32": "eager_cuda_float32", "eager_cuda_float64": "eager_cpu_float64"}


def emb_fn(cond, variant):
    pad, mn = cond["padding_idx"], cond["max_norm"]
    freq = cond["scale_grad_by_freq"] and variant != "unscaled"
    if cond["op"] == "embedding":
        return lambda w, idx, offsets, psw: F.embedding(idx, w, padding_idx=pad, max_norm=mn, scale_grad_by_freq=freq)
    psw_on = cond["per_sample_weights"]
    if variant == "via_embedding":
        def f(w, idx, offsets, psw):
            rows = F.embedding(idx, w)
            if psw_on:
                rows = rows * psw[:, None]
            bounds = list(offsets.tolist()) + [idx.shape[0]]
            outs = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                keep = [j for j in range(a, b) if pad is None or int(idx[j]) != pad]
                r = rows[keep] if keep else rows[:0]
                if cond["mode"] == "sum":
                    outs.append(r.sum(0))
                elif cond["mode"] == "mean":
                    outs.append(r.mean(0) if keep else r.sum(0))
                else:
                    outs.append(r.max(0).values if keep else torch.zeros(DIM, dtype=w.dtype, device=w.device))
            return torch.stack(outs)
        return f
    return lambda w, idx, offsets, psw: F.embedding_bag(idx, w, offsets, max_norm=mn, scale_grad_by_freq=freq, mode=cond["mode"],
                                                        per_sample_weights=psw if psw_on else None, padding_idx=pad)


def run_one(cand, cond, seed, variant, fn):
    dt, dev = getattr(torch, cand["dtype"]), cand["device"]
    inp = make_inputs(cond, seed)
    w = torch.tensor(inp["w"], dtype=dt, device=dev, requires_grad=True)
    idx = torch.tensor(inp["idx"], dtype=torch.long, device=dev)
    off = None if inp["offsets"] is None else torch.tensor(inp["offsets"], dtype=torch.long, device=dev)
    psw = torch.tensor(inp["psw"], dtype=dt, device=dev, requires_grad=True)
    out = fn(w, idx, off, psw)
    out.backward(torch.tensor(inp["g"], dtype=dt, device=dev))
    f = lambda a: a.detach().double().cpu().numpy()          # noqa: E731
    res = {"out": f(out), "dweight": f(w.grad), "weight_after": f(w)}
    if cond["op"] == "embedding_bag" and cond["per_sample_weights"]:
        res["dpsw"] = f(psw.grad)
    return res


def run(env):
    CACHE.mkdir(parents=True, exist_ok=True)
    for cand in candidates_for(env):
        path = CACHE / f"{cand['id']}.pkl"
        if path.exists():
            continue
        res, t0 = {}, time.time()
        for cond in conditions():
            fns = {}
            for var in variants_for(cond):
                fns[var] = emb_fn(cond, var)
                if cand.get("compiled"):
                    fns[var] = torch.compile(fns[var], dynamic=False)
            if cand.get("compiled"):
                torch._dynamo.reset()
            for seed in SEEDS:
                rec = {}
                for var in variants_for(cond):
                    try:
                        rec[var] = {"status": "ok", "outputs": run_one(cand, cond, seed, var, fns[var])}
                    except Exception as exc:  # noqa: BLE001
                        rec[var] = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
                res[(cond["id"], seed)] = rec
        meta = {"candidate": cand, "torch": torch.__version__, "seconds": round(time.time() - t0, 1), "env": env}
        path.write_bytes(pickle.dumps({"meta": meta, "res": res}))
        st = defaultdict(int)
        for rec in res.values():
            for r in rec.values():
                st[r["status"]] += 1
        print(cand["id"], dict(st), meta["seconds"], "s", flush=True)


# ------------------------------------------------------------------------------------------------ analysis

def deviation(k, r, dtype):
    k, r = np.asarray(k, float), np.asarray(r, float)
    if k.shape != r.shape:
        return k.size or 1, np.inf
    tau = TAU.get(dtype, 2.0 ** -8)
    fin = np.isfinite(k) & np.isfinite(r)
    cls = ~fin & ~((np.isnan(k) & np.isnan(r)) | (k == r))
    d = np.abs(k - r)
    lim = tau * (1 + np.abs(r))
    return int(((fin & (d > lim)) | cls).sum()), float(np.max(np.where(fin, d / lim, 0), initial=0))


def analyse():
    data = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(CACHE.glob("*.pkl"))}
    conds = {c["id"]: c for c in conditions()}
    out = {"conditions": len(conds), "candidates": {}, "examples": defaultdict(list)}
    for name, d in data.items():
        dtype = d["meta"]["candidate"]["dtype"]
        st = defaultdict(int)
        for rec in d["res"].values():
            for r in rec.values():
                st[r["status"]] += 1
                if r["status"] != "ok" and len(out["examples"]["errors"]) < 20:
                    out["examples"]["errors"].append({"candidate": name, "reason": r["reason"]})
        ref = REFERENCE.get(name)
        E, E64 = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        e_conds, e64_conds, e_max = set(), set(), 0.0
        P, H = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        tau = TAU[dtype]
        for (cid, seed), rec in d["res"].items():
            cond = conds[cid]
            b = rec.get("base")
            if not b or b["status"] != "ok":
                continue
            o = b["outputs"]
            for tgt, acc, cset in ((ref, E, e_conds), ("eager_cpu_float64", E64, e64_conds)):
                if tgt not in data or tgt == name:
                    continue
                rb = data[tgt]["res"].get((cid, seed), {}).get("base")
                if not rb or rb["status"] != "ok":
                    continue
                for x in o:
                    n, ratio = deviation(o[x], rb["outputs"][x], dtype)
                    acc[x][0] += 1
                    acc[x][1] += int(n > 0)
                    if tgt == ref:
                        e_max = max(e_max, ratio)
                    if n:
                        cset.add(cid)
                        if len(out["examples"]["E"]) < 40:
                            out["examples"]["E"].append({"candidate": name, "vs": tgt, "condition": cid, "seed": seed, "output": x,
                                                         "violating": n, "max_ratio": ratio})
            inp = make_inputs(cond, seed)
            pad = cond["padding_idx"]
            if pad is not None:
                z = bool(np.all(o["dweight"][pad] == 0))
                P["P_padding_row_zero_grad"][0] += 1
                P["P_padding_row_zero_grad"][1] += int(not z)
            un = rec.get("unscaled")
            if un and un["status"] == "ok":
                idx = np.asarray(inp["idx"]).ravel()
                cnt = np.bincount(idx, minlength=NUM).astype(float)
                expect = np.where(cnt[:, None] > 0, un["outputs"]["dweight"] / np.maximum(cnt[:, None], 1), 0)
                if pad is not None:
                    expect[pad] = 0
                ok = bool(np.all(np.abs(o["dweight"] - expect) <= tau * (1 + np.abs(expect))))
                P["P_scale_grad_by_freq"][0] += 1
                P["P_scale_grad_by_freq"][1] += int(not ok)
                if not ok and len(out["examples"]["P"]) < 40:
                    out["examples"]["P"].append({"candidate": name, "condition": cid, "seed": seed, "property": "P_scale_grad_by_freq",
                                                 "max_abs": float(np.max(np.abs(o["dweight"] - expect)))})
            ve = rec.get("via_embedding")
            if ve and ve["status"] == "ok":
                ok = True
                for x in ("out", "dweight") + (("dpsw",) if "dpsw" in o else ()):
                    n, _ = deviation(o[x], ve["outputs"][x], dtype)
                    ok &= n == 0
                key = "P_bag_sum_equals_embedding_sum" if cond["mode"] == "sum" else f"H_bag_{cond['mode']}_equals_embedding_{cond['mode']}"
                tally = P if cond["mode"] == "sum" else H
                tally[key][0] += 1
                tally[key][1] += int(not ok)
                if not ok and len(out["examples"]["P"]) < 40:
                    out["examples"]["P"].append({"candidate": name, "condition": cid, "seed": seed, "property": key})
            if cond["op"] == "embedding_bag" and cond["bags"] == "empty" and cond["mode"] in ("sum", "mean"):
                z = bool(np.all(o["out"][1] == 0))
                P["P_empty_bag_zero"][0] += 1
                P["P_empty_bag_zero"][1] += int(not z)
            if cond["op"] == "embedding_bag" and pad is not None and cond["mode"] in ("sum", "mean"):
                z = bool(np.all(o["out"][2] == 0))                    # bag 2: padding only
                H["H_padding_only_bag_zero"][0] += 1
                H["H_padding_only_bag_zero"][1] += int(not z)
            if cond["max_norm"]:
                mn = cond["max_norm"]
                w0, w1 = np.asarray(inp["w"], float), o["weight_after"]
                used = np.zeros(NUM, bool)
                used[np.asarray(inp["idx"]).ravel()] = True
                if pad is not None and cond["op"] == "embedding":
                    pass                                              # the padding row is renormalised too if referenced
                n0, n1 = np.linalg.norm(w0, axis=1), np.linalg.norm(w1, axis=1)
                ok_untouched = bool(np.array_equal(w1[~used], w0[~used].astype(np.float32).astype(float) if dtype == "float32" else w0[~used]))
                big = used & (n0 > mn)
                ok_rescaled = bool(np.all(np.abs(n1[big] - mn) <= 1e-5 * mn + tau * (1 + mn)))
                small = used & (n0 <= mn)
                ok_small = bool(np.array_equal(w1[small], w0[small]))
                outn = np.linalg.norm(o["out"].reshape(-1, DIM), axis=1)
                ok_out = bool(np.all(outn <= mn * (1 + 1e-5) + tau)) if cond["op"] == "embedding" else True
                ok = ok_untouched and ok_rescaled and ok_small and ok_out
                H["H_max_norm"][0] += 1
                H["H_max_norm"][1] += int(not ok)
                if not ok and len(out["examples"]["posthoc"]) < 40:
                    out["examples"]["posthoc"].append({"candidate": name, "condition": cid, "seed": seed, "untouched": ok_untouched,
                                                       "rescaled": ok_rescaled, "small_rows_kept": ok_small, "out_norm": ok_out})
        entry = {"statuses": dict(st), "seconds": d["meta"]["seconds"],
                 "E": {"reference": ref, "by_output": {k: f"{v[1]}/{v[0]}" for k, v in E.items()}, "violating_conditions": sorted(e_conds),
                       "max_ratio": e_max} if ref in data else None,
                 "E64": {"by_output": {k: f"{v[1]}/{v[0]}" for k, v in E64.items()}, "violating_conditions": sorted(e64_conds)}
                 if name != "eager_cpu_float64" else None,
                 "P": {k: f"{v[1]}/{v[0]}" for k, v in P.items()}, "posthoc": {k: f"{v[1]}/{v[0]}" for k, v in H.items()}}
        out["candidates"][name] = entry
        print(f"{name:30s} {dict(st)}")
        if entry["E"]:
            print(f"{'':30s} E vs {ref}: {entry['E']['by_output']} {entry['E']['violating_conditions']} max {e_max:.3g}")
        if entry["E64"]:
            print(f"{'':30s} E64: {entry['E64']['by_output']} {entry['E64']['violating_conditions']}")
        print(f"{'':30s} P {entry['P']}  H {entry['posthoc']}")
    out["examples"] = dict(out["examples"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse"])
    ap.add_argument("--env", default="ka_main")
    a = ap.parse_args()
    run(a.env) if a.stage == "run" else analyse()
