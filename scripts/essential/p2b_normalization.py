#!/usr/bin/env python3
"""2b family "normalization" (protocol v2 section 3; registry tier1.normalization): E and P, forward and backward.  F and
mode-B FR stay closed until the spec is delivered; mode-A FR for the Triton candidates is in p2b_fr_modeA.py.

Conditions: the coverage plan (23 distinct combinations of op, values, affine, batch) x 3 seeds.  Shapes: layer_norm and
rms_norm (batch, 5, 12) over the last axis; group_norm (batch, 6, 10) with 3 groups; batch_norm (batch, 6, 10) per channel,
momentum 0.1, given running statistics (mean N(0, 1), var U(0.5, 2)); eps = 1e-5 everywhere (rms_norm given explicitly).
Values: gauss N(0, 1); constant_rows (every normalised set constant, distinct constants); huge_offset 1e4 + N(0, 1); tiny
1e-20 x N(0, 1) (variance far below eps).  Outputs: y, dx, dweight, dbias (random upstream) and, for batch_norm_train, the
updated running_mean / running_var.

P (pre-registered): scale invariance norm(2x) == norm(x) where the set's variance is >= 1e3 eps (precondition of the
"eps -> 0 limit"; tolerance τ32 plus the eps term |y| eps / var); BN train: the running_var increment equals the biased batch
variance x n / (n - 1); BN eval: the output of sample 0 does not depend on the other samples and running statistics do not
change (bitwise; batch >= 2 only).  Post-hoc: per-row independence for layer/rms/group norm (perturbing other rows leaves
row 0 bitwise unchanged).

    python scripts/essential/p2b_normalization.py run --env ka_main
    PYTHONPATH=.cache/pylibs/nightly_np /data1/tzh/envs/pt_nightly_cu126/bin/python scripts/essential/p2b_normalization.py run --env nightly
    /data1/tzh/envs/liger/bin/python scripts/essential/p2b_normalization.py run --env liger
    python scripts/essential/p2b_normalization.py analyse
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
CACHE = ROOT / ".cache/essential/p2b/normalization"
OUT = ROOT / "results/essential/phase2b/normalization"
SEEDS = (0, 1, 2)
EPS, MOM = 1e-5, 0.1
TAU = {"float32": 2.0 ** -12, "float64": 1e-9}


def conditions():
    plan = json.loads((ROOT / "results/essential/phase2b/coverage_plan.json").read_text())["tier1"]["normalization"]["conditions"]
    out, seen = [], {}
    for c in plan:
        key = (c["op"], c["values"], c["affine"], c["batch"])
        if key in seen:
            seen[key]["high_risk"] |= bool(c.get("high_risk"))
            continue
        d = dict(c, high_risk=bool(c.get("high_risk")), id=f"norm_{c['op']}_{c['values']}_{'aff' if c['affine'] else 'noaff'}_b{c['batch']}")
        seen[key] = d
        out.append(d)
    return out


def _rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def shape_of(cond):
    return (cond["batch"], 5, 12) if cond["op"] in ("layer_norm", "rms_norm") else (cond["batch"], 6, 10)


def sets_axes(cond):
    """axes over which statistics are taken, for the input viewed as in shape_of (group_norm handled by reshaping)."""
    return {"layer_norm": (-1,), "rms_norm": (-1,), "batch_norm_train": (0, 2), "batch_norm_eval": (0, 2)}.get(cond["op"])


def make_inputs(cond, seed, variant):
    rng = _rng(f"norm/{cond['id']}/{seed}")
    shp = shape_of(cond)
    v = cond["values"]
    x = rng.normal(0, 1, shp)
    if v == "constant_rows":
        if cond["op"] in ("layer_norm", "rms_norm"):
            x = np.broadcast_to(rng.normal(0, 1, shp[:-1] + (1,)), shp).copy()
        elif cond["op"] == "group_norm":
            x = np.broadcast_to(np.repeat(rng.normal(0, 1, (shp[0], 3)), 2, axis=1)[:, :, None], shp).copy()
        else:
            x = np.broadcast_to(rng.normal(0, 1, (1, shp[1], 1)), shp).copy()
    elif v == "huge_offset":
        x = 1e4 + x
    elif v == "tiny":
        x = 1e-20 * x
    c = shp[-1] if cond["op"] in ("layer_norm", "rms_norm") else shp[1]
    w, b = rng.normal(1, 0.3, c), rng.normal(0, 0.3, c)
    rm, rv = rng.normal(0, 1, shp[1]), rng.uniform(0.5, 2.0, shp[1])
    g = rng.normal(0, 1, shp)
    x = x.astype(np.float32).astype(np.float64)               # every candidate receives float32-representable values
    w, b, rm, rv, g = (a.astype(np.float32).astype(np.float64) for a in (w, b, rm, rv, g))
    if variant.startswith("translated"):                      # exact in float32: Sterbenz for 1e4 + N(0, 1), x - x = 0
        x = x - 1e4 if v == "huge_offset" else x - x
    if variant.endswith("scaled"):
        x = 2 * x
    if variant == "others_perturbed":
        prng = _rng(f"norm-perturb/{cond['id']}/{seed}")
        if cond["op"].startswith("batch_norm"):
            x[1:] = x[1:] + 5 * prng.normal(0, 1, x[1:].shape)
        else:
            first_other = 2 if cond["op"] == "group_norm" else 1      # group_norm: channels 0, 1 are group 0
            x[:, first_other:] = x[:, first_other:] + 5 * prng.normal(0, 1, x[:, first_other:].shape)
    x = x.astype(np.float32).astype(np.float64)
    return {"x": x, "w": w, "b": b, "rm": rm, "rv": rv, "g": g}


TRANSLATION_INVARIANT = ("layer_norm", "group_norm", "batch_norm_train")


def variants_for(cond):
    v = ["base", "scaled"]
    if cond["op"] in TRANSLATION_INVARIANT and cond["values"] in ("huge_offset", "constant_rows"):
        v += ["translated", "translated_scaled"]          # numerical adjudication (protocol v2 deviation 9)
    if cond["op"].startswith("batch_norm"):
        if cond["batch"] >= 2:
            v.append("others_perturbed")
    else:
        v.append("others_perturbed")
    return v


# ------------------------------------------------------------------------------------------------ candidates

def candidates_for(env):
    if env == "ka_main":
        out = [{"id": f"eager_{d}_{t}", "device": d, "dtype": t} for d, t in
               (("cpu", "float64"), ("cuda", "float64"), ("cpu", "float32"), ("cuda", "float32"), ("cuda", "bfloat16"))]
        return out + [{"id": f"inductor_cuda_{t}", "device": "cuda", "dtype": t, "compiled": True} for t in ("float32", "bfloat16")]
    if env == "nightly":
        return [{"id": "nightly_eager_cuda_float32", "device": "cuda", "dtype": "float32"},
                {"id": "nightly_inductor_cuda_float32", "device": "cuda", "dtype": "float32", "compiled": True}]
    if env == "liger":
        return [{"id": "liger_rmsnorm", "device": "cuda", "dtype": "float32", "liger": "rms_norm"},
                {"id": "liger_layernorm", "device": "cuda", "dtype": "float32", "liger": "layer_norm"}]
    raise KeyError(env)


REFERENCE = {"inductor_cuda_float32": "eager_cuda_float32", "inductor_cuda_bfloat16": "eager_cuda_bfloat16",
             "nightly_inductor_cuda_float32": "nightly_eager_cuda_float32", "nightly_eager_cuda_float32": "eager_cuda_float32",
             "liger_rmsnorm": "eager_cuda_float32", "liger_layernorm": "eager_cuda_float32", "eager_cuda_float64": "eager_cpu_float64"}


def norm_fn(cond):
    op, aff = cond["op"], cond["affine"]

    def f(x, w, b, rm, rv):
        if op == "layer_norm":
            return F.layer_norm(x, (x.shape[-1],), w if aff else None, b if aff else None, EPS)
        if op == "rms_norm":
            return F.rms_norm(x, (x.shape[-1],), w if aff else None, EPS)
        if op == "group_norm":
            return F.group_norm(x, 3, w if aff else None, b if aff else None, EPS)
        return F.batch_norm(x, rm, rv, w if aff else None, b if aff else None, training=op == "batch_norm_train",
                            momentum=MOM, eps=EPS)
    return f


def run_one(cand, cond, seed, variant, fn):
    dt, dev = getattr(torch, cand["dtype"]), cand["device"]
    inp = make_inputs(cond, seed, variant)
    t = lambda a, rg=False: torch.tensor(a, dtype=dt, device=dev, requires_grad=rg)   # noqa: E731
    x, w, b = t(inp["x"], True), t(inp["w"], True), t(inp["b"], True)
    rm = torch.tensor(inp["rm"], dtype=dt, device=dev)
    rv = torch.tensor(inp["rv"], dtype=dt, device=dev)
    if cand.get("liger"):
        from liger_kernel.transformers import LigerLayerNorm, LigerRMSNorm
        if cand["liger"] != cond["op"] or not cond["affine"]:
            raise NotImplementedError(f"{cand['id']}: {cond['op']} / affine={cond['affine']} not provided")
        mod = (LigerRMSNorm(x.shape[-1], eps=EPS, in_place=False) if cond["op"] == "rms_norm"
               else LigerLayerNorm(x.shape[-1], eps=EPS, bias=True)).to(dev)
        with torch.no_grad():
            mod.weight.copy_(w)
            if cond["op"] == "layer_norm":
                mod.bias.copy_(b)
        y = mod(x)
        y.backward(t(inp["g"]))
        f = lambda a: a.detach().double().cpu().numpy()       # noqa: E731
        out = {"y": f(y), "dx": f(x.grad), "dw": f(mod.weight.grad)}
        if cond["op"] == "layer_norm":
            out["db"] = f(mod.bias.grad)
        return out
    y = fn(x, w, b, rm, rv)
    y.backward(t(inp["g"]))
    f = lambda a: a.detach().double().cpu().numpy()           # noqa: E731
    out = {"y": f(y), "dx": f(x.grad)}
    if cond["affine"]:
        out["dw"] = f(w.grad)
        if cond["op"] != "rms_norm":
            out["db"] = f(b.grad)
    if cond["op"].startswith("batch_norm"):
        out["rm"], out["rv"] = f(rm), f(rv)
    return out


def run(env):
    CACHE.mkdir(parents=True, exist_ok=True)
    for cand in candidates_for(env):
        path = CACHE / f"{cand['id']}.pkl"
        if path.exists():
            continue
        res, t0, slowest = {}, time.time(), (0.0, None)
        for cond in conditions():
            fn = norm_fn(cond)
            if cand.get("compiled"):
                torch._dynamo.reset()
                fn = torch.compile(fn, dynamic=False)
            for seed in SEEDS:
                rec = {}
                for var in variants_for(cond):
                    t1 = time.time()
                    try:
                        rec[var] = {"status": "ok", "outputs": run_one(cand, cond, seed, var, fn)}
                    except NotImplementedError as exc:
                        rec[var] = {"status": "unsupported", "reason": str(exc)}
                    except Exception as exc:  # noqa: BLE001
                        rec[var] = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
                    if time.time() - t1 > slowest[0]:
                        slowest = (time.time() - t1, f"{cond['id']}/{seed}/{var}")
                res[(cond["id"], seed)] = rec
        meta = {"candidate": cand, "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
                "slowest": [round(slowest[0], 2), slowest[1]], "env": env}
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


def set_variance(cond, x):
    """per-element variance of the element's normalisation set (float64), for the scale-invariance precondition."""
    op = cond["op"]
    if op in ("layer_norm",):
        return np.broadcast_to(x.var(-1, keepdims=True), x.shape)
    if op == "rms_norm":
        return np.broadcast_to((x ** 2).mean(-1, keepdims=True), x.shape)
    if op == "group_norm":
        xs = x.reshape(x.shape[0], 3, -1)
        return np.broadcast_to(xs.var(-1, keepdims=True), xs.shape).reshape(x.shape)
    return np.broadcast_to(x.var((0, 2), keepdims=True), x.shape)


def analyse():
    data = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(CACHE.glob("*.pkl"))}
    conds = {c["id"]: c for c in conditions()}
    out = {"conditions": len(conds), "candidates": {}, "examples": defaultdict(list)}
    for name, d in data.items():
        cand = d["meta"]["candidate"]
        dtype = cand["dtype"]
        judged = dtype in TAU
        st = defaultdict(int)
        for rec in d["res"].values():
            for r in rec.values():
                st[r["status"]] += 1
        ref = REFERENCE.get(name)
        e = defaultdict(lambda: [0, 0]) if ref in data else None
        e_max, e_conds = 0.0, set()
        e64 = defaultdict(lambda: [0, 0]) if "eager_cpu_float64" in data and name != "eager_cpu_float64" else None
        e64_conds = set()
        P = defaultdict(lambda: [0, 0])
        H = defaultdict(lambda: [0, 0])
        for (cid, seed), rec in d["res"].items():
            cond = conds[cid]
            b = rec.get("base")
            if not b or b["status"] != "ok":
                continue
            o = b["outputs"]
            for tgt, acc, cset in ((ref, e, e_conds), ("eager_cpu_float64", e64, e64_conds)):
                if acc is None:
                    continue
                rb = data[tgt]["res"].get((cid, seed), {}).get("base")
                if not rb or rb["status"] != "ok":
                    continue
                bad = False
                for x in o:
                    if x not in rb["outputs"]:
                        continue
                    n, ratio = deviation(o[x], rb["outputs"][x], dtype)
                    acc[x][0] += 1
                    acc[x][1] += int(n > 0)
                    if tgt == ref:
                        e_max = max(e_max, ratio)
                    bad |= n > 0
                    if n and len(out["examples"]["E" if tgt == ref else "E64"]) < 40:
                        out["examples"]["E" if tgt == ref else "E64"].append(
                            {"candidate": name, "condition": cid, "seed": seed, "output": x, "violating": n, "max_ratio": ratio})
                if bad:
                    cset.add(cid)
            inp = make_inputs(cond, seed, "base")
            tau = TAU.get(dtype, 2.0 ** -8)
            sc = rec.get("scaled")
            if sc and sc["status"] == "ok":
                var = set_variance(cond, np.asarray(inp["x"], float))
                ok_set = var >= 1e3 * EPS
                if ok_set.any():
                    y0, y1 = o["y"], sc["outputs"]["y"]
                    bias = 0.0
                    if cond["affine"] and cond["op"] != "rms_norm":
                        bshape = (-1,) if cond["op"] in ("layer_norm",) else (-1, 1)
                        bias = np.asarray(inp["b"], float).reshape(bshape)
                    lim = tau * (1 + np.abs(y0)) + np.abs(y0 - bias) * EPS / np.maximum(var, 1e-300)
                    if cond["op"].startswith("batch_norm") and cond["op"] == "batch_norm_eval":
                        pass                                   # eval uses running statistics: not scale invariant by design
                    else:
                        viol = bool(np.any(ok_set & (np.abs(y1 - y0) > lim)))
                        if not judged:
                            continue
                        P["P_scale_invariance"][0] += 1
                        P["P_scale_invariance"][1] += int(viol)
                        if viol and len(out["examples"]["P"]) < 40:
                            out["examples"]["P"].append({"candidate": name, "condition": cid, "seed": seed, "property": "P_scale_invariance",
                                                         "max_abs_change": float(np.max(np.abs(y1 - y0)[ok_set]))})
            if cond["op"] == "batch_norm_train" and "rv" in o:
                x = np.asarray(inp["x"], float)
                n = x.shape[0] * x.shape[2]
                vb = x.var((0, 2))
                inc = (o["rv"] - (1 - MOM) * np.asarray(inp["rv"], float)) / MOM
                expect = vb * n / (n - 1)
                ok = bool(np.all(np.abs(inc - expect) <= tau * (1 + np.abs(expect)) / MOM))
                if not judged:
                    continue
                P["P_bn_running_var_unbiased"][0] += 1
                P["P_bn_running_var_unbiased"][1] += int(not ok)
                if not ok and len(out["examples"]["P"]) < 40:
                    out["examples"]["P"].append({"candidate": name, "condition": cid, "seed": seed, "property": "P_bn_running_var_unbiased",
                                                 "increment_over_biased": (inc / np.maximum(vb, 1e-300)).tolist(), "n": n})
            op = rec.get("others_perturbed")
            if op and op["status"] == "ok":
                if cond["op"] == "batch_norm_eval":
                    unchanged = np.array_equal(o["rm"], op["outputs"]["rm"]) and np.array_equal(o["rv"], op["outputs"]["rv"])
                    stats_kept = (np.array_equal(o["rm"], _cast(inp["rm"], dtype)) and np.array_equal(o["rv"], _cast(inp["rv"], dtype)))
                    ok = bool(np.array_equal(o["y"][0], op["outputs"]["y"][0]) and unchanged and stats_kept)
                    P["P_bn_eval_running_stats_only"][0] += 1
                    P["P_bn_eval_running_stats_only"][1] += int(not ok)
                elif not cond["op"].startswith("batch_norm"):
                    ok = bool(np.array_equal(o["y"][:, 0], op["outputs"]["y"][:, 0]))
                    H["H_row_independence"][0] += 1
                    H["H_row_independence"][1] += int(not ok)
                    if not ok and len(out["examples"]["posthoc"]) < 40:
                        out["examples"]["posthoc"].append({"candidate": name, "condition": cid, "seed": seed,
                                                           "max_abs_change": float(np.max(np.abs(o["y"][:, 0] - op["outputs"]["y"][:, 0])))})
        adjud = {}
        if judged and "eager_cpu_float64" in data:
            for cid in sorted(e_conds | e64_conds | {x["condition"] for x in out["examples"].get("P", []) if x["candidate"] == name}):
                cond = conds[cid]
                if "translated" not in variants_for(cond):
                    adjud[cid] = "not adjudicated (no translated run)"
                    continue
                okc, oks = True, True
                for seed in SEEDS:
                    tr = d["res"][(cid, seed)].get("translated")
                    ts = d["res"][(cid, seed)].get("translated_scaled")
                    rb = data["eager_cpu_float64"]["res"][(cid, seed)]["base"]["outputs"]
                    if not tr or tr["status"] != "ok":
                        okc = False
                        continue
                    for x, val in tr["outputs"].items():
                        if x == "rm" or x not in rb:
                            continue
                        n_, _ = deviation(val, rb[x], dtype)
                        okc &= n_ == 0
                    if ts and ts["status"] == "ok":
                        y0, y1 = tr["outputs"]["y"], ts["outputs"]["y"]
                        oks &= bool(np.all(np.abs(y1 - y0) <= TAU[dtype] * (1 + np.abs(y0)) + np.abs(y0) * EPS))
                adjud[cid] = ("numerical (conditioning): compatible with the float64 run on the exactly translated input"
                              + ("" if oks else "; scale invariance NOT restored on the translated input")
                              if okc else "NOT explained: the translated input still deviates")
        entry = {"meta": {k: v for k, v in d["meta"].items() if k != "candidate"}, "statuses": dict(st), "judged": judged,

                 "E": ({"reference": ref, "by_output": {k: f"{v[1]}/{v[0]}" for k, v in e.items()}, "violating_conditions": sorted(e_conds),
                        "max_ratio": e_max} if e is not None else None),
                 "E64": ({"by_output": {k: f"{v[1]}/{v[0]}" for k, v in e64.items()}, "violating_conditions": sorted(e64_conds)}
                         if e64 is not None else None),
                 "P": {k: f"{v[1]}/{v[0]}" for k, v in P.items()}, "posthoc": {k: f"{v[1]}/{v[0]}" for k, v in H.items()},
                 "adjudication": adjud}
        out["candidates"][name] = entry
        print(f"{name:30s} {dict(st)}")
        if entry["E"]:
            print(f"{'':30s} E vs {ref}: {entry['E']['by_output']} conds {len(e_conds)} max {e_max:.3g}")
        if entry["E64"]:
            print(f"{'':30s} E64: {entry['E64']['by_output']} conds {sorted(e64_conds)[:6]}{'...' if len(e64_conds) > 6 else ''} ({len(e64_conds)})")
        print(f"{'':30s} P {entry['P']}  H {entry['posthoc']}")
        for cid, why in adjud.items():
            print(f"{'':30s} {cid}: {why}")
    out["examples"] = dict(out["examples"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


def _cast(a, dtype):
    return torch.tensor(a, dtype=getattr(torch, dtype)).double().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "analyse"])
    ap.add_argument("--env", default="ka_main")
    a = ap.parse_args()
    run(a.env) if a.stage == "run" else analyse()
