#!/usr/bin/env python3
"""F group (K - f, without K_R) and black-box precision invariance on the cached 2b runs, with the phase-2 specs
(specs/phase2, v0.2).  The spec is evaluated on the inputs each candidate received (the 2b harnesses rebuild them
deterministically from condition and seed; all candidates there receive float32-representable values).

Contract (unchanged from phase 1): a candidate value K deviates when it lies outside the spec enclosure [lo, hi] widened by
τ(1 + |mid|), τ64 = 1e-9, τ32 = 2^-12; bf16 recorded only; non-finite K against a finite f goes to column 4.  Elements the
spec declines (undefined / not established / refused) are counted apart.  Precision invariance (numeric contract v3
section 2) for the eager float32 / float64 pairs on the same device.

    python scripts/closure/f_eval.py normalization [embedding ...]
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "specs" / "phase2"))
sys.path.insert(0, str(ROOT / "scripts" / "essential"))
import contract_v3 as CV  # noqa: E402

OUT = ROOT / "results/closure/f_eval"
TAU = {"float64": 1e-9, "float32": 2.0 ** -12}


def bounds(v):
    """spec value -> (lo, hi) floats: Fraction, rigorous.Interval, mpmath interval, or a plain number."""
    if isinstance(v, Fr) or isinstance(v, (int, float)):
        x = float(v)
        return x, x
    if hasattr(v, "lo") and hasattr(v, "hi"):                    # rigorous.Interval (Decimal / Fraction endpoints)
        return float(v.lo), float(v.hi)
    if hasattr(v, "a") and hasattr(v, "b"):                      # mpmath iv
        return float(v.a), float(v.b)
    raise TypeError(type(v))


def spec_arrays(nested):
    flat = []

    def walk(t):
        if isinstance(t, (list, tuple)):
            for e in t:
                walk(e)
        else:
            flat.append(bounds(t))
    walk(nested)
    a = np.array(flat, dtype=np.float64).reshape(-1, 2)
    return a[:, 0], a[:, 1]


def compare(k, lo, hi, dtype):
    k = np.asarray(k, dtype=np.float64).ravel()
    lo, hi = np.asarray(lo).ravel(), np.asarray(hi).ravel()
    if k.size != lo.size:
        return {"error": f"shape mismatch {k.size} vs {lo.size}"}
    fin_f = np.isfinite(lo) & np.isfinite(hi)
    # f itself may be a special value the contract defines (logsumexp of an all -inf row = -inf): K equal to it agrees
    same_special = ~fin_f & (lo == hi) & (k == lo)
    special_mismatch = ~fin_f & ~same_special
    nonfin = ~np.isfinite(k) & fin_f                        # non-finite K against a finite f -> column 4
    with np.errstate(invalid="ignore"):
        mid = np.where(fin_f, 0.5 * (lo + hi), 0.0)
        tau = TAU.get(dtype)
        tol = (tau if tau else 2.0 ** -8) * (1 + np.abs(mid))
        dist = np.where(fin_f, np.maximum(np.maximum(lo - k, k - hi), 0.0), 0.0)
    bad = fin_f & np.isfinite(k) & (dist > tol)
    return {"elements": int(k.size), "beyond": int(bad.sum()), "nonfinite_column4": int(nonfin.sum()),
            "f_special_agree": int(same_special.sum()), "f_special_mismatch": int(special_mismatch.sum()),
            "max_dist_over_tol": float(np.max(np.where(fin_f & np.isfinite(k), dist / tol, 0), initial=0.0)),
            "judged": tau is not None}


def load(path):
    return pickle.loads(Path(path).read_bytes())


# ------------------------------------------------------------------------------------------------ normalization

def norm_spec(cond, inp):
    import spec_normalization as SN
    eps = Fr(1e-5)
    x = np.asarray(inp["x"], dtype=np.float64)
    w = [Fr(float(v)) for v in inp["w"]] if cond["affine"] else None
    b = [Fr(float(v)) for v in inp["b"]] if cond["affine"] else None
    op = cond["op"]
    out = {}
    if op in ("layer_norm", "rms_norm"):
        rows = []
        for n in range(x.shape[0]):
            for r in range(x.shape[1]):
                row = [Fr(float(v)) for v in x[n, r]]
                rows.append(SN.layer_norm(row, w, b, eps) if op == "layer_norm" else SN.rms_norm(row, w, eps))
        out["y"] = rows
    elif op == "group_norm":
        out["y"] = [SN.group_norm([[Fr(float(v)) for v in x[n, c]] for c in range(x.shape[1])], 3, w, b, eps)
                    for n in range(x.shape[0])]
    else:
        st = SN.BNState(x.shape[1])
        st.running_mean = [Fr(float(v)) for v in inp["rm"]]
        st.running_var = [Fr(float(v)) for v in inp["rv"]]
        xl = [[[Fr(float(v)) for v in x[n, c]] for c in range(x.shape[1])] for n in range(x.shape[0])]
        out["y"] = SN.batch_norm(xl, st, w, b, eps, momentum=Fr(1, 10), training=op == "batch_norm_train")
        if op == "batch_norm_train":
            out["rm"], out["rv"] = st.running_mean, st.running_var
    return out


def run_normalization():
    import p2b_normalization as NM
    cache = NM.CACHE
    cands = {p.stem: load(p) for p in sorted(cache.glob("*.pkl"))}
    res = {"family": "normalization", "spec": "specs/phase2/spec_normalization.py v0.1", "candidates": defaultdict(dict),
           "precision_invariance": {}, "spec_status": {}}
    for cond in NM.conditions():
        for seed in NM.SEEDS:
            inp = NM.make_inputs(cond, seed, "base")
            try:
                f = norm_spec(cond, inp)
                fa = {k: spec_arrays(v) for k, v in f.items()}
                res["spec_status"][f"{cond['id']}/{seed}"] = "ok"
            except Exception as exc:  # noqa: BLE001
                res["spec_status"][f"{cond['id']}/{seed}"] = f"{type(exc).__name__}: {exc}"[:200]
                continue
            for name, d in cands.items():
                rec = d["res"].get((cond["id"], seed), {}).get("base")
                if not rec or rec["status"] != "ok":
                    continue
                dtype = d["meta"]["candidate"]["dtype"]
                row = {}
                for k, (lo, hi) in fa.items():
                    if k in rec["outputs"]:
                        row[k] = compare(rec["outputs"][k], lo, hi, dtype)
                res["candidates"][name][f"{cond['id']}/{seed}"] = row
            for dev in ("cpu", "cuda"):
                a, b = cands.get(f"eager_{dev}_float32"), cands.get(f"eager_{dev}_float64")
                if not a or not b:
                    continue
                ra = a["res"].get((cond["id"], seed), {}).get("base")
                rb = b["res"].get((cond["id"], seed), {}).get("base")
                if ra and rb and ra["status"] == rb["status"] == "ok":
                    lo, hi = fa["y"]
                    res["precision_invariance"][f"{dev}/{cond['id']}/{seed}"] = CV.precision_invariance(
                        ra["outputs"]["y"], rb["outputs"]["y"], 0.5 * (lo + hi))
    return res


def run_s3_index():
    """Precision invariance for the S3 index / scatter forward (eager CPU float32 / float64 pair, phase-1 spec v0.4 cached on
    each candidate's received inputs), base and reversed-contribution runs (the order-invariance property)."""
    cache = ROOT / ".cache/essential/suite_s3"
    r32 = load(cache / "raw/index/eager_cpu32.pkl")
    r64 = load(cache / "raw/index/eager_cpu64.pkl")
    res = {"family": "index/scatter (S3)", "spec": "specs/phase1/spec_index_scatter.py v0.4 (cached)", "candidates": {},
           "precision_invariance": {}, "spec_status": {}}
    for (cid, seed), rec32 in r32.items():
        rec64 = r64.get((cid, seed))
        f = {}
        for dt in ("float32", "float64"):
            sp = cache / f"spec/index/{cid}_{seed}_base_{dt}_f64.pkl"
            s = load(sp) if sp.exists() else None
            rd = (s or {}).get("readings", {}).get("main", {})
            if not s or s.get("status") != "ok" or rd.get("status") != "ok" or rd.get("out") is None:
                res["spec_status"][f"{cid}/{seed}/{dt}"] = (s or {}).get("status", "missing")
                continue
            lo, hi = (np.asarray(v, dtype=np.float64).ravel() for v in rd["out"])
            f[dt] = 0.5 * (lo + hi)
            res["spec_status"][f"{cid}/{seed}/{dt}"] = "ok"
        if len(f) < 2 or not rec64:
            continue
        for var in ("base", "reverse"):
            a, b = rec32.get(var), rec64.get(var)
            if not a or not b or a["status"] != "ok" or b["status"] != "ok":
                continue
            k32 = np.asarray(a["outputs"]["out"], dtype=np.float64).ravel()
            k64 = np.asarray(b["outputs"]["out"], dtype=np.float64).ravel()
            # d32 against the spec on the float32-rounded inputs, d64 against the spec on the float64 inputs
            res["precision_invariance"][f"cpu/{cid}/{seed}/{var}"] = CV.precision_invariance(k32, k64, f["float32"], f["float64"])
    return res


FAMILIES = {"normalization": run_normalization, "s3_index": run_s3_index}


def summarize(res):
    per = {}
    for name, rows in res["candidates"].items():
        t = defaultdict(lambda: [0, 0, 0])
        conds = set()
        for key, row in rows.items():
            for k, c in row.items():
                if "error" in c:
                    continue
                t[k][0] += c["elements"]
                t[k][1] += c["beyond"]
                t[k][2] += c["nonfinite_column4"]
                if c["beyond"]:
                    conds.add(key.split("/")[0])
        per[name] = {"by_output": {k: {"elements": v[0], "beyond": v[1], "nonfinite": v[2]} for k, v in t.items()},
                     "conditions_with_beyond": sorted(conds)}
    pi = defaultdict(int)
    for key, r in res["precision_invariance"].items():
        pi[r["condition"]] += 1
    return {"per_candidate": per, "precision_invariance_conditions": dict(pi),
            "spec_not_ok": {k: v for k, v in res["spec_status"].items() if v != "ok"}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fam in sys.argv[1:]:
        t0 = time.time()
        res = FAMILIES[fam]()
        res["summary"] = summarize(res)
        res["seconds"] = round(time.time() - t0, 1)
        (OUT / f"{fam}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
        print(fam, json.dumps(res["summary"], default=str)[:3000])


if __name__ == "__main__":
    main()
