#!/usr/bin/env python3
"""Phase 1 of the essential-bug round (docs/protocol_essential_bugs_20261007.md): stages

    candidates  run one candidate on every condition x seed of a family, with the transformed-input runs of the
                property checks (P); raw outputs -> .cache/essential/raw/<family>/<candidate>.pkl
    spec        the independent specification on the inputs each dtype received (CPU, parallel)
                -> .cache/essential/spec/<family>/<cond>_<seed>_<dtype>_<eps>.pkl
    fr          Inductor candidates through the tool's mode B (K_R per element) -> .cache/essential/fr/...

    python scripts/essential/run_phase1.py candidates --family pool --candidate inductor_cuda32
    python scripts/essential/run_phase1.py spec --family ce --workers 24
    python scripts/essential/run_phase1.py fr --family index
    (nightly: run `candidates` with the nightly interpreter; the candidate is stored as nightly_<name>)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import conditions as C  # noqa: E402

ROOT = common.ROOT
# ESSENTIAL_SUITE (phase-2 protocol section 1): "phase1" (default, frozen), "s2" (phase-1 conditions with the
# supplementary property runs only), "s3" (supplementary boundary conditions, full E/F/P/FR)
SUITE = os.environ.get("ESSENTIAL_SUITE", "phase1")
CACHE = ROOT / ".cache/essential" if SUITE == "phase1" else ROOT / f".cache/essential/suite_{SUITE}"
FAMILIES = {"ce": (C.ce_conditions, C.make_ce_inputs), "pool": (C.pool_conditions, C.make_pool_inputs),
            "index": (C.index_conditions, C.make_index_inputs)}
if SUITE == "s3":
    import conditions_supplement as CS
    FAMILIES = {"pool": (CS.pool_conditions, CS.make_pool_inputs), "index": (CS.index_conditions, CS.make_index_inputs)}


def _env_tag():
    import torch
    return "nightly_" if ".dev" in torch.__version__ else ""


# ------------------------------------------------------------------------------------------------ transformed inputs (P)

def _ce_perm(cond, inp):
    """reverse the class order: logits columns, weights, index targets (and ignore_index when it is a class),
    probability-target columns.  Returns (cond', inp')."""
    Cn = cond["C"]
    c2 = dict(cond)
    ii = cond["ignore_index"]
    if 0 <= ii < Cn:
        c2["ignore_index"] = Cn - 1 - ii
    tgt = inp["target"]
    if cond["target"] == "prob":
        t2 = np.asarray(tgt)[:, ::-1].copy()
    else:
        t2 = [Cn - 1 - t if (t != ii and 0 <= t < Cn) or (t == ii and 0 <= ii < Cn) else t for t in tgt]
    w2 = None if inp["weights"] is None else list(reversed(inp["weights"]))
    return c2, dict(inp, logits=inp["logits"][:, ::-1].copy(), target=t2, weights=w2)


def _index_reverse(cond, inp):
    """reverse the order of the contributions (index entries with their source slices along dim)."""
    idx, src, dim = inp["index"], np.asarray(inp["source"]), inp["dim"]
    if cond["op"] in ("index_add", "index_reduce"):
        idx2 = list(reversed(idx))
        src2 = src[::-1].copy() if (src.ndim == 1 or dim == 0) else src[:, ::-1].copy()
    else:
        ia = np.asarray(idx)
        if ia.ndim == 1:
            idx2, src2 = list(ia[::-1]), src[::-1].copy()
        elif dim == 0:
            idx2, src2 = ia[::-1].tolist(), src[::-1].copy()
        else:
            idx2, src2 = ia[:, ::-1].tolist(), src[:, ::-1].copy()
    return dict(inp, index=idx2, source=src2)


def variants(family, cond, inp):
    """the base run and the transformed runs of the property checks (suite s2: the supplementary ones, S2)."""
    if SUITE == "s2":
        return variants_s2(family, cond, inp)
    out = {"base": (cond, inp)}
    if family == "ce":
        out["shift"] = (cond, dict(inp, logits=inp["logits"] + 2.0))
        out["perm"] = _ce_perm(cond, inp)
    elif family == "pool" and cond["op"] == "max_pool":
        out["shift"] = (cond, dict(inp, x=inp["x"] + 1.0))
    elif family == "index":
        out["ones"] = (cond, dict(inp, self=np.ones_like(inp["self"]), source=np.ones_like(inp["source"])))
        out["reverse"] = (cond, _index_reverse(cond, inp))
    return out


def variants_s2(family, cond, inp):
    """S2: properties phase 1 did not check.  CE: changing the logits of ignored rows changes neither the loss nor the
    other rows' gradients; avg pool: linearity f(2x + 3y) = 2 f(x) + 3 f(y); index_add / sum / mean: linearity in
    (self, source) jointly; pooling: equivariance under reversing the channels."""
    out = {"base": (cond, inp)}
    if family == "ce" and cond["target"] == "index":
        ign = np.array([t == cond["ignore_index"] for t in inp["target"]])
        if ign.any():
            lg = inp["logits"].copy()
            lg[ign] = lg[ign][:, ::-1] * 3.0 + 7.0                     # anything: these rows must not matter
            out["masked_rows_changed"] = (cond, dict(inp, logits=lg))
    elif family == "pool":
        y = np.flip(inp["x"], axis=-1).copy() * 0.5 + 1.0
        if cond["op"] == "avg_pool":
            out["lin_y"] = (cond, dict(inp, x=y))
            out["lin_2x3y"] = (cond, dict(inp, x=2.0 * inp["x"] + 3.0 * y))
        out["channels_reversed"] = (cond, dict(inp, x=inp["x"][::-1].copy()))
    elif family == "index" and (cond["op"] == "index_add" or cond.get("reduce") in ("sum", "mean")):
        ys, yv = np.flip(inp["self"]).copy() + 1.0, np.flip(inp["source"]).copy() - 1.0
        out["lin_y"] = (cond, dict(inp, self=ys, source=yv))
        out["lin_2x3y"] = (cond, dict(inp, self=2.0 * inp["self"] + 3.0 * ys, source=2.0 * inp["source"] + 3.0 * yv))
    return out


# ------------------------------------------------------------------------------------------------ stage: candidates

def stage_candidates(family, cand, limit=None):
    import candidates as K
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conds, make = FAMILIES[family]
    store = f"{_env_tag()}{cand}"
    path = CACHE / "raw" / family / f"{store}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = pickle.loads(path.read_bytes()) if path.exists() else {}
    cs = [c for c in conds() if c["op"] != "flce"][:limit]
    t0, n_new = time.time(), 0
    for i, cond in enumerate(cs):
        for seed in C.SEEDS:
            key = (cond["id"], seed)
            if key in done:
                continue
            inp = make(cond, seed)
            rec, lost = {}, False
            for vname, (c2, i2) in variants(family, cond, inp).items():
                if lost:                          # the CUDA context is gone: later runs would fail for that reason
                    rec[vname] = {"status": "not_run", "reason": "CUDA context lost after an earlier device-side error"}
                    continue
                try:
                    rec[vname] = K.run(family, c2, i2, cand, seed)
                except Exception as exc:  # noqa: BLE001
                    rec[vname] = {"status": "error", "reason": f"harness: {type(exc).__name__}: {exc}"[:300],
                                  "trace": traceback.format_exc()[-800:]}
                lost = rec[vname]["status"] == "error" and ("CUDA error" in rec[vname]["reason"] or "AcceleratorError" in rec[vname]["reason"])
            done[key] = rec
            n_new += 1
            if lost:                              # save and leave; the launcher restarts a fresh process
                path.write_bytes(pickle.dumps(done))
                print(f"{family}/{store}: device-side error at {key}; restarting", flush=True)
                sys.exit(3)
        if (i + 1) % 20 == 0 or i == len(cs) - 1:
            path.write_bytes(pickle.dumps(done))
            print(f"{family}/{store}: {i + 1}/{len(cs)} conditions, {time.time() - t0:.0f} s", flush=True)
    meta = {"torch": torch.__version__, "candidate": store, "entries": len(done), "seconds_this_run": round(time.time() - t0, 1)}
    (path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=1) + "\n")


# ------------------------------------------------------------------------------------------------ stage: spec

def received_inputs(family, cond, inp, dtype):
    """the values a candidate of this dtype receives (same rounding as candidates._t: float64 -> dtype on the CPU)."""
    import torch
    td = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]

    def r(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float64)).to(td).double().numpy()
    if family == "ce":
        tgt = inp["target"] if cond["target"] == "index" else r(inp["target"])
        return dict(logits=r(inp["logits"]), target=tgt, weights=None if inp["weights"] is None else r(inp["weights"]).tolist())
    if family == "pool":
        return dict(x=r(inp["x"]))
    return dict(self=r(inp["self"]), source=r(inp["source"]))


def _spec_job(args):
    family, cond, seed, variant, dtype, eps_kind = args
    import candidates as K
    import spec_eval as S

    out_path = CACHE / "spec" / family / f"{cond['id']}_{seed}_{variant}_{dtype}_{eps_kind}.pkl"
    if out_path.exists():
        return str(out_path), "cached"
    _, make = FAMILIES[family]
    inp = make(cond, seed)
    c2, i2 = variants(family, cond, inp)[variant]
    rx = received_inputs(family, c2, i2, dtype)
    t0 = time.time()
    try:
        if family == "ce":
            eps = float(np.float32(c2["eps"])) if eps_kind == "f32" else float(c2["eps"])
            res = S.ce(c2, rx["logits"], rx["target"], rx["weights"], eps, K.ce_upstream(cond, seed))
        elif family == "pool":
            # the upstream gradient the candidate uses (same generator, output shape from the spec's main reading)
            res = None
            import spec_pooling as spo
            kw_shape = None
            if c2["op"] == "avg_pool":
                y = spo.avg_pool(rx["x"].tolist(), list(c2["kernel"]), list(c2["stride"]), list(c2["padding"]),
                                 c2["ceil_mode"], c2["count_include_pad"], c2["divisor_override"])
            else:
                y = spo.max_pool(rx["x"].tolist(), list(c2["kernel"]), list(c2["stride"]), list(c2["padding"]),
                                 list(c2["dilation"]), c2["ceil_mode"])[0]
            kw_shape = np.array(y, dtype=object).shape
            v = K.pool_upstream(c2, i2, kw_shape)
            res = (S.avg_pool if c2["op"] == "avg_pool" else S.max_pool)(c2, rx["x"], v)
        else:
            v = K.index_upstream(i2, tuple(np.asarray(i2["self"]).shape))
            res = S.index_family(c2, rx["self"], i2["index"], rx["source"], i2["dim"], v)
    except Exception as exc:  # noqa: BLE001  (a spec exception other than its declared ones is reported, not hidden)
        res = {"status": "spec_exception", "reason": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-800:]}
    res["seconds"] = round(time.time() - t0, 3)
    res["received"] = rx
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pickle.dumps(res))
    return str(out_path), res.get("status")


def spec_jobs(family):
    import candidates as K
    conds, make = FAMILIES[family]
    jobs = []
    for cond in conds():
        if cond["op"] == "flce":
            continue
        dtypes = {K.DTYPE_NAME[K.CANDIDATES[c][1]] for c in K.FAMILY_CANDIDATES[family]}
        for seed in C.SEEDS:
            # the property checks compare a candidate's base and transformed runs with each other; only the base
            # inputs need the specification
            for variant in ("base",):
                for dtype in sorted(dtypes):
                    kinds = ["f64", "f32"] if (family == "ce" and cond["eps"] != 0 and dtype != "float64") else ["f64"]
                    for kind in kinds:
                        jobs.append((family, cond, seed, variant, dtype, kind))
    return jobs


def stage_spec(family, workers):
    from multiprocessing import get_context
    jobs = spec_jobs(family)
    # longest first (CE with large C)
    jobs.sort(key=lambda j: -(j[1].get("N", 1) * j[1].get("C", 1)))
    t0 = time.time()
    counts = {}
    with get_context("fork").Pool(workers) as pool:
        for k, (path, status) in enumerate(pool.imap_unordered(_spec_job, jobs, chunksize=1)):
            counts[status] = counts.get(status, 0) + 1
            if (k + 1) % 500 == 0 or k == len(jobs) - 1:
                print(f"spec {family}: {k + 1}/{len(jobs)} {counts} {time.time() - t0:.0f} s", flush=True)


def load_spec(family, cond_id, seed, variant, dtype, eps_kind="f64"):
    p = CACHE / "spec" / family / f"{cond_id}_{seed}_{variant}_{dtype}_{eps_kind}.pkl"
    return pickle.loads(p.read_bytes()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["candidates", "spec", "fr"])
    ap.add_argument("--family", choices=list(FAMILIES), required=True)
    ap.add_argument("--candidate")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if a.stage == "candidates":
        stage_candidates(a.family, a.candidate, a.limit)
    elif a.stage == "spec":
        stage_spec(a.family, a.workers)
    else:
        import fr_stage
        fr_stage.run(a.family, a.limit, a.candidate)


if __name__ == "__main__":
    main()
