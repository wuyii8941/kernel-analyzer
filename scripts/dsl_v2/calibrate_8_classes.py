#!/usr/bin/env python3
"""W6 (reference DSL v2 rc3 04): eight-class calibration of the frozen decision layer.

For every class a data-generating process gives per-unit residual vectors e_i (d coordinates) and reference vectors r_i;
the residual intervals are the points e_i +- a tiny width.  Each replicate runs the frozen layer (analysis.assess_units,
measure.class_statistics: endpoint-conservative t per rule, Holm inside each rule class, contract_v3 cannot-judge
rules) on 32 development + 64 confirmation units.  Null settings (the tested projections have zero mean) estimate the
false-positive rate per rule class; effect settings estimate the power; every setting reports the share of
cannot-judge / not-established outcomes and the cost.

    python scripts/dsl_v2/calibrate_8_classes.py --out results/dsl_v2/calibration/eight_classes.json --reps 300

Classes (rc3 04 W6): dense same-sign, sparse single coordinate, within-group cancellation, mean zero but aligned
nonzero, state dependent, discrete ties, rare large values, atomic execution fluctuation (8 executions averaged within
the input).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

D, NDEV, NCONF = 64, 32, 64
CLASSES = ("dense_same_sign", "sparse_single_coordinate", "within_group_cancellation", "mean_zero_aligned_nonzero",
           "state_dependent", "discrete_ties", "rare_large_values", "atomic_execution_fluctuation")


def sample(cls: str, mu: float, rng: np.random.Generator):
    """(residual points units x d, reference units x d) for one replicate; mu is the effect size in noise sd units."""
    n = NDEV + NCONF
    r = rng.standard_normal((n, D))
    noise = rng.standard_normal((n, D))
    if cls == "dense_same_sign":
        e = noise + mu
    elif cls == "sparse_single_coordinate":
        e = noise.copy()
        e[:, 0] += mu * math.sqrt(D)          # the same squared norm as a dense shift of mu
    elif cls == "within_group_cancellation":
        e = noise.copy()
        e[:, : D // 2] += mu
        e[:, D // 2:] -= mu                    # vector mean zero; a learned direction sees it
    elif cls == "mean_zero_aligned_nonzero":
        e = noise + mu * np.sign(r)            # pushes magnitudes away from zero; coordinate mean zero
    elif cls == "state_dependent":
        e = noise + mu * r                     # scales with the reference
    elif cls == "discrete_ties":
        e = np.round(noise * 2) / 2 + np.where(rng.random((n, D)) < 0.5, 0.0, mu)  # half-ulp grid, many ties
    elif cls == "rare_large_values":
        big = np.where(rng.random((n, D)) < 0.01, rng.standard_normal((n, D)) * 30, 0.0)
        e = noise + big + mu
    elif cls == "atomic_execution_fluctuation":
        execs = noise[:, None, :] + rng.standard_normal((n, 8, D)) * 2.0   # order-dependent execution noise
        e = execs.mean(axis=1) + mu                                        # averaged within the input
    else:
        raise KeyError(cls)
    return e, r


def one(args):
    cls, mu, seed = args
    from kernel_analyzer import measure
    from kernel_analyzer.reference_eval.analysis import assess_units
    rng = np.random.default_rng(seed)
    e, r = sample(cls, mu, rng)
    w = 1e-9
    t0 = time.time()
    rec, _ = assess_units("cal", e - w, e + w, r, np.ones(e.shape, bool), NDEV, ["R1", "R2", "R3", "R5"],
                          alignment_reference=r, run_detector=False)
    stats = measure.class_statistics(rec, measure.DEFAULT_RULE_CLASSES, 0.05)
    out = {}
    for c, v in stats.items():
        s = v["summary"]
        out[c] = "nonzero" if s.startswith("average effect nonzero") else (
            "cannot" if s.startswith("cannot judge") else "not_confirmed")
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    mus = [0.0, 0.1, 0.2, 0.4]
    jobs = [(c, m, 1000 * i + k) for i, (c, m) in enumerate((c, m) for c in CLASSES for m in mus) for k in range(a.reps)]
    with mp.Pool(a.workers) as pool:
        res = pool.map(one, jobs, chunksize=8)
    table = {}
    for (c, m, _), r in zip(jobs, res):
        t = table.setdefault(c, {}).setdefault(str(m), {"n": 0, "seconds": 0.0, "fixed_mean": {}, "aligned": {}})
        t["n"] += 1
        t["seconds"] += r["seconds"]
        for k in ("fixed_mean", "aligned"):
            t[k][r[k]] = t[k].get(r[k], 0) + 1
    for c in table.values():
        for t in c.values():
            for k in ("fixed_mean", "aligned"):
                t[k + "_rates"] = {o: v / t["n"] for o, v in t[k].items()}
            t["mean_seconds"] = t["seconds"] / t["n"]
    doc = {"note": "eight-class calibration of the frozen decision layer (rc3 04 W6); mu = 0 rows of classes whose "
                   "tested projections have zero mean give the false-positive rate for that rule class (which class "
                   "is null depends on the class: see 'null_for')", "d": D, "development": NDEV, "confirmation": NCONF,
           "alpha": 0.05, "reps": a.reps, "mus": mus,
           "null_for": {"dense_same_sign": ["fixed_mean@0", "aligned@0"], "sparse_single_coordinate": ["all@0"],
                        "within_group_cancellation": ["fixed_mean R1 at every mu (coordinate mean 0)", "all@0"],
                        "mean_zero_aligned_nonzero": ["fixed_mean at every mu (mean 0)", "all@0"],
                        "state_dependent": ["fixed_mean at every mu (E[r] = 0)", "all@0"],
                        "discrete_ties": ["all@0"], "rare_large_values": ["all@0"],
                        "atomic_execution_fluctuation": ["all@0"]},
           "table": table}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    for c, v in table.items():
        print(c, {m: (round(t["fixed_mean_rates"].get("nonzero", 0), 3), round(t["aligned_rates"].get("nonzero", 0), 3),
                      round(t["fixed_mean_rates"].get("cannot", 0) + t["aligned_rates"].get("cannot", 0), 3))
                  for m, t in v.items()})


if __name__ == "__main__":
    main()
