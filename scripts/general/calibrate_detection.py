#!/usr/bin/env python3
"""Calibration of the default detection (protocol_general_capability_v1 section 4 and deviation 1).

Five synthetic residual distributions x {null, effect 0.25 / 0.02 / 0.005} x 1000 replicates; 32 development + 64
confirmation units, 256 coordinates, residual intervals of width 1e-3.  Every replicate goes through the same code the
unified entry runs: analysis.assess_units (frozen rules R1, R2, R3, R5) and measure.class_statistics (Holm inside the
fixed-mean and the alignment class, contract_v3 cannot-judge rules).  Reported per cell and rule class: rejection rate
(false-positive rate under the null; power under an effect the class is built to see), cannot-judge / not-established
rate.  The bounded route (Hoeffding with M = 100 sum|w|) is reported for the rare-tail distribution.

    python scripts/general/calibrate_detection.py [--reps 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "essential"))

N_DEV, N_CONF, D, WIDTH = 32, 64, 256, 1e-3
EFFECTS = (0.0, 0.25, 0.02, 0.005)
DISTS = ("dense_small", "sparse_large", "group_cancellation", "state_dependent", "rare_tail")
RULES = ["R1", "R2", "R3", "R5"]
GROUP_SIGN = np.tile(np.r_[np.ones(8), -np.ones(8)], D // 16)


def sample(dist, eff, rng):
    n = N_DEV + N_CONF
    G = rng.normal(0, 1, (n, D))
    if dist == "dense_small":
        e = rng.normal(0, 1, (n, D)) + eff * 1.0
    elif dist == "sparse_large":
        p, s = 0.05, 10.0
        sd = np.sqrt(p) * s
        B = rng.random((n, D)) < p
        e = np.where(B, rng.normal(0, s, (n, D)) + eff * sd / p, 0.0)
    elif dist == "group_cancellation":
        e = rng.normal(0, 1, (n, D)) + eff * 1.0 * GROUP_SIGN
    elif dist == "state_dependent":
        e = rng.normal(0, 1, (n, D)) - eff * 1.0 * np.sign(G)
    else:                                                    # rare_tail
        p, big = 0.001, 100.0
        sd = np.sqrt(p) * big
        q = 0.5 + 0.5 * eff * sd / (p * big)                 # mean = p * big * (2q - 1) = eff * sd
        B = rng.random((n, D)) < p
        sign = np.where(rng.random((n, D)) < q, 1.0, -1.0)
        e = np.where(B, big * sign, 0.0)
    return e, G


def one(args):
    dist, eff, rep = args
    from kernel_analyzer import measure
    from kernel_analyzer.reference_eval.analysis import assess_units
    rng = np.random.default_rng([hash(dist) % 2 ** 31, int(eff * 1e4), rep])
    e, G = sample(dist, eff, rng)
    lo, hi = e - WIDTH / 2, e + WIDTH / 2
    ok = np.ones_like(e, dtype=bool)
    rec, _ = assess_units("cal", lo, hi, G, ok, N_DEV, RULES, alignment_reference=G, run_detector=False)
    cls = measure.class_statistics(rec, measure.DEFAULT_RULE_CLASSES, 0.05)
    out = {c: {r: v["judgment"] for r, v in cls[c]["rules"].items()} for c in cls}
    if dist == "rare_tail":                                  # bounded route on R1 (w = -1/sqrt(D)): |a_i| <= 100 * sqrt(D)
        a = -(e[N_DEV:].sum(axis=1)) / np.sqrt(D)
        out["bounded_R1"] = measure.bounded_mean_test(a, np.full(N_CONF, 100.0 * np.sqrt(D)), 0.05)["verdict"]
    return dist, eff, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    jobs = [(d, e, r) for d in DISTS for e in EFFECTS for r in range(a.reps)]
    t0 = time.time()
    tally = defaultdict(Counter)
    with ProcessPoolExecutor(max_workers=24) as ex:
        for dist, eff, out in ex.map(one, jobs, chunksize=20):
            for cls in ("fixed_mean", "aligned"):
                js = out[cls]
                for r, j in js.items():
                    tally[(dist, eff, r)][j.split(" (")[0]] += 1
                k = ("nonzero" if any(v.startswith("nonzero") for v in js.values()) else
                     "cannot judge" if all(v.startswith(("cannot", "not est")) for v in js.values()) else "not confirmed")
                tally[(dist, eff, cls)][k] += 1
            if "bounded_R1" in out:
                tally[(dist, eff, "bounded_R1")][out["bounded_R1"]] += 1
    rows = []
    for (dist, eff, key), c in sorted(tally.items()):
        n = sum(c.values())
        rows.append({"distribution": dist, "effect": eff, "rule_or_class": key, "replicates": n,
                     "reject_rate": (c.get("nonzero", 0) + c.get("DETECTED_POSITIVE", 0) + c.get("DETECTED_NEGATIVE", 0)) / n,
                     "cannot_judge_rate": sum(v for k2, v in c.items() if k2.startswith(("cannot", "not est", "NOT_EST"))) / n,
                     "counts": dict(c)})
    out = {"protocol": "docs/protocol_general_capability_v1_20261008.md section 4 + deviation 1",
           "design": {"development": N_DEV, "confirmation": N_CONF, "coordinates": D, "interval_width": WIDTH,
                      "replicates": a.reps, "effects": EFFECTS}, "seconds": round(time.time() - t0, 1), "rows": rows}
    p = ROOT / "results/general/calibration_five_distributions.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        if r["rule_or_class"] in ("fixed_mean", "aligned", "bounded_R1"):
            print(f"{r['distribution']:20s} eff {r['effect']:<6} {r['rule_or_class']:11s} reject {r['reject_rate']:.3f} "
                  f"cannot {r['cannot_judge_rate']:.3f}")
    print("seconds", out["seconds"])


if __name__ == "__main__":
    main()
