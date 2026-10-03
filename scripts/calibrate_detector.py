#!/usr/bin/env python3
"""Calibrate the default detector (detect.py) on data with known answers (CPU).

Units: e = mean pattern + noise in d = 32 x 32 coordinates, 128 units (64 development, 64 confirmation),
the sizes the blind test uses by default.  The output K is drawn independently (N(0, 1) per coordinate)
except under the scaling mechanism, where e = eps * K + noise.

Reported per noise type: the rate of each verdict under a zero mean (false detections), and the detection
rate of every test and of each family for mean patterns the detector is not told about.

    python scripts/calibrate_detector.py --out results/reference_eval/detector_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_decision_rule import clopper_pearson, noise  # noqa: E402
from kernel_analyzer.reference_eval.detect import detect  # noqa: E402

R = C = 32
D = R * C
N, N_DEV = 128, 64
NOISES = ("gaussian", "student_t3", "sparse_spike", "skewed_q05", "skewed_q01")


def pattern(name, mu, rng):
    v = np.zeros(D)
    if name == "dense":
        v[:] = 1.0
    elif name == "sparse":
        v[rng.integers(D)] = 1.0
    elif name == "one_row":
        r = rng.integers(R)
        v.reshape(R, C)[r] = 1.0
    elif name == "cancelling":  # +c, -c alternating: the coordinate sum is exactly zero
        v[0::2], v[1::2] = 1.0, -1.0
    elif name == "random_signs":
        v = rng.choice([-1.0, 1.0], D)
    return mu * v / np.linalg.norm(v)


def run(rng, kind, mean_pattern=None, mu=0.0, eps=0.0):
    k = rng.standard_normal((N, D))
    e = noise(rng, kind, (N, D))
    if mean_pattern is not None:
        e = e + pattern(mean_pattern, mu, rng)
    if eps:
        e = e + eps * k
    return detect(e, k, N_DEV, shape=(R, C), seed=int(rng.integers(1 << 31)))


def tally(results):
    out = {}
    for fam in ("vector_mean", "alignment"):
        verdicts = [r[fam]["verdict"] for r in results]
        out[fam] = {v: verdicts.count(v) / len(results) for v in ("DETECTED", "EXPLORATORY_ONLY", "NOT_CONFIRMED")}
        tests = {}
        for r in results:
            for t in r[fam]["tests"]:
                tests.setdefault(t["test"], []).append(t["verdict"] == "DETECTED")
        out[fam]["per_test_detected"] = {k: float(np.mean(v)) for k, v in tests.items()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--null-reps", type=int, default=1000)
    parser.add_argument("--power-reps", type=int, default=300)
    args = parser.parse_args()
    rng = np.random.default_rng(20261012)
    report = {"schema": "kernel-analyzer-detector-calibration-v1", "units": N, "development_units": N_DEV,
              "coordinates": D, "shape": [R, C], "null": {}, "power": {}}
    for kind in NOISES:
        res = [run(rng, kind) for _ in range(args.null_reps)]
        t = tally(res)
        for fam in ("vector_mean", "alignment"):
            k = int(round(t[fam]["DETECTED"] * args.null_reps))
            t[fam]["false_detection_interval_95"] = clopper_pearson(k, args.null_reps)
        report["null"][kind] = t
        print("null", kind, {f: (t[f]["DETECTED"], t[f]["EXPLORATORY_ONLY"]) for f in ("vector_mean", "alignment")},
              flush=True)
    for kind in ("gaussian", "student_t3", "skewed_q05"):
        for pat in ("dense", "sparse", "one_row", "cancelling", "random_signs"):
            for mu in (1.0, 2.0, 4.0):
                res = [run(rng, kind, pat, mu) for _ in range(args.power_reps)]
                report["power"][f"{kind}/{pat}/mu={mu}"] = tally(res)
            print("power", kind, pat, {mu: report["power"][f"{kind}/{pat}/mu={mu}"]["vector_mean"]["DETECTED"]
                                       for mu in (1.0, 2.0, 4.0)}, flush=True)
        for eps in (0.01, 0.03, 0.1):
            res = [run(rng, kind, eps=eps) for _ in range(args.power_reps)]
            report["power"][f"{kind}/scaling/eps={eps}"] = tally(res)
        print("power", kind, "scaling", {eps: report["power"][f"{kind}/scaling/eps={eps}"]["alignment"]["DETECTED"]
                                         for eps in (0.01, 0.03, 0.1)}, flush=True)
    report["notes"] = ["mu is |E[e]| in units of the per-coordinate noise sd; eps scales e = eps * K",
                       "family verdict DETECTED requires a Holm rejection with tail diagnostics supported"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
