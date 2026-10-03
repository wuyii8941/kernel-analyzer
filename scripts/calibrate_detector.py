#!/usr/bin/env python3
"""Calibrate the default detector (detect.py) on data with known answers (CPU).

Units: e = mean pattern + noise in d = 32 x 32 coordinates, 128 units (64 development, 64 confirmation),
the sizes the blind test uses by default.  The output K is drawn independently (N(0, 1) per coordinate)
except under the scaling mechanism, where e = eps * K + noise.

Reported per noise type: the rate of each verdict under a zero mean (false detections), and the detection
rate of every test and of each family for mean patterns the detector is not told about.

    python scripts/calibrate_detector.py --out results/reference_eval/detector_calibration_v2.json
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


def run(rng, kind, mean_pattern=None, mu=0.0, eps=0.0, box=None):
    """box: None (point residuals), or (artifact, radius) in units of the noise sd: the reference enclosure
    is [e + a - r, e + a + r] with the midpoint offset a = artifact * tanh(K) (a stable evaluation artefact
    correlated with K) and r = |a| + radius, so the box always contains the true residual e.  kind "zero"
    is a residual that is exactly zero (K_R = f)."""

    k = rng.standard_normal((N, D))
    e = np.zeros((N, D)) if kind == "zero" else noise(rng, kind, (N, D))
    if mean_pattern is not None:
        e = e + pattern(mean_pattern, mu, rng)
    if eps:
        e = e + eps * k
    lo = hi = e
    if box is not None:
        artifact, radius = box
        a = artifact * np.tanh(k)
        r = np.abs(a) + radius
        lo, hi = e + a - r, e + a + r
    return detect(lo, hi, k, N_DEV, shape=(R, C), seed=int(rng.integers(1 << 31)))


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


def _task(args):
    key, kind, pat, mu, eps, box, reps, seed = args
    rng = np.random.default_rng(seed)
    return key, tally([run(rng, kind, pat, mu, eps, box) for _ in range(reps)]), reps


def main():
    from multiprocessing import Pool

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--null-reps", type=int, default=1000)
    parser.add_argument("--power-reps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    from kernel_analyzer.reference_eval.detect import VERSION

    report = {"schema": "kernel-analyzer-detector-calibration-v2", "detector_version": VERSION, "units": N,
              "development_units": N_DEV, "coordinates": D, "shape": [R, C], "null": {}, "power": {},
              "reference_intervals": {}}
    # (section, key, noise, pattern, mu, eps, box, reps); every task has its own seed
    tasks = [("null", kind, kind, None, 0.0, 0.0, None, args.null_reps) for kind in NOISES]
    for kind in ("gaussian", "student_t3", "skewed_q05"):
        for pat in ("dense", "sparse", "one_row", "cancelling", "random_signs"):
            for mu in (1.0, 2.0, 4.0):
                tasks.append(("power", f"{kind}/{pat}/mu={mu}", kind, pat, mu, 0.0, None, args.power_reps))
        for eps in (0.01, 0.03, 0.1):
            tasks.append(("power", f"{kind}/scaling/eps={eps}", kind, None, 0.0, eps, None, args.power_reps))
    # reference intervals: nulls whose midpoints carry a stable artefact correlated with K (the situation in
    # which version 1, on midpoints, reported detections), and the power cost of box width
    for name, kind, box in (("zero_residual/artifact=1e-3", "zero", (1e-3, 1e-4)),
                            ("gaussian/artifact=0.2/radius=0.1", "gaussian", (0.2, 0.1)),
                            ("gaussian/artifact=0/radius=0.5", "gaussian", (0.0, 0.5))):
        tasks.append(("reference_intervals", f"null/{name}", kind, None, 0.0, 0.0, box, args.null_reps))
    for radius in (0.0, 0.1, 0.5):
        for pat, mu in (("dense", 2.0), ("cancelling", 4.0)):
            tasks.append(("reference_intervals", f"power/gaussian/{pat}/mu={mu}/radius={radius}", "gaussian", pat,
                          mu, 0.0, (0.0, radius), args.power_reps))
        tasks.append(("reference_intervals", f"power/gaussian/scaling/eps=0.03/radius={radius}", "gaussian", None,
                      0.0, 0.03, (0.0, radius), args.power_reps))
    jobs = [(t[1], t[2], t[3], t[4], t[5], t[6], t[7], [20261012, i]) for i, t in enumerate(tasks)]
    with Pool(args.workers) as pool:
        results = {key: (t, reps) for key, t, reps in pool.imap_unordered(_task, jobs)}
    for section, key, *_ in tasks:
        t, reps = results[key]
        if key.startswith("null/") or section == "null":
            for fam in ("vector_mean", "alignment"):
                k = int(round(t[fam]["DETECTED"] * reps))
                t[fam]["false_detection_interval_95"] = clopper_pearson(k, reps)
        report[section][key] = t
        print(section, key, {f: t[f]["DETECTED"] for f in ("vector_mean", "alignment")}, flush=True)
    report["notes"] = ["mu is |E[e]| in units of the per-coordinate noise sd; eps scales e = eps * K",
                       "family verdict DETECTED requires a Holm rejection with tail diagnostics supported",
                       "reference_intervals: box = [e + a - r, e + a + r], a = artifact * tanh(K), "
                       "r = |a| + radius (noise-sd units); the box always contains the true residual",
                       "every scenario has its own seed [20261012, task index]"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
