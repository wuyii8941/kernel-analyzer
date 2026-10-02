#!/usr/bin/env python3
"""Calibrate the bias decision rule on data with a known mean (CPU only).

The rule is the one used by the unified entry: learn a fixed direction from
the calibration units (normalized mean), project the confirmation units on
it, two-sided t interval at alpha, endpoint-conservative verdict on [l, h],
Holm over a declared family.  Units are u_i = mu * v + noise_i in R^d with v
unknown to the rule.  Noise per coordinate has unit variance and is Gaussian,
Student-t(3), or sparse with rare large values.

Reported: false-positive rate at mu = 0 with a Clopper-Pearson interval,
family-wise error under Holm, and detection probability as a function of
effect size, dimension and confirmation sample size.  The interval width w
models reference uncertainty entering the endpoints.

    python scripts/calibrate_decision_rule.py --out results/reference_eval/decision_rule_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.analysis import _holm, _summarize  # noqa: E402

ALPHA = 0.05


def noise(rng, kind, shape):
    if kind == "gaussian":
        return rng.standard_normal(shape)
    if kind == "student_t3":
        return rng.standard_t(3, shape) / math.sqrt(3.0)
    if kind == "sparse_spike":
        # 98% small Gaussian, 2% large; unit variance overall.
        small, big, p = 0.3, 1.0, 0.02
        big_scale = math.sqrt((1 - (1 - p) * small ** 2) / p)
        mask = rng.random(shape) < p
        return np.where(mask, rng.standard_normal(shape) * big_scale, rng.standard_normal(shape) * small)
    raise ValueError(kind)


def simulate(rng, kind, d, mu, n_cal, n_conf, reps, chunk=200):
    """Projections of the confirmation units on the learned direction: (reps, n_conf)."""

    out = []
    for start in range(0, reps, chunk):
        r = min(chunk, reps - start)
        cal = noise(rng, kind, (r, n_cal, d))
        cal[:, :, 0] += mu
        direction = cal.mean(axis=1)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        conf = noise(rng, kind, (r, n_conf, d))
        conf[:, :, 0] += mu
        out.append(np.einsum("rnd,rd->rn", conf, direction))
    return np.concatenate(out)


def verdicts(proj, width):
    """Vectorized endpoint-conservative verdicts: +1 positive, -1 negative, 0 not confirmed."""

    n = proj.shape[1]
    mean = proj.mean(axis=1)
    sd = proj.std(axis=1, ddof=1)
    half = t.ppf(1 - ALPHA / 2, n - 1) * sd / math.sqrt(n)
    lower = mean - width - half
    upper = mean + width + half
    return np.where(lower > 0, 1, np.where(upper < 0, -1, 0)), mean, sd


def p_values(proj, width):
    n = proj.shape[1]
    sd = proj.std(axis=1, ddof=1)
    se = sd / math.sqrt(n)
    mean = proj.mean(axis=1)
    p_l = 2 * t.sf(np.abs(mean - width) / se, n - 1)
    p_h = 2 * t.sf(np.abs(mean + width) / se, n - 1)
    return np.maximum(p_l, p_h)


def clopper_pearson(k, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return [float(lo), float(hi)]


def check_equivalence(rng):
    """The vectorized verdict equals analysis._summarize on sampled trials."""

    mismatches = 0
    for trial in range(200):
        proj = rng.standard_normal((1, 64)) * 1.0 + rng.choice([0.0, 0.2, 0.5])
        w = rng.choice([0.0, 0.05])
        v, _, _ = verdicts(proj, w)
        s = _summarize("x", "fixed_direction", proj[0] - w, proj[0] + w, ALPHA)
        expect = {"DETECTED_POSITIVE": 1, "DETECTED_NEGATIVE": -1, "NOT_CONFIRMED": 0}[s["verdict"]]
        mismatches += int(v[0] != expect)
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path, default=ROOT / "results/reference_eval/figures/decision_rule_power.png")
    parser.add_argument("--null-reps", type=int, default=4000)
    parser.add_argument("--power-reps", type=int, default=1000)
    args = parser.parse_args()
    rng = np.random.default_rng(20261003)
    report = {"schema": "kernel-analyzer-decision-rule-calibration-v1", "alpha": ALPHA,
              "rule": "fixed direction from 32 calibration units; t interval on confirmation projections; "
                      "endpoint-conservative on [a - w, a + w]",
              "equivalence_mismatches_vs_analysis_summarize": check_equivalence(rng)}
    noises = ("gaussian", "student_t3", "sparse_spike")
    dims = (1, 64, 1024)

    null_rows = []
    for kind in noises:
        for d in dims:
            for w in (0.0, 0.1):
                proj = simulate(rng, kind, d, 0.0, 32, 64, args.null_reps)
                v, _, _ = verdicts(proj, w)
                k = int((v != 0).sum())
                null_rows.append({"noise": kind, "d": d, "width": w, "reps": args.null_reps, "false_positives": k,
                                  "rate": k / args.null_reps, "interval_95": clopper_pearson(k, args.null_reps)})
                print("null", kind, d, w, k / args.null_reps)
    report["false_positive_rate"] = null_rows

    # Family-wise error with Holm over six independent null comparisons.
    fam = []
    for kind in noises:
        reps = 2000
        pvals = np.stack([p_values(simulate(rng, kind, 64, 0.0, 32, 64, reps), 0.0) for _ in range(6)], axis=1)
        rejected = sum(any(_holm(list(row), ALPHA)) for row in pvals)
        fam.append({"noise": kind, "family_size": 6, "reps": reps, "familywise_errors": int(rejected),
                    "rate": rejected / reps, "interval_95": clopper_pearson(int(rejected), reps)})
        print("holm", kind, rejected / reps)
    report["holm_familywise_error"] = fam

    effects = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    power_rows = []
    for kind in noises:
        for d in dims:
            for mu in effects[1:]:
                proj = simulate(rng, kind, d, mu, 32, 64, args.power_reps)
                v, _, _ = verdicts(proj, 0.0)
                power_rows.append({"noise": kind, "d": d, "n_conf": 64, "mu": mu, "width": 0.0,
                                   "detect_positive": float((v == 1).mean()), "detect_negative": float((v == -1).mean())})
    for n_conf in (16, 32, 128):
        for mu in effects[1:]:
            proj = simulate(rng, "gaussian", 64, mu, 32, n_conf, args.power_reps)
            v, _, _ = verdicts(proj, 0.0)
            power_rows.append({"noise": "gaussian", "d": 64, "n_conf": n_conf, "mu": mu, "width": 0.0,
                               "detect_positive": float((v == 1).mean()), "detect_negative": float((v == -1).mean())})
    for w in (0.05, 0.1, 0.2):
        for mu in effects[1:]:
            proj = simulate(rng, "gaussian", 64, mu, 32, 64, args.power_reps)
            v, _, _ = verdicts(proj, w)
            power_rows.append({"noise": "gaussian", "d": 64, "n_conf": 64, "mu": mu, "width": w,
                               "detect_positive": float((v == 1).mean()), "detect_negative": float((v == -1).mean())})
    report["detection_probability"] = power_rows
    report["notes"] = [
        "mu is the mean of each unit along the true direction, in units of the per-coordinate noise sd.",
        "In d dimensions the learned direction is noisy, so the same mu is harder to detect as d grows; "
        "the curves are the tool's sensitivity, not an a-priori magnitude bound (no bound M is used).",
        "Wrong-sign detections are reported separately (detect_negative at mu > 0).",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    plot(report, args.figure)


def plot(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["detection_probability"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, kind in zip(axes, ("gaussian", "student_t3", "sparse_spike")):
        for d, color in zip((1, 64, 1024), ("#1f5f8b", "#d1495b", "#3d8f5f")):
            pts = sorted((r["mu"], r["detect_positive"]) for r in rows
                         if r["noise"] == kind and r["d"] == d and r["n_conf"] == 64 and r["width"] == 0.0)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=color, label=f"d = {d}")
        ax.axhline(0.05, color="#888", lw=0.8, ls="--")
        ax.set_title(kind.replace("_", " "))
        ax.set_xscale("log")
        ax.set_xlabel("mean along the true direction (noise sd units)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("detection probability (64 confirmation units)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
