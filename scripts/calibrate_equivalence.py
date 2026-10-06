#!/usr/bin/env python3
"""Calibration of the decision layer's two axes (evaluation plan, work item D / milestone M1):

1. coverage: the endpoint-conservative two-sided (1 - alpha) interval [lower bound of E[l], upper bound of E[h]]
   contains the true mean projection mu;
2. false equivalence: P(WITHIN_DELTA) when |mu| >= delta, checked at mu = +-delta and +-1.2 delta (must be <= alpha
   at the boundary), with the power at mu = 0 and +-0.5 delta for reference.

Per-unit projections x_i = mu + noise_i (normal, Student t with 3 degrees of freedom, centred lognormal), either as
points (l = h = x) or as boxes (l = x - w_i, h = x + w_i, w_i ~ U(0, 0.4) sd) -- the interval case of the tool.

    python scripts/calibrate_equivalence.py --reps 4000 --out results/reference_eval/calibration_equivalence.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.reference_eval.analysis import _summarize, equivalence  # noqa: E402

ALPHA = 0.05


def noise(rng, kind, size):
    if kind == "normal":
        return rng.normal(size=size)
    if kind == "t3":
        return rng.standard_t(3, size=size) / np.sqrt(3.0)  # unit variance
    s = 0.75  # centred lognormal, unit variance
    z = rng.lognormal(0.0, s, size=size)
    return (z - np.exp(s * s / 2)) / np.sqrt((np.exp(s * s) - 1) * np.exp(s * s))


def run(reps, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for kind in ("normal", "t3", "lognormal"):
        for box in (False, True):
            for n in (16, 32, 64):
                delta = 0.5  # in units of the per-unit noise sd
                for frac in (0.0, 0.5, 1.0, 1.2):
                    for sign in ((1,) if frac == 0 else (1, -1)):
                        mu = sign * frac * delta
                        cover = within = det = 0
                        for _ in range(reps):
                            x = mu + noise(rng, kind, n)
                            w = rng.uniform(0, 0.4, n) if box else np.zeros(n)
                            l, h = x - w, x + w
                            s = _summarize("c", "R", l, h, ALPHA)
                            cover += s["lower_bound_of_E_l"] <= mu <= s["upper_bound_of_E_h"]
                            det += s["verdict"].startswith("DETECTED")
                            within += equivalence(l, h, delta, ALPHA)["verdict"] == "WITHIN_DELTA"
                        rows.append({"noise": kind, "box": box, "n": n, "mu_over_delta": sign * frac,
                                     "coverage": cover / reps, "p_within_delta": within / reps,
                                     "p_detected_nonzero": det / reps})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/reference_eval/calibration_equivalence.json")
    a = ap.parse_args()
    rows = run(a.reps, a.seed)
    a.out.write_text(json.dumps({"alpha": ALPHA, "reps": a.reps, "delta_over_sd": 0.5, "rows": rows}, indent=1))
    worst_cov = min(r["coverage"] for r in rows)
    boundary = [r for r in rows if abs(abs(r["mu_over_delta"]) - 1.0) < 1e-9]
    beyond = [r for r in rows if abs(r["mu_over_delta"]) > 1.0]
    print(f"coverage: min {worst_cov:.4f} over {len(rows)} settings (nominal {1 - ALPHA})")
    print(f"P(within delta) at |mu| = delta: max {max(r['p_within_delta'] for r in boundary):.4f} (must be <= {ALPHA})")
    print(f"P(within delta) at |mu| = 1.2 delta: max {max(r['p_within_delta'] for r in beyond):.4f}")
    for r in rows:
        if r["mu_over_delta"] in (0.0, 1.0):
            print(f"  {r['noise']:9s} box={r['box']!s:5s} n={r['n']:3d} mu/delta={r['mu_over_delta']:+.1f} "
                  f"coverage={r['coverage']:.3f} P(within)={r['p_within_delta']:.3f} P(nonzero)={r['p_detected_nonzero']:.3f}")


if __name__ == "__main__":
    main()
