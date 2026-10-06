#!/usr/bin/env python3
"""Sensitivity curves (evaluation plan, figure 3 / work item D): detection probability of each rule under
declared effect structures, through the production decision layer (analysis.assess_units).

Residuals e[u, j] = b * s_j(u) + sigma * eps[u, j] (sigma = 1) on d coordinates, 96 units (32 development, 64
confirmation), reference r[u, j] = 3 + N(0, 1) (``--reference positive``) or 3 N(0, 1) (``--reference mixed``).  With
the positive reference -sign(r) is -1 almost everywhere, so toward_zero coincides with a uniform negative shift and
R1, R2, R3 cannot be told apart; the mixed reference separates them (uniform: no alignment with the reference;
toward_zero: zero coordinate mean):
  uniform        s_j = 1 for every coordinate (target of R1, the fixed -1/sqrt(n) direction)
  toward_zero    s_j = -sign(r[u, j]) (shrinkage; target of R2 / R3)
  fixed_random   s_j = a random +-1 pattern fixed across units (no declared rule targets it; R5 learns it)
The effect is reported as theta = b * sqrt(d) / sigma, the mean of the projection on the best unit direction in
units of its per-unit noise; with 64 confirmation units a t test has ~50% power near theta = 0.25.

    python scripts/sensitivity_curves.py --reps 200 --out results/reference_eval/sensitivity_curves.json
    python scripts/sensitivity_curves.py --reps 200 --reference mixed --out results/reference_eval/sensitivity_curves_mixed.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.reference_eval.analysis import assess_units  # noqa: E402

RULES = ["R1", "R2", "R3", "R5"]
THETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/reference_eval/sensitivity_curves.json")
    ap.add_argument("--reference", choices=("positive", "mixed"), default="positive")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for structure in ("uniform", "toward_zero", "fixed_random"):
        for d in (16, 256, 4096):
            pattern = rng.choice([-1.0, 1.0], size=d)
            for theta in THETAS:
                b = theta / np.sqrt(d)
                hits = {r: 0 for r in RULES}
                for _ in range(a.reps):
                    ref = (3.0 + rng.normal(size=(96, d))) if a.reference == "positive" else 3.0 * rng.normal(size=(96, d))
                    if structure == "uniform":
                        s = np.ones((96, d))
                    elif structure == "toward_zero":
                        s = -np.sign(ref)
                    else:
                        s = np.broadcast_to(pattern, (96, d))
                    e = b * s + rng.normal(size=(96, d))
                    rec, _ = assess_units("s", e, e, ref, np.ones((96, d), bool), 32, RULES, run_detector=False)
                    for r in rec["rules"]:
                        hits[r["rule"]] += str(r.get("verdict", "")).startswith("DETECTED")
                rows.append({"structure": structure, "d": d, "theta": theta,
                             **{f"p_{k}": v / a.reps for k, v in hits.items()}})
                with open(a.out.with_suffix(".partial.jsonl"), "a") as fh:  # survive interruption
                    fh.write(json.dumps(rows[-1]) + "\n")
                print(f"{structure:12s} d={d:5d} theta={theta:4.2f} " + " ".join(f"{k}={v / a.reps:.2f}" for k, v in hits.items()),
                      flush=True)
    a.out.write_text(json.dumps({"reps": a.reps, "reference": a.reference, "units": {"development": 32, "confirmation": 64}, "rules": RULES,
                                 "alpha_per_test": 0.05, "multiplicity": "none (per test)", "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
