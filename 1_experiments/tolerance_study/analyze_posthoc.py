#!/usr/bin/env python3
"""Post-hoc controls (added after the paired results were seen; reported as post-hoc, not as part of the frozen
confirmation): the run-to-run divergence of identical configurations, against which the trajectory distances of B-2
(compiled vs eager RAdam) and A-2 (pure bf16 vs fp32 master) are read, and the diagnostic learning-rate runs.

    python analyze_posthoc.py --out results/posthoc/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import analyze_paired as P

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / ".cache/round2"


def distance_rows(dir_a, dir_b, dir_ref, seeds, steps):
    """Per seed and step: ||theta_a - theta_b|| / ||theta_ref - theta_0|| for both coordinate groups."""
    out = {g: {str(t): [] for t in steps} for g in ("matrices", "vectors")}
    for s in seeds:
        caps = {k: {t: P.load(d / f"seed{s}" / f"capture_step{t:05d}.pt") for t in (1,) + steps}
                for k, d in (("a", dir_a), ("b", dir_b), ("ref", dir_ref))}
        for g, names in (("matrices", P.MATRICES), ("vectors", P.VECTORS)):
            th0 = P.vec(caps["ref"][1], names)
            for t in steps:
                d_ref = float((P.vec(caps["ref"][t], names) - th0).norm())
                d = float((P.vec(caps["a"][t], names) - P.vec(caps["b"][t], names)).norm())
                out[g][str(t)].append(d / d_ref if d_ref else None)
    return {g: {t: P._dist(v) for t, v in rows.items()} for g, rows in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"label": "post-hoc (added after seeing the paired results)"}
    seeds_b = list(range(5000, 5008))
    steps_b = (10, 50, 100, 200)
    res["B_run_to_run"] = {
        "eager_vs_eager_rerun": distance_rows(C / "paired_b/eager", C / "posthoc/eager_rerun", C / "paired_b/eager",
                                              seeds_b, steps_b),
        "compiled_vs_eager": distance_rows(C / "paired_b/compiled", C / "paired_b/eager", C / "paired_b/eager",
                                           seeds_b, steps_b),
        "eval_loss_eager_rerun_minus_eager": P.paired_axes(
            [json.loads((C / "posthoc/eager_rerun" / f"seed{s}" / "log.json").read_text())["log"][-1]["eval_loss"] -
             json.loads((C / "paired_b/eager" / f"seed{s}" / "log.json").read_text())["log"][-1]["eval_loss"]
             for s in seeds_b])}
    seeds_a = [s for s in range(3000, 3004) if (C / "posthoc/bf16_rerun" / f"seed{s}" / "log.json").exists()]
    steps_a = (10, 50, 100, 300, 600)
    if seeds_a:
        res["A_run_to_run"] = {
            "seeds": seeds_a,
            "bf16_vs_bf16_rerun": distance_rows(C / "paired/bf16", C / "posthoc/bf16_rerun", C / "paired/master",
                                                seeds_a, steps_a),
            "bf16_vs_master": distance_rows(C / "paired/bf16", C / "paired/master", C / "paired/master", seeds_a,
                                            steps_a),
            "eval_loss_bf16_rerun_minus_bf16": [
                json.loads((C / "posthoc/bf16_rerun" / f"seed{s}" / "log.json").read_text())["log"][-1]["eval_loss"] -
                json.loads((C / "paired/bf16" / f"seed{s}" / "log.json").read_text())["log"][-1]["eval_loss"]
                for s in seeds_a]}
    diag = {}
    for name, d in (("bf16_lr2e-5 (paired seed 3000)", C / "paired/bf16/seed3000"),
                    ("fp32_master_lr2e-5 (paired seed 3000)", C / "paired/master/seed3000"),
                    ("fp32_everywhere_lr2e-5", C / "posthoc/fp32_all_s3000"),
                    ("fp32_master_lr2e-6", C / "posthoc/master_lr2e-6_s3000"),
                    ("bf16_lr2e-6", C / "posthoc/bf16_lr2e-6_s3000")):
        tl = np.array([r["train_loss"] for r in json.loads((d / "log.json").read_text())["log"][:50]])
        diag[name] = {"train_loss_steps_2_10": float(tl[1:10].mean()), "steps_11_30": float(tl[10:30].mean()),
                      "steps_31_50": float(tl[30:50].mean())}
    res["learning_rate_diagnostic_seed3000_50_steps"] = diag
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "learning_rate_diagnostic_seed3000_50_steps"},
                     indent=1)[:4000])


if __name__ == "__main__":
    main()
