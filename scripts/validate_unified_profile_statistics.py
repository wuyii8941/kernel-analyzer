#!/usr/bin/env python3
"""Synthetic calibration for the frozen additive/aligned/residual profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def signflip_p(values: np.ndarray, rng: np.random.Generator, draws: int = 999) -> float:
    observed = abs(float(values.mean()))
    signs = rng.choice((-1.0, 1.0), size=(draws, len(values)))
    return (1.0 + float(np.sum(np.abs((signs * values).mean(axis=1)) >= observed))) / (draws + 1.0)


def interval(values: np.ndarray, rng: np.random.Generator, draws: int = 500) -> tuple[float, float]:
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate(u: np.ndarray, r: np.ndarray, rng: np.random.Generator) -> dict:
    cal_u, test_u = u[:16], u[16:]
    cal_r, test_r = r[:16], r[16:]
    repair_rms = float(np.sqrt(np.mean(np.sum(test_r * test_r, axis=1))))
    if repair_rms <= 1e-12:
        return {"status": "ABSTAIN_REPAIR_SCALE"}
    direction = cal_u.mean(axis=0)
    if np.linalg.norm(direction) <= 1e-15:
        additive = np.zeros(16)
    else:
        additive = test_u @ (direction / np.linalg.norm(direction)) / repair_rms
    energy = np.sum(test_r * test_r, axis=1)
    aligned = np.sum(test_u * test_r, axis=1) / energy
    cal_alpha = np.sum(cal_u * cal_r, axis=1) / np.sum(cal_r * cal_r, axis=1)
    cal_residual = cal_u - cal_alpha[:, None] * cal_r
    residual_direction = cal_residual.mean(axis=0)
    test_residual = test_u - aligned[:, None] * test_r
    residual = (
        test_residual @ (residual_direction / np.linalg.norm(residual_direction)) / repair_rms
        if np.linalg.norm(residual_direction) > 1e-15 else np.zeros(16)
    )
    branches = {"additive": additive, "aligned": aligned, "orthogonal": residual}
    raw_p = {name: signflip_p(values, rng) for name, values in branches.items()}
    ordered = sorted(raw_p.items(), key=lambda item: item[1])
    corrected = {}; running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (3 - rank) * p)); corrected[name] = running
    return {
        "status": "COMPLETE",
        "estimate": {name: float(values.mean()) for name, values in branches.items()},
        "interval": {name: interval(values, rng) for name, values in branches.items()},
        "holm_p": corrected,
        "detected": {name: corrected[name] < 0.05 for name in branches},
    }


def generate(name: str, rng: np.random.Generator, n: int = 32, d: int = 128):
    r = rng.normal(size=(n, d)); r /= np.linalg.norm(r, axis=1, keepdims=True)
    noise = 0.03 * rng.normal(size=(n, d))
    truth = {"additive": False, "aligned": False, "orthogonal": False}
    if name == "centered_high_variance": u = 0.30 * rng.normal(size=(n, d))
    elif name == "fixed_additive":
        b = np.zeros(d); b[:8] = 0.04
        u = b + noise; truth.update(additive=True, orthogonal=True)
    elif name == "rotating_relative":
        u = 0.08 * r + noise; truth["aligned"] = True
    elif name == "alternating":
        b = np.zeros(d); b[:8] = 0.08
        u = np.asarray([b if i % 2 == 0 else -b for i in range(n)]) + noise
    elif name == "heavy_tail_centered": u = 0.02 * rng.standard_t(df=2.5, size=(n, d))
    elif name == "sparse_additive":
        b = np.zeros(d); b[3] = 0.15
        u = b + noise; truth.update(additive=True, orthogonal=True)
    elif name == "small_repair":
        r *= 1e-14; u = noise
    else: raise ValueError(name)
    return u, r, truth


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=200); args = parser.parse_args()
    scenarios = ("centered_high_variance", "fixed_additive", "rotating_relative", "alternating",
                 "heavy_tail_centered", "sparse_additive", "small_repair")
    rows = {}
    for sidx, name in enumerate(scenarios):
        counts = {"complete": 0, "abstain": 0, "any_detection": 0,
                  "additive": 0, "aligned": 0, "orthogonal": 0}
        coverage = {"additive": 0, "aligned": 0, "orthogonal": 0}
        for trial in range(args.trials):
            rng = np.random.default_rng(20260920 + sidx * 10000 + trial)
            u, r, truth = generate(name, rng)
            result = evaluate(u, r, rng)
            if result["status"] != "COMPLETE": counts["abstain"] += 1; continue
            counts["complete"] += 1
            detected = result["detected"]
            counts["any_detection"] += int(any(detected.values()))
            for branch in ("additive", "aligned", "orthogonal"):
                counts[branch] += int(detected[branch])
                target = 0.0 if not truth[branch] else None
                if target is not None:
                    lo, hi = result["interval"][branch]; coverage[branch] += int(lo <= target <= hi)
        denom = max(counts["complete"], 1)
        rows[name] = {
            "trials": args.trials, "complete": counts["complete"], "abstain": counts["abstain"],
            "family_detection_rate": counts["any_detection"] / denom,
            "branch_detection_rate": {b: counts[b] / denom for b in ("additive", "aligned", "orthogonal")},
            "zero_effect_interval_coverage": {b: coverage[b] / denom for b in coverage},
        }
    null_names = ("centered_high_variance", "alternating", "heavy_tail_centered")
    go = all(rows[x]["family_detection_rate"] <= 0.075 for x in null_names)
    go &= rows["fixed_additive"]["branch_detection_rate"]["additive"] >= 0.80
    go &= rows["rotating_relative"]["branch_detection_rate"]["aligned"] >= 0.80
    go &= rows["small_repair"]["abstain"] == args.trials
    payload = {"schema": "kernel-analyzer-unified-profile-synthetic-validation-v1",
               "status": "GO" if go else "NO_GO", "trials_per_scenario": args.trials,
               "rows": rows,
               "gates": {"null_family_false_positive_lte": 0.075, "target_power_gte": 0.80,
                         "small_repair_all_abstain": True},
               "claim_boundary": "Independent synthetic states; correlated-state inference remains out of scope and must use run/cluster-aware resampling."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__": main()
