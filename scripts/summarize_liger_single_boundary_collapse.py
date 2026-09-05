#!/usr/bin/env python3
"""Summarize paired Liger collapse experiments without rerunning training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer.single_boundary_collapse import classify_paired_loss_collapse  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def losses(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray([row["training_loss"] for row in payload["rows"]], dtype=np.float64)


def paired_summary(candidate_path: Path, repair_path: Path) -> dict[str, Any]:
    candidate = load(candidate_path)
    repair = load(repair_path)
    if candidate["stream"] != repair["stream"]:
        raise ValueError("candidate and repair must use the same stream")
    c = losses(candidate)
    r = losses(repair)
    if c.shape != r.shape:
        raise ValueError("candidate and repair must contain the same number of steps")
    if any(
        left["offset_digest"] != right["offset_digest"]
        for left, right in zip(candidate["rows"], repair["rows"], strict=True)
    ):
        raise ValueError("candidate and repair do not use the same batches")
    window = min(128, len(c))
    kernel = np.ones(window, dtype=np.float64) / window
    c_roll = np.convolve(c, kernel, mode="valid")
    r_roll = np.convolve(r, kernel, mode="valid")
    delta = c - r
    tail = delta[-window:]
    return {
        "candidate": str(candidate_path),
        "repair": str(repair_path),
        "stream": candidate["stream"],
        "steps": len(c),
        "collapse_rule": classify_paired_loss_collapse(c, r),
        "maximum_128_step_mean_loss_ratio": float(np.max(c_roll / r_roll)),
        "minimum_128_step_mean_loss_ratio": float(np.min(c_roll / r_roll)),
        "final_128_step_mean_loss_difference": float(np.mean(tail)),
        "final_128_step_mean_loss_ratio": float(np.mean(c[-window:]) / np.mean(r[-window:])),
        "whole_run_mean_loss_difference": float(np.mean(delta)),
        "candidate_validation_loss": candidate["validation_loss_mean"],
        "repair_validation_loss": repair["validation_loss_mean"],
        "validation_loss_difference": (
            candidate["validation_loss_mean"] - repair["validation_loss_mean"]
        ),
        "candidate_finite": bool(np.all(np.isfinite(c))),
        "repair_finite": bool(np.all(np.isfinite(r))),
    }


def checkpoint_difference(candidate_path: Path, repair_path: Path) -> dict[str, float]:
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    repair = torch.load(repair_path, map_location="cpu", weights_only=True)
    squared_difference = 0.0
    squared_repair = 0.0
    maximum = 0.0
    for name, repair_value in repair["master"].items():
        difference = candidate["master"][name].double() - repair_value.double()
        squared_difference += float(torch.sum(difference.square()).item())
        squared_repair += float(torch.sum(repair_value.double().square()).item())
        maximum = max(maximum, float(torch.max(torch.abs(difference)).item()))
    return {
        "parameter_difference_l2": math.sqrt(squared_difference),
        "parameter_difference_relative_l2": math.sqrt(squared_difference / squared_repair),
        "parameter_difference_max_abs": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, action="append", metavar=("CANDIDATE", "REPAIR"), required=True)
    parser.add_argument("--checkpoint-pair", nargs=2, metavar=("CANDIDATE", "REPAIR"))
    args = parser.parse_args()
    pairs = [paired_summary(Path(c), Path(r)) for c, r in args.pair]
    if args.checkpoint_pair:
        pairs[0]["final_parameters"] = checkpoint_difference(
            Path(args.checkpoint_pair[0]), Path(args.checkpoint_pair[1])
        )
    payload = {
        "schema": "kernel-analyzer-liger-single-boundary-collapse-summary-v1",
        "status": "COMPLETE",
        "paired_runs": pairs,
        "natural_candidate_collapse_reproduced": any(
            pair["collapse_rule"]["collapsed"] for pair in pairs
        ),
        "claim_boundary": (
            "A finite validation-loss difference establishes paired trajectory non-identity, "
            "not catastrophic training failure or population-level quality degradation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
