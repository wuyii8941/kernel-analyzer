#!/usr/bin/env python3
"""Summarize the five-checkpoint Go = R + M causal split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.long_horizon_trigger import atomic_json, under_root


METRICS = (
    "candidate_minus_eager_projection",
    "r_shapley_removal_projection",
    "m_shapley_removal_projection",
    "total_go_path_removal_projection",
    "reference_r_candidate_m_residual_projection",
    "candidate_r_reference_m_residual_projection",
    "reference_r_reference_m_residual_projection",
)


def interval(values: list[float], draws: int = 20000) -> list[float]:
    rng = random.Random(20260805)
    n = len(values)
    samples = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)
    )
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [under_root(path, "input") for path in args.input]
    output = under_root(args.output, "output")
    payloads = [json.loads(path.read_text()) for path in paths]
    steps = [int(value["checkpoint_step"]) for value in payloads]
    layers = {int(value.get("target_layer", 23)) for value in payloads}
    if sorted(steps) != [64, 256, 1024, 2048, 4096] or len(layers) != 1:
        raise ValueError(f"unexpected checkpoint set: {steps}")
    if any(
        value.get("status") != "COMPLETE" or len(value.get("rows", [])) != 32
        for value in payloads
    ):
        raise ValueError("every input must contain 32 complete rows")
    by_state = {index: [] for index in range(8, 40)}
    for value in payloads:
        for row in value["rows"]:
            by_state[int(row["state_index"])].append(row)
    if any(len(rows) != 5 for rows in by_state.values()):
        raise ValueError("state/checkpoint cross product is incomplete")

    summary = {}
    for metric in METRICS:
        aggregates = [
            sum(row[metric] for row in rows) / len(rows) for rows in by_state.values()
        ]
        summary[metric] = {
            "state_cluster_mean": sum(aggregates) / len(aggregates),
            "state_cluster_bootstrap_95": interval(aggregates),
            "positive_states": sum(value > 0 for value in aggregates),
            "states": len(aggregates),
            "per_checkpoint_mean": {
                str(value["checkpoint_step"]): sum(
                    row[metric] for row in value["rows"]
                )
                / 32
                for value in payloads
            },
        }

    original_total = summary["candidate_minus_eager_projection"]["state_cluster_mean"]
    go_removal = summary["total_go_path_removal_projection"]["state_cluster_mean"]
    result = {
        "schema": "kernel-analyzer-l23-go-path-summary-v1",
        "status": "COMPLETE",
        "target_layer": next(iter(layers)),
        "checkpoints": sorted(steps),
        "states": list(range(8, 40)),
        "metric_summary": summary,
        "ratios": {
            "r_shapley_over_original_total": summary["r_shapley_removal_projection"]["state_cluster_mean"] / original_total,
            "m_shapley_over_original_total": summary["m_shapley_removal_projection"]["state_cluster_mean"] / original_total,
            "go_path_removal_over_original_total": go_removal / original_total,
            "r_shapley_over_go_path_removal": summary["r_shapley_removal_projection"]["state_cluster_mean"] / go_removal,
            "m_shapley_over_go_path_removal": summary["m_shapley_removal_projection"]["state_cluster_mean"] / go_removal,
        },
        "validation": {
            "all_reference_go_replays_bitwise_exact": all(
                row["reference_go_replay_matches"]
                for value in payloads
                for row in value["rows"]
            ),
            "all_rr_go_matches_reference": all(
                row["rr_go_matches_reference"]
                for value in payloads
                for row in value["rows"]
            ),
            "max_shapley_closure_abs": max(
                row["shapley_closure_max_abs"]
                for value in payloads
                for row in value["rows"]
            ),
            "max_candidate_restoration_sham_abs": max(
                row["candidate_restoration_sham_max_abs"]
                for value in payloads
                for row in value["rows"]
            ),
            "max_candidate_go_sham_abs": max(
                row["candidate_go_sham_max_abs"]
                for value in payloads
                for row in value["rows"]
            ),
            "tensor_values_saved": False,
        },
        "inputs": [str(path) for path in paths],
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "ratios": result["ratios"]}, sort_keys=True))


if __name__ == "__main__":
    main()
