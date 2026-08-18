#!/usr/bin/env python3
"""Compact the five-checkpoint bmm_76 campaign with state-cluster intervals."""

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
    "s_shapley_removal_projection",
    "k_shapley_removal_projection",
    "total_bmm_operand_removal_projection",
    "reference_s_reference_k_residual_projection",
    "u_shapley_removal_projection",
    "p_program_shapley_removal_projection",
    "softmax_factor_total_removal_projection",
    "reference_u_reference_p_program_residual_projection",
    "d_shapley_removal_projection",
    "v_shapley_removal_projection",
    "u_bmm_total_removal_projection",
    "reference_d_reference_v_residual_projection",
    "go_removal_projection",
    "reference_go_residual_projection",
)


def interval(values: list[float], draws: int = 20000) -> list[float]:
    rng = random.Random(20260805)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
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
    if sorted(steps) != [64, 256, 1024, 2048, 4096]:
        raise ValueError(f"unexpected checkpoint set: {steps}")
    if any(value.get("status") != "COMPLETE" or len(value.get("rows", [])) != 32 for value in payloads):
        raise ValueError("every input must contain 32 complete rows")
    by_state = {index: [] for index in range(8, 40)}
    for value in payloads:
        for row in value["rows"]:
            by_state[int(row["state_index"])].append(row)
    if any(len(rows) != 5 for rows in by_state.values()):
        raise ValueError("state/checkpoint cross product is incomplete")
    summary = {}
    for metric in METRICS:
        aggregates = [sum(row[metric] for row in rows) / len(rows) for rows in by_state.values()]
        summary[metric] = {
            "state_cluster_mean": sum(aggregates) / len(aggregates),
            "state_cluster_bootstrap_95": interval(aggregates),
            "positive_states": sum(value > 0 for value in aggregates),
            "states": len(aggregates),
            "per_checkpoint_mean": {
                str(value["checkpoint_step"]): sum(row[metric] for row in value["rows"]) / 32
                for value in payloads
            },
        }
    result = {
        "schema": "kernel-analyzer-l23-attention-bmm-summary-v1",
        "status": "COMPLETE",
        "checkpoints": sorted(steps),
        "states": list(range(8, 40)),
        "metric_summary": summary,
        "validation": {
            "all_reference_bmm_replays_bitwise_exact": all(
                row["reference_bmm_replay_matches_eager_query_gradient"]
                for value in payloads for row in value["rows"]
            ),
            "max_shapley_closure_abs": max(
                row["shapley_closure_max_abs"] for value in payloads for row in value["rows"]
            ),
            "max_candidate_restoration_sham_abs": max(
                row["candidate_restoration_sham_max_abs"] for value in payloads for row in value["rows"]
            ),
            "tensor_values_saved": False,
        },
        "inputs": [str(path) for path in paths],
    }
    total = summary["candidate_minus_eager_projection"]["state_cluster_mean"]
    result["ratios"] = {
        "s_shapley_over_total": summary["s_shapley_removal_projection"]["state_cluster_mean"] / total,
        "k_shapley_over_total": summary["k_shapley_removal_projection"]["state_cluster_mean"] / total,
        "rr_residual_over_total": summary["reference_s_reference_k_residual_projection"]["state_cluster_mean"] / total,
        "u_shapley_over_total": summary["u_shapley_removal_projection"]["state_cluster_mean"] / total,
        "p_program_shapley_over_total": summary["p_program_shapley_removal_projection"]["state_cluster_mean"] / total,
        "softmax_rr_residual_over_total": summary["reference_u_reference_p_program_residual_projection"]["state_cluster_mean"] / total,
        "d_shapley_over_total": summary["d_shapley_removal_projection"]["state_cluster_mean"] / total,
        "v_shapley_over_total": summary["v_shapley_removal_projection"]["state_cluster_mean"] / total,
        "u_bmm_rr_residual_over_total": summary["reference_d_reference_v_residual_projection"]["state_cluster_mean"] / total,
        "go_removal_over_total": summary["go_removal_projection"]["state_cluster_mean"] / total,
        "go_residual_over_total": summary["reference_go_residual_projection"]["state_cluster_mean"] / total,
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "ratios": result["ratios"]}, sort_keys=True))


if __name__ == "__main__":
    main()
