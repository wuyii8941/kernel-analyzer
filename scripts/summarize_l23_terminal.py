#!/usr/bin/env python3
"""Summarize the five-checkpoint terminal nested-repair campaign."""

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
    "logits_vjp_stage_removal_projection",
    "local_logits_vjp_removal_projection",
    "upstream_logits_removal_projection",
    "lm_head_mm_stage_removal_projection",
    "final_norm_stage_removal_projection",
    "terminal_total_removal_projection",
    "reference_g_residual_projection",
    "reference_dn_residual_projection",
    "reference_t_residual_projection",
)

def interval(values, draws=20000):
    rng = random.Random(20260805); n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return [samples[int(.025 * draws)], samples[int(.975 * draws)]]

def main():
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
        for row in value["rows"]: by_state[int(row["state_index"])].append(row)
    if any(len(rows) != 5 for rows in by_state.values()):
        raise ValueError("state/checkpoint cross product is incomplete")
    summary = {}
    for metric in METRICS:
        aggregates = [sum(row[metric] for row in rows) / 5 for rows in by_state.values()]
        summary[metric] = {
            "state_cluster_mean": sum(aggregates) / 32,
            "state_cluster_bootstrap_95": interval(aggregates),
            "positive_states": sum(value > 0 for value in aggregates),
            "states": 32,
            "per_checkpoint_mean": {str(value["checkpoint_step"]): sum(row[metric] for row in value["rows"]) / 32 for value in payloads},
        }
    total = summary["candidate_minus_eager_projection"]["state_cluster_mean"]
    result = {
        "schema": "kernel-analyzer-l23-terminal-summary-v1",
        "status": "COMPLETE",
        "checkpoints": sorted(steps),
        "states": list(range(8, 40)),
        "metric_summary": summary,
        "ratios_over_original_total": {
            "logits_vjp": summary["logits_vjp_stage_removal_projection"]["state_cluster_mean"] / total,
            "local_logits_vjp": summary["local_logits_vjp_removal_projection"]["state_cluster_mean"] / total,
            "upstream_logits": summary["upstream_logits_removal_projection"]["state_cluster_mean"] / total,
            "lm_head_mm": summary["lm_head_mm_stage_removal_projection"]["state_cluster_mean"] / total,
            "final_norm": summary["final_norm_stage_removal_projection"]["state_cluster_mean"] / total,
            "terminal_total": summary["terminal_total_removal_projection"]["state_cluster_mean"] / total,
            "post_terminal_residual": summary["reference_t_residual_projection"]["state_cluster_mean"] / total,
        },
        "validation": {
            "all_reference_lm_head_input_vjp_replays_bitwise_exact": all(row["reference_lm_head_input_vjp_replay_matches"] for value in payloads for row in value["rows"]),
            "all_analytic_logits_vjp_replays_bitwise_exact": all(row["analytic_logits_vjp_matches_eager"] for value in payloads for row in value["rows"]),
            "all_t_ref_matches_reference": all(row["t_ref_matches_reference"] for value in payloads for row in value["rows"]),
            "max_stage_closure_abs": max(row["terminal_stage_closure_max_abs"] for value in payloads for row in value["rows"]),
            "max_logits_substage_closure_abs": max(row["logits_substage_closure_max_abs"] for value in payloads for row in value["rows"]),
            "max_candidate_restoration_sham_abs": max(row["candidate_restoration_sham_max_abs"] for value in payloads for row in value["rows"]),
            "max_candidate_g_sham_abs": max(row["candidate_g_sham_max_abs"] for value in payloads for row in value["rows"]),
            "max_candidate_dn_sham_abs": max(row["candidate_dn_sham_max_abs"] for value in payloads for row in value["rows"]),
            "tensor_values_saved": False,
        },
        "inputs": [str(path) for path in paths],
    }
    result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(output, result)
    print(json.dumps({"output": str(output), "ratios": result["ratios_over_original_total"]}, sort_keys=True))

if __name__ == "__main__": main()
