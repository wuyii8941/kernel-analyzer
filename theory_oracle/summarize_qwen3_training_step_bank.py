#!/usr/bin/env python
"""Summarize Qwen3 finite-bank shift, state heterogeneity and repeat variability."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_bank_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return {
        "mean": statistics.fmean(values),
        "sample_variance_across_state_means": statistics.variance(values)
        if len(values) > 1 else 0.0,
        "sample_sd_across_state_means": statistics.stdev(values)
        if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def repeat_stats(grouped_values: list[list[float]]) -> dict[str, float]:
    ranges = [max(values) - min(values) for values in grouped_values]
    within_variances = [
        statistics.variance(values) if len(values) > 1 else 0.0
        for values in grouped_values
    ]
    return {
        "max_within_state_repeat_range": max(ranges, default=0.0),
        "mean_within_state_sample_variance": statistics.fmean(within_variances)
        if within_variances else 0.0,
    }


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    rows = read_jsonl(result_dir / "states.jsonl")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    sample_rows = read_jsonl(Path(args.samples))
    sample_by_id = {str(row["case_id"]): row for row in sample_rows}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["state_id"]), []).append(row)
    if len(grouped) != int(summary["states"]):
        raise ValueError("state count does not match executor summary")
    if not grouped:
        raise ValueError("result bank is empty")
    repeat_counts = {len(state_rows) for state_rows in grouped.values()}
    if len(repeat_counts) != 1 or min(repeat_counts) < 2:
        raise ValueError(f"inconsistent or insufficient repeats: {sorted(repeat_counts)}")

    state_cards = []
    for state_id, state_rows in sorted(grouped.items()):
        state_rows.sort(key=lambda row: int(row["repeat"]))
        sample = sample_by_id.get(state_id)
        if sample is None:
            raise ValueError(f"missing frozen sample metadata for {state_id}")
        metadata = sample.get("metadata") or {}
        card = {
            "state_id": state_id,
            "rollout_batch": metadata.get("rollout_batch"),
            "repeat_count": len(state_rows),
            "loss_signed_delta_mean": statistics.fmean(
                float(row["loss_signed_delta"]) for row in state_rows
            ),
            "model_next_l2_mean": statistics.fmean(
                float(row["model_next"]["floating_difference_l2"])
                for row in state_rows
            ),
            "optimizer_moment_l2_mean": statistics.fmean(
                float(row["optimizer_next"]["floating_moment_difference_l2"])
                for row in state_rows
            ),
            "greedy_token_disagreements": max(
                int(row["greedy_token_disagreement_count"]) for row in state_rows
            ),
            "exact_verdicts": sorted(
                {str(row["exact_transition_verdict"]) for row in state_rows}
            ),
        }
        state_cards.append(card)

    loss_by_state = [card["loss_signed_delta_mean"] for card in state_cards]
    model_l2_by_state = [card["model_next_l2_mean"] for card in state_cards]
    moment_l2_by_state = [card["optimizer_moment_l2_mean"] for card in state_cards]
    rollout_batches = sorted(
        {card["rollout_batch"] for card in state_cards}, key=lambda value: str(value)
    )

    output = {
        "schema_version": "forkcert.qwen3-training-step-bank-summary.v0.1",
        "source_result_dir": str(result_dir.resolve()),
        "states": len(state_cards),
        "repeats_per_state": next(iter(repeat_counts)),
        "rollout_batch_clusters": rollout_batches,
        "rollout_batch_cluster_count": len(rollout_batches),
        "average_relative_shift_on_frozen_bank": {
            "loss_signed_delta": finite_bank_stats(loss_by_state),
            "positive_states": sum(value > 0 for value in loss_by_state),
            "negative_states": sum(value < 0 for value in loss_by_state),
            "zero_states": sum(value == 0 for value in loss_by_state),
        },
        "state_heterogeneity_on_frozen_bank": {
            "loss_signed_delta": finite_bank_stats(loss_by_state),
            "model_next_l2": finite_bank_stats(model_l2_by_state),
            "optimizer_moment_l2": finite_bank_stats(moment_l2_by_state),
        },
        "within_state_runtime_variability": {
            "loss_signed_delta": repeat_stats([
                [float(row["loss_signed_delta"]) for row in grouped[card["state_id"]]]
                for card in state_cards
            ]),
            "model_next_l2": repeat_stats([
                [float(row["model_next"]["floating_difference_l2"])
                 for row in grouped[card["state_id"]]]
                for card in state_cards
            ]),
            "optimizer_moment_l2": repeat_stats([
                [float(row["optimizer_next"]["floating_moment_difference_l2"])
                 for row in grouped[card["state_id"]]]
                for card in state_cards
            ]),
        },
        "sampling_uncertainty": {
            "verdict": "INDETERMINATE",
            "reason": (
                "The bank contains multiple responses nested within only "
                f"{len(rollout_batches)} rollout-batch prompt clusters. It is a frozen "
                "mechanics bank, not an independently sampled target-state population; "
                "a row-wise bootstrap or t interval would be pseudo-replication."
            ),
        },
        "oracle_boundaries": {
            "exact_transition_counts": {
                "accept": int(summary["exact_accept_states"]),
                "reject": int(summary["exact_reject_states"]),
                "invalid": int(summary["invalid_states"]),
            },
            "numerical_transition": str(summary["numerical_transition_verdict"]),
            "impact": str(summary["impact_verdict"]),
            "greedy_sequence_disagreement_states": int(
                summary["greedy_sequence_disagreement_states"]
            ),
        },
        "state_cards": state_cards,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

