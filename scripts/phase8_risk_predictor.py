#!/usr/bin/env python
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from forkcert.io import read_jsonl


def group_id(row: dict[str, Any]) -> int:
    return int(row["metadata"]["phase1_metadata"]["online_state"]["rollout_batch"])


def signed_values(row: dict[str, Any]) -> tuple[float, float]:
    margin = float(row["logp_ref"]) - float(row["old_logp"]) - float(row["clip_boundary"])
    delta = float(row["logp_alt"]) - float(row["logp_ref"])
    return margin, delta


def stratum(row: dict[str, Any]) -> tuple[int, int]:
    return (int(row["advantage_sign"]), min(int(row["token_index"]) // 16, 7))


def crossing_probability(sorted_deltas: list[float], margin: float) -> float:
    if not sorted_deltas:
        return 0.0
    boundary = -margin
    if margin <= 0:
        index = bisect.bisect_right(sorted_deltas, boundary)
        return (len(sorted_deltas) - index) / len(sorted_deltas)
    index = bisect.bisect_right(sorted_deltas, boundary)
    return index / len(sorted_deltas)


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if not positives:
        return float("nan")
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    found = 0
    total = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            found += 1
            total += found / rank
    return total / positives


def expected_calibration_error(labels: list[int], probabilities: list[float], bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    order = sorted(range(len(labels)), key=lambda index: probabilities[index])
    rows = []
    ece = 0.0
    for bin_index in range(bins):
        lo = bin_index * len(order) // bins
        hi = (bin_index + 1) * len(order) // bins
        indices = order[lo:hi]
        if not indices:
            continue
        predicted = sum(probabilities[index] for index in indices) / len(indices)
        observed = sum(labels[index] for index in indices) / len(indices)
        ece += len(indices) / len(order) * abs(predicted - observed)
        rows.append({"decile": bin_index + 1, "n": len(indices), "predicted_rate": predicted, "observed_rate": observed, "observed_forks": sum(labels[index] for index in indices)})
    return ece, rows


def calibration_split(group: int, test_group: int) -> bool:
    digest = hashlib.sha256(f"forkcert-cal-v1|{test_group}|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 5 == 0


def predict_fold(test_group: int, rows_by_group: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    remaining = [group for group in rows_by_group if group != test_group]
    calibration_groups = {group for group in remaining if calibration_split(group, test_group)}
    fit_groups = set(remaining) - calibration_groups
    if not calibration_groups or not fit_groups:
        raise RuntimeError("empty fit/calibration split")
    global_deltas = sorted(signed_values(row)[1] for group in fit_groups for row in rows_by_group[group])
    conditional: dict[tuple[int, int], list[float]] = defaultdict(list)
    for group in fit_groups:
        for row in rows_by_group[group]:
            conditional[stratum(row)].append(signed_values(row)[1])
    for values in conditional.values():
        values.sort()

    def raw_probability(row: dict[str, Any], conditional_mode: bool) -> float:
        margin, _ = signed_values(row)
        values = conditional.get(stratum(row), global_deltas) if conditional_mode else global_deltas
        return crossing_probability(values, margin)

    calibration_rows = [row for group in calibration_groups for row in rows_by_group[group]]
    calibration_raw = [raw_probability(row, True) for row in calibration_rows]
    observed = sum(bool(row["actual_fork"]) for row in calibration_rows)
    predicted = sum(calibration_raw)
    # Jeffreys-style stabilization prevents zeroing all test risks when a
    # small calibration partition contains no fork.
    scale = (observed + 0.5) / (predicted + 0.5)
    outputs = []
    for row in rows_by_group[test_group]:
        margin, delta = signed_values(row)
        conditional_raw = raw_probability(row, True)
        outputs.append(
            {
                "group": test_group,
                "case_id": row["case_id"],
                "token_index": row["token_index"],
                "label": int(bool(row["actual_fork"])),
                "signed_margin": margin,
                "signed_delta": delta,
                "scores": {
                    "fixed_tolerance_1e-3": float(abs(delta) > 1e-3),
                    "absolute_delta": abs(delta),
                    "margin_only": 1.0 / (abs(margin) + 1e-12),
                    "independence_baseline": raw_probability(row, False),
                    "signed_conditional_calibrated": min(1.0, conditional_raw * scale),
                    "observed_signed_crossing_oracle": float((margin <= 0 < margin + delta) or (margin > 0 >= margin + delta)),
                },
                "calibration": {"fit_groups": len(fit_groups), "calibration_groups": len(calibration_groups), "scale": scale},
            }
        )
    return outputs


def metric_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row["label"] for row in predictions]
    names = list(predictions[0]["scores"])
    result = {}
    for name in names:
        scores = [float(row["scores"][name]) for row in predictions]
        ordered = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        top_one_percent = ordered[: max(1, math.ceil(0.01 * len(ordered)))]
        found = sum(labels[index] for index in top_one_percent)
        item = {
            "average_precision": average_precision(labels, scores),
            "top_1pct_n": len(top_one_percent),
            "top_1pct_precision": found / len(top_one_percent),
            "top_1pct_recall": found / sum(labels) if sum(labels) else None,
            "top_1pct_forks": found,
        }
        if name == "fixed_tolerance_1e-3":
            flagged = [index for index, score in enumerate(scores) if score > 0]
            true = sum(labels[index] for index in flagged)
            item.update({"flagged": len(flagged), "precision": true / len(flagged) if flagged else None, "recall": true / sum(labels) if sum(labels) else None})
        if name in {"independence_baseline", "signed_conditional_calibrated"}:
            ece, deciles = expected_calibration_error(labels, scores)
            item.update({"predicted_forks": sum(scores), "observed_forks": sum(labels), "ece_10bin": ece, "risk_deciles": deciles})
        result[name] = item
    return result


def cluster_bootstrap(predictions: list[dict[str, Any]], draws: int = 200) -> dict[str, list[float]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[int(row["group"])].append(row)
    groups = sorted(grouped)
    rng = random.Random(0)
    values = defaultdict(list)
    for _ in range(draws):
        sample = [rng.choice(groups) for _ in groups]
        rows = [row for group in sample for row in grouped[group]]
        labels = [row["label"] for row in rows]
        values["observed_forks"].append(float(sum(labels)))
        for name in ["independence_baseline", "signed_conditional_calibrated"]:
            values[f"{name}_predicted_forks"].append(sum(float(row["scores"][name]) for row in rows))
            ap = average_precision(labels, [float(row["scores"][name]) for row in rows])
            if not math.isnan(ap):
                values[f"{name}_average_precision"].append(ap)
    output = {}
    for key, sequence in values.items():
        sequence.sort()
        output[key] = [sequence[int(0.025 * (len(sequence) - 1))], sequence[int(0.975 * (len(sequence) - 1))]]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-rollout-out signed-margin fork-risk predictor.")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--out", default="results/phase8_risk_predictions.jsonl")
    parser.add_argument("--summary", default="results/phase8_risk_summary.json")
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.certificates) if int(row.get("advantage_sign", 0)) != 0]
    rows_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_group[group_id(row)].append(row)
    predictions = [prediction for group in sorted(rows_by_group) for prediction in predict_fold(group, rows_by_group)]
    with Path(args.out).open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": "forkcert.risk.v1",
        "protocol": "leave-one-rollout-out test; remaining rollout groups split into fit and calibration; no token-random split",
        "tokens": len(predictions),
        "rollout_groups": len(rows_by_group),
        "actual_forks": sum(row["label"] for row in predictions),
        "positive_rollout_groups": sorted({row["group"] for row in predictions if row["label"]}),
        "metrics": metric_summary(predictions),
        "cluster_bootstrap_95pct": cluster_bootstrap(predictions),
        "limitations": ["Only five positives in three early rollout groups.", "Absolute-delta and signed-crossing oracle require the alternative path and are retrospective comparators, not pre-execution predictors."],
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
