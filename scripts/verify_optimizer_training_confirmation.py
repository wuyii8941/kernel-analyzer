#!/usr/bin/env python3
"""Independently verify the frozen AdamW8bit training confirmation records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CONDITIONS = ("FP32_ADAMW", "ADAMW8BIT_BLOCK256", "ADAMW8BIT_BLOCK64")


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def t_interval(values: list[float]) -> list[float]:
    import scipy.stats

    center = mean(values)
    variance = math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)
    if variance == 0:
        return [center, center]
    critical = float(scipy.stats.t.ppf(0.975, len(values) - 1))
    half = critical * math.sqrt(variance / len(values))
    return [center - half, center + half]


def recompute(root: Path) -> dict[str, Any]:
    protocol_path = root / "protocol.json"
    protocol = load(protocol_path)
    if protocol.get("schema") != "optimizer-training-confirmation-v1":
        raise ValueError("unexpected protocol schema")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError("frozen dependency changed: " + name)

    steps = int(protocol["steps"])
    evaluation_steps = {str(value) for value in protocol["evaluation_steps"]}
    evaluation_states = int(protocol["evaluation_states"])
    primary_values: list[float] = []
    modified_values: list[float] = []
    improvement_values: list[float] = []
    condition_endpoints = {condition: [] for condition in CONDITIONS}
    speeds = {condition: [] for condition in CONDITIONS}
    peaks = {condition: [] for condition in CONDITIONS}
    collapses: list[dict[str, Any]] = []
    stream_rows = []

    for index in range(int(protocol["stream_count"])):
        path = root / "streams" / f"stream_{index:02d}.json"
        stream = load(path)
        if stream.get("status") != "COMPLETE" or stream.get("stream_index") != index:
            raise ValueError(f"stream {index} is incomplete")
        if stream.get("protocol_sha256") != digest(protocol_path):
            raise ValueError(f"stream {index} used a different protocol")
        records = {row["condition"]: row for row in stream["records"]}
        if set(records) != set(CONDITIONS):
            raise ValueError(f"stream {index} has incomplete conditions")
        endpoints = {}
        for condition in CONDITIONS:
            record = records[condition]
            training = [float(value) for value in record["training_loss"]]
            evaluations = record["evaluation_loss_by_step"]
            if len(training) != steps or set(evaluations) != evaluation_steps:
                raise ValueError(f"stream {index} {condition} has incomplete measurements")
            if not all(len(values) == evaluation_states for values in evaluations.values()):
                raise ValueError(f"stream {index} {condition} has incomplete evaluation rows")
            all_values = training + [float(value) for values in evaluations.values() for value in values]
            if not all(math.isfinite(value) for value in all_values):
                collapses.append({"stream_index": index, "condition": condition,
                                  "reason": "NONFINITE"})
            initial = mean([float(value) for value in evaluations["0"]])
            final = mean([float(value) for value in evaluations[str(steps)]])
            if final > initial + 1.0:
                collapses.append({"stream_index": index, "condition": condition,
                                  "reason": "FINAL_EVALUATION_LOSS_INCREASE_GT_1"})
            endpoints[condition] = final
            condition_endpoints[condition].append(final)
            speeds[condition].append(float(record["training_steps_per_second_with_evaluation_overhead"]))
            peaks[condition].append(int(record["peak_allocated_bytes"]))
        primary = endpoints["ADAMW8BIT_BLOCK256"] - endpoints["FP32_ADAMW"]
        modified = endpoints["ADAMW8BIT_BLOCK64"] - endpoints["FP32_ADAMW"]
        improvement = abs(primary) - abs(modified)
        primary_values.append(primary)
        modified_values.append(modified)
        improvement_values.append(improvement)
        stream_rows.append({
            "stream_index": index,
            "final_mean_evaluation_loss": endpoints,
            "block256_minus_reference": primary,
            "block64_minus_reference": modified,
            "absolute_gap_reduction": improvement,
        })

    primary_interval = t_interval(primary_values)
    improvement_interval = t_interval(improvement_values)
    margin = float(protocol["material_loss_margin"])
    primary_decision = (
        "MATERIAL_EFFECT" if primary_interval[0] > margin or primary_interval[1] < -margin
        else "DIFFERENT_BUT_BELOW_MATERIAL_MARGIN"
        if primary_interval[0] > 0 or primary_interval[1] < 0
        else "INCONCLUSIVE"
    )
    return {
        "schema": "optimizer-training-confirmation-verification-v1",
        "status": "VERIFIED",
        "protocol_sha256": digest(protocol_path),
        "stream_count": len(stream_rows),
        "stream_rows": stream_rows,
        "primary": {
            "paired_values": primary_values,
            "mean": mean(primary_values),
            "interval_95": primary_interval,
            "material_margin": margin,
            "decision": primary_decision,
            "all_streams_same_positive_direction": all(value > 0 for value in primary_values),
            "perplexity_ratio_from_mean_loss_difference": math.exp(mean(primary_values)),
            "perplexity_ratio_interval_from_loss_interval": [math.exp(value) for value in primary_interval],
        },
        "modified_variant": {
            "paired_values": modified_values,
            "mean": mean(modified_values),
            "absolute_gap_reduction_values": improvement_values,
            "absolute_gap_reduction_mean": mean(improvement_values),
            "absolute_gap_reduction_interval_95": improvement_interval,
            "decision": "CONFIRMED_CLOSER_TO_REFERENCE" if improvement_interval[0] > 0
            else "NOT_CONFIRMED",
        },
        "condition_summary": {
            condition: {
                "mean_final_evaluation_loss": mean(condition_endpoints[condition]),
                "mean_training_steps_per_second_with_evaluation_overhead": mean(speeds[condition]),
                "max_peak_allocated_bytes": max(peaks[condition]),
            }
            for condition in CONDITIONS
        },
        "collapse_events": collapses,
        "collapse_decision": "NO_COLLAPSE_OBSERVED" if not collapses else "COLLAPSE_OBSERVED",
        "inference_boundary": (
            "Paired t interval over eight seeded, nonoverlapping WikiText token streams at "
            "one fixed checkpoint. It assumes these streams are suitable independent units; "
            "it is not a guarantee over checkpoints, initializations, datasets, or models."
        ),
    }


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    return left == right


def verify_published(root: Path, recomputed: dict[str, Any]) -> None:
    published = load(root / "summary.json")
    checks = (
        (published["primary"]["paired_values"], recomputed["primary"]["paired_values"]),
        (published["primary"]["mean"], recomputed["primary"]["mean"]),
        (published["primary"]["interval_95"], recomputed["primary"]["interval_95"]),
        (published["primary"]["decision"], recomputed["primary"]["decision"]),
        (published["modified_variant"]["absolute_gap_reduction_values"],
         recomputed["modified_variant"]["absolute_gap_reduction_values"]),
        (published["modified_variant"]["absolute_gap_reduction_interval_95"],
         recomputed["modified_variant"]["absolute_gap_reduction_interval_95"]),
        (published["modified_variant"]["decision"], recomputed["modified_variant"]["decision"]),
    )
    if not all(close(left, right) for left, right in checks):
        raise ValueError("published summary differs from independent recomputation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_relative_to(Path("/data1/tzh")) or not output.is_relative_to(Path("/data1/tzh")):
        raise ValueError("all paths must remain under /data1/tzh")
    result = recompute(root)
    verify_published(root, result)
    save_new(output, result)


if __name__ == "__main__":
    main()
