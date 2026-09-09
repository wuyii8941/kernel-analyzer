#!/usr/bin/env python3
"""Independently verify the frozen FP32-first-moment training confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CONDITIONS = ("ADAMW8BIT_BLOCK256", "FP32_FIRST_MOMENT_BLOCK256", "FP32_ADAMW")


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

    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("at least two finite independent values are required")
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
    if protocol.get("schema") != "optimizer-hybrid-training-confirmation-v1":
        raise ValueError("unexpected protocol schema")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError("frozen dependency changed: " + name)

    steps = int(protocol["steps"])
    evaluation_steps = {str(value) for value in protocol["evaluation_steps"]}
    evaluation_states = int(protocol["evaluation_states"])
    improvements: list[float] = []
    default_reference_gaps: list[float] = []
    hybrid_reference_gaps: list[float] = []
    stream_rows = []
    condition_endpoints = {condition: [] for condition in CONDITIONS}
    speeds = {condition: [] for condition in CONDITIONS}
    peaks = {condition: [] for condition in CONDITIONS}
    collapses = []

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
            if (record.get("status") != "COMPLETE" or len(training) != steps
                    or set(evaluations) != evaluation_steps):
                raise ValueError(f"stream {index} {condition} has incomplete measurements")
            if not all(len(values) == evaluation_states for values in evaluations.values()):
                raise ValueError(f"stream {index} {condition} has incomplete evaluation rows")
            values = training + [float(value) for rows in evaluations.values() for value in rows]
            if not all(math.isfinite(value) for value in values):
                collapses.append({"stream_index": index, "condition": condition,
                                  "reason": "NONFINITE"})
            initial = mean([float(value) for value in evaluations["0"]])
            final = mean([float(value) for value in evaluations[str(steps)]])
            if final > initial + 1.0:
                collapses.append({"stream_index": index, "condition": condition,
                                  "reason": "FINAL_EVALUATION_LOSS_INCREASE_GT_1"})
            endpoints[condition] = final
            condition_endpoints[condition].append(final)
            speeds[condition].append(float(record["steps_per_second_with_evaluation_overhead"]))
            peaks[condition].append(int(record["peak_allocated_bytes"]))
        improvement = endpoints["ADAMW8BIT_BLOCK256"] - endpoints["FP32_FIRST_MOMENT_BLOCK256"]
        default_reference_gap = endpoints["ADAMW8BIT_BLOCK256"] - endpoints["FP32_ADAMW"]
        hybrid_reference_gap = endpoints["FP32_FIRST_MOMENT_BLOCK256"] - endpoints["FP32_ADAMW"]
        improvements.append(improvement)
        default_reference_gaps.append(default_reference_gap)
        hybrid_reference_gaps.append(hybrid_reference_gap)
        stream_rows.append({
            "stream_index": index,
            "final_mean_evaluation_loss": endpoints,
            "default_minus_hybrid": improvement,
            "default_minus_fp32": default_reference_gap,
            "hybrid_minus_fp32": hybrid_reference_gap,
        })

    interval = t_interval(improvements)
    margin = float(protocol["material_improvement_margin"])
    decision = ("MATERIAL_IMPROVEMENT" if interval[0] > margin else
                "DETECTABLE_IMPROVEMENT" if interval[0] > 0 else "NOT_CONFIRMED")
    default_interval = t_interval(default_reference_gaps)
    hybrid_interval = t_interval(hybrid_reference_gaps)
    return {
        "schema": "optimizer-hybrid-training-verification-v1",
        "status": "VERIFIED",
        "protocol_sha256": digest(protocol_path),
        "stream_count": len(stream_rows),
        "streams": stream_rows,
        "primary": {
            "paired_values": improvements,
            "mean": mean(improvements),
            "interval_95": interval,
            "material_margin": margin,
            "decision": decision,
            "positive_stream_count": sum(value > 0 for value in improvements),
            "perplexity_ratio_default_over_hybrid": math.exp(mean(improvements)),
            "perplexity_ratio_interval": [math.exp(value) for value in interval],
        },
        "secondary_reference_contrasts": {
            "default_minus_fp32": {
                "paired_values": default_reference_gaps,
                "mean": mean(default_reference_gaps),
                "interval_95": default_interval,
                "positive_stream_count": sum(value > 0 for value in default_reference_gaps),
                "perplexity_ratio": math.exp(mean(default_reference_gaps)),
                "interpretation": (
                    "MATERIAL_EFFECT" if default_interval[0] > margin else
                    "DETECTABLE_BELOW_MATERIAL_MARGIN" if default_interval[0] > 0 else
                    "NOT_CONFIRMED"
                ),
            },
            "hybrid_minus_fp32": {
                "paired_values": hybrid_reference_gaps,
                "mean": mean(hybrid_reference_gaps),
                "interval_95": hybrid_interval,
                "positive_stream_count": sum(value > 0 for value in hybrid_reference_gaps),
                "perplexity_ratio": math.exp(mean(hybrid_reference_gaps)),
                "interpretation": (
                    "MATERIAL_EFFECT" if hybrid_interval[0] > margin else
                    "DETECTABLE_BELOW_MATERIAL_MARGIN" if hybrid_interval[0] > 0 else
                    "NOT_CONFIRMED"
                ),
            },
        },
        "condition_summary": {
            condition: {
                "mean_final_evaluation_loss": mean(condition_endpoints[condition]),
                "mean_steps_per_second_with_evaluation_overhead": mean(speeds[condition]),
                "max_peak_allocated_bytes": max(peaks[condition]),
            }
            for condition in CONDITIONS
        },
        "collapse_events": collapses,
        "collapse_decision": "NO_COLLAPSE_OBSERVED" if not collapses else "COLLAPSE_OBSERVED",
        "inference_boundary": (
            "Paired t interval over eight new, nonoverlapping WikiText token streams at one "
            "fixed Mamba checkpoint. It assumes these streams are suitable independent units; "
            "it is not a guarantee over checkpoints, initializations, datasets, or models."
        ),
    }


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close(left[key], right[key], tolerance) for key in left)
    return left == right


def verify_published(root: Path, result: dict[str, Any]) -> None:
    published = load(root / "summary.json")
    expected = {key: published[key] for key in ("schema", "streams", "primary", "scope")}
    observed = {key: result[key] for key in ("streams", "primary")}
    if not close(expected["streams"], observed["streams"]):
        raise ValueError("published stream rows differ from independent recomputation")
    if not close(expected["primary"], observed["primary"]):
        # The verifier adds transparent derived fields not present in the producer.
        common = {key: result["primary"][key] for key in published["primary"]}
        if not close(published["primary"], common):
            raise ValueError("published primary result differs from independent recomputation")


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
