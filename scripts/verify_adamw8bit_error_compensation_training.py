#!/usr/bin/env python3
"""Independently recompute the error-compensation training conclusion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
from pathlib import Path


CONDITIONS = ("FP32_ADAMW", "ADAMW8BIT_BLOCK256", "ADAMW8BIT_COMPENSATED_BLOCK256")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> list[float]:
    import scipy.stats
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance == 0:
        return [mean, mean]
    half = float(scipy.stats.t.ppf(0.975, len(values) - 1)) * math.sqrt(
        variance / len(values)
    )
    return [mean - half, mean + half]


def equivalent_recorded_value(reported: object, recomputed: object) -> bool:
    """Compare JSON results without rejecting harmless float round-off.

    Categorical fields remain exact.  Numeric values are only allowed the
    machine-scale variation produced by independently repeating the same
    arithmetic with a different SciPy/Python build.
    """
    if isinstance(reported, numbers.Real) and isinstance(recomputed, numbers.Real):
        return math.isclose(
            float(reported), float(recomputed), rel_tol=1e-12, abs_tol=1e-15
        )
    if isinstance(reported, list) and isinstance(recomputed, list):
        return len(reported) == len(recomputed) and all(
            equivalent_recorded_value(left, right)
            for left, right in zip(reported, recomputed, strict=True)
        )
    return reported == recomputed


def verify(root: Path) -> dict:
    protocol = load(root / "protocol.json")
    summary = load(root / "summary.json")
    errors: list[str] = []
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            errors.append("frozen source or input changed: " + name)
    gains = []
    default_reference = []
    compensated_reference = []
    for index in range(int(protocol["stream_count"])):
        path = root / "streams" / f"stream_{index:02d}.json"
        if not path.is_file():
            errors.append(f"missing stream {index}")
            continue
        stream = load(path)
        if (stream.get("status") != "COMPLETE" or stream.get("stream_index") != index
                or stream.get("protocol_sha256") != sha(root / "protocol.json")):
            errors.append(f"invalid stream record {index}")
            continue
        records = {row.get("condition"): row for row in stream.get("records", [])}
        if set(records) != set(CONDITIONS):
            errors.append(f"incomplete conditions for stream {index}")
            continue
        endpoints = {}
        for condition in CONDITIONS:
            values = records[condition]["evaluation_loss_by_step"].get(
                str(protocol["steps"]), []
            )
            if (len(values) != int(protocol["evaluation_states"])
                    or not all(math.isfinite(float(value)) for value in values)):
                errors.append(f"invalid endpoint for stream {index} condition {condition}")
                break
            endpoints[condition] = sum(values) / len(values)
        if len(endpoints) != len(CONDITIONS):
            continue
        gains.append(endpoints["ADAMW8BIT_BLOCK256"] -
                     endpoints["ADAMW8BIT_COMPENSATED_BLOCK256"])
        default_reference.append(endpoints["ADAMW8BIT_BLOCK256"] -
                                 endpoints["FP32_ADAMW"])
        compensated_reference.append(endpoints["ADAMW8BIT_COMPENSATED_BLOCK256"] -
                                     endpoints["FP32_ADAMW"])
    recomputed = None
    if len(gains) == int(protocol["stream_count"]):
        primary_interval = interval(gains)
        margin = float(protocol["material_improvement_margin"])
        if primary_interval[0] > margin:
            decision = "MATERIAL_IMPROVEMENT"
        elif primary_interval[0] > 0:
            decision = "DETECTABLE_IMPROVEMENT"
        else:
            decision = "NOT_CONFIRMED"
        recomputed = {
            "paired_values": gains,
            "mean": sum(gains) / len(gains),
            "interval_95": primary_interval,
            "decision": decision,
            "default_minus_reference_mean": sum(default_reference) / len(default_reference),
            "compensated_minus_reference_mean":
                sum(compensated_reference) / len(compensated_reference),
        }
        reported = summary.get("primary", {})
        for key in ("paired_values", "mean", "interval_95", "decision"):
            if not equivalent_recorded_value(reported.get(key), recomputed[key]):
                errors.append("reported primary result differs: " + key)
    return {
        "schema": "adamw8bit-error-compensation-training-verification-v1",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "stream_count": len(gains),
        "recomputed": recomputed,
        "protocol_sha256": sha(root / "protocol.json"),
        "summary_sha256": sha(root / "summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("choose a new output under /data1/tzh")
    result = verify(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))
    if result["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
