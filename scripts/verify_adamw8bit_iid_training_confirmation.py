#!/usr/bin/env python3
"""Independently recompute the prospective iid training confirmation result."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> list[float]:
    import scipy.stats

    center = math.fsum(values) / len(values)
    variance = math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)
    if variance == 0.0:
        return [center, center]
    half = float(scipy.stats.t.ppf(0.975, len(values) - 1)) * math.sqrt(
        variance / len(values)
    )
    return [center - half, center + half]


def close(left: Any, right: Any) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def verify(root: Path) -> dict[str, Any]:
    protocol = load(root / "protocol.json")
    summary = load(root / "summary.json")
    errors: list[str] = []
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            errors.append("frozen source or input changed: " + name)

    streams = int(protocol["stream_count"])
    conditions = tuple(protocol["conditions"])
    endpoint = str(protocol["steps"])
    expected_values = int(protocol["evaluation_states"])
    task_statuses: list[str] = []
    paired: list[float] = []
    for stream in range(streams):
        means: dict[str, float] = {}
        for condition in conditions:
            path = root / "runs" / f"stream_{stream:02d}_{condition}.json"
            if not path.is_file():
                task_statuses.append("NOT_COMPLETED")
                continue
            row = load(path)
            status = str(row.get("status"))
            task_statuses.append(status)
            if row.get("stream") != stream or row.get("condition") != condition:
                errors.append(f"task identity differs in {path.name}")
                continue
            if status != "COMPLETE":
                continue
            values = row.get("evaluation_loss_by_step", {}).get(endpoint, [])
            if len(values) != expected_values or not all(math.isfinite(float(x)) for x in values):
                errors.append(f"invalid finite endpoint in {path.name}")
                continue
            means[condition] = math.fsum(float(x) for x in values) / len(values)
        if set(means) == set(conditions):
            paired.append(means["OFF"] - means["ON"])

    counts = dict(Counter(task_statuses))
    all_finite = len(paired) == streams and counts == {"COMPLETE": 2 * streams}
    recomputed_primary: dict[str, Any] = {
        "paired_finite_count": len(paired),
        "paired_values": paired,
    }
    if len(paired) >= 2:
        confidence_interval = interval(paired)
        recomputed_primary.update({
            "mean": math.fsum(paired) / len(paired),
            "interval_95": confidence_interval,
        })
        if all_finite:
            margin = float(protocol["material_improvement_margin"])
            recomputed_primary["decision"] = (
                "MATERIAL_IMPROVEMENT" if confidence_interval[0] > margin else
                "DETECTABLE_IMPROVEMENT" if confidence_interval[0] > 0.0 else
                "NOT_CONFIRMED"
            )
    if all_finite:
        reported = summary.get("primary", {})
        for key in ("paired_finite_count", "paired_values"):
            if not close(reported.get(key), recomputed_primary[key]):
                errors.append("reported primary differs: " + key)
        reported_description = reported.get("complete_case_description", {})
        for key in ("mean", "interval_95"):
            if not close(reported_description.get(key), recomputed_primary[key]):
                errors.append("reported primary description differs: " + key)
        if reported.get("decision") != recomputed_primary.get("decision"):
            errors.append("reported primary differs: decision")
    return {
        "schema": "adamw8bit-iid-training-confirmation-verification-v1",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "task_counts": counts,
        "all_frozen_pairs_have_finite_endpoints": all_finite,
        "recomputed_primary": recomputed_primary,
        "protocol_sha256": sha(root / "protocol.json"),
        "summary_sha256": sha(root / "summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not root.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        parser.error("inputs and outputs must stay inside kernel-analyzer")
    if output.exists():
        parser.error("choose a new output; verifier records are immutable")
    result = verify(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
