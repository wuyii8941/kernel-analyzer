#!/usr/bin/env python3
"""Recompute the AdamW8bit aligned source-link result from saved unit statistics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from kernel_analyzer.bias_evidence import binary_prevalence_summary, statewise_aligned_summary


def load(path: Path):
    return json.loads(path.read_text())


def _records_equal(actual, expected) -> bool:
    """Compare saved sufficient statistics without rejecting print-roundoff."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15)
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _records_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _records_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def recompute(root: Path) -> dict:
    protocol = load(root / "protocol.json")
    recorded = load(root / "result.json")
    rows = [load(root / "units" / f"unit-{index:03d}.json")
            for index in range(protocol["unit_count"])]
    errors = []
    for index, row in enumerate(rows):
        if row["unit_id"] != f"source-link-history-{index:03d}":
            errors.append(f"unit identity differs at {index}")
        if row["population_indices"] != protocol["unit_population_indices"][index]:
            errors.append(f"history differs at {index}")
        expected = abs(row["compensated_aligned_gain"]) < abs(row["default_aligned_gain"])
        if row["absolute_aligned_gain_reduced"] != expected:
            errors.append(f"aligned-reduction flag differs at {index}")
    default = statewise_aligned_summary(
        [row["default_effect_repair_inner_product"] for row in rows],
        [row["repair_energy"] for row in rows],
    )
    compensated = statewise_aligned_summary(
        [row["compensated_effect_repair_inner_product"] for row in rows],
        [row["repair_energy"] for row in rows],
    )
    reduction = binary_prevalence_summary(
        [row["absolute_aligned_gain_reduced"] for row in rows],
        null_probability=.5, alpha=.05,
        estimand="PROBABILITY_RESIDUAL_READBACK_REDUCES_ABSOLUTE_ALIGNED_GAIN",
    )
    for name, expected in (("default_aligned", default),
                           ("compensated_aligned", compensated),
                           ("absolute_aligned_reduction_prevalence", reduction)):
        if not _records_equal(recorded.get(name), expected):
            errors.append(name + " differs from unit recomputation")
    return {
        "schema": "adamw8bit-bias-source-link-verification-v1",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "unit_count": len(rows),
        "default_positive_count": default["positive_direction_frequency"]["positive_count"],
        "compensated_positive_count": compensated["positive_direction_frequency"]["positive_count"],
        "absolute_gain_reduction_count": reduction["success_count"],
        "scope": "saved unit sufficient-statistic recomputation; not a second GPU execution",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output path")
    result = recompute(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
