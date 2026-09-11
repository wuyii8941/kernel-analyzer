#!/usr/bin/env python3
"""Independently verify the saved full-model block-size mechanism result."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from kernel_analyzer.training_equivalence import exact_binomial_one_sided_bounds


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    protocol = load(root / "protocol.json")
    recorded = load(root / "result.json")
    rows = [load(root / "units" / f"unit-{index:03d}.json")
            for index in range(protocol["unit_count"])]
    errors = []
    for index, row in enumerate(rows):
        if row.get("population_indices") != protocol["unit_population_indices"][index]:
            errors.append(f"FROZEN_DRAW_DIFFERS:{index}")
        rms = row["full_model_write_rms"]
        expected = rms["block64"] < rms["block256"]
        if row.get("block64_lower_rms") is not expected:
            errors.append(f"COMPARISON_DIFFERS:{index}")
    successes = sum(row["full_model_write_rms"]["block64"]
                    < row["full_model_write_rms"]["block256"] for row in rows)
    bounds = exact_binomial_one_sided_bounds(successes, len(rows), alpha=0.05)
    means = {key: math.fsum(row["full_model_write_rms"][key] for row in rows) / len(rows)
             for key in ("block64", "block256")}
    gradient_inner_negative_counts = {
        key: sum(row["effect_final_gradient_inner_product"][key] < 0 for row in rows)
        for key in ("block64", "block256")
    }
    if successes != recorded.get("block64_lower_rms_count"):
        errors.append("SUCCESS_COUNT_DIFFERS")
    if list(bounds) != recorded.get("one_sided_probability_bounds"):
        errors.append("PROBABILITY_BOUNDS_DIFFER")
    if means != recorded.get("mean_full_model_write_rms"):
        errors.append("MEANS_DIFFER")
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            errors.append("FROZEN_SOURCE_DIFFERS:" + name)
    return {
        "schema": "adamw8bit-full-model-block-probe-verification-v1",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "independent_unit_count": len(rows),
        "block64_lower_rms_count": successes,
        "one_sided_probability_bounds": list(bounds),
        "mean_full_model_write_rms": means,
        "negative_final_gradient_inner_product_counts": gradient_inner_negative_counts,
        "gradient_inner_product_role": "DESCRIPTIVE_FIRST_ORDER_DIAGNOSTIC",
        "prediction_result": "CONFIRMED" if bounds[0] > 0.5 else "NOT_CONFIRMED",
        "protocol_sha256": sha(root / "protocol.json"),
        "result_sha256": sha(root / "result.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("choose a new output under /data1/tzh")
    result = verify(args.root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
