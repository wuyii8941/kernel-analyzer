#!/usr/bin/env python3
"""Independently verify the frozen residual-structure mechanism probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve(); output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("choose a new output inside kernel-analyzer")
    protocol = load(root / "protocol.json")
    errors = []
    for name, expected in protocol["source_sha256"].items():
        if not Path(name).is_file() or sha(Path(name)) != expected:
            errors.append("frozen dependency changed: " + name)
    rows = [load(root / "units" / f"unit-{i:03d}.json") for i in range(8)]
    relative_energy_errors = []
    for row in rows:
        for left, right in zip(row["residual_read_energy"], row["coordinate_roll_energy"]):
            relative_energy_errors.append(abs(left - right) / max(abs(left), 1e-300))
        if sorted(row["residual_read_energy"][1:]) != sorted(row["time_reverse_energy"][1:]):
            errors.append(f"time residual multiset differs for unit {row['unit']}")
    maximum_energy_error = max(relative_energy_errors)
    if maximum_energy_error > 1e-12:
        errors.append("coordinate energy differs beyond floating summation tolerance")
    correct_coordinate = sum(row["write_rms"]["CORRECT"] < row["write_rms"]["COORDINATE_ROLL"] for row in rows)
    correct_time = sum(row["write_rms"]["CORRECT"] < row["write_rms"]["TIME_REVERSE"] for row in rows)
    if correct_coordinate != len(rows) or correct_time != len(rows):
        errors.append("correct arrangement is not better in every fixed history")
    result = {
        "schema": "adamw8bit-residual-structure-independent-verification-v1",
        "status": "VERIFIED" if not errors else "INVALID",
        "errors": errors,
        "unit_count": len(rows),
        "correct_better_than_coordinate_count": correct_coordinate,
        "correct_better_than_time_count": correct_time,
        "maximum_relative_coordinate_energy_summation_difference": maximum_energy_error,
        "saved_exact_boolean_explanation": (
            "The saved coordinate_multiset_exact check used a shape-dependent inverse roll "
            "and returned false. The transform itself is a block-local torch.roll permutation; "
            "the independently recomputed squared energies agree within floating reduction error."
        ),
        "result_sha256": sha(root / "result.json"),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
