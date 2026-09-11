#!/usr/bin/env python3
"""Independently recompute the AdamW8bit direction-prevalence result."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from kernel_analyzer.population_direction import population_positive_direction_prevalence


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path, *, exclude_protocols: tuple[Path, ...] = ()) -> dict:
    protocol = load(root / "protocol.json")
    result = load(root / "result.json")
    errors = []
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            errors.append("frozen source or input changed: " + name)
    rows = []
    for index in range(int(protocol["unit_count"])):
        path = root / "units" / f"unit-{index:03d}.json"
        if not path.is_file():
            errors.append(f"missing unit {index}")
            continue
        row = load(path)
        if row.get("repeatability") != "EXACT":
            errors.append(f"unit {index} repeatability failed")
        if not all(math.isfinite(float(row[key])) for key in (
            "effect_repair_inner_product", "effect_energy", "repair_energy"
        )):
            errors.append(f"unit {index} contains nonfinite statistics")
        rows.append(row)
    if len(rows) == int(protocol["unit_count"]):
        recomputed = population_positive_direction_prevalence(
            [row["effect_repair_inner_product"] for row in rows],
            null_positive_probability=float(protocol["null_positive_probability"]),
            alpha=float(protocol["alpha"]),
        )
        if recomputed != result.get("primary_endpoint"):
            errors.append("primary endpoint does not recompute")
    else:
        recomputed = None
    excluded_histories: set[tuple[int, ...]] = set()
    excluded_protocol_sha256 = {}
    for path in exclude_protocols:
        other = load(path)
        excluded_protocol_sha256[str(path)] = sha(path)
        excluded_histories.update(
            tuple(int(value) for value in indices)
            for indices in other.get("unit_population_indices", [])
        )
    unseen_rows = [
        row for row in rows
        if tuple(int(value) for value in row.get("population_indices", []))
        not in excluded_histories
    ]
    unseen_endpoint = (
        population_positive_direction_prevalence(
            [row["effect_repair_inner_product"] for row in unseen_rows],
            null_positive_probability=float(protocol["null_positive_probability"]),
            alpha=float(protocol["alpha"]),
        )
        if unseen_rows else None
    )
    return {
        "schema": "adamw8bit-population-direction-verification-v1",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "unit_count": len(rows),
        "recomputed_primary_endpoint": recomputed,
        "cross_experiment_independence_audit": {
            "excluded_protocol_sha256": excluded_protocol_sha256,
            "overlapping_unit_count": len(rows) - len(unseen_rows),
            "unseen_unit_count": len(unseen_rows),
            "unseen_subset_endpoint": unseen_endpoint,
            "role": (
                "Conservative confirmation after excluding histories present in "
                "the declared earlier protocols; it does not rewrite the original result."
            ),
        },
        "protocol_sha256": sha(root / "protocol.json"),
        "result_sha256": sha(root / "result.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-protocol", type=Path, action="append", default=[],
        help="Exclude units whose population-index history appears in an earlier protocol.",
    )
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("choose a new output under /data1/tzh")
    result = verify(args.root, exclude_protocols=tuple(args.exclude_protocol))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))
    if result["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
