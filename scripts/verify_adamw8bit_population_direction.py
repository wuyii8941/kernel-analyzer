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


def verify(root: Path) -> dict:
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
    return {
        "schema": "adamw8bit-population-direction-verification-v1",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "unit_count": len(rows),
        "recomputed_primary_endpoint": recomputed,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))
    if result["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
