#!/usr/bin/env python3
"""Independently recompute the AdamW8bit population decision from saved units."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from kernel_analyzer.training_numerical_analysis import (
    analyze_population_exceedance_artifact,
)


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
    raw = load(root / "raw.json")
    recorded = load(root / "analysis.json")
    units = [load(root / "units" / f"unit-{index:03d}.json")
             for index in range(protocol["unit_count"])]
    errors = []
    expected_ids = [f"iid-history-{index:03d}" for index in range(protocol["unit_count"])]
    if raw.get("state_ids") != expected_ids or raw.get("inference_unit_ids") != expected_ids:
        errors.append("INFERENCE_UNIT_IDS_DIFFER")
    for index, unit in enumerate(units):
        if unit.get("unit_id") != expected_ids[index]:
            errors.append(f"UNIT_ID_DIFFERS:{index}")
        if unit.get("population_indices") != protocol["unit_population_indices"][index]:
            errors.append(f"FROZEN_DRAW_DIFFERS:{index}")
        expected_rms = math.sqrt(unit["effect_energy"] / unit["repair_energy"])
        if not math.isclose(expected_rms, unit["statewise_rms"], rel_tol=1e-14):
            errors.append(f"UNIT_RMS_DIFFERS:{index}")
    expected_rows = [{
        "effect_energy": unit["effect_energy"],
        "repair_energy": unit["repair_energy"],
        "effect_repair_inner_product": unit["effect_repair_inner_product"],
    } for unit in units]
    if raw.get("original_coordinate_statistics", {}).get("PARAMETER_WRITE") != expected_rows:
        errors.append("RAW_SUFFICIENT_STATISTICS_DIFFER")
    recomputed = analyze_population_exceedance_artifact(raw, protocol)
    for key in ("measurement_status", "claim_scope", "equivalence_decision",
                "mandatory_endpoints", "bias_analysis"):
        if recorded.get(key) != recomputed.get(key):
            errors.append("ANALYSIS_DIFFERS:" + key)
    source_errors = []
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            source_errors.append(name)
    errors.extend("FROZEN_SOURCE_DIFFERS:" + name for name in source_errors)
    return {
        "schema": "adamw8bit-population-update-verification-v1",
        "status": "VERIFIED" if not errors else "FAILED",
        "errors": errors,
        "unit_count": len(units),
        "exceedance_count": recomputed.get("bias_analysis", {}).get(
            "population_endpoint", {}
        ).get("observed_exceedance_count"),
        "one_sided_probability_bounds": recomputed.get("bias_analysis", {}).get(
            "population_endpoint", {}
        ).get("one_sided_probability_bounds"),
        "decision": recomputed.get("equivalence_decision"),
        "scope": recomputed.get("claim_scope"),
        "protocol_sha256": sha(root / "protocol.json"),
        "raw_sha256": sha(root / "raw.json"),
        "analysis_sha256": sha(root / "analysis.json"),
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
