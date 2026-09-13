#!/usr/bin/env python3
"""Recompute integrity and decision checks for priority analyses 1 and 2."""
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


def verify(root: Path) -> dict:
    errors = []

    moment_root = root / "moment_reconstruction"
    moment_protocol = load(moment_root / "protocol.json")
    moment = load(moment_root / "result.json")
    for name, expected in moment_protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            errors.append("moment source changed: " + name)
    expected_rows = (len(moment_protocol["sizes"]) * len(moment_protocol["seeds"])
                     * moment_protocol["steps"])
    if moment.get("status") != "COMPLETE" or len(moment.get("rows", [])) != expected_rows:
        errors.append("moment rows incomplete")
    max_reconstruction = max(
        row["conditions"][mode][part]["relative_reconstruction_residual"]
        for row in moment.get("rows", []) for mode in ("OFF", "ON")
        for part in ("first_moment", "second_moment")
    )
    if max_reconstruction > 1e-6:
        errors.append("moment reconstruction residual exceeds tolerance")

    selective_root = root / "selective_parameter_compensation"
    selective_protocol = load(selective_root / "protocol.json")
    selective = load(selective_root / "result.json")
    for name, expected in selective_protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            errors.append("selective source changed: " + name)
    units = [load(selective_root / "units" / f"unit-{index:03d}.json")
             for index in range(selective_protocol["unit_count"])]
    if selective.get("status") != "COMPLETE" or len(units) != selective_protocol["unit_count"]:
        errors.append("selective units incomplete")
    if not all(row["all_complement_checks"] for row in units):
        errors.append("selective complementary paths differ")
    for mode, saved in selective["mean_write_rms"].items():
        recomputed = math.fsum(row["write_rms"][mode] for row in units) / len(units)
        if not math.isclose(saved, recomputed, rel_tol=1e-14, abs_tol=0.0):
            errors.append("selective mean mismatch: " + mode)
    key_values = [row["off_effect_energy_fraction_in_key"] for row in units]
    if selective["key_fraction_of_off_effect_energy"]["values"] != key_values:
        errors.append("selective key fractions differ")

    method_path = root / "method_incremental_value.json"
    method = load(method_path)
    for name, expected in method.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            errors.append("method source changed: " + name)
    for name, expected in method.get("empirical_evidence_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            errors.append("method evidence changed: " + name)
    counts = method.get("fixed_suite_equivalence_correct_count", {})
    if counts.get("FULL_ANALYSIS") != method.get("scenario_count"):
        errors.append("full method controlled checks failed")
    if counts.get("UPDATE_RMS") != counts.get("FULL_ANALYSIS"):
        errors.append("fixed-suite equivalence comparison no longer has declared equality")
    by_name = {row["name"]: row for row in method.get("controlled_scenarios", [])}
    required = {
        "small_local_material_update", "zero_mean_material_energy",
        "single_state_false_reassurance", "calibration_confirmation_direction_change",
    }
    if not required.issubset(by_name):
        errors.append("controlled diagnostic scenarios missing")
    modifications = {row["modification"]: row for row in method["empirical_modification_evidence"]}
    if modifications["BLOCK64"]["training_decision"] != "NOT_CONFIRMED":
        errors.append("block64 negative result changed")
    if modifications["FP32_FIRST_MOMENT"]["training_decision"] != "NOT_CONFIRMED":
        errors.append("first-moment negative result changed")
    if modifications["CROSS_STEP_RESIDUAL_COMPENSATION"]["training_decision"] != "MATERIAL_IMPROVEMENT":
        errors.append("compensation result changed")

    return {
        "schema": "priority-analysis-v2-independent-verification-v1",
        "status": "VERIFIED" if not errors else "INVALID",
        "errors": errors,
        "checks": {
            "moment_row_count": len(moment.get("rows", [])),
            "maximum_moment_relative_reconstruction_residual": max_reconstruction,
            "selective_unit_count": len(units),
            "all_selective_complement_checks": all(row["all_complement_checks"] for row in units),
            "method_scenario_count": method.get("scenario_count"),
            "full_and_update_rms_equivalence_correct_count": counts.get("FULL_ANALYSIS"),
        },
        "source_sha256": {
            str(moment_root / "protocol.json"): sha(moment_root / "protocol.json"),
            str(moment_root / "result.json"): sha(moment_root / "result.json"),
            str(selective_root / "protocol.json"): sha(selective_root / "protocol.json"),
            str(selective_root / "result.json"): sha(selective_root / "result.json"),
            str(method_path): sha(method_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        parser.error("choose a new output inside kernel-analyzer")
    result = verify(root)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"]}))
    if result["status"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
