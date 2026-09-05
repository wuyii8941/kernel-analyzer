#!/usr/bin/env python3
"""Inventory exact, summarized, and newly required measurement fields."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = (
    ROOT / "results/property/training_bias_profile_v2/five_case_raw",
    ROOT / "results/property/training_bias_profile_v2/prospective_batch_1/raw",
    ROOT / "results/property/training_bias_profile_v2/prospective_batch_2/raw",
    ROOT / "results/property/training_numerical_analysis_v1/recapture",
)
OUTPUT = ROOT / "results/property/training_numerical_analysis_v1/measurement_provenance.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = []
    for search_root in SEARCH_ROOTS:
        if not search_root.is_dir():
            continue
        for path in sorted(search_root.glob("*.json")):
            payload = json.loads(path.read_text())
            if payload.get("schema") != "kernel-analyzer-training-bias-profile-v2-raw-case":
                continue
            stages = payload.get("stages", {})
            is_new_recapture = "training_numerical_analysis_v1/recapture" in str(path)
            stage_geometry = {}
            for stage, views in stages.items():
                geometries = []
                for view in views:
                    if view == "EXACT":
                        geometries.append("FULL_VECTOR")
                    elif view.startswith("COUNT_SKETCH_V2"):
                        geometries.append("COUNT_SKETCH_V2")
                    elif is_new_recapture:
                        # Jobs launched under protocol v1 used the corrected mapping
                        # before its external view label was renamed.
                        geometries.append("COUNT_SKETCH_V2_LEGACY_VIEW_NAME")
                    else:
                        geometries.append("LEGACY_COUNT_SKETCH")
                stage_geometry[stage] = sorted(set(geometries))
            uses_legacy_sketch = any(
                "LEGACY_COUNT_SKETCH" in geometries
                for geometries in stage_geometry.values()
            )
            has_original = bool(payload.get("original_coordinate_statistics"))
            has_write = payload.get("primary_update_endpoint") == "PARAMETER_WRITE"
            rows.append({
                "case_id": payload.get("case_id", path.stem),
                "artifact": str(path.relative_to(ROOT)),
                "artifact_sha256": _sha(path),
                "status": payload.get("status", "UNKNOWN"),
                "stage_geometry": stage_geometry,
                "original_coordinate_statistics": has_original,
                "actual_parameter_write": has_write,
                "new_protocol_reuse": (
                    "DIRECT_RECOMPUTE_AVAILABLE" if has_original and has_write
                    else "RECAPTURE_REQUIRED_FOR_ACTUAL_WRITE_AND_ORIGINAL_ENERGY"
                    if uses_legacy_sketch
                    else "RECAPTURE_REQUIRED_FOR_ACTUAL_PARAMETER_WRITE"
                ),
                "historical_result_policy": "RETAIN_WITH_ORIGINAL_MEASUREMENT_SEMANTICS",
            })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "schema": "kernel-analyzer-measurement-provenance-audit-v1",
        "legacy_sketch_issue": (
            "The historical empirical v2 sketch derived bucket and sign from the same "
            "hash; for power-of-two dimensions the sign was determined by bucket parity."
        ),
        "new_capture_requirement": (
            "Save original-coordinate effect energy, repair energy, and their inner "
            "product before any summary; measure the actual stored-parameter write."
        ),
        "cases": rows,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
