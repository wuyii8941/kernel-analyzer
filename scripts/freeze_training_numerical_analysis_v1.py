#!/usr/bin/env python3
"""Freeze the first reusable-analysis confirmation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/training_numerical_analysis_v1/protocol.json"
SOURCES = (
    "src/kernel_analyzer/training_bias_profile.py",
    "src/kernel_analyzer/training_equivalence.py",
    "src/kernel_analyzer/short_persistence.py",
    "scripts/run_training_bias_profile_v2_empirical.py",
    "scripts/capture_bound_endpoint_bias_formation_v21.py",
    "scripts/recompute_training_numerical_report.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    payload = {
        "schema": "kernel-analyzer-training-numerical-analysis-protocol-v1",
        "status": "FROZEN_BEFORE_NEW_RECAPTURE_RESULTS",
        "primary_stage": "PARAMETER_WRITE",
        "secondary_stages": ["LOCAL", "PARAMETER_GRADIENT", "ADAMW_UPDATE"],
        "fixed_suite_margins": {
            "full_update_rms": 0.01,
            "additive": 0.001,
            "repair_aligned": 0.01,
            "residual_direction": 0.001,
            "policy": "Historical engineering margins retained; not universal constants.",
        },
        "confirmation_cases": [
            {"case_id": "phi4_seq64_lmhead_dx", "role": "CONVENTIONAL_CANDIDATE"},
            {"case_id": "gemma4_text128_scan_0037", "role": "TRITON_NORMALIZATION_REDUCTION"},
            {"case_id": "llama32_text128_scan_0000", "role": "TRITON_SOFTMAX_BACKWARD"},
        ],
        "selection_basis": (
            "Existing bound cases selected for implementation-family coverage, not by "
            "new parameter-write outcome. Liger remains a mixed-path support case."
        ),
        "required_measurements": [
            "original_coordinate_effect_energy",
            "original_coordinate_repair_energy",
            "original_coordinate_effect_repair_inner_product",
            "actual_stored_parameter_write",
            "runtime_execution_identity",
        ],
        "claim_scope": (
            "The frozen 32-state suite and declared parameter only. Population inference "
            "requires separately declared independent units. Training outcome is not declared."
        ),
        "unresolved_policy": "Retain the original case ID and reason; do not replace it.",
        "source_sha256": {name: _sha(ROOT / name) for name in SOURCES},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
