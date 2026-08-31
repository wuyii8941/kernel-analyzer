#!/usr/bin/env python3
"""Align unified 32-state profiles with already completed 4096-step outcomes."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "results/property/three_mechanism_profiles_v1/summary.json"
OUT = ROOT / "results/property/three_mechanism_profiles_v1/consequence_alignment.json"
LONG = ROOT / "results/property/declared_persistent_4096/operator_scan_targets"
CASES = {
    "gemma4_text128_scan_0037": "normalization reduction",
    "llama32_text128_scan_0000": "softmax backward",
    "llama32_text512_scan_extern_dc4ef40f35eb": "attention BMM",
}


def main() -> None:
    profiles = json.loads(PROFILE.read_text())
    rows = []
    for case_id, mechanism in CASES.items():
        long_path = LONG / case_id / "consequence.json"
        long = json.loads(long_path.read_text())
        compact_profile = {}
        for stage, value in profiles["cases"][case_id]["stages"].items():
            compact_profile[stage] = {
                key: item for key, item in value.items()
                if key not in {"joint_gram", "effect_vector_digests", "repair_vector_digests"}
            }
        row = {
            "case_id": case_id,
            "mechanism_family": mechanism,
            "profile": compact_profile,
            "long_horizon": {
                "steps": long["steps"],
                "statistics": long["statistics"],
                "final_drift_l2": long["final_drift_l2"],
                "loss_audit": long["loss_audit"],
                "artifact": str(long_path.relative_to(ROOT)),
            },
        }
        rows.append(row)
    payload = {
        "schema": "kernel-analyzer-three-mechanism-consequence-alignment-v1",
        "status": "COMPLETE_THREE_EXACT_TARGETS",
        "profile_protocol": "32 distinct matched states; 16 calibration + 16 untouched confirmation",
        "inference": "95% state bootstrap intervals; sign-flip p-values; Holm across 3 cases x 3 stages x 3 effect types",
        "long_horizon_role": "paired consequence only; not used to select profile directions",
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
