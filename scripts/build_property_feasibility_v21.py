#!/usr/bin/env python3
"""Build case-by-property feasibility, without assigning any verdict."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/bias_formation_v2_1/property_feasibility.json"
PROPERTIES = ("P1_CONDITIONAL_SOURCE_ASYMMETRY", "P2_SOURCE_TRANSPORT_ALIGNMENT", "P3_FORWARD_BACKWARD_NUMERICAL_CONSISTENCY")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    roster = json.loads((ROOT / "results/property/bias_formation_v2_1/roster_bound.json").read_text())
    case_rows = []
    for case in roster["cases"]:
        cid = case["case_id"]
        if cid == "liger_fused_ce_t128":
            statuses = {PROPERTIES[0]: "BLOCKED_MISSING_SHAM", PROPERTIES[1]: "NOT_APPLICABLE", PROPERTIES[2]: "NOT_APPLICABLE"}
        elif cid == "phi4_lm_head_dx_seq64":
            statuses = {PROPERTIES[0]: "BLOCKED_MISSING_ATOMIZATION", PROPERTIES[1]: "BLOCKED_MISSING_TRANSPORT", PROPERTIES[2]: "NOT_APPLICABLE"}
        elif cid == "qwen_saved_p_seq128":
            statuses = {PROPERTIES[0]: "NOT_APPLICABLE", PROPERTIES[1]: "CAPTURE_READY", PROPERTIES[2]: "CAPTURE_READY"}
        elif cid == "qwen_bmm_seq64":
            statuses = {PROPERTIES[0]: "NOT_APPLICABLE", PROPERTIES[1]: "BLOCKED_MISSING_SHAM", PROPERTIES[2]: "NOT_APPLICABLE"}
        else:
            statuses = {prop: "BLOCKED_MISSING_ATOMIZATION" for prop in PROPERTIES}
        case_rows.append({"case_id": cid, "role": case.get("role"), "properties": statuses, "source_status": case.get("source_status")})
    specs = ROOT / "results/property/bias_formation_v2/property_specs.json"
    return {
        "schema": "kernel-analyzer-bias-property-feasibility-v2_1",
        "status": "PRE_MEASUREMENT_STATISTICAL_CORRECTION",
        "supersedes": "bias_formation_v2",
        "v2_gpu_measurements": 0,
        "property_ids": list(PROPERTIES),
        "property_specs_sha256": _sha(specs),
        "statuses": ["CAPTURE_READY", "INTERVENTION_READY", "NOT_APPLICABLE", "BLOCKED_MISSING_ATOMIZATION", "BLOCKED_MISSING_TRANSPORT", "BLOCKED_MISSING_SHAM"],
        "cases": case_rows,
    }


if __name__ == "__main__":
    result = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_name("." + OUT.name + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(OUT)
    print(json.dumps({"output": str(OUT), "gpu_campaign_started": False}, sort_keys=True))
