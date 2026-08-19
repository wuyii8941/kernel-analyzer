#!/usr/bin/env python3
"""Summarize the corrected semantic-region F+B rescreen without raw hashes."""

from __future__ import annotations

import glob
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/bias_formation/hotspot_search"
OUTPUT = BASE / "semantic_region_rescreen_summary.json"


def main() -> None:
    rows = []
    paths = glob.glob(str(BASE / "semantic_region_screen/*/screening_gram.json"))
    paths += glob.glob(str(BASE / "semantic_region_anchor/*_screen/screening_gram.json"))
    paths += glob.glob(str(BASE / "multishape_screen/*_closure_gap/screening_gram.json"))
    for path_string in sorted(paths):
        path = Path(path_string)
        data = json.loads(path.read_text())
        for case in data["cases"]:
            layers = case["layers"]
            local = layers["LOCAL_ENDPOINT"]
            gradient = layers["PARAMETER_GRADIENT"]
            rows.append({
                "screen": path.parent.name,
                "case_id": case["case_id"],
                "task_id": case["task_id"],
                "carrier": case["carrier"],
                "states": int(data["state_count"]),
                "local_energy": local["average_state_energy"],
                "local_ratio": local["cross_state_ratio"],
                "local_status": local["status"],
                "gradient_energy": gradient["average_state_energy"],
                "gradient_ratio": gradient["cross_state_ratio"],
                "gradient_status": gradient["status"],
                "downstream_reach": gradient["average_state_energy"] > 0,
            })
    result = {
        "schema": "kernel-analyzer-semantic-region-rescreen-summary-v1",
        "status": "PARTIAL_SCREENING",
        "complete_fb_unit_required": True,
        "screen_count": len(rows),
        "downstream_reach_count": sum(row["downstream_reach"] for row in rows),
        "confirmed_new_bias_case_count": 0,
        "rows": rows,
        "interpretation": {
            "phi_anchor": (
                "The corrected region path recovers the known Phi lm_head dX sensitivity: "
                "local centered with positive downstream gradient geometry."
            ),
            "normalization_partial_reductions": (
                "Qwen and DeepSeek partial FP32 reduction discrepancies are real but are erased "
                "by the following reduction/writeback under the tested states."
            ),
            "qwen_loss_addmm": (
                "The compiler-added loss-gradient addmm reaches the tied embedding gradient, "
                "but its four-state direction is centered at seq64/128/256."
            ),
            "mamba_loss_mm": (
                "The compiler-added loss-head MM reaches the tied embedding gradient, but its "
                "four-state direction is centered at seq64/128/256."
            ),
        },
        "provenance_policy": (
            "No transient-vector or derived-report SHA256; frozen release/input/common-state "
            "identity remains checked by the runner."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUTPUT), "screens": len(rows),
        "downstream_reach": result["downstream_reach_count"],
    }))


if __name__ == "__main__":
    main()
