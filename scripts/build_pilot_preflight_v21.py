#!/usr/bin/env python3
"""Aggregate the four 2-state engineering preflights; never emits verdicts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/property/bias_formation_v2_1/pilot"
CASE_IDS = ("liger_fused_ce_t128", "phi4_lm_head_dx_seq64", "qwen_saved_p_seq128", "qwen_bmm_seq64")


def build() -> dict:
    rows = []
    for case_id in CASE_IDS:
        path = PILOT / f"{case_id}.preflight.json"
        if not path.exists():
            rows.append({"case_id": case_id, "status": "MISSING_PREFLIGHT"})
        else:
            rows.append(json.loads(path.read_text()))
    ready = all(row.get("status") == "PREFLIGHT_READY" for row in rows)
    return {
        "schema": "kernel-analyzer-bias-property-pilot-preflight-v2_1",
        "status": "PREFLIGHT_READY" if ready else "BLOCKED_PROVENANCE",
        "case_count": len(CASE_IDS),
        "dry_run_states": 2,
        "scientific_verdict": False,
        "gpu_execution_started": False,
        "formation_16_plus_16_started": False,
        "cases": rows,
        "claim_boundary": "engineering preflight only; no formation or consequence measurement",
    }


if __name__ == "__main__":
    result = build()
    path = PILOT / "pilot_preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
    print(json.dumps({"output": str(path), "status": result["status"], "gpu_execution_started": False}, sort_keys=True))
