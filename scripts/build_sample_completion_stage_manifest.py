#!/usr/bin/env python3
"""Report which sample-completion stages are already reusable.

This manifest is a planning/engineering artifact.  A previous 2-state reach or
16-state formation file is not silently promoted to a 32-step uniform case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/sample_completion_v1/stage_manifest.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    roster = load(ROOT / "results/property/sample_completion_v1/roster.json")
    stages: list[dict[str, Any]] = []
    existing_reach = {
        "qwen": ROOT / "results/property/bias_formation/hotspot_search/equivalence_reach/qwen/engineering_reach.json",
        "phi4": ROOT / "results/property/bias_formation/hotspot_search/equivalence_reach/phi4/engineering_reach.json",
        "deepseek8b": ROOT / "results/property/bias_formation/hotspot_search/equivalence_reach/deepseek8b/engineering_reach.json",
        "mamba": ROOT / "results/property/bias_formation/hotspot_search/equivalence_reach/mamba/engineering_reach.json",
    }
    reach_by_case: dict[str, dict[str, Any]] = {}
    for model, path in existing_reach.items():
        if not path.exists():
            continue
        payload = load(path)
        for case in payload.get("cases", []):
            reach_by_case[str(case.get("task_id"))] = {
                "model": model,
                "states": int(payload.get("states", 0)),
                "carrier_reached_any_state": bool(case.get("carrier_reached_any_state", False)),
                "artifact": str(path.relative_to(ROOT)),
            }

    formation_dirs = {
        "qwen": ROOT / "results/property/bias_formation/hotspot_search/qwen_seq64_rescreen_v2",
        "phi4": ROOT / "results/property/bias_formation/hotspot_search/phi4_seq64_rescreen",
        "deepseek8b": ROOT / "results/property/bias_formation/hotspot_search/deepseek8b_seq64_rescreen",
    }
    formation_by_case: dict[str, str] = {}
    for directory in formation_dirs.values():
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                payload = load(path)
            except (OSError, json.JSONDecodeError):
                continue
            case_id = payload.get("case_id")
            if case_id:
                formation_by_case[str(case_id)] = str(path.relative_to(ROOT))

    for row in roster["base_cases"] + roster["search_units"]:
        case_id = str(row["case_id"])
        # The search-unit case IDs are already the exact IDs in the formation
        # rescreen files; for base cases there is no uniform trace yet.
        prior = formation_by_case.get(case_id)
        reach = reach_by_case.get(str(row.get("task_id")))
        stages.append({
            "case_id": case_id,
            "role": row["role"],
            "operator_family": row["implementation_family"],
            "engineering_2_state": "EXISTING" if reach else "NOT_BOUND_TO_NEW_ROSTER",
            "formation_8_state": "NOT_STARTED",
            "formation_16_state": "EXISTING_LEGACY_NOT_UNIFORM" if prior else "NOT_STARTED",
            "consequence_32_step": "NOT_STARTED",
            "uniform_trace_ready": False,
            "scientific_label": None,
            "prior_reach": reach,
            "prior_formation": prior,
        })
    result = {
        "schema": "kernel-analyzer-sample-completion-stage-manifest-v1",
        "status": "PRE_CAMPAIGN_NO_NEW_UNIFORM_LABELS",
        "stages": stages,
        "counts": {
            "roster_units": len(stages),
            "engineering_existing": sum(r["engineering_2_state"] == "EXISTING" for r in stages),
            "legacy_formation_existing": sum(r["formation_16_state"] == "EXISTING_LEGACY_NOT_UNIFORM" for r in stages),
            "uniform_32_complete": 0,
        },
        "claim_boundary": "Prior reach/formation artifacts identify reusable work only. They do not satisfy the sample_completion_v1 uniform trace or 32-step label gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
