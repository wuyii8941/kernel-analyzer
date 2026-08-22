#!/usr/bin/env python3
"""Summarize bounded open-loop formation controls without calling them live negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results/property/joint_bias_formation_v1/negative_openloop_formation"
DEFAULT_OUTPUT = ROOT / "results/property/joint_bias_formation_v1/negative_openloop_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cases = []
    for path in sorted(args.root.glob("**/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        confirmation = data.get("populations", {}).get("confirmation", {})
        cases.append({
            "case_id": data.get("case_id"),
            "artifact": str(path.relative_to(ROOT)),
            "overall_status": data.get("status"),
            "confirmation": {
                layer: confirmation.get(layer, {}).get("status")
                for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")
            },
            "formation_point": data.get("formation_point"),
            "first_observed_biased_stage": data.get("first_observed_biased_stage"),
        })
    payload = {
        "schema": "kernel-analyzer-negative-openloop-formation-summary-v1",
        "status": "COMPLETE_BOUNDED_OPEN_LOOP",
        "case_count": len(cases),
        "status_counts": {
            status: sum(case.get("overall_status") == status for case in cases)
            for status in sorted({case.get("overall_status") for case in cases})
        },
        "cases": cases,
        "claim_boundary": "These are 32-state open-loop formation controls. They do not estimate live candidate/repair trajectory recall or prove persistence safety.",
        "next_step": "Run a separate live consequence adapter only if the full-state candidate/repair recurrence contract is available.",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(cases), "output": str(args.output), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
