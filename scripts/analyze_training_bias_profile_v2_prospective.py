#!/usr/bin/env python3
"""Analyze one frozen prospective Training Bias Profile v2 batch.

Every case named by the frozen protocol remains in the multiplicity family.
Missing or failed measurements contribute p=1 and are reported as ABSTAIN;
they are never silently replaced or removed after results are known.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.training_bias_profile import BRANCHES, holm_adjusted_p  # noqa: E402
from scripts.analyze_training_bias_profile_v2_empirical import (  # noqa: E402
    _branch_summary,
    _primary_view,
)


PRIMARY_STAGE = "ADAMW_UPDATE"
EXPLANATION_STAGES = ("LOCAL", "PARAMETER_GRADIENT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    frozen = protocol["cases"]

    payloads: dict[str, dict[str, Any]] = {}
    status_rows: dict[str, dict[str, Any]] = {}
    for row in frozen:
        case_id = f"{row['model']}_seq{row['sequence_length']}_{row['task_id'].replace(':', '_')}"
        path = args.raw_dir / f"{case_id}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("status") != "COMPLETE":
                raise RuntimeError(f"incomplete raw artifact: {path}")
            payloads[case_id] = payload
        elif args.status_dir is not None and (args.status_dir / f"{case_id}.json").exists():
            status_rows[case_id] = json.loads(
                (args.status_dir / f"{case_id}.json").read_text()
            )

    primary_raw: dict[str, float] = {}
    explanation_raw: dict[str, float] = {}
    for row in frozen:
        case_id = f"{row['model']}_seq{row['sequence_length']}_{row['task_id'].replace(':', '_')}"
        payload = payloads.get(case_id)
        for stage in (PRIMARY_STAGE, *EXPLANATION_STAGES):
            for branch in BRANCHES:
                key = f"{case_id}|{stage}|{branch}"
                target = primary_raw if stage == PRIMARY_STAGE else explanation_raw
                if payload is None:
                    target[key] = 1.0
                    continue
                views = payload["stages"][stage]
                profile = views[_primary_view(views)]["profile"]
                if profile["status"] != "POPULATION_INFERENCE_COMPLETE":
                    target[key] = 1.0
                else:
                    target[key] = float(
                        profile["population_inference"]["branches"][branch][
                            "raw_studentized_signflip_p"
                        ]
                    )

    primary_adjusted = holm_adjusted_p(primary_raw)
    explanation_adjusted = holm_adjusted_p(explanation_raw)
    cases: dict[str, Any] = {}
    for row in frozen:
        case_id = f"{row['model']}_seq{row['sequence_length']}_{row['task_id'].replace(':', '_')}"
        payload = payloads.get(case_id)
        if payload is None:
            status = status_rows.get(case_id, {})
            cases[case_id] = {
                "model": row["model"],
                "family": row["family"],
                "pool_rank": row["pool_rank"],
                "primary_update_result": "ABSTAIN",
                "reason": status.get("reason", "MEASUREMENT_NOT_COMPLETE"),
                "status_artifact": status or None,
            }
            continue
        stages: dict[str, Any] = {}
        for stage in (*EXPLANATION_STAGES, PRIMARY_STAGE):
            views = payload["stages"][stage]
            family = primary_adjusted if stage == PRIMARY_STAGE else explanation_adjusted
            stages[stage] = {
                "suite": views[_primary_view(views)]["profile"]["suite"],
                "branches": {
                    branch: _branch_summary(
                        views, branch, family[f"{case_id}|{stage}|{branch}"]
                    )
                    for branch in BRANCHES
                },
            }
        confirmed = [
            branch for branch, item in stages[PRIMARY_STAGE]["branches"].items()
            if item["status"] == "CONFIRMED"
        ]
        cases[case_id] = {
            "model": row["model"],
            "family": row["family"],
            "pool_rank": row["pool_rank"],
            "carrier": payload["carrier"],
            "primary_update_result": (
                "CONFIRMED_TRAINING_UPDATE_EFFECT"
                if confirmed else "NO_CONFIRMED_UPDATE_EFFECT_UNDER_PROTOCOL"
            ),
            "confirmed_update_branches": confirmed,
            "stages": stages,
        }

    result = {
        "schema": "kernel-analyzer-training-bias-profile-v2-prospective-batch-summary",
        "status": (
            "COMPLETE_ALL_FROZEN_CASES"
            if len(payloads) == len(frozen) else "COMPLETE_WITH_ABSTENTIONS"
        ),
        "selection": "frozen before new profile measurement",
        "protocol": str(args.protocol),
        "multiplicity": {
            "method": "Holm family-wise adjustment",
            "primary_family": "all frozen cases x three branches at ADAMW_UPDATE",
            "primary_tests": len(primary_raw),
            "explanation_family": "all frozen cases x two explanation stages x three branches",
            "explanation_tests": len(explanation_raw),
            "abstentions_retained_as_p_one": True,
        },
        "cases": cases,
        "claim_boundary": (
            "Prospective relative to Training Bias Profile v2 method development. "
            "The four frozen rows remain in the denominator regardless of outcome."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "completed": len(payloads),
        "frozen": len(frozen),
        "confirmed": [
            key for key, value in cases.items()
            if value["primary_update_result"] == "CONFIRMED_TRAINING_UPDATE_EFFECT"
        ],
    }))


if __name__ == "__main__":
    main()
