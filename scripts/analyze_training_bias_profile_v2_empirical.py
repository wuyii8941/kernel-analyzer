#!/usr/bin/env python3
"""Apply the frozen five-case multiplicity and robustness rules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.training_bias_profile import BRANCHES, holm_adjusted_p  # noqa: E402


CASES = (
    "liger",
    "phi",
    "qwen_lmhead",
    "qwen_vproj",
    "mamba_inproj",
)
PRIMARY_STAGE = "ADAMW_UPDATE"
EXPLANATION_STAGES = ("LOCAL", "PARAMETER_GRADIENT")
PRIMARY_VIEW = "SKETCH_SEED_20260831"


def _primary_view(views: dict[str, Any]) -> str:
    if "EXACT" in views:
        if len(views) != 1:
            raise RuntimeError("an exact stage must not also contain sketches")
        return "EXACT"
    if PRIMARY_VIEW not in views:
        raise RuntimeError(f"large-vector stage is missing {PRIMARY_VIEW}")
    return PRIMARY_VIEW


def _sign(value: float) -> int:
    return 1 if value > 0.0 else (-1 if value < 0.0 else 0)


def _branch_summary(
    views: dict[str, Any], branch: str, adjusted_p: float,
) -> dict[str, Any]:
    primary_name = _primary_view(views)
    primary_profile = views[primary_name]["profile"]
    if primary_profile["status"] != "POPULATION_INFERENCE_COMPLETE":
        return {
            "status": "ABSTAIN",
            "reason": primary_profile["status"],
            "primary_view": primary_name,
        }
    primary = primary_profile["population_inference"]["branches"][branch]
    estimates: dict[str, float] = {}
    raw_confirmed: dict[str, bool] = {}
    statuses: dict[str, str] = {}
    for name, wrapped in sorted(views.items()):
        profile = wrapped["profile"]
        statuses[name] = str(profile["status"])
        if profile["status"] != "POPULATION_INFERENCE_COMPLETE":
            continue
        item = profile["population_inference"]["branches"][branch]
        estimates[name] = float(item["estimate"])
        raw_confirmed[name] = bool(item["raw_confirmed"])
    signs = {_sign(value) for value in estimates.values()}
    direction_robust = len(estimates) == len(views) and len(signs) == 1 and 0 not in signs
    lower, upper = primary["confidence_interval_95"]
    interval_excludes_zero = lower > 0.0 or upper < 0.0
    confirmed = bool(
        adjusted_p <= 0.05
        and interval_excludes_zero
        and primary["confirmation_direction_matches_calibration"]
        and direction_robust
    )
    return {
        "status": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
        "primary_view": primary_name,
        "estimate": primary["estimate"],
        "confidence_interval_95": primary["confidence_interval_95"],
        "raw_studentized_signflip_p": primary["raw_studentized_signflip_p"],
        "holm_adjusted_p": adjusted_p,
        "confirmation_direction_matches_calibration": primary[
            "confirmation_direction_matches_calibration"
        ],
        "interval_excludes_zero": interval_excludes_zero,
        "summary_direction_robust": direction_robust,
        "estimate_by_view": estimates,
        "raw_confirmed_by_view": raw_confirmed,
        "view_status": statuses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: dict[str, dict[str, Any]] = {}
    for case in CASES:
        path = args.input_dir / f"{case}.json"
        payload = json.loads(path.read_text())
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"{case} recapture is incomplete: {payload}")
        if not payload["determinism"]["all_exact"]:
            raise RuntimeError(f"{case} determinism gate failed")
        rows[case] = payload

    primary_raw: dict[str, float] = {}
    explanation_raw: dict[str, float] = {}
    for case, payload in rows.items():
        for stage in (PRIMARY_STAGE, *EXPLANATION_STAGES):
            views = payload["stages"][stage]
            primary_name = _primary_view(views)
            profile = views[primary_name]["profile"]
            if profile["status"] != "POPULATION_INFERENCE_COMPLETE":
                raise RuntimeError(f"{case}/{stage} did not complete population inference")
            for branch in BRANCHES:
                key = f"{case}|{stage}|{branch}"
                value = float(
                    profile["population_inference"]["branches"][branch][
                        "raw_studentized_signflip_p"
                    ]
                )
                (primary_raw if stage == PRIMARY_STAGE else explanation_raw)[key] = value
    if len(primary_raw) != 15 or len(explanation_raw) != 30:
        raise RuntimeError("frozen multiplicity family is incomplete")
    primary_adjusted = holm_adjusted_p(primary_raw)
    explanation_adjusted = holm_adjusted_p(explanation_raw)

    cases: dict[str, Any] = {}
    for case, payload in rows.items():
        stages: dict[str, Any] = {}
        for stage in (*EXPLANATION_STAGES, PRIMARY_STAGE):
            family = primary_adjusted if stage == PRIMARY_STAGE else explanation_adjusted
            stages[stage] = {
                "suite": payload["stages"][stage][
                    _primary_view(payload["stages"][stage])
                ]["profile"]["suite"],
                "branches": {
                    branch: _branch_summary(
                        payload["stages"][stage],
                        branch,
                        family[f"{case}|{stage}|{branch}"],
                    )
                    for branch in BRANCHES
                },
            }
        confirmed_update = [
            branch for branch, item in stages[PRIMARY_STAGE]["branches"].items()
            if item["status"] == "CONFIRMED"
        ]
        cases[case] = {
            "case_id": payload["case_id"],
            "primary_update_result": (
                "CONFIRMED_TRAINING_UPDATE_EFFECT"
                if confirmed_update else "NO_CONFIRMED_UPDATE_EFFECT_UNDER_PROTOCOL"
            ),
            "confirmed_update_branches": confirmed_update,
            "stages": stages,
            "optimizer": payload["optimizer"],
            "carrier": payload["carrier"],
            "claim_boundary": payload["claim_boundary"],
        }

    output = {
        "schema": "kernel-analyzer-training-bias-profile-v2-five-case-summary",
        "status": "COMPLETE",
        "result_selection": "rules frozen before empirical recapture",
        "multiplicity": {
            "primary_family": "five cases x three branches at ADAMW_UPDATE",
            "primary_tests": len(primary_raw),
            "explanation_family": "five cases x two stages x three branches",
            "explanation_tests": len(explanation_raw),
            "method": "Holm family-wise adjustment",
        },
        "cases": cases,
        "boundaries": [
            "The population is frozen input states at one checkpoint, not independent training runs.",
            "The optimizer is cold-start AdamW with zero weight decay and zero moments at every state.",
            "A non-confirmed branch is not a SAFE verdict.",
            "LOCAL and PARAMETER_GRADIENT locate effects but are not the primary training-equivalence endpoint.",
            "Historical labels and 4096-step outcomes were not inputs to the decision rule."
        ],
    }
    if any(not math.isfinite(float(value)) for value in (*primary_raw.values(), *explanation_raw.values())):
        raise RuntimeError("nonfinite p-value")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": output["status"],
        "confirmed_update_cases": [
            case for case, item in cases.items()
            if item["primary_update_result"] == "CONFIRMED_TRAINING_UPDATE_EFFECT"
        ],
    }))


if __name__ == "__main__":
    main()
