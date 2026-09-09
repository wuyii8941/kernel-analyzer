#!/usr/bin/env python3
"""Recompute a frozen optimizer-state follow-up from stage baseline records."""
import argparse
import hashlib
import json
import math
from pathlib import Path


STAGES = ("LOCAL", "PARAMETER_GRADIENT", "PARAMETER_WRITE")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_by_case(document, selected):
    grouped = {case_id: {} for case_id in selected}
    for row in document.get("rows", []):
        case_id = row.get("case_id")
        if case_id not in grouped:
            continue
        stage = row.get("stage")
        if stage in grouped[case_id]:
            raise ValueError("Duplicate case-stage baseline")
        if stage not in STAGES or row.get("status") != "VALID":
            raise ValueError("Selected case has invalid stage baseline")
        value = row.get("relative_rms")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid relative RMS")
        grouped[case_id][stage] = float(value)
    if any(set(stages) != set(STAGES) for stages in grouped.values()):
        raise ValueError("Every selected case requires all three stages")
    return grouped


def summarize(plan, cold, warm8, warm32):
    selected = [case["case_id"] for case in plan.get("cases", [])]
    if (plan.get("schema") != "family-optimizer-state-followup-v1" or not selected
            or len(set(selected)) != len(selected)):
        raise ValueError("Frozen optimizer follow-up plan required")
    sources = {
        "ZERO_MOMENTS_REPRODUCTION": rows_by_case(cold, selected),
        "8_STEP_TARGET_PARAMETER_WARM_STATE": rows_by_case(warm8, selected),
        "32_STEP_TARGET_PARAMETER_WARM_STATE": rows_by_case(warm32, selected),
    }
    records = []
    directionally_consistent = True
    for case_id in selected:
        base = sources["ZERO_MOMENTS_REPRODUCTION"][case_id]
        conditions = []
        for condition in ("8_STEP_TARGET_PARAMETER_WARM_STATE",
                          "32_STEP_TARGET_PARAMETER_WARM_STATE"):
            current = sources[condition][case_id]
            ratios = {}
            for stage in STAGES:
                ratios[stage] = (
                    current[stage] / base[stage] if base[stage] > 0 else None
                )
            matched = (
                ratios["PARAMETER_WRITE"] is not None
                and ratios["PARAMETER_GRADIENT"] is not None
                and ratios["PARAMETER_WRITE"] < 1
                and ratios["PARAMETER_WRITE"] < ratios["PARAMETER_GRADIENT"]
            )
            directionally_consistent &= matched
            conditions.append({
                "condition": condition,
                "relative_rms": current,
                "warm_to_zero_moments_ratio": ratios,
                "update_falls_more_than_gradient": matched,
            })
        records.append({
            "case_id": case_id,
            "zero_moments_relative_rms": base,
            "warm_conditions": conditions,
        })
    return {
        "schema": "optimizer-state-followup-summary-v1",
        "frozen_prediction": plan["prediction_fixed_before_followup"],
        "records": records,
        "directional_prediction_result": (
            "DIRECTIONALLY_CONSISTENT_MAGNITUDE_THRESHOLD_NOT_PREDECLARED"
            if directionally_consistent else "NOT_DIRECTIONALLY_CONSISTENT"
        ),
        "formal_prediction_test": "NOT_ASSESSED_NO_NUMERIC_SUCCESS_THRESHOLD_PREDECLARED",
        "state_scope": "TARGET_PARAMETER_ONLY_WARM_STATE_NOT_FULL_MODEL_NATURAL_TRAINING",
        "training_outcome_established": False,
        "scope": (
            "Verified fixed-suite descriptive comparison; not a random-state inference, "
            "mechanism proof, or training-quality result"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--warm8", type=Path, required=True)
    parser.add_argument("--warm32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    paths = [args.plan, args.cold, args.warm8, args.warm32]
    documents = [json.loads(path.read_text()) for path in paths]
    result = summarize(*documents)
    result["input_sha256"] = {str(path.resolve()): sha(path) for path in paths}
    result["summarizer_sha256"] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({
        "cases": len(result["records"]),
        "directional_prediction_result": result["directional_prediction_result"],
    }))


if __name__ == "__main__":
    main()
