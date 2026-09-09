#!/usr/bin/env python3
"""Freeze an optimizer-state follow-up from a completed direction summary."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(plan, direction_summary, *, minimum_update_rms, maximum_cases,
           required_direction):
    if (not math.isfinite(minimum_update_rms) or minimum_update_rms<=0
            or type(maximum_cases) is not int or maximum_cases<=0):
        raise ValueError("Positive frozen selection limits required")
    if required_direction not in {
        "additive_direction","repair_aligned_direction","residual_direction"
    }:
        raise ValueError("Unknown direction requirement")
    cases={case["case_id"]:case for case in plan.get("cases",[])}
    if len(cases)!=len(plan.get("cases",[])):
        raise ValueError("Unique source cases required")
    eligible=[]
    for row in direction_summary.get("records",[]):
        if (row.get(required_direction) in {"POSITIVE","NEGATIVE"}
                and row.get("update_rms_relative",-1)>=minimum_update_rms
                and row.get("case_id") in cases):
            eligible.append(row)
    eligible.sort(key=lambda row:(-row["update_rms_relative"],row["case_id"]))
    chosen=eligible[:maximum_cases]
    return {
        "schema":"family-optimizer-state-followup-v1",
        "cases":[cases[row["case_id"]] for row in chosen],
        "selection_rule":{
            "required_direction":required_direction,
            "minimum_update_rms":minimum_update_rms,
            "maximum_cases":maximum_cases,
            "ranking":"DESCENDING_FIXED_SUITE_PARAMETER_WRITE_RMS_THEN_CASE_ID",
        },
        "selected_evidence":[{
            "case_id":row["case_id"],"update_rms_relative":row["update_rms_relative"],
            "direction":row[required_direction],
        } for row in chosen],
        "followup_conditions":["ZERO_MOMENTS_REPRODUCTION","8_STEP_TARGET_PARAMETER_WARM_STATE",
                               "32_STEP_TARGET_PARAMETER_WARM_STATE"],
        "prediction_fixed_before_followup":(
            "If cold-start AdamW response causes most of the parameter-write RMS, warm moments "
            "will reduce parameter-write RMS substantially while local and gradient RMS do not "
            "fall by the same factor."
        ),
        "data_use":"RESULT_AWARE_MECHANISM_FOLLOWUP_NOT_BLIND_DISCOVERY",
        "training_outcome_established":False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--direction-summary",type=Path,required=True)
    parser.add_argument("--minimum-update-rms",type=float,required=True)
    parser.add_argument("--maximum-cases",type=int,required=True)
    parser.add_argument("--required-direction",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    plan=json.loads(args.plan.read_text())
    summary=json.loads(args.direction_summary.read_text())
    result=select(plan,summary,minimum_update_rms=args.minimum_update_rms,
                  maximum_cases=args.maximum_cases,required_direction=args.required_direction)
    result["source_plan_sha256"]=sha(args.plan)
    result["direction_summary_sha256"]=sha(args.direction_summary)
    result["freezing_script_sha256"]=sha(Path(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps({"selected_cases":len(result["cases"]),
                      "case_ids":[case["case_id"] for case in result["cases"]]}))


if __name__=="__main__":
    main()
