#!/usr/bin/env python3
"""Summarize reproducible directions from verified family measurements.

This is a fixed-suite diagnostic, not a population test or training selector.
Total parameter-write magnitude and direction replication remain separate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from kernel_analyzer.numerical_baselines import stage_baselines
from kernel_analyzer.training_equivalence import simultaneous_intervals_from_joint_gram


def interval_sign(interval):
    low, high = map(float, interval)
    if not all(map(math.isfinite, (low, high))) or low > high:
        raise ValueError("Invalid direction interval")
    return 1 if low > 0 else -1 if high < 0 else 0


def direction_intervals(payload, stage="PARAMETER_WRITE"):
    ids=list(payload["state_ids"])
    calibration=list(payload["calibration_state_ids"])
    confirmation=list(payload["confirmation_state_ids"])
    if (not calibration or not confirmation or len(set(ids))!=len(ids)
            or set(ids)!=set(calibration+confirmation) or set(calibration)&set(confirmation)):
        raise ValueError("Invalid fixed-suite partitions")
    order=[ids.index(value) for value in calibration+confirmation]
    results={}; unavailable={}
    for view,record in sorted(payload.get("stages",{}).get(stage,{}).items()):
        if view!="EXACT" and not view.startswith(("COUNT_SKETCH_V3_FLOAT64", "COUNT_SKETCH_V2")):
            continue
        profile=record.get("profile",{})
        gram=profile.get("suite",profile).get("joint_gram")
        if gram is None:
            continue
        ordered={}
        for key in ("effect_effect","repair_repair","effect_repair"):
            matrix=np.asarray(gram[key],dtype=float)
            if matrix.shape!=(len(ids),len(ids)) or not np.isfinite(matrix).all():
                raise ValueError("Invalid joint Gram")
            ordered[key]=matrix[np.ix_(order,order)]
        try:
            results[view]=simultaneous_intervals_from_joint_gram(
                ordered,calibration_count=len(calibration))
        except ValueError as error:
            unavailable[view]=str(error)
    return {"intervals":results,"unavailable":unavailable}


def replicated_sign(intervals, endpoint):
    signs=[interval_sign(value[endpoint]) for value in intervals.values()]
    if not signs or 0 in signs or len(set(signs))!=1:
        return "NOT_REPRODUCED_ACROSS_ALL_RECORDED_VIEWS"
    return "POSITIVE" if signs[0]>0 else "NEGATIVE"


def summarize(summary_paths):
    rows=[]; inputs={}
    seen=set()
    for summary_path in summary_paths:
        data=summary_path.read_bytes(); inputs[str(summary_path.resolve())]=hashlib.sha256(data).hexdigest()
        summary=json.loads(data)
        for observation in summary.get("observations",[]):
            identity=(str(Path(observation["raw_artifact"]).resolve()),observation["raw_sha256"])
            if identity in seen:
                raise ValueError("Duplicate verified raw measurement")
            seen.add(identity)
            raw_path=Path(identity[0]); raw_data=raw_path.read_bytes()
            if hashlib.sha256(raw_data).hexdigest()!=identity[1]:
                raise ValueError("Raw measurement changed")
            payload=json.loads(raw_data)
            stage={row["stage"]:row for row in stage_baselines(payload)}["PARAMETER_WRITE"]
            if stage["status"]!="VALID":
                raise ValueError("Parameter-write statistics are unavailable")
            diagnostics=direction_intervals(payload)
            intervals=diagnostics["intervals"]
            rows.append({
                "case_id":observation["case_id"],
                "update_rms_relative":stage["relative_rms"],
                "aligned_ratio_of_sums":stage["aligned_ratio_of_sums"],
                "recorded_direction_views":len(intervals),
                "unavailable_direction_views":diagnostics["unavailable"],
                "additive_direction":replicated_sign(intervals,"additive"),
                "repair_aligned_direction":replicated_sign(intervals,"repair_aligned"),
                "residual_direction":replicated_sign(intervals,"residual_direction"),
                "intervals_by_view":intervals,
                "raw_artifact":str(raw_path),"raw_sha256":identity[1],
            })
    rows.sort(key=lambda row:(-row["update_rms_relative"],row["case_id"]))
    return {
        "schema":"fixed-suite-direction-candidate-summary-v1",
        "records":rows,"record_count":len(rows),
        "direction_reproduced_counts":{
            endpoint:sum(row[endpoint] in {"POSITIVE","NEGATIVE"} for row in rows)
            for endpoint in ("additive_direction","repair_aligned_direction","residual_direction")
        },
        "input_sha256":inputs,
        "training_selection_performed":False,
        "scope":"Fixed-suite descriptive replication across recorded views; not a population bias verdict or training outcome",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-summary",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    result=summarize(args.measurement_summary)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps({"record_count":result["record_count"],
                      "direction_reproduced_counts":result["direction_reproduced_counts"]}))


if __name__=="__main__":
    main()
