#!/usr/bin/env python3
"""Bind every reviewed channel-bias call to its exact bias parameter.

The generated cases use the existing common-input capture and analysis path.
No numerical result is read and model/layer names do not select cases.
"""
import argparse
import json
from pathlib import Path

from scripts.run_numerical_coverage import read, save, sha


def bind(pairs, paths, tasks):
    task_by_id = {row["task_id"]: row for row in tasks["rows"]}
    if len(task_by_id) != len(tasks["rows"]):
        raise ValueError("Duplicate task identity")
    parameter_by_primal = {
        row["primal"]: row for row in paths["parameter_binding_records"]
    }
    if len(parameter_by_primal) != len(paths["parameter_binding_records"]):
        raise ValueError("Duplicate primal binding")
    path_by_endpoint = {row["endpoint"]: row for row in paths["rows"]}
    if len(path_by_endpoint) != len(paths["rows"]):
        raise ValueError("Duplicate endpoint path")
    cases, unresolved = [], []
    for record in pairs["records"]:
        task = task_by_id.get(record["closed_task_id"])
        binding = parameter_by_primal.get(record["bias_name"])
        endpoint = path_by_endpoint.get(record["exact_aot_endpoint_id"])
        reasons = []
        if task is None:
            reasons.append("CLOSED_TASK_NOT_FOUND")
        elif (task.get("implementation_kind") != "TRITON"
              or task.get("formal_pointer") != "in_out_ptr0"
              or task.get("symbol") != record["bias_symbol"]
              or task.get("status") != "EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT"
              or task.get("exact_aot_endpoint_id") != record["exact_aot_endpoint_id"]):
            reasons.append("TASK_BOUNDARY_DIFFERS")
        if binding is None or binding.get("status") != "EXACT_MODULE_STACK_PARAMETER_BINDING":
            reasons.append("BIAS_PARAMETER_NOT_EXACTLY_BOUND")
        if endpoint is None or binding is None or binding["name"] not in {
                item["name"] for item in endpoint.get("parameters", [])}:
            reasons.append("BIAS_PARAMETER_NOT_REACHABLE_FROM_ENDPOINT")
        if reasons:
            unresolved.append({"closed_task_id": record["closed_task_id"],
                               "reasons": reasons})
            continue
        cases.append({
            "case_id": record["closed_task_id"].replace(":", "_") + "-channel-bias-common-input",
            "task_id": record["closed_task_id"],
            "carrier": binding["name"],
            "reference_method": "PARTIAL_REDUCTION_FROM_BOUND_INPUT",
            "reference_contract_symbol": record["bias_symbol"],
            "implementation_kind": "TRITON",
            "exact_aot_endpoint_id": record["exact_aot_endpoint_id"],
            "parameter_selection_rule": "EXACT_BIAS_PRIMAL_TO_MODULE_PARAMETER",
            "parameter_selection_evidence": {
                "primal": record["bias_name"],
                "binding_status": binding["status"],
                "aliases": binding["aliases"],
            },
            "runtime_parameter_reach": "NOT_YET_MEASURED",
            "comparison_scope": "COMMON_PRE_CALL_BASE_AND_BIAS_INPUTS",
        })
    return cases, unresolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--parameter-paths", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    pairs, paths, tasks = map(read, (args.pairs, args.parameter_paths, args.tasks))
    cases, unresolved = bind(pairs, paths, tasks)
    result = {
        "schema": "channel-bias-training-cases-v1",
        "cases": cases,
        "unresolved": unresolved,
        "source_selection_uses_numerical_results": False,
        "all_reviewed_positions_bound": not unresolved and len(cases) == len(pairs["records"]),
        "scope": "Reviewed in-place BF16 channel-bias Triton calls in one saved Mamba release",
        "source_sha256": {
            str(path.resolve()): sha(path)
            for path in (args.pairs, args.parameter_paths, args.tasks, Path(__file__))
        },
    }
    save(args.output, result)
    print(json.dumps({"cases": len(cases), "unresolved": len(unresolved)}))


if __name__ == "__main__":
    main()
