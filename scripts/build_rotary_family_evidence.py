"""Build a checked compact record for the Ministral fused-RoPE measurements.

The record is reporting evidence, not a population or training-loss claim.  It
binds the fixed-suite analyses to the audited Triton source contract and to the
matched cold/warm/reset-moment comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def one_case(plan: dict) -> dict:
    cases = plan.get("cases", [])
    if len(cases) != 1:
        raise ValueError("Expected one selected fused-RoPE case")
    case = cases[0]
    if case.get("implementation_kind") != "TRITON":
        raise ValueError("Selected implementation is not recorded as Triton")
    if case.get("reference_family") != "ATTENTION_POSITION_SCALING":
        raise ValueError("Unexpected registered-reference family")
    if case.get("reference_method") != "REGISTERED_SAME_INPUT_REFERENCE":
        raise ValueError("Fused-RoPE comparison is not same-input")
    return case


def checked_analysis(payload: dict) -> dict:
    if payload.get("schema") != "kernel-analyzer-analysis-result-v1":
        raise ValueError("Unexpected unified-analysis schema")
    if payload.get("measurement_status") != "VALID":
        raise ValueError("Unified analysis is not valid")
    if payload.get("claim_scope") != "FIXED_SUITE_UPDATE":
        raise ValueError("Only the fixed-suite result may be summarized")
    if payload.get("training_outcome") != "NOT_MEASURED":
        raise ValueError("This evidence record must not imply a training outcome")
    bias = payload["bias_analysis"]
    if bias.get("decision_rule") != "ORIGINAL_COORDINATE_Q_ONLY_FIXED_SUITE":
        raise ValueError("Expected original-coordinate fixed-suite decision")
    if bias.get("population_guarantee") is not False:
        raise ValueError("Population guarantee must remain disabled")
    return {
        "parameter_write_total_rms": bias["fixed_suite_total_rms"],
        "parameter_write_aligned_ratio_of_sums": bias[
            "fixed_suite_aligned_ratio_of_sums"
        ],
        "equivalence_decision": payload["equivalence_decision"],
    }


def build(
    high_analysis_path: Path,
    low_analysis_path: Path,
    high_plan_path: Path,
    low_plan_path: Path,
    optimizer_summary_path: Path,
) -> dict:
    high_analysis = load(high_analysis_path)
    low_analysis = load(low_analysis_path)
    high_case = one_case(load(high_plan_path))
    low_case = one_case(load(low_plan_path))
    optimizer = load(optimizer_summary_path)

    identities = {
        high_analysis["case_id"],
        low_analysis["case_id"],
        high_case["case_id"],
        low_case["case_id"],
        optimizer["case_id"],
    }
    if len(identities) != 1:
        raise ValueError("High, low and optimizer-condition records differ by case")
    if high_case.get("carrier") != low_case.get("carrier"):
        raise ValueError("High- and low-position comparisons use different parameters")
    high_contract = high_case["reference_contract"]
    low_contract = low_case["reference_contract"]
    semantic_hashes = {
        high_contract.get("function_semantic_ast_sha256"),
        low_contract.get("function_semantic_ast_sha256"),
    }
    if None in semantic_hashes or len(semantic_hashes) != 1:
        raise ValueError("High and low positions do not bind the same semantic source")
    if optimizer.get("status") != "COMPLETE_FIXED_SUITE_DESCRIPTIVE_COMPARISON":
        raise ValueError("Optimizer-condition comparison is incomplete")
    if optimizer.get("prediction_status") != "POST_OUTCOME_DIAGNOSTIC_NOT_PREREGISTERED":
        raise ValueError("Optimizer-condition data use was not preserved")
    if len(optimizer.get("state_ids", [])) != 32:
        raise ValueError("Expected the frozen 32-state optimizer comparison")
    comparisons = optimizer["comparisons"]
    if comparisons.get("warm_and_reset_gradient_statistics_exactly_equal") is not True:
        raise ValueError("Warm and reset comparisons do not share gradient statistics")

    high = checked_analysis(high_analysis)
    low = checked_analysis(low_analysis)
    condition_write = {
        name: values["PARAMETER_WRITE"]
        for name, values in optimizer["conditions"].items()
    }
    record = {
        "family": "ROTARY",
        "artifact_path": str(optimizer_summary_path),
        "artifact_sha256": sha(optimizer_summary_path),
        "evidence_kind": "FUSED_ROTARY_POSITION_SCALING_FIXED_SUITE",
        "backend": "INDUCTOR_GENERATED_TRITON",
        "reference_family": "ATTENTION_POSITION_SCALING",
        "reference_scope": "REGISTERED_SAME_INPUT_REFERENCE",
        "case_id": identities.pop(),
        "target_parameter": high_case["carrier"],
        "semantic_source_sha256": semantic_hashes.pop(),
        "state_count": 32,
        "measurement_status": "VALID_FIXED_SUITE_ENGINEERING_MEASUREMENT",
        "claim_scope": "FIXED_SUITE_UPDATE",
        "high_position_result": high,
        "low_position_result": low,
        "optimizer_condition_parameter_write": condition_write,
        "optimizer_condition_comparisons": comparisons,
        "position_scaling_is_unique_root_cause": False,
        "population_guarantee": False,
        "training_quality_claim": False,
        "prediction_status": optimizer["prediction_status"],
        "position_inventory_count": 0,
        "scope": (
            "One Ministral fused-RoPE output and one target parameter. High/low "
            "position results and cold/warm/reset-moment results are fixed-suite "
            "comparisons; they do not establish a population effect or loss outcome."
        ),
    }
    inputs = [
        high_analysis_path,
        low_analysis_path,
        high_plan_path,
        low_plan_path,
        optimizer_summary_path,
    ]
    return {
        "schema": "operator-family-additional-evidence-v1",
        "records": [record],
        "input_sha256": {str(path): sha(path) for path in inputs},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-analysis", type=Path, required=True)
    parser.add_argument("--low-analysis", type=Path, required=True)
    parser.add_argument("--high-plan", type=Path, required=True)
    parser.add_argument("--low-plan", type=Path, required=True)
    parser.add_argument("--optimizer-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("New output under /data1/tzh required")
    result = build(
        args.high_analysis,
        args.low_analysis,
        args.high_plan,
        args.low_plan,
        args.optimizer_summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result["records"][0], indent=2))


if __name__ == "__main__":
    main()
