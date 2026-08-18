#!/usr/bin/env python3
"""Freeze the candidate-property tournament roster.

This command writes only a compact protocol and roster.  It does not read
candidate values and it does not assign a property label.  The eight project
cases are fixed before any bias-formation trace is collected; Flash Attention
is recorded as an external positive control and can never count as a project
discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.bias import BiasProperty  # noqa: E402


PROTOCOL_ID = "bias_formation_v1"
CALIBRATION_COUNT = 16
EVALUATION_COUNT = 16
STATE_SPLIT_SEED = 20260818


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def protocol() -> dict[str, Any]:
    return {
        "schema": "kernel-analyzer-bias-formation-protocol-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_PROTOCOL_NO_PROPERTY_RESULT",
        "unit": "ONE_EXACT_FORWARD_PLUS_ACTUAL_BACKWARD_ENDPOINT",
        "transition": [
            "local_endpoint_residual",
            "parameter_gradient_residual",
            "effective_optimizer_update_residual",
            "paired_trajectory_drift",
        ],
        "state_split": {
            "calibration_count": CALIBRATION_COUNT,
            "evaluation_count": EVALUATION_COUNT,
            "split_seed": STATE_SPLIT_SEED,
            "calibration_is_open_loop": True,
            "evaluation_is_closed_loop": True,
            "state_ids_must_be_bound_from_frozen_artifact": True,
            "calibration_and_evaluation_must_be_disjoint": True,
        },
        "candidate_properties": [
            {
                "id": prop.value,
                "short_name": prop.name,
                "role": {
                    BiasProperty.CONDITIONAL_SOURCE_ASYMMETRY: "baseline",
                    BiasProperty.SOURCE_TRANSPORT_ALIGNMENT: "formation_candidate",
                    BiasProperty.FB_NUMERICAL_CONSISTENCY: "formation_candidate",
                    BiasProperty.NONLINEAR_RECTIFICATION: "formation_candidate",
                    BiasProperty.OPTIMIZER_RECTIFICATION: "formation_candidate",
                    BiasProperty.SEMANTIC_ORBIT_CENTERING: "method_candidate",
                }[prop],
                "status": "HYPOTHESIS_ONLY",
                "uses_seup_as_label": False,
                "uses_t4_as_predictor": False,
            }
            for prop in BiasProperty
        ],
        "transition_certificate": {
            "labels": [
                "LOCAL_CENTERED",
                "LOCAL_BIASED",
                "GRADIENT_CENTERED",
                "GRADIENT_BIASED",
                "UPDATE_CENTERED",
                "UPDATE_BIASED",
                "TEMPORALLY_CANCELING",
                "PERSISTENT",
                "ABSTAIN_UNRESOLVED",
            ],
            "continuous_metrics_are_primary": True,
            "missing_layer_fails_closed": True,
            "ordinary_atol_rtol_is_baseline_only": True,
        },
        "freeze_rules": {
            "case_roster_frozen_before_trace": True,
            "property_ranking_not_used_to_change_roster": True,
            "old_t1_t4_and_seup_verdicts_immutable": True,
            "external_control_cannot_count_as_project_discovery": True,
            "no_single_property_is_required_to_pass": True,
        },
        "decision_rules": [
            "NON_CIRCULAR_CALIBRATION_ONLY",
            "CROSS_MECHANISM_REPLICATION",
            "POSITIVE_AND_NEGATIVE_DISCRIMINATION",
            "MATCHED_CAUSAL_INTERVENTION",
            "BEATS_TOLERANCE_AND_NORM_BASELINES",
        ],
    }


def roster() -> dict[str, Any]:
    project_cases = [
        {
            "case_id": "liger_fused_ce",
            "model": "qwen3_1p7b",
            "sequence_lengths": [64, 128, 256],
            "role": "DEEP_POSITIVE_ANCHOR",
            "primary_properties": [BiasProperty.CONDITIONAL_SOURCE_ASYMMETRY.value],
            "secondary_properties": [BiasProperty.OPTIMIZER_RECTIFICATION.value],
            "mechanism_boundary": "chunk_geometry_and_accumulation_endpoint",
        },
        {
            "case_id": "phi_lm_head_mm",
            "model": "phi4",
            "sequence_lengths": [64, 128, 256],
            "role": "BREADTH_POSITIVE",
            "primary_properties": [BiasProperty.CONDITIONAL_SOURCE_ASYMMETRY.value],
            "secondary_properties": [BiasProperty.SOURCE_TRANSPORT_ALIGNMENT.value],
            "mechanism_boundary": "mm_arithmetic_endpoint",
        },
        {
            "case_id": "qwen_saved_p_softmax",
            "model": "qwen3_1p7b",
            "sequence_lengths": [128],
            "role": "BREADTH_POSITIVE_OR_ROTATING",
            "primary_properties": [BiasProperty.SOURCE_TRANSPORT_ALIGNMENT.value, BiasProperty.FB_NUMERICAL_CONSISTENCY.value],
            "secondary_properties": [BiasProperty.NONLINEAR_RECTIFICATION.value],
            "mechanism_boundary": "saved_probability_to_softmax_backward_region",
        },
        {
            "case_id": "qwen_l23_key_materialization",
            "model": "qwen3_1p7b",
            "sequence_lengths": [64, 128, 256],
            "role": "BREADTH_POSITIVE_OR_ORBIT",
            "primary_properties": [BiasProperty.FB_NUMERICAL_CONSISTENCY.value, BiasProperty.SEMANTIC_ORBIT_CENTERING.value],
            "secondary_properties": [BiasProperty.SOURCE_TRANSPORT_ALIGNMENT.value],
            "mechanism_boundary": "delayed_key_materialization_region",
        },
        {
            "case_id": "qwen_rsqrt",
            "model": "qwen3_1p7b",
            "sequence_lengths": [128],
            "role": "CRITICAL_BOUNDARY",
            "primary_properties": [BiasProperty.NONLINEAR_RECTIFICATION.value, BiasProperty.OPTIMIZER_RECTIFICATION.value],
            "secondary_properties": [BiasProperty.SOURCE_TRANSPORT_ALIGNMENT.value],
            "mechanism_boundary": "rsqrt_backward_region",
        },
        {
            "case_id": "qwen_bmm",
            "model": "qwen3_1p7b",
            "sequence_lengths": [128],
            "role": "LOCAL_VARIANCE_NEGATIVE",
            "primary_properties": [BiasProperty.SOURCE_TRANSPORT_ALIGNMENT.value],
            "secondary_properties": [BiasProperty.SEMANTIC_ORBIT_CENTERING.value],
            "mechanism_boundary": "batched_mm_endpoint",
        },
        {
            "case_id": "qwen_seq128_vproj_rounding",
            "model": "qwen3_1p7b",
            "sequence_lengths": [128],
            "role": "NONPERSISTENT_SOURCE_BOUNDARY",
            "primary_properties": [BiasProperty.CONDITIONAL_SOURCE_ASYMMETRY.value, BiasProperty.SOURCE_TRANSPORT_ALIGNMENT.value],
            "secondary_properties": [BiasProperty.SEMANTIC_ORBIT_CENTERING.value],
            "mechanism_boundary": "v_projection_output_rounding_endpoint",
        },
        {
            "case_id": "qwen3vl_silu_backward",
            "model": "qwen3_vl",
            "sequence_lengths": [128],
            "role": "LOCAL_ERROR_NEGATIVE",
            "primary_properties": [BiasProperty.NONLINEAR_RECTIFICATION.value],
            "secondary_properties": [BiasProperty.OPTIMIZER_RECTIFICATION.value],
            "mechanism_boundary": "silu_backward_endpoint",
        },
    ]
    external = {
        "case_id": "flash_attention_external_control",
        "model": "external_control",
        "sequence_lengths": [],
        "role": "EXTERNAL_POSITIVE_CONTROL",
        "primary_properties": [BiasProperty.CONDITIONAL_SOURCE_ASYMMETRY.value],
        "secondary_properties": [BiasProperty.SOURCE_TRANSPORT_ALIGNMENT.value],
        "mechanism_boundary": "paper_defined_flash_attention_rounding_chain",
        "counts_as_project_discovery": False,
    }
    rows = []
    for row in project_cases + [external]:
        rows.append({
            **row,
            "counts_as_project_discovery": bool(row.get("counts_as_project_discovery", True)),
            "state_slots": {
                "calibration": [f"calibration_{i:02d}" for i in range(CALIBRATION_COUNT)],
                "evaluation": [f"evaluation_{i:02d}" for i in range(EVALUATION_COUNT)],
            },
            "state_binding": "runner_must_bind_exact_frozen_artifact_ids",
            "old_verdicts_frozen": True,
            "property_result": "NOT_RUN",
        })
    return {
        "schema": "kernel-analyzer-bias-property-roster-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_ROSTER_NO_PROPERTY_RESULT",
        "project_case_count": len(project_cases),
        "external_control_count": 1,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/property/bias_formation_v1")
    args = parser.parse_args()
    protocol_payload = protocol()
    roster_payload = roster()
    protocol_payload["protocol_sha256"] = hashlib.sha256(_canonical(protocol_payload)).hexdigest()
    roster_payload["roster_sha256"] = hashlib.sha256(_canonical(roster_payload)).hexdigest()
    _write(args.output_dir / "protocol.json", protocol_payload)
    _write(args.output_dir / "roster.json", roster_payload)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "protocol_sha256": protocol_payload["protocol_sha256"],
        "roster_sha256": roster_payload["roster_sha256"],
        "project_cases": roster_payload["project_case_count"],
        "external_controls": roster_payload["external_control_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
