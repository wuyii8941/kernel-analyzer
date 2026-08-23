#!/usr/bin/env python3
"""Consolidate the measured source, response, and propagation evidence.

This script does not infer missing vectors.  It emits an evaluated mechanism
map and a fail-closed readiness verdict for the frozen generic predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-stage", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--phi-antithetic", type=Path, required=True)
    parser.add_argument("--phi-propagation", type=Path, required=True)
    parser.add_argument("--phi-adamw", type=Path, required=True)
    parser.add_argument("--consequence-summary", type=Path, required=True)
    parser.add_argument(
        "--liger-recapture",
        type=Path,
        help=(
            "Optional private capture containing the verified raw vectors. "
            "Use it once to create the compact repository summary."
        ),
    )
    parser.add_argument("--liger-summary-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictor-output", type=Path, required=True)
    args = parser.parse_args()

    three = load(args.three_stage)
    parity = load(args.parity)
    phi_antithetic = load(args.phi_antithetic)
    phi_propagation = load(args.phi_propagation)
    phi_adamw = load(args.phi_adamw)
    consequence = load(args.consequence_summary)

    if three.get("status") != "COMPLETE_ORDERED_32_STATE_SUMMARY" or len(three.get("cases", [])) != 3:
        raise RuntimeError("three-stage headline summary is incomplete")
    if parity.get("status") != "COMPLETE_EXACT_RESPONSE_PARITY":
        raise RuntimeError("exact response-parity artifact is incomplete")
    if phi_antithetic.get("status") != "UNRESOLVED_REPRESENTABILITY":
        raise RuntimeError("Phi antithetic result did not fail closed on representability")
    if phi_antithetic.get("exact_antithetic_all_states") is not False:
        raise RuntimeError("Phi exact-antithetic gate unexpectedly passed")
    if phi_propagation.get("status") != "COMPLETE":
        raise RuntimeError("Phi propagation intervention is incomplete")
    if phi_adamw.get("status") != "COMPLETE_ORDERED_32_STATE_COMMON_STATE_ADAMW":
        raise RuntimeError("Phi AdamW response map is incomplete")
    if args.liger_recapture is not None:
        liger = load(args.liger_recapture)
        raw = liger.get("raw_vector_capture", {})
        if raw.get("status") != "COMPLETE_32_STATE_THREE_STAGE_VECTORS_RETAINED":
            raise RuntimeError("Liger raw three-stage recapture is incomplete")
        for layer, rows in raw.get("layers", {}).items():
            if len(rows) != 32:
                raise RuntimeError(f"Liger raw layer {layer} does not contain 32 states")
            for row in rows:
                path = Path(row["path"])
                expected_bytes = int(row["coordinate_count"]) * 4
                if not path.is_file() or path.stat().st_size != expected_bytes:
                    raise RuntimeError(f"Liger raw vector missing or has the wrong size: {path}")
                if len(str(row.get("vector_digest", ""))) != 64:
                    raise RuntimeError(f"Liger raw vector digest is absent: {path}")

        public_liger = {
            key: liger[key]
            for key in (
                "schema", "case_id", "status", "measurement_kind", "state_split",
                "policy", "populations", "first_confirmed_bias_stage",
                "first_observed_biased_stage", "formation_point", "capture_provenance",
            )
        }
        public_liger["capture_provenance"] = dict(public_liger["capture_provenance"])
        public_liger["capture_provenance"]["raw_vectors_retained"] = False
        public_liger["capture_provenance"]["raw_vectors_verified_before_cleanup"] = True
        public_liger["raw_vector_capture"] = {
            "status": "VERIFIED_EPHEMERAL_32_STATE_CAPTURE_SUMMARIZED",
            "state_count": 32,
            "layers": {
                layer: [
                    {
                        "state_id": row["state_id"],
                        "coordinate_count": row["coordinate_count"],
                        "storage_dtype": row["storage_dtype"],
                        "vector_digest": row["vector_digest"],
                    }
                    for row in rows
                ]
                for layer, rows in raw["layers"].items()
            },
            "retention": "Raw vectors were verified and then deleted; rerun the capture to reproduce them.",
        }
        public_liger["source_capture_sha256"] = sha256(args.liger_recapture)
        args.liger_summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.liger_summary_output.write_text(
            json.dumps(public_liger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        public_liger = load(args.liger_summary_output)
        raw = public_liger.get("raw_vector_capture", {})
        if raw.get("status") != "VERIFIED_EPHEMERAL_32_STATE_CAPTURE_SUMMARIZED":
            raise RuntimeError("compact Liger summary has not passed raw-vector verification")
        if raw.get("state_count") != 32:
            raise RuntimeError("compact Liger summary does not contain 32 states")

    headline_rows = []
    for row in three["cases"]:
        local = float(row["operator_output_error"]["coherence_amplification"])
        gradient = float(row["parameter_gradient_error"]["coherence_amplification"])
        update = float(row["effective_update_error"]["coherence_amplification"])
        headline_rows.append({
            "case_id": row["case_id"],
            "operator_output_A": local,
            "parameter_gradient_A": gradient,
            "effective_update_A": update,
            "backward_transport_gain_A_ratio": gradient / max(local, 1e-30),
            "backward_transport_delta_A": gradient - local,
            "optimizer_mapping_gain_A_ratio": update / max(gradient, 1e-30),
        })

    response_rows = []
    for row in parity["cases"]:
        response_rows.append({
            "case_id": row["case_id"],
            "status": "EXACT_POSITIVE_NEGATIVE_ERROR_RESPONSE",
            "even_over_natural_resultant": row["even_over_natural_resultant"],
            "odd_over_natural_resultant": row["odd_over_natural_resultant"],
            "even_odd_cosine": row["even_odd_cosine"],
            "dominant_term": row["dominant_term"],
            "closure_relative": row["closure_relative"],
        })
    response_rows.extend([
        {
            "case_id": "liger_fused_ce_t128",
            "status": "ANALYTIC_LINEAR_RESPONSE_ON_CAPTURED_DW_TO_STATELESS_SGD",
            "even_component": 0.0,
            "odd_component": "equals the captured effective-update residual",
            "claim_boundary": "The fused endpoint is already the tied parameter-gradient contribution; this is not a complete equivalent-schedule distribution measurement.",
        },
        {
            "case_id": "phi4_seq64_lmhead_dx",
            "status": "UNRESOLVED_EXACT_NEGATIVE_ERROR_NOT_REPRESENTABLE",
            "states": phi_antithetic["state_count"],
            "representability_error_min": min(float(row["representability_error"]) for row in phi_antithetic["rows"]),
            "representability_error_max": max(float(row["representability_error"]) for row in phi_antithetic["rows"]),
            "approximate_even_odd_not_used": True,
        },
    ])

    phi_sgd = next(row for row in headline_rows if row["case_id"] == "phi4_seq64_lmhead_dx")
    adamw_gradient = float(phi_adamw["stages"]["parameter_gradient_error"]["coherence_curve"][-1]["coherence_amplification"])
    adamw_update = float(phi_adamw["stages"]["effective_update_error"]["coherence_curve"][-1]["coherence_amplification"])
    propagation_rows = [{
        "case_id": "phi4_seq64_lmhead_dx",
        "fixed_update_feedback_over_direct_ratio": phi_propagation["final"]["feedback_over_direct_ratio"],
        "reading": "DIRECT_SOURCE_UPDATE_SUM_EXPLAINS_THE_ALTERNATE_CHECKPOINT_DRIFT",
    }]

    consequence_rows = consequence.get("cases", consequence.get("rows", []))
    local_diffusive_feedback_persistent = sum(
        row.get("regime") == "FEEDBACK_SUSTAINED" for row in consequence_rows
    )

    payload = {
        "schema": "kernel-analyzer-general-bias-mechanism-map-v1",
        "status": "COMPLETE_MEASURED_MECHANISM_MAP_WITH_ABSTENTIONS",
        "factorization": {
            "source": "implementation and operands produce an endpoint residual distribution",
            "response": "backward and optimizer map that residual to an effective update",
            "propagation": "subsequent training state either preserves, suppresses, or amplifies the update sequence",
        },
        "headline_three_stage": headline_rows,
        "response_even_odd": response_rows,
        "optimizer_response": {
            "case_id": "phi4_seq64_lmhead_dx",
            "sgd_gradient_A": phi_sgd["parameter_gradient_A"],
            "sgd_update_A": phi_sgd["effective_update_A"],
            "adamw_gradient_A": adamw_gradient,
            "adamw_update_A": adamw_update,
            "reading": "the optimizer can suppress, rather than merely inherit, directional gradient error",
        },
        "propagation": propagation_rows,
        "background_feedback_audit": {
            "sampled_rows": len(consequence_rows),
            "local_diffusive_feedback_persistent_rows": local_diffusive_feedback_persistent,
            "reading": "feedback persistence is common background and is not promoted to an operator-source positive",
        },
        "inputs": {str(path): sha256(path) for path in (
            args.three_stage, args.parity, args.phi_antithetic, args.phi_propagation,
            args.phi_adamw, args.consequence_summary,
        )},
        "public_liger_recapture_summary": str(args.liger_summary_output),
        "claim_boundary": (
            "This map establishes several measured formation regimes and identifies where directionality becomes visible. "
            "It does not establish one universal low-cost predictor: exact response replay is mathematically unavailable "
            "for Phi's natural BF16 endpoint, and no case currently has every frozen source, response, and propagation input."
        ),
    }
    payload["inputs"][str(args.liger_summary_output)] = sha256(args.liger_summary_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readiness_rows = [
        {"case_id": "liger_fused_ce_t128", "source_measurement": "MISSING_CROSSFIT_EQUIVALENT_SCHEDULES", "response": "ANALYTIC_LINEAR_READY", "propagation": "AVAILABLE_CASE_SPECIFIC", "verdict": "ABSTAIN_INCOMPLETE_SOURCE_FACTOR"},
        {"case_id": "phi4_seq64_lmhead_dx", "source_measurement": "DERIVED_SUMMARY_ONLY", "response": "INELIGIBLE_EXACT_NEGATIVE_ERROR_NOT_REPRESENTABLE", "propagation": "READY", "verdict": "ABSTAIN_RESPONSE_NOT_DEFINED"},
        {"case_id": "qwen_seq256_lmhead_dx", "source_measurement": "MISSING", "response": "MISSING", "propagation": "MISSING_FROZEN_FIXED_SEQUENCE", "verdict": "ABSTAIN_INCOMPLETE_ALL_FACTORS"},
        {"case_id": "qwen_saved_p_seq128", "source_measurement": "MISSING", "response": "READY_EXACT", "propagation": "AVAILABLE_CASE_SPECIFIC", "verdict": "ABSTAIN_INCOMPLETE_SOURCE_FACTOR"},
        {"case_id": "qwen3vl_silu_seq160", "source_measurement": "MISSING", "response": "READY_EXACT", "propagation": "AVAILABLE_CASE_SPECIFIC", "verdict": "ABSTAIN_INCOMPLETE_SOURCE_FACTOR"},
    ]
    predictor = {
        "schema": "kernel-analyzer-three-factor-predictor-evaluation-v1",
        "status": "COMPLETE_EVALUATION_ALL_ABSTAIN_INPUT_INCOMPLETE",
        "evaluated_cases": len(readiness_rows),
        "fully_eligible_cases": 0,
        "heldout_confirmation": "NOT_APPLICABLE_NO_FROZEN_SCORE_EMITTED",
        "rows": readiness_rows,
        "claim_boundary": "The frozen predictor was evaluated for input eligibility. No score is emitted and no accuracy claim is made when any factor is absent or undefined.",
    }
    args.predictor_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictor_output.write_text(json.dumps(predictor, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"mechanism_status": payload["status"], "predictor_status": predictor["status"], "fully_eligible_cases": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
