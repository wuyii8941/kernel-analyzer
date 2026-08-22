#!/usr/bin/env python3
"""Build a fail-closed offline summary of the three-factor artifacts.

This script deliberately does not reconstruct missing epsilon vectors.  It
only aggregates case-specific antithetic summaries that were actually saved;
generic even/odd decomposition remains unresolved when raw +/- response
vectors are absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "results/property/joint_bias_formation_v1/joint_bias_formation_map.json"
SAVED_P = ROOT / "results/property/bias_property_search/saved_p_pairing_work_v2.json"
SAVED_P_RESPONSE = ROOT / "results/property/joint_bias_formation_v1/qwen_saved_p_pairing_response_vectors.json"
SILU = ROOT / "results/property/bias_property_search/vl_silu_optimizer_oddness_v2.json"
PHI = ROOT / "results/property/bias_formation/interventions/phi4_mm_transport_pairing.json"
PHI_FIXED = ROOT / "results/property/joint_bias_formation_v1/phi_fixed_update_propagation.json"
PHI_ANTITHETIC = ROOT / "results/property/joint_bias_formation_v1/phi_antithetic_response_capture.json"
OUTPUT = ROOT / "results/property/joint_bias_formation_v1/offline_factor_summary.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_antithetic(path: Path, case_id: str) -> dict[str, Any]:
    data = load(path)
    aggregate = data["aggregate"]
    records = data.get("records", [])
    return {
        "case_id": case_id,
        "artifact": str(path.relative_to(ROOT)),
        "generic_even_odd": "UNRESOLVED_MISSING_RAW_VECTORS",
        "case_specific_response": {
            "forward_losses_equal": aggregate.get("all_forward_losses_equal"),
            "response_even_resultant_l2": aggregate.get("optimizer_oddness_resultant_l2"),
            "response_even_resultant_ratio": aggregate.get("optimizer_oddness_resultant_ratio"),
            "mean_step_response_even_energy_fraction": aggregate.get("mean_step_response_even_energy_fraction"),
            "mean_step_response_even_energy_on_sign_crossings": aggregate.get("mean_step_response_even_energy_on_sign_crossings"),
            "mean_step_response_oddness_ratio": aggregate.get("mean_step_optimizer_oddness_ratio"),
            "stateless_sgd_resultant_l2": aggregate.get("stateless_sgd_natural_resultant_l2", aggregate.get("stateless_sgd_resultant_l2")),
            "steps": len(records),
        },
        "raw_vectors_retained": False,
        "prefix_to_32_backtest": "UNAVAILABLE_FROM_SAVED_SUMMARY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    map_data = load(MAP)
    cases: list[dict[str, Any]] = []
    saved_p_case = summarize_antithetic(SAVED_P, "qwen_saved_p_seq128")
    if SAVED_P_RESPONSE.exists():
        response = load(SAVED_P_RESPONSE)
        saved_p_case["trajectory_response_capture"] = {
            "artifact": str(SAVED_P_RESPONSE.relative_to(ROOT)),
            "status": response.get("response_vector_capture", {}).get("status"),
            "state_count": response.get("response_vector_capture", {}).get("state_count"),
            "response_even_status": response.get("response_even_population", {}).get("status"),
            "response_even_cross_state_ratio": response.get("response_even_population", {}).get("cross_state_ratio"),
            "response_odd_status": response.get("response_odd_population", {}).get("status"),
            "response_odd_cross_state_ratio": response.get("response_odd_population", {}).get("cross_state_ratio"),
            "claim_boundary": "Trajectory-conditioned response decomposition; not a common-state formation label.",
        }
    cases.append(saved_p_case)
    cases.append(summarize_antithetic(SILU, "qwen3vl_silu_seq160"))

    phi = load(PHI)
    phi_fixed = load(PHI_FIXED)
    phi_antithetic = load(PHI_ANTITHETIC) if PHI_ANTITHETIC.exists() else None
    cases.append({
        "case_id": "phi4_lm_head_dx_seq64",
        "artifact": str(PHI.relative_to(ROOT)),
        "generic_even_odd": "UNRESOLVED_MISSING_RAW_ANTITHETIC_VECTORS",
        "case_specific_response": {
            "natural_gradient_status": phi["natural_gradient_population"].get("status"),
            "shuffled_gradient_status": phi["shuffled_gradient_population"].get("status"),
            "natural_cross_state_ratio": phi["natural_gradient_population"].get("cross_state_ratio"),
            "shuffled_cross_state_ratio": phi["shuffled_gradient_population"].get("cross_state_ratio"),
            "transport_pairing_status": phi.get("status"),
        },
        "propagation": {
            "artifact": str(PHI_FIXED.relative_to(ROOT)),
            "direct_sum_l2": phi_fixed["final"]["fixed_sum_l2"],
            "drift_l2": phi_fixed["final"]["alternate_feedback_drift_l2"],
            "drift_over_direct": phi_fixed["final"]["feedback_over_direct_ratio"],
            "interpretation": "POSITIVE_PROPAGATION_CLOSURE",
        },
        "raw_vectors_retained": False,
        "prefix_to_32_backtest": "UNAVAILABLE_FROM_SAVED_SUMMARY",
    })
    if phi_antithetic is not None:
        cases[-1]["antithetic_response_capture"] = {
            "artifact": str(PHI_ANTITHETIC.relative_to(ROOT)),
            "status": phi_antithetic.get("status"),
            "state_count": phi_antithetic.get("state_count"),
            "exact_antithetic_all_states": phi_antithetic.get("exact_antithetic_all_states"),
            "representability_floor": phi_antithetic.get("representability_floor"),
            "representability_error_min": min(
                row["representability_error"] for row in phi_antithetic.get("rows", [])
            ) if phi_antithetic.get("rows") else None,
            "representability_error_max": max(
                row["representability_error"] for row in phi_antithetic.get("rows", [])
            ) if phi_antithetic.get("rows") else None,
            "response_even_status": phi_antithetic.get("response_even_population", {}).get("status"),
            "response_even_cross_state_ratio": phi_antithetic.get("response_even_population", {}).get("cross_state_ratio"),
            "response_odd_status": phi_antithetic.get("response_odd_population", {}).get("status"),
            "response_odd_cross_state_ratio": phi_antithetic.get("response_odd_population", {}).get("cross_state_ratio"),
            "claim_boundary": "Representability unresolved; +/- response is an approximate BF16 reflection, not an exact causal intervention.",
        }
        cases[-1]["case_specific_response"] = {
            **cases[-1]["case_specific_response"],
            "antithetic_capture_status": phi_antithetic.get("status"),
        }

    for case_id in ("liger_fused_ce_t128", "mamba_seq64_input_proj", "qwen_vproj_seq128",
                    "qwen_layer23_attention_state", "qwen_bmm_seq64", "qwen_rsqrt_seq128",
                    "qwen_l23_key_materialization_seq1024"):
        row = map_data["cases"].get(case_id, {})
        cases.append({
            "case_id": case_id,
            "generic_even_odd": "UNRESOLVED_MISSING_RAW_VECTORS",
            "case_specific_source_status": row.get("source", {}).get("status"),
            "case_specific_response_status": row.get("response", {}).get("status"),
            "case_specific_propagation_status": row.get("propagation", {}).get("status"),
            "raw_vectors_retained": bool(row.get("source_artifact_has_raw_vectors", False)),
            "prefix_to_32_backtest": "UNAVAILABLE",
        })

    payload = {
        "schema": "kernel-analyzer-three-factor-offline-summary-v1",
        "status": "PARTIAL_BOUNDED",
        "generic_even_odd_formula": {
            "response_even": "(F(+epsilon)+F(-epsilon))/2",
            "response_odd": "(F(+epsilon)-F(-epsilon))/2",
            "event_even": "(w_plus+w_minus)/2",
            "event_odd": "(w_plus-w_minus)/2",
            "closure": "mu = sum(event_even*response_even) + sum(event_odd*response_odd)",
        },
        "case_count": len(cases),
        "generic_ready_count": 0,
        "case_specific_summary_count": 4 if phi_antithetic is not None else 3,
        "cases": cases,
        "claim_boundary": "Case-specific response summaries are reported where saved. Missing raw epsilon and +/- response vectors are unresolved; no generic predictor or zero vector is imputed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": payload["status"], "generic_ready_count": 0, "case_specific_summary_count": payload["case_specific_summary_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
