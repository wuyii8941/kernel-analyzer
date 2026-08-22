#!/usr/bin/env python3
"""Build the three-factor Bias Formation Map from existing replay artifacts.

This is an analysis pass, not a relabeling pass.  It only promotes a factor
when the input artifact declares the corresponding matched intervention.  A
missing raw vector, response pair, or propagation trace is emitted as
UNRESOLVED rather than inferred from a T4/SEUP label.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.joint_bias import prefix_resultant  # noqa: E402


OUT = ROOT / "results/property/joint_bias_formation_v1"
EVIDENCE = ROOT / "results/property/bias_property_search/property_evidence.json"
SAVED_P = ROOT / "results/property/bias_property_search/saved_p_pairing_work_v2.json"
SILU = ROOT / "results/property/bias_property_search/vl_silu_optimizer_oddness_v2.json"
PHI_TRAJ = ROOT / "results/property/bias_formation/consequence/phi4_lm_head_dx_trajectory.json"
SEUP_SUMMARY = ROOT / "results/property/bias_formation_final/seup_consequence_summary.json"
ROSTER = ROOT / "results/property/bias_formation_v2_1/roster_bound.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def response_record(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        return {"status": "UNRESOLVED_MISSING_ARTIFACT", "artifact": str(path)}
    data = load(path)
    agg = data.get("aggregate", {})
    if not agg:
        return {"status": "UNRESOLVED_MALFORMED", "artifact": str(path)}
    return {
        "status": "MEASURED_ANTITHETIC_RESPONSE",
        "case_label": label,
        "artifact": str(path.relative_to(ROOT)),
        "state_count": len(data.get("records", [])),
        "all_forward_losses_equal": bool(agg.get("all_forward_losses_equal", False)),
        "optimizer_oddness_resultant_ratio": agg.get("optimizer_oddness_resultant_ratio"),
        "response_even_energy_fraction": agg.get("mean_step_response_even_energy_fraction"),
        "response_even_energy_on_sign_crossings": agg.get("mean_step_response_even_energy_on_sign_crossings"),
        "stateless_sgd_resultant_l2": agg.get("stateless_sgd_resultant_l2"),
        "adam_oddness_resultant_l2": agg.get("optimizer_oddness_resultant_l2"),
        "claim_boundary": data.get("claim_boundary"),
    }


def propagation_summary() -> dict[str, Any]:
    result: dict[str, Any] = {}
    if PHI_TRAJ.exists():
        data = load(PHI_TRAJ)
        result["phi4_lm_head_dx_seq64"] = {
            "status": "MEASURED_TRAJECTORY",
            "artifact": str(PHI_TRAJ.relative_to(ROOT)),
            "steps": len(data.get("steps", [])),
            "evaluation": data.get("evaluation", {}),
            "prefix": prefix_resultant(data.get("steps", []), "drift_l2"),
            "claim_boundary": data.get("claim_boundary"),
        }
    if SEUP_SUMMARY.exists():
        data = load(SEUP_SUMMARY)
        for case_id, value in data.get("cases", {}).items():
            row = dict(value) if isinstance(value, dict) else {"value": value}
            row["artifact"] = str(SEUP_SUMMARY.relative_to(ROOT))
            row.setdefault("status", "MEASURED_TRAJECTORY_SUMMARY")
            result.setdefault(case_id, {}).update(row)
    for path, case_id in ((SAVED_P, "qwen_saved_p_seq128"), (SILU, "qwen3vl_silu")):
        if path.exists():
            data = load(path)
            agg = data.get("aggregate", {})
            result[case_id] = {
                "status": "MEASURED_TRAJECTORY_FEEDBACK_SIGNATURE",
                "artifact": str(path.relative_to(ROOT)),
                "steps": len(data.get("records", [])),
                "natural_update_persistence": agg.get("natural_update_persistence"),
                "antithetic_update_persistence": agg.get("antithetic_update_persistence"),
                "natural_update_resultant_l2": agg.get("natural_update_resultant_l2"),
                "claim_boundary": data.get("claim_boundary"),
                "prefix_vector_trace": "UNAVAILABLE: records retain norms, not full vectors",
            }
    return result


def build() -> dict[str, Any]:
    evidence = load(EVIDENCE)
    cases: dict[str, Any] = {}
    for item in evidence.get("cases", []):
        case_id = str(item["case"])
        cases[case_id] = {
            "source": {
                "status": "MEASURED_CASE_INTERVENTION" if item.get("status") not in {"SUPPORTING_SOURCE_OBSERVATION_NOT_TRAJECTORY_MATCHED", "SUPPORTING_CROSS_ARCHITECTURE_SOURCE_OBSERVATION"} else "SUPPORTING_ONLY",
                "mechanism": item.get("mechanism"),
                "artifact": item.get("artifact"),
                "natural_effect": item.get("natural_effect", item.get("kernel_bootstrap_95")),
                "causal_reading": item.get("causal_reading"),
                "claim_boundary": item.get("claim_boundary"),
            },
            "response": {
                "status": "UNRESOLVED_MISSING_ANTITHETIC_REPLAY",
                "reason": "formation artifacts retain Gram/digests but not raw epsilon and +/- response vectors",
            },
            "propagation": {
                "status": "UNRESOLVED_NO_SEPARATE_VECTOR_PROPAGATOR",
            },
            "source_artifact_has_raw_vectors": False,
        }

    # The formal v2.1 roster is the denominator for this map.  Keep the
    # literature anchor out of the experimental denominator, and retain
    # roster entries with no replay artifact as explicit unresolved rows.
    roster_ids: list[str] = []
    if ROSTER.exists():
        roster = load(ROSTER)
        for item in roster.get("cases", []):
            case_id = str(item.get("case_id", ""))
            if not case_id or case_id == "flash_attention_literature_anchor":
                continue
            roster_ids.append(case_id)
            if case_id not in cases:
                cases[case_id] = {
                    "source": {"status": "UNRESOLVED_NO_FORMATION_ARTIFACT", "artifact": None},
                    "response": {"status": "UNRESOLVED_NO_REPLAY_ARTIFACT"},
                    "propagation": {"status": "UNRESOLVED_NO_PROPAGATION_ARTIFACT"},
                    "source_artifact_has_raw_vectors": False,
                }

    aliases = {
        "liger_fused_ce_t128": "liger_fused_ce",
        "phi4_lm_head_dx_seq64": "phi4_seq64_lmhead_dx",
        "qwen_vproj_seq128": "qwen128_vproj_mm",
        "qwen3vl_silu_seq160": "qwen3vl_silu",
    }
    for roster_id, evidence_id in aliases.items():
        if evidence_id in cases:
            evidence_row = cases.pop(evidence_id)
            cases[roster_id] = evidence_row

    # Existing, exact response replays are promoted only for these two cases.
    cases["qwen_saved_p_seq128"]["response"] = response_record(SAVED_P, "saved-P")
    silu_key = "qwen3vl_silu_seq160" if "qwen3vl_silu_seq160" in cases else "qwen3vl_silu"
    cases[silu_key]["response"] = response_record(SILU, "SiLU")
    # Existing intervention records are valid response/transport evidence, but
    # they do not provide the generic +/- replay requested by this protocol.
    phi_key = "phi4_lm_head_dx_seq64" if "phi4_lm_head_dx_seq64" in cases else "phi4_seq64_lmhead_dx"
    cases[phi_key]["response"] = {
        "status": "MEASURED_TRANSPORT_PAIRING_INTERVENTION",
        "artifact": "results/property/bias_formation/interventions/phi4_mm_transport_pairing.json",
        "generic_antithetic_response": "UNRESOLVED",
        "claim_boundary": "case-level composite transport intervention; no closed universal response law",
    }
    phi_sr = ROOT / "results/property/joint_bias_formation_v1/phi_sr_intervention.json"
    if phi_sr.exists():
        sr = load(phi_sr)
        cases[phi_key]["source"].update({
            "sr_intervention_status": sr.get("status"),
            "sr_artifact": str(phi_sr.relative_to(ROOT)),
            "natural_coherence_amplification": sr.get("natural_vs_fp32", {}).get("coherence_amplification"),
            "sr_coherence_amplification_mean": sr.get("sr_amplification_mean"),
            "sr_to_natural_amplification_ratio": sr.get("sr_to_natural_amplification_ratio"),
            "sr_claim_boundary": "real backward SR intervention; effective-update energy is not exactly norm-matched",
        })
    liger_key = "liger_fused_ce_t128" if "liger_fused_ce_t128" in cases else "liger_fused_ce"
    cases[liger_key]["response"] = {
        "status": "UNRESOLVED_MISSING_ANTITHETIC_REPLAY",
        "artifact": "results/property/bias_oracle_recovery/liger_joint_event.json (not present in repository)",
        "reason": "event screen is source-only and does not contain +/- F+B responses",
    }

    # Apply the same aliases to the already attached evidence records.
    # Add source-event feasibility to Liger without upgrading it to an
    # antithetic response result.
    liger_event = ROOT / "results/property/bias_oracle_recovery/liger_joint_event.json"
    if liger_key in cases and liger_event.exists():
        event = load(liger_event)
        cases[liger_key]["source"].update({
            "event_screen_status": event.get("status"),
            "event_artifact": str(liger_event.relative_to(ROOT)),
            "event_closure": event.get("closure"),
            "event_rounding_vs_contribution": event.get("event_rounding_vs_chunk_contribution"),
        })

    # Host-GPU rerun and same-semantics chunk intervention are recorded as
    # separate evidence.  The rerun validates the exact F+B execution path;
    # it does not override the frozen confirmation gate when its confirmation
    # population remains unresolved.
    liger_rerun = ROOT / "results/property/bias_formation/formation/liger_fused_ce_t128_host_rerun.json"
    if liger_key in cases and liger_rerun.exists():
        rerun = load(liger_rerun)
        cases[liger_key]["source"].update({
            "host_gpu_rerun_status": rerun.get("status"),
            "host_gpu_rerun_artifact": str(liger_rerun.relative_to(ROOT)),
            "host_gpu_confirmation_statuses": {
                layer: rerun.get("populations", {}).get("confirmation", {}).get(layer + "_status")
                for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")
            },
            "host_gpu_first_confirmed_bias_stage": rerun.get("first_confirmed_bias_stage"),
            "host_gpu_claim_boundary": "宿主机 RTX A6000 重跑；confirmation 区间未越过冻结门槛，不升级 formation verdict",
        })
    liger_chunk = ROOT / "results/property/joint_bias_formation_v1/liger_chunk_host_certificate.json"
    if liger_key in cases and liger_chunk.exists():
        chunk = load(liger_chunk)
        cases[liger_key]["source"].update({
            "chunk_geometry_intervention_status": chunk.get("verdict"),
            "chunk_geometry_intervention_artifact": str(liger_chunk.relative_to(ROOT)),
            "chunk_bf16_signs": next((row.get("positive"), row.get("negative"), row.get("tied"))
                                      for row in chunk.get("directional_results", [])
                                      if row.get("accumulator") == "bf16"),
            "chunk_fp32_signs": next((row.get("positive"), row.get("negative"), row.get("tied"))
                                      for row in chunk.get("directional_results", [])
                                      if row.get("accumulator") == "fp32"),
            "chunk_bf16_over_fp32_mean_rms": chunk.get("accumulator_amplification", {}).get("ratio_of_mean_rms"),
            "chunk_claim_boundary": chunk.get("claim_boundary"),
        })
    liger_sr = ROOT / "results/property/joint_bias_formation_v1/liger_sr_intervention.json"
    if liger_key in cases and liger_sr.exists():
        sr = load(liger_sr)
        cases[liger_key]["source"].update({
            "sr_intervention_status": sr.get("status"),
            "sr_intervention_artifact": str(liger_sr.relative_to(ROOT)),
            "sr_natural_coherence_amplification": sr.get("natural_rn", {}).get("coherence_amplification"),
            "sr_coherence_amplification": [row.get("coherence_amplification") for row in sr.get("sr", [])],
            "sr_to_natural_amplification": sr.get("sr_to_rn_amplification"),
            "sr_claim_boundary": sr.get("claim_boundary"),
            "sr_interpretation": "RN residual is already diffusive on this confirmation split; SR is a completed negative/diagnostic intervention, not evidence of removing a confirmed persistent source",
        })

    propagation = propagation_summary()
    propagation_aliases = {
        "phi4_lm_head_dx_seq64": "phi4_lm_head_dx_seq64",
        "phi4_seq64_lmhead_dx": "phi4_lm_head_dx_seq64",
        "qwen3vl_silu_seq160": "qwen3vl_silu",
        "qwen_layer23_attention_state": "layer23_qproj_attention_state_region",
    }
    for case_id, row in cases.items():
        source_id = propagation_aliases.get(case_id, case_id)
        if source_id in propagation:
            row["propagation"] = propagation[source_id]

    return {
        "schema": "kernel-analyzer-joint-bias-formation-map-v1",
        "protocol": {
            "property": "implementation error distribution + conditional F+B/optimizer response + closed-loop propagation",
            "formation_definition": "mu_t = E_pi[R_s(epsilon_pi,t)]",
            "response_decomposition": "R_e=(R(+epsilon)+R(-epsilon))/2; R_o=(R(+epsilon)-R(-epsilon))/2",
            "propagation_definition": "D_H_hat = sum_t Phi_hat[H,t] mu_hat_t",
            "formation_and_persistence_separate": True,
            "missing_raw_vectors_fail_closed": True,
        },
        "case_count": len(cases),
        "formal_roster_case_count": len(roster_ids),
        "cases": cases,
        "propagation_records": propagation,
        "claim_boundary": "This is an auditable factor map over existing artifacts, not yet a universal low-cost predictor.",
    }


def render_md(data: dict[str, Any]) -> str:
    lines = [
        "# Joint Bias Formation Map v1",
        "",
        "This report separates source distribution, F+B/optimizer response, and closed-loop propagation.",
        "Missing replay vectors are unresolved; no formation label is inferred from SEUP or trajectory drift.",
        "",
        "| Case | Source | Response | Propagation |",
        "|---|---|---|---|",
    ]
    for case_id, row in data["cases"].items():
        lines.append("| `{}` | `{}` | `{}` | `{}` |".format(
            case_id,
            row["source"].get("status", "UNRESOLVED"),
            row["response"].get("status", "UNRESOLVED"),
            row["propagation"].get("status", "UNRESOLVED"),
        ))
    lines += [
        "",
        "## Current interpretation",
        "",
        "The existing repository has exact antithetic response replays for saved-P and SiLU, a composite transport intervention for Phi, and source-event/chunk-geometry evidence for Liger. A host-GPU Liger 32-state rerun completed, but its confirmation population remained unresolved under the frozen gate. The 24-state chunk intervention confirmed BF16-specific directional geometry (24/24 versus FP32 13/11), while the separate RN→SR default residual screen was diffusive (A=0.942 versus SR=0.975/1.001) and is retained as a negative/diagnostic result.",
        "",
        "The next causal measurements are therefore: real Liger SR/order-breaking, a generic response replay for Liger/Phi, and fixed-update propagation probes. No missing vector is imputed.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    payload = build()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "joint_bias_formation_map.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "joint_bias_formation_map.md").write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "case_count": payload["case_count"]}, sort_keys=True))
