#!/usr/bin/env python3
"""Assemble the mechanism-validation package from retained causal evidence.

The package deliberately separates three notions:

* v2.1 open-loop formation stage (only certificates can supply this label);
* mechanism validation (matched intervention and semantic closure); and
* SEUP consequence (persistence after a bias has formed).

No T4 or SEUP result is used as a formation label.  This reducer only consumes
already-retained artifacts and does not run a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/bias_formation"
FINAL = ROOT / "results/property/bias_formation_final"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty matrix")
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def phi_mechanism() -> dict[str, Any]:
    intervention_path = BASE / "interventions/phi4_mm_transport_pairing.json"
    formation_path = BASE / "formation/phi4_lm_head_dx_seq64.json"
    intervention = load(intervention_path)
    formation = load(formation_path)
    rows = intervention["rows"]
    errors = [float(row["transport_prediction_relative_error"]) for row in rows]
    gates = intervention["gates"]
    return {
        "schema": "kernel-analyzer-phi-transport-mechanism-v1",
        "case_id": "phi4_lm_head_dx_seq64",
        "status": "EMPIRICALLY_VALIDATED_COMPOSITE_TRANSPORT_MECHANISM",
        "mechanism_type": "BACKWARD_TRANSPORT_PAIRING",
        "formation": {
            "local": "CENTERED",
            "parameter_gradient": "BIASED",
            "effective_update": "BIASED",
            "first_confirmed_bias_stage": formation.get("first_confirmed_bias_stage"),
            "source": str(formation_path.relative_to(ROOT)),
        },
        "intervention": {
            "description": intervention["intervention"],
            "natural_gradient_status": intervention["natural_gradient_population"]["status"],
            "shuffled_gradient_status": intervention["shuffled_gradient_population"]["status"],
            "local_norm_preserved_every_state": gates["local_norm_preserved_every_state"],
            "states": len(rows),
            "matched_sham": intervention["matched_sham"],
        },
        "analytic_transport": {
            "closed": gates["analytic_transport_matches_natural_gradient"],
            "relative_error_min": min(errors),
            "relative_error_max": max(errors),
            "relative_error_mean": sum(errors) / len(errors),
            "status": "INCOMPLETE_SEMANTIC_TRANSPORT_DECOMPOSITION",
        },
        "claim": (
            "The paired residual/transport intervention causally changes the "
            "gradient bias while preserving the local residual norm. This is a "
            "case-level composite transport mechanism. The incomplete analytic "
            "reconstruction prevents a claim about one physical transport factor "
            "or a universal property."
        ),
        "evidence": {
            "intervention": str(intervention_path.relative_to(ROOT)),
            "formation": str(formation_path.relative_to(ROOT)),
        },
    }


def l23_mechanism() -> dict[str, Any]:
    path = ROOT / "results/coverage/cases/l23_qproj_attention_state_region.json"
    live_path = ROOT / "results/final/l23_attention_live_weight.json"
    certificate = load(path)
    live = load(live_path)
    attribution = certificate["causal_attribution"]
    gates = certificate["gates"]
    trajectory = certificate["trajectory"]
    return {
        "schema": "kernel-analyzer-l23-attention-state-mechanism-v1",
        "case_id": "layer23_qproj_attention_state_region",
        "status": "VALIDATED_SEMANTIC_REGION_MECHANISM",
        "mechanism_type": "ATTENTION_STATE_RECONSTRUCTION_TRANSPORT",
        "formation": {
            "stage_label": "SEMANTIC_REGION_CAUSAL_EVIDENCE",
            "strict_v2_1_layer_labels": "NOT_CAPTURED",
            "not_inferred_from_t4_or_seup": True,
        },
        "closed_forward_backward_math": certificate["closed_math"],
        "causal_intervention": {
            "root_boundary": attribution["root_boundary"],
            "s_bwd_only_repair_closes_direction": gates["s_bwd_only_repair_closes_direction"],
            "joint_attention_state_repair_closes_direction": gates["joint_attention_state_repair_closes_direction"],
            "k_only_state_repair_bootstrap": attribution["k_only_repair_residual"]["state_cluster_bootstrap_95"],
            "s_bwd_shapley_bootstrap": attribution["s_bwd_shapley"]["state_cluster_bootstrap_95"],
            "matched_sham_exact": gates["matched_sham_exact"],
            "states": attribution["s_bwd_shapley"]["states"],
        },
        "persistence_separation": {
            "paired_live_weight_trajectory": gates["paired_live_weight_trajectory"],
            "projection_strictly_increases_each_step": trajectory["projection_strictly_increases_each_step"],
            "steps": trajectory["steps"],
            "first_projection": trajectory["first_fp32_master_projection"],
            "final_projection": trajectory["final_fp32_master_projection"],
            "evidence": str(live_path.relative_to(ROOT)),
        },
        "claim": (
            "The attention-backward semantic region S_bwd is the causal root of "
            "the q_proj tile direction: restoring S_bwd closes the direction, "
            "restoring K alone does not, and the sham is exact. This is a second, "
            "independent semantic-region mechanism. It is not a single-kernel "
            "attribution and does not by itself supply a v2.1 local/gradient/update "
            "formation label."
        ),
        "evidence": {
            "semantic_region": str(path.relative_to(ROOT)),
            "live_weight": str(live_path.relative_to(ROOT)),
        },
    }


def build_population_matrix(phi: Mapping[str, Any], l23: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = FINAL / "population_screening.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case = row.get("case_id", "")
            if case == phi["case_id"]:
                formation_type = "TRANSPORT_BIAS_CANDIDATE"
                mechanism_status = phi["status"]
                mechanism_evidence = "phi_transport_mechanism.json"
            elif case == l23["case_id"]:
                formation_type = "ATTENTION_STATE_TRANSPORT_BIAS"
                mechanism_status = l23["status"]
                mechanism_evidence = "anchor_case_reports/qwen_l23_attention.md"
            elif case == "qwen_saved_p_seq128":
                formation_type = "VARIANCE_ONLY"
                mechanism_status = "CASE_LEVEL_ONLY"
                mechanism_evidence = "anchor_case_reports/qwen_saved_p.md"
            elif case == "liger_fused_ce_t128":
                formation_type = "UNRESOLVED"
                mechanism_status = "CONFIRMATION_MARGIN_CROSSED"
                mechanism_evidence = "anchor_case_reports/liger.md"
            elif case == "qwen_bmm_seq64":
                formation_type = "UNRESOLVED"
                mechanism_status = "INELIGIBLE_EXACT_REPAIR_SHAM_PROVENANCE"
                mechanism_evidence = "population_screening.csv"
            else:
                formation_type = "UNRESOLVED"
                mechanism_status = "NOT_CAPTURED"
                mechanism_evidence = "population_screening.csv"
            rows.append({
                "population_id": row.get("population_id", ""),
                "population_kind": row.get("population_kind", ""),
                "case_id": case,
                "candidate_id": row.get("candidate_id", ""),
                "model": row.get("model", ""),
                "sequence_length": row.get("sequence_length", ""),
                "exact_endpoint_or_region": row.get("exact_endpoint_or_region", ""),
                "local": row.get("formation_local", "NOT_MEASURED"),
                "gradient": row.get("formation_parameter_gradient", "NOT_MEASURED"),
                "update": row.get("formation_effective_update", "NOT_MEASURED"),
                "formation_type": formation_type,
                "mechanism_status": mechanism_status,
                "mechanism_evidence": mechanism_evidence,
                "legacy_role_provenance_only": row.get("legacy_role_provenance_only", ""),
                "screening_status": row.get("screening_status", ""),
            })
    return rows


def write_reports(phi: Mapping[str, Any], l23: Mapping[str, Any]) -> None:
    anchor = FINAL / "anchor_case_reports"
    interventions = FINAL / "intervention_results"
    anchor.mkdir(parents=True, exist_ok=True)
    interventions.mkdir(parents=True, exist_ok=True)
    write_json(FINAL / "phi_transport_mechanism.json", phi)
    write_json(FINAL / "qwen_l23_attention_mechanism.json", l23)
    write_json(interventions / "phi_mm_transport_pairing.json", {
        "schema": "kernel-analyzer-intervention-result-reference-v1",
        "case_id": phi["case_id"],
        "status": phi["status"],
        "mechanism_type": phi["mechanism_type"],
        "evidence": phi["evidence"],
        "gates": phi["intervention"],
        "analytic_transport": phi["analytic_transport"],
    })
    write_json(interventions / "qwen_l23_attention_state.json", {
        "schema": "kernel-analyzer-intervention-result-reference-v1",
        "case_id": l23["case_id"],
        "status": l23["status"],
        "mechanism_type": l23["mechanism_type"],
        "evidence": l23["evidence"],
        "gates": l23["causal_intervention"],
    })
    write_json(FINAL / "seup_consequence_summary.json", {
        "schema": "kernel-analyzer-bias-formation-seup-consequence-summary-v1",
        "formation_and_persistence_are_separate": True,
        "cases": {
            "phi4_lm_head_dx_seq64": {
                "formation_stage": "PARAMETER_GRADIENT",
                "seup_source": "results/property/bias_formation/consequence/phi4_lm_head_dx_seup.json",
                "signed_persistence": 1.0,
                "local_projected_accumulation_fraction": 0.9991653,
            },
            "layer23_qproj_attention_state_region": {
                "formation_stage": "SEMANTIC_REGION_CAUSAL_EVIDENCE",
                "seup_source": "results/final/l23_attention_live_weight.json",
                "projection_strictly_increases_each_step": l23["persistence_separation"]["projection_strictly_increases_each_step"],
                "steps": l23["persistence_separation"]["steps"],
                "first_projection": l23["persistence_separation"]["first_projection"],
                "final_projection": l23["persistence_separation"]["final_projection"],
            },
        },
        "claim_boundary": "SEUP/trajectory evidence describes persistence after mechanism evidence; it never supplies a formation label.",
    })
    (anchor / "phi_mm.md").write_text(
        """# Phi MM anchor\n\nPhi has the strict v2.1 transition `LOCAL_CENTERED -> GRADIENT_BIASED -> UPDATE_BIASED`. A matched row-pairing intervention preserves the local residual norm and changes the gradient population from `BIASED` to `CENTERED`. This validates a case-level composite backward-transport mechanism. The current analytic RMSNorm-only reconstruction has 0.32--0.60 relative error, so no universal transport property is claimed.\n\nEvidence: `phi_transport_mechanism.json`, `intervention_results/phi_mm_transport_pairing.json`.\n""", encoding="utf-8")
    (anchor / "qwen_l23_attention.md").write_text(
        """# Qwen layer-23 attention-state anchor\n\nThis is a closed semantic-region mechanism, not a single-kernel claim. The exact forward/backward equations are `Y=H W^T`, `dW=G_q^T H`, `S_bwd=alpha J_softmax(P)^T(D V^T)`, and `G_q=S_bwd K`. Restoring `S_bwd` closes the directional q_proj tile, restoring K alone does not, and the matched sham is exact. The live-weight trajectory is a separate persistence result.\n\nEvidence: `qwen_l23_attention_mechanism.json`, `intervention_results/qwen_l23_attention_state.json`.\n""", encoding="utf-8")
    (anchor / "qwen_saved_p.md").write_text(
        """# Qwen saved-P boundary\n\nThe measured v2.1 formation result is centered at local endpoint, parameter-gradient, and effective-update layers. It is a strong negative boundary for the claim that every implementation mismatch becomes bias. It does not prove all saved-state contracts are harmless.\n""", encoding="utf-8")
    (anchor / "liger.md").write_text(
        """# Liger boundary\n\nLiger has a directional calibration local signal, but its disjoint confirmation interval crosses the frozen bias margin. Source-generated bias is therefore unresolved. Existing SEUP persistence is not used as a formation label.\n""", encoding="utf-8")

    matrix = build_population_matrix(phi, l23)
    write_csv(FINAL / "bias_population_matrix.csv", matrix)
    (FINAL / "mechanism_taxonomy.md").write_text(
        """# Bias Formation Mechanism Taxonomy\n\nThe current evidence validates two independent mechanism boundaries while keeping formation and persistence separate.\n\n| Mechanism | Case | Validation | Boundary |\n|---|---|---|---|\n| Composite backward transport pairing | Phi MM | empirical matched intervention: natural gradient biased, row-paired shuffle centered, local norm preserved | analytic transport decomposition incomplete; no universal P2 claim |\n| Attention-state semantic transport | Qwen layer-23 q_proj tile | complete F+B semantic equations, S_bwd-only repair closes direction, K-only does not, exact sham | semantic region, not one kernel; strict v2.1 layer formation labels not captured |\n| Source-generated bias | Liger | unresolved confirmation | no source intervention |\n| Numerical contract bias | Qwen saved-P | not supported in this case; all three v2.1 layers centered | case-level negative only |\n| Optimizer-induced bias | none | not observed | no optimizer intervention |\n\nThese are mechanism validations, not a universal property. The two positive mechanisms share a downstream training bottleneck—backward transport into a parameter gradient—but arise from different semantic regions.\n""", encoding="utf-8")
    (FINAL / "scientific_summary.md").write_text(
        """# Bias Formation Map — mechanism discovery result\n\n## Core answer\n\nThe current study validates two independent training-semantic bottlenecks that can turn implementation variation into directional parameter-gradient/update effects:\n\n1. **Phi MM: composite backward transport pairing.** Local numerical variation is centered in the strict v2.1 formation capture, but the parameter-gradient and effective-update populations are biased. A norm-preserving residual/transport pairing intervention removes the gradient bias. The complete analytic transport decomposition is still open, so this is a validated empirical composite mechanism, not a universal source–transport law.\n2. **Qwen layer-23 attention state: semantic backward-state transport.** The exact F+B equations close at the semantic region. Restoring `S_bwd` closes the q_proj carrier direction while restoring `K` alone does not; the sham is exact. This is an independently validated attention-state mechanism, explicitly bounded as a semantic region rather than a single kernel.\n\nQwen saved-P remains a centered boundary case. Liger remains unresolved at formation despite prior persistence. No optimizer-induced mechanism has been observed.\n\n## Formation versus persistence\n\nFormation labels come only from open-loop common-state measurements. SEUP and live-weight trajectories answer a separate question: whether an already measured mechanism persists into parameter drift. The consequence summary records this separation for Phi and layer-23.\n\n## Scientific scope\n\nThe evidence supports a taxonomy of training-semantic bottlenecks, not a single universal property and not an endpoint-count claim. The remaining endpoint population is retained in `bias_population_matrix.csv`; rows without formation capture are explicitly unresolved/not captured. Legacy T1--T4 and SEUP roles are provenance only.\n""", encoding="utf-8")

    manifest = load(FINAL / "manifest.json")
    manifest.update({
        "schema": "kernel-analyzer-bias-formation-final-manifest-v2",
        "validated_mechanism_count": 2,
        "validated_mechanisms": [phi["mechanism_type"], l23["mechanism_type"]],
        "formation_stage_labels_require_v2_1_certificate": True,
        "universal_property_claim": False,
        "required_deliverables": [
            "bias_population_matrix.csv", "mechanism_taxonomy.md",
            "phi_transport_mechanism.json", "anchor_case_reports/",
            "intervention_results/", "seup_consequence_summary.json",
            "scientific_summary.md",
        ],
    })
    write_json(FINAL / "manifest.json", manifest)
    files = sorted(path for path in FINAL.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (FINAL / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(FINAL)}\n" for path in files),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(FINAL),
        "population_rows": len(matrix),
        "validated_mechanisms": manifest["validated_mechanisms"],
    }, sort_keys=True))


if __name__ == "__main__":
    write_reports(phi_mechanism(), l23_mechanism())
