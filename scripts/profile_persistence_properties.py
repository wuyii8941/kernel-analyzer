#!/usr/bin/env python3
"""Profile candidate persistence properties on frozen development artifacts.

This script is intentionally retrospective and development-only.  It does not
read T4/SEUP labels to manufacture a predictor, and it never turns a missing
measurement into a negative result.  The output is a compact inventory of what
the current artifacts actually measure for four candidate properties:

* source asymmetry;
* source--transport coupling;
* transported-error concentration;
* trajectory carrier stability.

The report is a property-screening artifact, not a held-out oracle verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/property/bias_property_search/property_evidence.json"
SCOPE = ROOT / "results/property/tcmp_allop_v1/scope_extension_20260822.json"
OUT = ROOT / "results/property/bias_property_search"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def gram_profile(layer: dict[str, Any] | None) -> dict[str, Any]:
    if not layer:
        return {"status": "UNMEASURED"}
    gram = np.asarray(layer.get("complete_gram", []), dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] < 2 or gram.shape[0] != gram.shape[1]:
        return {"status": "UNMEASURED_MISSING_COMPLETE_GRAM"}
    if not np.isfinite(gram).all():
        return {"status": "INVALID_NONFINITE"}
    gram = (gram + gram.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return {"status": "UNMEASURED_ZERO_ENERGY"}
    participation = float(total * total / max(float(np.square(eigenvalues).sum()), 1e-30))
    return {
        "status": "MEASURED_PROXY",
        "state_count": int(gram.shape[0]),
        "top_eigen_fraction": float(eigenvalues[-1] / total),
        "top_2_eigen_fraction": float(eigenvalues[-2:].sum() / total),
        "effective_rank_participation": participation,
        "cross_state_ratio": float(layer.get("cross_state_ratio", float("nan"))),
        "cross_state_u_statistic": float(layer.get("cross_state_u_statistic", float("nan"))),
    }


def formation_profile(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"status": "UNMEASURED"}
    document = load(path)
    populations = document.get("populations")
    if populations is None:
        # The older v2.1 certificate uses the same population shape under
        # ``calibration`` and ``confirmation`` in the final taxonomy.
        populations = {
            "calibration": document.get("calibration", {}),
            "confirmation": document.get("confirmation", {}),
        }
    result: dict[str, Any] = {"status": "MEASURED", "artifact": relative(path)}
    for population_name in ("calibration", "confirmation"):
        population = populations.get(population_name, {})
        layers: dict[str, Any] = {}
        for layer_name in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE"):
            layer = population.get(layer_name)
            if layer is None:
                layers[layer_name] = {"status": "UNMEASURED"}
                continue
            layers[layer_name] = {
                "status": layer.get("status", population.get(layer_name + "_status", "UNRESOLVED")),
                "cross_state_ratio": layer.get("cross_state_ratio"),
                "bootstrap_95": [layer.get("bootstrap_lower"), layer.get("bootstrap_upper")],
                "transport_concentration": gram_profile(layer),
            }
        result[population_name] = layers
    return result


def _formation_transition(formation: dict[str, Any]) -> dict[str, Any]:
    """Return the measured stage statuses without imputing missing layers."""

    confirmation = formation.get("confirmation", {}) if formation else {}
    result: dict[str, Any] = {}
    for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE"):
        value = confirmation.get(layer)
        if isinstance(value, dict):
            result[layer] = value.get("status", "UNRESOLVED")
        else:
            result[layer] = "UNMEASURED"
    return result


def source_asymmetry(
    case_id: str,
    evidence: dict[str, Any],
    formation: dict[str, Any] | None = None,
    development_role: str | None = None,
) -> dict[str, Any]:
    # Scope-extension controls have no source-atom decomposition, but their
    # complete formation certificates do provide a pre-trajectory local stage
    # status.  Handle this before consulting the mechanism-evidence registry,
    # which intentionally contains only the older mechanism cases.
    if development_role == "CENTERED_CONTROL" and formation:
        transition = _formation_transition(formation)
        return {
            "status": (
                "CENTERED_LOCAL_CONTROL"
                if transition["LOCAL_ENDPOINT"] == "CENTERED"
                else "UNRESOLVED_LOCAL_CONTROL"
            ),
            "formation_transition": transition,
            "evidence_kind": "complete_open_loop_formation_certificate",
        }
    case = next((row for row in evidence.get("cases", []) if row.get("case") == case_id), None)
    if case is None:
        return {"status": "UNMEASURED"}
    natural = case.get("natural_effect", {})
    control = case.get("antithetic_control", {})
    if case_id == "liger_fused_ce":
        positive = natural.get("positive_states") == 24 and natural.get("mean_projection", 0.0) > 0
        centered_control = control.get("negative_states") == 11 and control.get("mean_projection", 1.0) <= 0.001
        return {
            "status": "SUPPORTED_DEVELOPMENT_CASE" if positive and centered_control else "UNRESOLVED",
            "mean_projection": natural.get("mean_projection"),
            "bootstrap_95": natural.get("bootstrap_95"),
            "orbit_control_mean_projection": control.get("mean_projection"),
            "orbit_control_bootstrap_95": control.get("bootstrap_95"),
            "raw_coordinate_signed_mean": natural.get("raw_coordinate_signed_mean"),
        }
    if case_id == "qwen128_vproj_mm":
        kernel = case.get("kernel_u")
        rounding = case.get("output_rounding_u")
        return {
            "status": "SUPPORTED_CONDITIONAL_SOURCE_OBSERVATION",
            "kernel_u": kernel,
            "output_rounding_u": rounding,
            "output_rounding_to_kernel_u_ratio": abs(rounding) / max(abs(kernel), 1e-30),
        }
    if case_id == "mamba_seq64_input_proj":
        return {
            "status": "SUPPORTED_CONDITIONAL_SOURCE_OBSERVATION",
            "kernel_u": case.get("kernel_u"),
            "output_rounding_u": case.get("output_rounding_u"),
        }
    if natural.get("verdict") == "CENTERED":
        return {"status": "CENTERED_LOCAL_CONTROL", "evidence_kind": "local_effect"}
    return {"status": "UNMEASURED_OR_NOT_APPLICABLE"}


def source_transport(case_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    case = next((row for row in evidence.get("cases", []) if row.get("case") == case_id), None)
    if case is None:
        return {"status": "UNMEASURED"}
    if case_id == "phi4_seq64_lmhead_dx":
        natural = case["natural_effect"]["gradient_cross_state_ratio"]
        shuffled = case["antithetic_control"]["gradient_cross_state_ratio"]
        return {
            "status": "SUPPORTED_CASE_LEVEL_COUPLING",
            "natural_gradient_ratio": natural,
            "shuffled_gradient_ratio": shuffled,
            "pairing_suppression": 1.0 - shuffled / max(abs(natural), 1e-30),
            "local_norm_preserved": bool(case["antithetic_control"]["local_norm_preserved_every_state"]),
            "analytic_transport_closed": False,
        }
    if case_id == "qwen_saved_p_seq128":
        suppression = case.get("update_pairing_suppression")
        return {
            "status": "REJECTED_FOR_THIS_CASE" if suppression is not None and suppression <= 0.0 else "UNRESOLVED",
            "gradient_pairing_suppression": case.get("gradient_pairing_suppression"),
            "update_pairing_suppression": suppression,
            "optimizer_rectification_ratio": case.get("optimizer_oddness", {}).get("resultant_ratio"),
        }
    if case_id == "qwen_layer23_attention_state":
        return {
            "status": "BOUNDARY_NOT_MARGINAL_PRESERVING",
            "s_bwd_positive_states": case["natural_effect"].get("s_bwd_shapley_positive_states"),
            "s_bwd_states": case["natural_effect"].get("states"),
        }
    return {"status": "NOT_MEASURED_OR_NOT_APPLICABLE"}


def carrier_stability(path: Path | None, case_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    if case_id == "qwen_layer23_attention_state":
        case = next(row for row in evidence["cases"] if row.get("case") == case_id)
        persistence = case.get("persistence_separation", {})
        return {
            "status": "MEASURED_TRAJECTORY_STABLE",
            "projection_strictly_increases": persistence.get("projection_strictly_increases_each_step"),
            "steps": persistence.get("steps"),
        }
    if path is None or not path.exists():
        return {"status": "UNMEASURED"}
    document = load(path)
    gates = document.get("gates", {})
    projections = document.get("directional_projections", [])
    stable_gate = gates.get("same_weight_carrier_direction_stable")
    stable_gate = stable_gate if stable_gate is not None else gates.get("directional_live_weight_accumulation")
    if stable_gate is None:
        stable_gate = gates.get("all_64_frozen_carrier_projections_positive")
    finite = [float(value) for value in projections if isinstance(value, (int, float))]
    signs = sum(value > 0.0 for value in finite) / len(finite) if finite else None
    monotonic = all(right > left for left, right in zip(finite, finite[1:])) if len(finite) > 1 else None
    return {
        "status": "MEASURED_TRAJECTORY_STABLE" if stable_gate else "MEASURED_TRAJECTORY_NOT_STABLE",
        "gate": stable_gate,
        "checkpoint_projection_sign_fraction": signs,
        "checkpoint_projection_strictly_increasing": monotonic,
        "trajectory_status": document.get("status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    evidence = load(EVIDENCE)
    scope = load(SCOPE)

    formation_paths: dict[str, Path] = {
        "liger_fused_ce": ROOT / "results/property/bias_formation/formation/liger_fused_ce_t128.json",
        "phi4_seq64_lmhead_dx": ROOT / "results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json",
        "qwen_saved_p_seq128": ROOT / "results/property/bias_formation/formation/qwen_saved_p_seq128.json",
        "deepseek8b_seq64_backward_1665_in_out_ptr0": ROOT / "results/property/tcmp_allop_v1/heldout/deepseek8b_seq64_backward_1665_in_out_ptr0/deepseek8b_seq64_backward_1665_in_out_ptr0.json",
        "phi4_seq64_backward_495_out_ptr1": ROOT / "results/property/tcmp_allop_v1/heldout/phi4_seq64_backward_495_out_ptr1/phi4_seq64_backward_495_out_ptr1.json",
        "phi4_seq64_backward_1031_in_out_ptr0": ROOT / "results/property/tcmp_allop_v1/heldout/phi4_seq64_backward_1031_in_out_ptr0/phi4_seq64_backward_1031_in_out_ptr0.json",
        "qwen_seq64_backward_1293_in_out_ptr0": ROOT / "results/property/tcmp_allop_v1/heldout/qwen_seq64_backward_1293_in_out_ptr0/qwen_seq64_backward_1293_in_out_ptr0.json",
        "qwen_seq64_backward_1308_output_0": ROOT / "results/property/tcmp_allop_v1/heldout/qwen_seq64_backward_1308_output_0/qwen_seq64_backward_1308_output_0.json",
    }
    trajectory_paths: dict[str, Path] = {
        "liger_fused_ce": ROOT / "results/trajectory/liger_trajectory.json",
        "phi4_seq64_lmhead_dx": ROOT / "results/coverage/cases/phi4_seq64_lmhead_dx_trajectory.json",
        "qwen_saved_p_seq128": ROOT / "results/coverage/cases/qwen128_softmax_saved_p_trajectory.json",
        "qwen128_vproj_mm": ROOT / "results/coverage/cases/qwen128_vproj_trajectory.json",
        "qwen64_vproj_mm": ROOT / "results/coverage/cases/qwen64_vproj_trajectory.json",
        "mamba_seq64_input_proj": ROOT / "results/coverage/cases/mamba_seq64_input_proj_trajectory.json",
        "qwen3vl_silu": ROOT / "results/coverage/cases/qwen3vl_layer0_silu_trajectory.json",
    }
    conditional_cases = ["qwen128_vproj_mm", "qwen64_vproj_mm", "mamba_seq64_input_proj"]
    cases: list[dict[str, Any]] = []

    # The first six rows have existing mechanism artifacts; the five scope
    # extension rows are complete centered controls.  All are development-only.
    identifiers = [
        ("liger_fused_ce", "KNOWN_PERSISTENCE_ANCHOR"),
        ("phi4_seq64_lmhead_dx", "KNOWN_FORMATION_AND_PERSISTENCE"),
        ("qwen_saved_p_seq128", "FEEDBACK_BOUNDARY"),
        ("qwen128_vproj_mm", "CONDITIONAL_SOURCE_FORMATION"),
        ("qwen64_vproj_mm", "CONDITIONAL_SOURCE_FORMATION"),
        ("mamba_seq64_input_proj", "PARTIAL_CROSS_ARCHITECTURE_FORMATION"),
        ("qwen_layer23_attention_state", "SEMANTIC_REGION_PERSISTENCE"),
        ("qwen3vl_silu", "OPTIMIZER_RECTIFICATION_BOUNDARY"),
        ("deepseek8b_seq64_backward_1665_in_out_ptr0", "CENTERED_CONTROL"),
        ("phi4_seq64_backward_495_out_ptr1", "CENTERED_CONTROL"),
        ("phi4_seq64_backward_1031_in_out_ptr0", "CENTERED_CONTROL"),
        ("qwen_seq64_backward_1293_in_out_ptr0", "CENTERED_CONTROL"),
        ("qwen_seq64_backward_1308_output_0", "CENTERED_CONTROL"),
    ]
    for case_id, role in identifiers:
        formation = formation_profile(formation_paths.get(case_id))
        cases.append({
            "case_id": case_id,
            "development_role": role,
            "source_asymmetry": source_asymmetry(
                case_id, evidence, formation=formation, development_role=role
            ),
            "source_transport_coupling": source_transport(case_id, evidence),
            "transport_concentration": formation,
            "carrier_stability": carrier_stability(trajectory_paths.get(case_id), case_id, evidence),
            "conditional_source_artifact": case_id in conditional_cases,
            "formation_artifact": formation.get("artifact"),
            "trajectory_artifact": relative(trajectory_paths[case_id]) if case_id in trajectory_paths and trajectory_paths[case_id].exists() else None,
        })

    # The scope extension is a denominator check, not a label source.
    output = {
        "schema": "kernel-analyzer-development-property-profile-v1",
        "status": "DEVELOPMENT_ONLY_NOT_FROZEN",
        "development_only": True,
        "historical_verdicts_used_as_predictors": False,
        "scope_extension_sha256": scope.get("result_sha256"),
        "properties": {
            "source_asymmetry": {
                "definition": "A conditional event/schedule population has a nonzero signed effective local component.",
                "candidate_inputs": ["declared source decomposition", "orbit/intervention residuals"],
            },
            "source_transport_coupling": {
                "definition": "Preserving residual marginals but permuting their real F+B transport pairing removes the directional gradient/update component.",
                "candidate_inputs": ["local residual multiset", "transport pairing intervention", "complete VJP boundary"],
            },
            "transport_concentration": {
                "definition": "Transported error vectors concentrate into a low-dimensional state-by-parameter subspace.",
                "candidate_inputs": ["complete cross-state Gram", "top-eigen energy", "participation ratio"],
                "warning": "This is a proxy; state Gram rank is not a proof of a fixed global parameter carrier.",
            },
            "carrier_stability": {
                "definition": "The transported update carrier remains aligned across adjacent ordered trajectory states.",
                "candidate_inputs": ["short ordered reference trajectory", "frozen carrier or subspace overlap"],
            },
        },
        "cases": cases,
        "decision": {
            "source_asymmetry": "RETAIN_AS_CANDIDATE",
            "source_transport_coupling": "RETAIN_CASE_LEVEL_CANDIDATE",
            "transport_concentration": "RETAIN_AS_SUPPORTING_FEATURE_NOT_STANDALONE",
            "carrier_stability": "RETAIN_AS_CONSEQUENCE_SCREEN_REQUIRES_SHORT_TRAJECTORY",
            "reason": "Current artifacts separate source/transport evidence, but no property is frozen for held-out prediction yet.",
        },
        "next_gate": "Implement shared short-trajectory random-projection screen only after this development profile is reviewed and frozen.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "development_property_profile.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_root / "development_property_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["case_id", "role", "source_status", "transport_status", "concentration_status", "stability_status"])
        for row in cases:
            writer.writerow([
                row["case_id"], row["development_role"],
                row["source_asymmetry"]["status"],
                row["source_transport_coupling"]["status"],
                row["transport_concentration"]["status"],
                row["carrier_stability"]["status"],
            ])
    summary_lines = [
        "# Development property profile",
        "",
        "This is a retrospective development artifact. It does not freeze a held-out predictor.",
        "",
        "| case | role | source asymmetry | source-transport | concentration | carrier stability |",
        "|---|---|---|---|---|---|",
    ]
    for row in cases:
        summary_lines.append(
            f"| {row['case_id']} | {row['development_role']} | "
            f"{row['source_asymmetry']['status']} | "
            f"{row['source_transport_coupling']['status']} | "
            f"{row['transport_concentration']['status']} | "
            f"{row['carrier_stability']['status']} |"
        )
    summary_lines += [
        "",
        "Current decision: retain source asymmetry and source-transport coupling as candidate formation properties; use concentration as a supporting feature; measure carrier stability with the shared short trajectory.",
        "Missing data are reported as UNMEASURED and are not interpreted as safe or negative.",
    ]
    (args.output_root / "development_property_profile.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": relative(args.output_root / "development_property_profile.json"), "cases": len(cases), "status": output["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
