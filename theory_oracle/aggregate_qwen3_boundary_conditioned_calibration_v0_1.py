#!/usr/bin/env python3
"""Calibration-only PPO boundary diagnostic for complete matched-state records.

This artifact is deliberately descriptive.  It reconstructs a signed reference
margin from the frozen pre-state inputs, validates the recorded eager/compiled
clip decisions, and reports candidate-reference margin shifts inside an
explicit tau grid.  It does not select tau, confirm a population effect, or
authorize operator attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from theory_oracle.aggregate_qwen3_calibration_records_v0_1 import (
    load_complete_state_bundles,
)


SCHEMA_VERSION = "forkcert.qwen3-boundary-conditioned-calibration.v0.1"
WEIGHTING_CONTRACT_ID = (
    "EQUAL_TRAJECTORY_PHASE_EXPOSED_STATE_NEAR_TOKEN_V0_1"
)
ALL_ELIGIBLE_WEIGHTING_CONTRACT_ID = (
    "EQUAL_TRAJECTORY_PHASE_SAMPLED_STATE_ELIGIBLE_DECISION_V0_1"
)
EPSILON = 0.2
REQUIRED_PHASES = ("early", "middle", "late")
SCRIPT_PATH = Path(__file__).resolve()
RECORD_AGGREGATOR_PATH = (
    SCRIPT_PATH.parent / "aggregate_qwen3_calibration_records_v0_1.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signed_clip_margin(
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Positive iff the GRPO/PPO clipping branch is active for a nonzero advantage."""
    ratio = torch.exp(logps - old_logps)
    sign = torch.sign(advantages).unsqueeze(-1)
    return sign * (ratio - 1.0) - EPSILON


def clip_decisions_from_margin(
    margin: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    eligible = (advantages != 0).unsqueeze(-1) & completion_mask.bool()
    return (margin > 0) & eligible


def state_profile(
    *,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    reference_logps: torch.Tensor,
    candidate_logps: torch.Tensor,
    recorded_reference_decisions: torch.Tensor,
    recorded_candidate_decisions: torch.Tensor,
    taus: list[float],
    condition_reference_logps: torch.Tensor | None = None,
) -> dict[str, Any]:
    ref_margin = signed_clip_margin(reference_logps, old_logps, advantages)
    cand_margin = signed_clip_margin(candidate_logps, old_logps, advantages)
    condition_margin = signed_clip_margin(
        condition_reference_logps
        if condition_reference_logps is not None
        else reference_logps,
        old_logps,
        advantages,
    )
    eligible = (advantages != 0).unsqueeze(-1) & completion_mask.bool()
    reconstructed_ref = clip_decisions_from_margin(
        ref_margin, advantages, completion_mask
    )
    reconstructed_cand = clip_decisions_from_margin(
        cand_margin, advantages, completion_mask
    )
    if not torch.equal(reconstructed_ref, recorded_reference_decisions.bool()):
        raise ValueError("reference clip decisions do not match reconstructed margin")
    if not torch.equal(reconstructed_cand, recorded_candidate_decisions.bool()):
        raise ValueError("candidate clip decisions do not match reconstructed margin")

    delta = cand_margin - ref_margin
    eligible_count = int(eligible.sum().item())
    if eligible_count > 0:
        for name, value in (
            ("reference margin", ref_margin),
            ("candidate margin", cand_margin),
            ("condition margin", condition_margin),
            ("margin shift", delta),
        ):
            if not torch.isfinite(value[eligible]).all().item():
                raise ValueError(f"{name} is nonfinite on an eligible decision")
    result: dict[str, Any] = {
        "eligible_decisions": eligible_count,
        "all_eligible_endpoint_status": (
            "IDENTIFIED"
            if eligible_count > 0
            else "UNINSTANTIATED_NO_ELIGIBLE_DECISIONS"
        ),
        "all_eligible_mean_margin_shift": (
            float(delta[eligible].double().mean().item())
            if eligible_count > 0
            else None
        ),
        "reference_to_candidate_on_rate": (
            float(
                ((~reconstructed_ref) & reconstructed_cand & eligible).sum().item()
                / eligible_count
            )
            if eligible_count > 0
            else None
        ),
        "candidate_to_reference_off_rate": (
            float(
                (reconstructed_ref & (~reconstructed_cand) & eligible).sum().item()
                / eligible_count
            )
            if eligible_count > 0
            else None
        ),
        "tau_profiles": {},
    }
    for tau in taus:
        near = eligible & (condition_margin.abs() <= tau)
        count = int(near.sum().item())
        if count == 0:
            result["tau_profiles"][str(tau)] = {
                "exposures": 0,
                "condition_mask_sha256": boolean_mask_sha256(near),
                "mean_margin_shift": None,
                "directional_event_shift": None,
                "semantic_disagreement": None,
            }
            continue
        upward = ((~reconstructed_ref) & reconstructed_cand & near).sum().item()
        downward = (reconstructed_ref & (~reconstructed_cand) & near).sum().item()
        result["tau_profiles"][str(tau)] = {
            "exposures": count,
            "condition_mask_sha256": boolean_mask_sha256(near),
            "mean_margin_shift": float(delta[near].double().mean().item()),
            "directional_event_shift": float((upward - downward) / count),
            "semantic_disagreement": float((upward + downward) / count),
        }
    return result


def mean(values: list[float]) -> float | None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError("mean received a nonfinite value")
    return sum(values) / len(values) if values else None


def sample_variance(values: list[float]) -> float:
    if any(not math.isfinite(value) for value in values):
        raise ValueError("sample_variance received a nonfinite value")
    if len(values) < 2:
        return 0.0
    center = sum(values) / len(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def boolean_mask_sha256(mask: torch.Tensor) -> str:
    value = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def aggregate_state_profiles(rows: list[dict[str, Any]], taus: list[float]) -> dict[str, Any]:
    phase_rows: dict[str, list[dict[str, Any]]] = {
        phase: [row for row in rows if row.get("phase") == phase]
        for phase in REQUIRED_PHASES
    }
    all_eligible_phase_identification: dict[str, Any] = {}
    all_eligible_phase_means: dict[str, float | None] = {}
    for phase, current in phase_rows.items():
        identified = [
            row
            for row in current
            if row.get("all_eligible_endpoint_status") in (None, "IDENTIFIED")
            and isinstance(row.get("all_eligible_mean_margin_shift"), (int, float))
            and math.isfinite(float(row["all_eligible_mean_margin_shift"]))
        ]
        uninstantiated = [
            str(row.get("state_id")) for row in current if row not in identified
        ]
        phase_identified = bool(current) and len(identified) == len(current)
        all_eligible_phase_identification[phase] = {
            "sampled_states": len(current),
            "identified_states": len(identified),
            "uninstantiated_state_ids": uninstantiated,
            "phase_estimand_identified": phase_identified,
        }
        all_eligible_phase_means[phase] = (
            mean(
                [float(row["all_eligible_mean_margin_shift"]) for row in identified]
            )
            if phase_identified
            else None
        )
    all_phases_identified = all(
        all_eligible_phase_identification[phase]["phase_estimand_identified"]
        for phase in REQUIRED_PHASES
    )
    tau_results: dict[str, Any] = {}
    for tau in taus:
        key = str(tau)
        available = [
            row["tau_profiles"][key]
            for row in rows
            if row["tau_profiles"][key]["exposures"] > 0
        ]
        shifts = [float(row["mean_margin_shift"]) for row in available]
        directional = [float(row["directional_event_shift"]) for row in available]
        disagreement = [float(row["semantic_disagreement"]) for row in available]
        phase_profiles: dict[str, Any] = {}
        for phase, current_rows in phase_rows.items():
            current = [
                row["tau_profiles"][key]
                for row in current_rows
                if row["tau_profiles"][key]["exposures"] > 0
            ]
            phase_profiles[phase] = {
                "sampled_states": len(current_rows),
                "states_with_exposure": len(current),
                "total_exposures_descriptive_only": sum(
                    row["exposures"] for row in current
                ),
                "state_weighted_mean_margin_shift": mean(
                    [float(row["mean_margin_shift"]) for row in current]
                ),
                "state_weighted_directional_event_shift": mean(
                    [float(row["directional_event_shift"]) for row in current]
                ),
                "state_weighted_semantic_disagreement": mean(
                    [float(row["semantic_disagreement"]) for row in current]
                ),
            }
        all_phase_conditionals_identified = all(
            phase_profiles[phase]["states_with_exposure"] >= 2
            for phase in REQUIRED_PHASES
        )
        tau_results[key] = {
            "states_with_exposure": len(available),
            "total_exposures_descriptive_only": sum(row["exposures"] for row in available),
            "state_weighted_mean_margin_shift": mean(shifts),
            "state_effect_sign_counts": {
                "positive": sum(value > 0 for value in shifts),
                "zero": sum(value == 0 for value in shifts),
                "negative": sum(value < 0 for value in shifts),
            },
            "state_weighted_directional_event_shift": mean(directional),
            "state_weighted_semantic_disagreement": mean(disagreement),
            "phase_profiles": phase_profiles,
            "all_phase_conditionals_identified": all_phase_conditionals_identified,
            "phase_balanced_state_weighted_mean_margin_shift": (
                mean(
                    [
                        float(
                            phase_profiles[phase][
                                "state_weighted_mean_margin_shift"
                            ]
                        )
                        for phase in REQUIRED_PHASES
                    ]
                )
                if all_phase_conditionals_identified
                else None
            ),
            "phase_balanced_directional_event_shift": (
                mean(
                    [
                        float(
                            phase_profiles[phase][
                                "state_weighted_directional_event_shift"
                            ]
                        )
                        for phase in REQUIRED_PHASES
                    ]
                )
                if all_phase_conditionals_identified
                else None
            ),
            "phase_balanced_semantic_disagreement": (
                mean(
                    [
                        float(
                            phase_profiles[phase][
                                "state_weighted_semantic_disagreement"
                            ]
                        )
                        for phase in REQUIRED_PHASES
                    ]
                )
                if all_phase_conditionals_identified
                else None
            ),
        }
    return {
        "complete_states": len(rows),
        "all_eligible_phase_identification": all_eligible_phase_identification,
        "all_eligible_phase_means": all_eligible_phase_means,
        "all_eligible_phase_balanced_state_weighted_mean_margin_shift": (
            mean(
                [float(all_eligible_phase_means[phase]) for phase in REQUIRED_PHASES]
            )
            if all_phases_identified
            else None
        ),
        "tau_profiles": tau_results,
    }


def load_state(
    target: dict[str, Any], bundle: dict[str, Any], taus: list[float]
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    transition_evidence: list[dict[str, str]] = []
    arm_names = {"reference": "eager", "candidate": "compiled"}
    arm_records = bundle.get("arm_records", [])
    if len(arm_records) != 4:
        raise ValueError("record bundle must contain four arm records")
    for record in arm_records:
        identity = record.get("identity", {})
        arm = arm_names.get(identity.get("arm"))
        repeat = identity.get("repeat_id")
        if arm is None or repeat not in (1, 2):
            raise ValueError("invalid arm/repeat identity in record bundle")
        key = f"{arm}_{repeat}"
        link = record.get("provenance", {}).get("transition_result", {})
        path = Path(link.get("path", "")).resolve()
        if not path.is_file() or sha256_file(path) != link.get("sha256"):
            raise ValueError(f"transition-result provenance failed for {key}")
        results[key] = json.loads(path.read_text(encoding="utf-8"))
        if results[key].get("valid") is not True:
            raise ValueError(f"invalid arm result: {path}")
        transition_evidence.append(
            {"arm_repeat": key, "path": str(path), "sha256": link["sha256"]}
        )
    if set(results) != {"eager_1", "compiled_1", "eager_2", "compiled_2"}:
        raise ValueError("record bundle arm/repeat grid is incomplete")
    snapshot_path = Path(results["eager_1"]["snapshot"]["path"])
    snapshot_metadata_path = snapshot_path / "forkcert_transition_snapshot.json"
    if (
        not snapshot_metadata_path.is_file()
        or sha256_file(snapshot_metadata_path)
        != results["eager_1"]["snapshot"]["metadata_sha256"]
    ):
        raise ValueError("snapshot metadata provenance failed")
    snapshot = json.loads(snapshot_metadata_path.read_text(encoding="utf-8"))
    target_minibatch_path = Path(snapshot["target_minibatch_path"]).resolve()
    if (
        not target_minibatch_path.is_file()
        or sha256_file(target_minibatch_path)
        != results["eager_1"]["snapshot"]["target_minibatch_sha256"]
    ):
        raise ValueError("target minibatch provenance failed")
    inputs = torch.load(
        target_minibatch_path, map_location="cpu", weights_only=False
    )

    repeat_profiles = []
    reference_logps_by_repeat: list[torch.Tensor] = []
    condition_reference_logps = torch.tensor(
        results["eager_1"]["continuous"]["scorer_logps"]
    )
    for repeat in (1, 2):
        ref = results[f"eager_{repeat}"]
        cand = results[f"compiled_{repeat}"]
        current_reference_logps = torch.tensor(ref["continuous"]["scorer_logps"])
        reference_logps_by_repeat.append(current_reference_logps)
        repeat_profiles.append(
            state_profile(
                old_logps=inputs["old_per_token_logps"].float(),
                advantages=inputs["advantages"].float(),
                completion_mask=inputs["completion_mask"],
                reference_logps=current_reference_logps,
                candidate_logps=torch.tensor(cand["continuous"]["scorer_logps"]),
                recorded_reference_decisions=torch.tensor(ref["semantic"]["clip_decisions"]),
                recorded_candidate_decisions=torch.tensor(cand["semantic"]["clip_decisions"]),
                taus=taus,
                condition_reference_logps=condition_reference_logps,
            )
        )
    native_reference_mask_hashes_by_repeat: dict[str, dict[str, str]] = {}
    eligible = (inputs["advantages"].float() != 0).unsqueeze(-1) & inputs[
        "completion_mask"
    ].bool()
    for repeat, reference_logps in enumerate(reference_logps_by_repeat, start=1):
        native_margin = signed_clip_margin(
            reference_logps,
            inputs["old_per_token_logps"].float(),
            inputs["advantages"].float(),
        )
        native_reference_mask_hashes_by_repeat[str(repeat)] = {
            str(tau): boolean_mask_sha256(eligible & (native_margin.abs() <= tau))
            for tau in taus
        }
    reference_margin_exact = torch.equal(
        reference_logps_by_repeat[0], reference_logps_by_repeat[1]
    )
    tau_profiles: dict[str, Any] = {}
    tau_runtime: dict[str, Any] = {}
    for tau in taus:
        key = str(tau)
        current = [profile["tau_profiles"][key] for profile in repeat_profiles]
        exposure_counts = {profile["exposures"] for profile in current}
        if len(exposure_counts) != 1:
            raise ValueError("anchor-defined boundary exposure changed across repeats")
        exposures = next(iter(exposure_counts))
        mask_hashes = {profile["condition_mask_sha256"] for profile in current}
        if len(mask_hashes) != 1:
            raise ValueError("anchor-defined boundary mask changed across repeats")
        condition_mask_sha256 = next(iter(mask_hashes))
        if exposures == 0:
            tau_profiles[key] = current[0]
            tau_runtime[key] = {
                "mean_margin_shift_variance": None,
                "directional_event_shift_variance": None,
                "semantic_disagreement_variance": None,
            }
            continue
        tau_profiles[key] = {
            "exposures": exposures,
            "condition_mask_sha256": condition_mask_sha256,
            "mean_margin_shift": mean(
                [float(profile["mean_margin_shift"]) for profile in current]
            ),
            "directional_event_shift": mean(
                [float(profile["directional_event_shift"]) for profile in current]
            ),
            "semantic_disagreement": mean(
                [float(profile["semantic_disagreement"]) for profile in current]
            ),
        }
        tau_runtime[key] = {
            "mean_margin_shift_variance": sample_variance(
                [float(profile["mean_margin_shift"]) for profile in current]
            ),
            "directional_event_shift_variance": sample_variance(
                [float(profile["directional_event_shift"]) for profile in current]
            ),
            "semantic_disagreement_variance": sample_variance(
                [float(profile["semantic_disagreement"]) for profile in current]
            ),
        }
    identity = snapshot["capture_target_identity"]
    for field in ("state_id", "phase"):
        if identity.get(field) != target.get(field):
            raise ValueError(f"snapshot/plan {field} mismatch")
    if int(snapshot["optimizer_step"]) != int(target["optimizer_step"]):
        raise ValueError("snapshot/plan optimizer_step mismatch")
    return {
        "state_id": identity["state_id"],
        "trajectory_id": identity["trajectory_id"],
        "phase": identity["phase"],
        "optimizer_step": snapshot["optimizer_step"],
        "evidence": {
            "snapshot_metadata": {
                "path": str(snapshot_metadata_path),
                "sha256": sha256_file(snapshot_metadata_path),
            },
            "target_minibatch": {
                "path": str(target_minibatch_path),
                "sha256": sha256_file(target_minibatch_path),
            },
            "transition_results": transition_evidence,
        },
        "eligible_decisions": repeat_profiles[0]["eligible_decisions"],
        "all_eligible_endpoint_status": repeat_profiles[0][
            "all_eligible_endpoint_status"
        ],
        "all_eligible_mean_margin_shift": (
            mean(
                [
                    float(profile["all_eligible_mean_margin_shift"])
                    for profile in repeat_profiles
                ]
            )
            if repeat_profiles[0]["all_eligible_endpoint_status"] == "IDENTIFIED"
            else None
        ),
        "reference_to_candidate_on_rate": (
            mean(
                [
                    float(profile["reference_to_candidate_on_rate"])
                    for profile in repeat_profiles
                ]
            )
            if repeat_profiles[0]["all_eligible_endpoint_status"] == "IDENTIFIED"
            else None
        ),
        "candidate_to_reference_off_rate": (
            mean(
                [
                    float(profile["candidate_to_reference_off_rate"])
                    for profile in repeat_profiles
                ]
            )
            if repeat_profiles[0]["all_eligible_endpoint_status"] == "IDENTIFIED"
            else None
        ),
        "tau_profiles": tau_profiles,
        "repeat_profiles": [
            {"repeat_id": repeat, **profile}
            for repeat, profile in enumerate(repeat_profiles, start=1)
        ],
        "same_state_runtime_variability": {
            "all_eligible_mean_margin_shift_variance": (
                sample_variance(
                    [
                        float(profile["all_eligible_mean_margin_shift"])
                        for profile in repeat_profiles
                    ]
                )
                if repeat_profiles[0]["all_eligible_endpoint_status"] == "IDENTIFIED"
                else None
            ),
            "tau_profiles": tau_runtime,
            "condition_anchor": (
                "REFERENCE_REPEAT_1_MARGIN_MASK_ALLOWED_ONLY_WHEN_REFERENCE_"
                "MARGINS_ARE_EXACT_ACROSS_REPEATS"
            ),
        },
        "reference_anchor_stability": {
            "reference_scorer_logps_exact_across_repeats": reference_margin_exact,
            "native_condition_mask_sha256_by_repeat": native_reference_mask_hashes_by_repeat,
            "formal_confirmation_allowed": reference_margin_exact,
            "failure_disposition": (
                None
                if reference_margin_exact
                else "UNINSTANTIATED_STOCHASTIC_REFERENCE_ANCHOR_REQUIRES_"
                "INDEPENDENT_ANCHOR_PROTOCOL"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--taus", nargs="+", type=float, required=True)
    parser.add_argument("--allow-partial-calibration", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    taus = sorted(set(args.taus))
    if not taus or any(not math.isfinite(tau) or tau <= 0 for tau in taus):
        raise SystemExit("taus must be finite positive values")
    root = Path(args.results_root).resolve()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    targets = plan.get("targets", [])
    present_targets = []
    missing_state_ids = []
    errors: list[str] = []
    identity = plan.get("identity", {})
    if plan.get("schema_version") != "forkcert.multi-transition-capture-plan.v0.1":
        errors.append("unsupported frozen capture plan schema")
    if identity.get("query_id") != "Q-R" or identity.get("trajectory_anchor") != "EAGER_TRAJECTORY":
        errors.append("boundary calibration requires the frozen eager-anchored Q-R query")
    if not isinstance(targets, list) or len(targets) != 24:
        errors.append("frozen capture plan must contain exactly 24 states")
        targets = targets if isinstance(targets, list) else []
    target_ids = [target.get("state_id") for target in targets]
    target_steps = [target.get("optimizer_step") for target in targets]
    if (
        None in target_ids
        or len(target_ids) != len(set(target_ids))
        or None in target_steps
        or len(target_steps) != len(set(target_steps))
    ):
        errors.append("frozen capture plan state IDs and optimizer steps must be unique")
    if Counter(target.get("phase") for target in targets) != Counter(
        {phase: 8 for phase in REQUIRED_PHASES}
    ):
        errors.append("frozen capture plan must contain 8 states in every phase")
    for target in targets:
        step_dir = root / f"step{int(target['optimizer_step']):03d}"
        bundle_exists = (step_dir / "record_bundle.json").is_file()
        validation_exists = (step_dir / "record_validation.json").is_file()
        if bundle_exists and validation_exists:
            present_targets.append(target)
        elif bundle_exists or validation_exists:
            errors.append(f"partial record evidence for {target['state_id']}")
        else:
            missing_state_ids.append(target["state_id"])
    if missing_state_ids and not args.allow_partial_calibration:
        errors.append(f"missing {len(missing_state_ids)} frozen-plan states")
    subplan = {**plan, "targets": present_targets}
    state_bundles, state_evidence, load_errors = load_complete_state_bundles(
        subplan, root
    )
    errors.extend(load_errors)
    rows = []
    for target, bundle in state_bundles:
        try:
            rows.append(load_state(target, bundle, taus))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{target['state_id']}: {error}")
    complete = not errors and len(rows) == len(targets) == 24
    exact_reference_anchor_states = sum(
        row.get("reference_anchor_stability", {}).get(
            "reference_scorer_logps_exact_across_repeats"
        )
        is True
        for row in rows
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": not errors,
        "status": (
            "COMPLETE_CALIBRATION_BOUNDARY_DESCRIPTION"
            if complete
            else "PARTIAL_CALIBRATION_BOUNDARY_DESCRIPTION"
        ),
        "construction": {
            "expected_states": len(targets),
            "observed_complete_states": len(rows),
            "missing_state_ids": missing_state_ids,
            "errors": errors,
            "tau_grid": taus,
            "boundary_anchor": "REFERENCE_PRE_INTERVENTION_SIGNED_CLIP_MARGIN",
            "reference_anchor_identification": {
                "observed_states_with_exact_reference_scorer_logps_across_repeats": exact_reference_anchor_states,
                "observed_states_total": len(rows),
                "all_observed_states_exact": exact_reference_anchor_states == len(rows),
                "formal_v0_1_rule": (
                    "repeat-1 may anchor the condition only when reference scorer "
                    "log-probs are exact across repeats; otherwise an independent "
                    "anchor protocol is required"
                ),
            },
            "state_weighting": "equal weight over states with at least one reference-near-boundary eligible decision",
            "hierarchical_weighting": "equal phase; equal exposed state within phase; equal eligible near-boundary token within state",
            "weighting_contract_id": WEIGHTING_CONTRACT_ID,
            "missing_phase_rule": "conditional estimand is uninstantiated unless every frozen phase has at least two exposed states, so within-phase state dispersion is identifiable",
            "all_eligible_weighting_contract_id": ALL_ELIGIBLE_WEIGHTING_CONTRACT_ID,
            "all_eligible_hierarchical_weighting": "equal trajectory; equal phase; equal sampled state; equal eligible decision within state",
            "all_eligible_missing_state_rule": "the state-balanced estimand is uninstantiated if any sampled state has no eligible decisions; exposure-weighted or exposed-state-conditioned alternatives require a distinct predeclared contract",
        },
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "analysis_code": {
            "boundary_aggregator": {
                "path": str(SCRIPT_PATH),
                "sha256": sha256_file(SCRIPT_PATH),
            },
            "record_loader": {
                "path": str(RECORD_AGGREGATOR_PATH),
                "sha256": sha256_file(RECORD_AGGREGATOR_PATH),
            },
        },
        "state_evidence": state_evidence,
        "aggregate": aggregate_state_profiles(rows, taus),
        "states": rows,
        "claims_allowed": {
            "calibration_description": True,
            "population_global_B": False,
            "population_conditional_B": False,
            "operator_attribution": False,
            "correctness": False,
        },
        "next_gate": "freeze tau/endpoint family from calibration-only evidence, then test on independent trajectories",
        "nonclaims": [
            "a tau with a large observed effect is not a confirmed conditional Bias",
            "reference anchoring does not make eager a correctness authority",
            "semantic disagreement is not a signed Bias endpoint",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "construction": payload["construction"]}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
