#!/usr/bin/env python
"""Aggregate complete per-state record bundles without changing the frozen population."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.bias_oracle_population_v0_2 import (
    EffectRecord,
    estimate_scalar_population,
)
from theory_oracle.bias_oracle_record_v0_2 import validate_record_bundle


SCHEMA_VERSION = "forkcert.qwen3-calibration-record-aggregate.v0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def linked_arms(bundle: dict[str, Any], pair: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_digest = {row["record_digest"]: row for row in bundle["arm_records"]}
    reference = by_digest[pair["links"]["reference_arm_record_digest"]]
    candidate = by_digest[pair["links"]["candidate_arm_record_digest"]]
    return reference, candidate


def measured_endpoint(value: dict[str, Any]) -> float | None:
    return float(value["value"]) if value.get("status") == "MEASURED" else None


def reference_aligned_dot(
    pair: dict[str, Any], reference: dict[str, Any], candidate: dict[str, Any]
) -> float | None:
    del candidate
    reference_l2 = float(
        reference["outcomes"]["propagation_ledgers"]["parameter_update_l2"]
    )
    relative = measured_endpoint(pair["effects"]["U1"])
    if reference_l2 == 0.0:
        return 0.0
    return None if relative is None else relative * reference_l2 * reference_l2


def clip_decision_metric(pair: dict[str, Any], metric: str) -> float | None:
    """Derive directional and disagreement rates from paired token decisions."""

    events = pair.get("effects", {}).get("paired_semantic_events", {})
    reference = events.get("reference", {}).get("clip_decisions")
    candidate = events.get("candidate", {}).get("clip_decisions")
    if not isinstance(reference, list) or not isinstance(candidate, list):
        return None
    if len(reference) != len(candidate):
        return None
    reference_flat: list[bool] = []
    candidate_flat: list[bool] = []
    for reference_row, candidate_row in zip(reference, candidate, strict=True):
        if not isinstance(reference_row, list) or not isinstance(candidate_row, list):
            return None
        if len(reference_row) != len(candidate_row):
            return None
        if any(not isinstance(value, bool) for value in reference_row + candidate_row):
            return None
        reference_flat.extend(reference_row)
        candidate_flat.extend(candidate_row)
    if not reference_flat:
        return None
    off_to_on = sum(
        (not reference_value) and candidate_value
        for reference_value, candidate_value in zip(
            reference_flat, candidate_flat, strict=True
        )
    )
    on_to_off = sum(
        reference_value and (not candidate_value)
        for reference_value, candidate_value in zip(
            reference_flat, candidate_flat, strict=True
        )
    )
    denominator = len(reference_flat)
    values = {
        "directional_rate_shift": (off_to_on - on_to_off) / denominator,
        "disagreement_rate": (off_to_on + on_to_off) / denominator,
        "off_to_on_rate": off_to_on / denominator,
        "on_to_off_rate": on_to_off / denominator,
        "exposure_count": denominator,
    }
    return float(values[metric])


def binary_event_disagreement(pair: dict[str, Any], difference_name: str) -> float | None:
    events = pair.get("effects", {}).get("paired_semantic_events", {})
    difference = events.get(difference_name)
    if not isinstance(difference, (int, float)):
        return None
    return float(abs(difference))


def endpoint_extractors() -> dict[
    str,
    tuple[
        str,
        Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], float | None],
    ],
]:
    return {
        "training_loss_shift": (
            "SIGNED_SCALAR_IMPLEMENTATION_SHIFT",
            lambda pair, ref, cand: float(cand["outcomes"]["propagation_ledgers"]["training_loss"])
            - float(ref["outcomes"]["propagation_ledgers"]["training_loss"]),
        ),
        "U1_reference_aligned_shift": (
            "SIGNED_UPDATE_GEOMETRY_ENDPOINT",
            lambda pair, ref, cand: measured_endpoint(pair["effects"]["U1"]),
        ),
        "U1_reference_aligned_dot": (
            "SIGNED_UPDATE_ALIGNED_FORCING_ENDPOINT",
            reference_aligned_dot,
        ),
        "T1a_heldout_grpo_shift": (
            "SIGNED_STATE_ADAPTIVE_TASK_ENDPOINT",
            lambda pair, ref, cand: measured_endpoint(pair["effects"]["T1a_shift"]),
        ),
        "T1b_correct_answer_nll_shift": (
            "SIGNED_FIXED_BANK_TASK_ENDPOINT",
            lambda pair, ref, cand: measured_endpoint(pair["effects"]["T1b_shift"]),
        ),
        "U2_paired_delta_l2": (
            "NONNEGATIVE_MAGNITUDE_PROFILE_NOT_B",
            lambda pair, ref, cand: measured_endpoint(pair["effects"]["U2_delta"]),
        ),
        "clip_count_shift": (
            "SIGNED_EVENT_COUNT_SHIFT",
            lambda pair, ref, cand: float(
                pair["effects"]["paired_semantic_events"]["clip_count_difference"]
            ),
        ),
        "clip_directional_rate_shift": (
            "SIGNED_EVENT_PROBABILITY_SHIFT",
            lambda pair, ref, cand: clip_decision_metric(
                pair, "directional_rate_shift"
            ),
        ),
        "clip_disagreement_rate": (
            "NONNEGATIVE_EVENT_DISAGREEMENT_PROFILE_NOT_B",
            lambda pair, ref, cand: clip_decision_metric(pair, "disagreement_rate"),
        ),
        "clip_off_to_on_rate": (
            "ONE_SIDED_EVENT_TRANSITION_RATE",
            lambda pair, ref, cand: clip_decision_metric(pair, "off_to_on_rate"),
        ),
        "clip_on_to_off_rate": (
            "ONE_SIDED_EVENT_TRANSITION_RATE",
            lambda pair, ref, cand: clip_decision_metric(pair, "on_to_off_rate"),
        ),
        "clip_decision_exposure_count": (
            "DENOMINATOR_PROFILE_NOT_B",
            lambda pair, ref, cand: clip_decision_metric(pair, "exposure_count"),
        ),
        "gradient_clip_trigger_shift": (
            "SIGNED_EVENT_PROBABILITY_SHIFT",
            lambda pair, ref, cand: float(
                pair["effects"]["paired_semantic_events"][
                    "gradient_clip_trigger_difference"
                ]
            ),
        ),
        "gradient_clip_trigger_disagreement": (
            "NONNEGATIVE_EVENT_DISAGREEMENT_PROFILE_NOT_B",
            lambda pair, ref, cand: binary_event_disagreement(
                pair, "gradient_clip_trigger_difference"
            ),
        ),
        "optimizer_skip_shift": (
            "SIGNED_EVENT_PROBABILITY_SHIFT",
            lambda pair, ref, cand: float(
                pair["effects"]["paired_semantic_events"]["optimizer_skip_difference"]
            ),
        ),
        "optimizer_skip_disagreement": (
            "NONNEGATIVE_EVENT_DISAGREEMENT_PROFILE_NOT_B",
            lambda pair, ref, cand: binary_event_disagreement(
                pair, "optimizer_skip_difference"
            ),
        ),
    }


def collect_endpoint(
    state_bundles: list[tuple[dict[str, Any], dict[str, Any]]],
    extractor: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], float | None],
) -> tuple[list[EffectRecord], Counter[str], list[dict[str, Any]]]:
    records: list[EffectRecord] = []
    availability: Counter[str] = Counter()
    unavailable_records: list[dict[str, Any]] = []
    for target, bundle in state_bundles:
        for pair in bundle["paired_effect_records"]:
            reference, candidate = linked_arms(bundle, pair)
            value = extractor(pair, reference, candidate)
            if value is None:
                availability["UNAVAILABLE"] += 1
                unavailable_records.append(dict(pair.get("identity", {})))
                continue
            availability["MEASURED"] += 1
            identity = pair["identity"]
            records.append(
                EffectRecord(
                    trajectory_id=str(identity["trajectory_id"]),
                    phase=str(identity["phase"]),
                    state_id=str(identity["state_id"]),
                    repeat_id=int(identity["repeat_id"]),
                    effect=float(value),
                )
            )
    return records, availability, unavailable_records


def load_complete_state_bundles(
    plan: dict[str, Any], results_root: Path
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
    list[str],
]:
    """Load only records that exactly belong to the frozen capture plan.

    The returned errors are part of the fail-closed construction verdict.  In
    particular, this routine never silently shrinks the target state bank.
    """

    targets = plan.get("targets", [])
    plan_identity = plan.get("identity", {})
    state_bundles: list[tuple[dict[str, Any], dict[str, Any]]] = []
    state_evidence: list[dict[str, Any]] = []
    errors: list[str] = []
    if not isinstance(targets, list) or not targets:
        return [], [], ["plan.targets must be a non-empty list"]
    target_ids = [target.get("state_id") for target in targets]
    if None in target_ids or len(target_ids) != len(set(target_ids)):
        return [], [], ["plan state_id values must be present and unique"]

    for target in targets:
        state_root = results_root / f"step{int(target['optimizer_step']):03d}"
        bundle_path = state_root / "record_bundle.json"
        validation_path = state_root / "record_validation.json"
        if not bundle_path.is_file() or not validation_path.is_file():
            errors.append(f"missing complete record for {target['state_id']}")
            continue
        validation = load_json(validation_path)
        bundle = load_json(bundle_path)
        bundle_sha = sha256_file(bundle_path)
        fresh_validation = validate_record_bundle(bundle, verify_artifacts=False)
        pairs = bundle.get("paired_effect_records", [])
        expected_identity = {
            **plan_identity,
            **{
                key: target[key]
                for key in (
                    "state_id",
                    "optimizer_step",
                    "phase",
                    "eligible_step_population",
                )
            },
        }
        identity_fields = tuple(expected_identity)
        repeat_ids = {
            int(pair.get("identity", {}).get("repeat_id", -1)) for pair in pairs
        }
        evidence_valid = all(
            (
                validation.get("valid") is True,
                validation.get("population_eligible") is True,
                validation.get("bundle", {}).get("sha256") == bundle_sha,
                fresh_validation.get("valid") is True,
                bundle.get("scope", {}).get("population_eligible") is True,
                len(pairs) == 2,
                repeat_ids == {1, 2},
                all(
                    all(pair.get("identity", {}).get(key) == value for key, value in expected_identity.items())
                    for pair in pairs
                ),
                all(
                    all(key in pair.get("identity", {}) for key in identity_fields)
                    for pair in pairs
                ),
            )
        )
        if not evidence_valid:
            errors.append(f"invalid record evidence for {target['state_id']}")
            continue
        state_bundles.append((target, bundle))
        state_evidence.append(
            {
                "state_id": target["state_id"],
                "optimizer_step": int(target["optimizer_step"]),
                "phase": target["phase"],
                "bundle": str(bundle_path),
                "bundle_sha256": bundle_sha,
                "validation": str(validation_path),
                "validation_sha256": sha256_file(validation_path),
            }
        )
    return state_bundles, state_evidence, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    results_root = Path(args.results_root).resolve()
    plan = load_json(plan_path)
    targets = plan["targets"]
    state_bundles, state_evidence, errors = load_complete_state_bundles(
        plan, results_root
    )

    phase_counts = Counter(target["phase"] for target, _ in state_bundles)
    complete = (
        not errors
        and len(state_bundles) == len(targets)
        and phase_counts == Counter(target["phase"] for target in targets)
    )

    endpoint_results: dict[str, Any] = {}
    expected_effect_records = len(targets) * 2
    for endpoint_name, (endpoint_class, extractor) in endpoint_extractors().items():
        records, availability, unavailable_records = collect_endpoint(
            state_bundles, extractor
        )
        endpoint_complete = complete and len(records) == expected_effect_records
        if endpoint_complete:
            estimate = estimate_scalar_population(
                records,
                required_phases=("early", "middle", "late"),
                min_confirmation_trajectories=8,
            )
            endpoint_results[endpoint_name] = {
                "status": "COMPLETE_ONE_TRAJECTORY_DESCRIPTION",
                "endpoint_class": endpoint_class,
                "availability": dict(availability),
                "unavailable_records": unavailable_records,
                "trajectory_mean": estimate["trajectory_rows"][0]["mean_effect"],
                "phase_means": estimate["trajectory_rows"][0]["phase_effects"],
                "profile": estimate,
                "population_B_claim_allowed": False,
            }
        else:
            endpoint_results[endpoint_name] = {
                "status": "UNAVAILABLE_FULL_FROZEN_POPULATION",
                "endpoint_class": endpoint_class,
                "availability": dict(availability),
                "unavailable_records": unavailable_records,
                "missing_effect_records": expected_effect_records - len(records),
                "population_B_claim_allowed": False,
                "reason": "states with undefined/uninstantiated endpoints are retained; no complete-case deletion",
            }

    valid = complete
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": "VALID_COMPLETE_TRAJECTORY_DESCRIPTION" if valid else "INVALID_INCOMPLETE_TRAJECTORY",
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "results_root": str(results_root),
        "construction": {
            "complete": complete,
            "errors": errors,
                "states": len(state_bundles),
                "expected_states": len(targets),
            "phase_counts": dict(phase_counts),
            "repeats_per_state": 2,
            "weighting_contract": {
                "trajectory": "one trajectory description only",
                "phase": "equal phase",
                "state_within_phase": "equal state",
                "clip_rate_within_state": "normalize by that state's recorded completion-token clip-decision positions before state averaging",
                "clip_rate_is_eligibility_conditioned": False,
                "eligibility_conditioned_clip_rate_location": "separate reference-anchored boundary-conditional family",
                "clip_rate_is_exposure_pooled": False,
            },
            "runtime_variability_scope": {
                "conditioned_fixed_observed_realization_fields": [
                    "compiler_config_digest",
                    "graph_family_digest"
                ],
                "unobserved_realization_sources_not_separately_identified": [
                    "generated-kernel identity",
                    "autotuning variant identity"
                ],
                "interpretation": "N is conditional on observed config/graph identity; any residual uninstrumented realization variation remains mixed into N"
            },
            "task_endpoint_randomness_scope": {
                "T1a_bank_sampling_variance": "UNIDENTIFIED_ONE_FROZEN_BANK_PER_STATE",
                "T1a_H_includes": "state variation plus state-adaptive frozen-bank content variation",
                "T1b_scope": "fixed correct-answer-bank functional",
                "evaluator_repeats": "nested within transition repeats and not top-level samples"
            },
            "trajectory_count": len({bundle["paired_effect_records"][0]["identity"]["trajectory_id"] for _, bundle in state_bundles}) if state_bundles else 0,
        },
        "endpoints": endpoint_results,
        "state_evidence": state_evidence,
        "interpretation": (
            "Each endpoint is a complete calibration-0 trajectory description. One trajectory cannot identify between-trajectory variance or support population B."
            if valid
            else "Aggregation failed closed; partial states are not used to redefine the frozen population."
        ),
        "nonclaims": [
            "trajectory mean is not confirmed population B",
            "U2 delta L2 is a magnitude distribution, not a signed bias",
            "event disagreement is a nonnegative semantic-impact profile, not a signed bias",
            "missing endpoint states are not silently dropped",
            "event count shift is endpoint-specific and not correctness",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "construction": payload["construction"]}, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
