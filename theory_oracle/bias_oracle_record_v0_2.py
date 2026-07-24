#!/usr/bin/env python
"""Fail-closed validation for two-level Bias Oracle arm/pair record bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "forkcert.bias-oracle-record-bundle.v0.2"
ARM_LABELS = {"reference", "candidate"}
ENDPOINT_STATUSES = {"MEASURED", "UNINSTANTIATED", "UNDEFINED"}

ARM_IDENTITY_FIELDS = (
    "query_id",
    "trajectory_id",
    "trajectory_anchor",
    "trajectory_seed",
    "data_slice_id",
    "phase",
    "eligible_step_population",
    "state_selection_prng_seed",
    "state_id",
    "optimizer_step",
    "repeat_id",
    "arm",
)
PAIR_IDENTITY_FIELDS = ARM_IDENTITY_FIELDS[:-1]
PRE_STATE_FIELDS = (
    "model_digest",
    "buffer_digest",
    "optimizer_digest",
    "scheduler_digest",
    "scaler_digest",
    "rng_digest",
    "minibatch_digest",
)
REALIZATION_FIELDS = ("compiler_config_digest", "graph_family_digest")
FORBIDDEN_ARM_PAIRED_FIELDS = {"U1", "U2_delta", "T1a_shift", "T1b_shift"}


def canonical_digest(value: dict[str, Any]) -> str:
    content = {key: item for key, item in value.items() if key != "record_digest"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def attach_record_digest(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["record_digest"] = canonical_digest(result)
    return result


def endpoint(status: str, value: float | None = None, **metadata: Any) -> dict[str, Any]:
    if status not in ENDPOINT_STATUSES:
        raise ValueError(f"unsupported endpoint status: {status}")
    if status == "MEASURED":
        if value is None or not math.isfinite(float(value)):
            raise ValueError("MEASURED endpoint requires a finite value")
        result: dict[str, Any] = {"status": status, "value": float(value)}
    else:
        if value is not None:
            raise ValueError(f"{status} endpoint cannot carry a numeric value")
        result = {"status": status, "value": None}
    result.update(metadata)
    return result


def _missing(mapping: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if field not in mapping]


def _identity_key(identity: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(identity.get(field) for field in fields)


def _endpoint_valid(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("status") not in ENDPOINT_STATUSES:
        return False
    if value["status"] == "MEASURED":
        return isinstance(value.get("value"), (int, float)) and math.isfinite(float(value["value"]))
    return value.get("value") is None


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def _artifact_valid(artifact: Any) -> bool:
    if not isinstance(artifact, dict):
        return False
    path = artifact.get("path")
    expected = artifact.get("sha256")
    if not isinstance(path, str) or not isinstance(expected, str):
        return False
    target = Path(path)
    if not target.is_file():
        return False
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def _validate_semantic_events(
    value: Any, prefix: str, errors: list[str]
) -> tuple[int, ...] | None:
    if not isinstance(value, dict):
        errors.append(f"{prefix} is not an object")
        return None
    required = (
        "clip_count",
        "clip_decisions",
        "gradient_clip_triggered",
        "optimizer_step_skipped",
    )
    missing = [field for field in required if field not in value]
    if missing:
        errors.append(f"{prefix} missing {missing}")
        return None
    clip_count = value.get("clip_count")
    decisions = value.get("clip_decisions")
    if isinstance(clip_count, bool) or not isinstance(clip_count, int) or clip_count < 0:
        errors.append(f"{prefix}.clip_count is invalid")
    if not isinstance(decisions, list) or not decisions:
        errors.append(f"{prefix}.clip_decisions must be a non-empty nested list")
        return None
    shape: list[int] = []
    observed_count = 0
    for row in decisions:
        if not isinstance(row, list) or not row or any(
            not isinstance(item, bool) for item in row
        ):
            errors.append(f"{prefix}.clip_decisions contains an invalid row")
            return None
        shape.append(len(row))
        observed_count += sum(row)
    if isinstance(clip_count, int) and not isinstance(clip_count, bool):
        if observed_count != clip_count:
            errors.append(f"{prefix}.clip_count does not equal clip_decisions")
    for field in ("gradient_clip_triggered", "optimizer_step_skipped"):
        if not isinstance(value.get(field), bool):
            errors.append(f"{prefix}.{field} is not boolean")
    return tuple(shape)


def _validate_paired_semantic_events(
    value: Any,
    reference_events: Any,
    candidate_events: Any,
    prefix: str,
    errors: list[str],
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix} is not an object")
        return
    if value.get("reference") != reference_events:
        errors.append(f"{prefix}.reference does not equal linked reference arm")
    if value.get("candidate") != candidate_events:
        errors.append(f"{prefix}.candidate does not equal linked candidate arm")
    if not isinstance(reference_events, dict) or not isinstance(candidate_events, dict):
        return
    reference_shape = _validate_semantic_events(
        reference_events, f"{prefix}.linked_reference", errors
    )
    candidate_shape = _validate_semantic_events(
        candidate_events, f"{prefix}.linked_candidate", errors
    )
    if reference_shape is not None and candidate_shape is not None:
        if reference_shape != candidate_shape:
            errors.append(f"{prefix} linked clip decision shapes differ")
    expected = {
        "clip_count_difference": candidate_events.get("clip_count", 0)
        - reference_events.get("clip_count", 0),
        "gradient_clip_trigger_difference": int(
            candidate_events.get("gradient_clip_triggered", False)
        )
        - int(reference_events.get("gradient_clip_triggered", False)),
        "optimizer_skip_difference": int(
            candidate_events.get("optimizer_step_skipped", False)
        )
        - int(reference_events.get("optimizer_step_skipped", False)),
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(f"{prefix}.{field} does not equal candidate-reference")


def validate_record_bundle(
    bundle: dict[str, Any],
    *,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if bundle.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported bundle schema_version")
    scope = bundle.get("scope")
    if not isinstance(scope, dict) or not isinstance(scope.get("population_eligible"), bool):
        errors.append("scope.population_eligible must be boolean")
        population_eligible = False
    else:
        population_eligible = bool(scope["population_eligible"])

    arms = bundle.get("arm_records")
    pairs = bundle.get("paired_effect_records")
    if not isinstance(arms, list) or not arms:
        errors.append("arm_records must be a non-empty list")
        arms = []
    if not isinstance(pairs, list) or not pairs:
        errors.append("paired_effect_records must be a non-empty list")
        pairs = []

    arm_keys: list[tuple[Any, ...]] = []
    arm_by_digest: dict[str, dict[str, Any]] = {}
    repeat_realizations: dict[tuple[Any, ...], set[tuple[Any, ...]]] = defaultdict(set)
    for index, record in enumerate(arms):
        prefix = f"arm_records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} is not an object")
            continue
        identity = record.get("identity", {})
        pre_state = record.get("pre_state", {})
        realization = record.get("realization", {})
        outcomes = record.get("outcomes", {})
        for label, mapping, fields in (
            ("identity", identity, ARM_IDENTITY_FIELDS),
            ("pre_state", pre_state, PRE_STATE_FIELDS),
            ("realization", realization, REALIZATION_FIELDS),
        ):
            if not isinstance(mapping, dict):
                errors.append(f"{prefix}.{label} is not an object")
                continue
            missing = _missing(mapping, fields)
            if missing:
                errors.append(f"{prefix}.{label} missing {missing}")
        if identity.get("arm") not in ARM_LABELS:
            errors.append(f"{prefix}.identity.arm is invalid")
        arm_keys.append(_identity_key(identity, ARM_IDENTITY_FIELDS))
        repeat_realization_key = tuple(
            identity.get(field)
            for field in ARM_IDENTITY_FIELDS
            if field != "repeat_id"
        )
        repeat_realizations[repeat_realization_key].add(
            tuple(realization.get(field) for field in REALIZATION_FIELDS)
        )
        if not isinstance(outcomes, dict):
            errors.append(f"{prefix}.outcomes is not an object")
            continue
        forbidden = sorted(FORBIDDEN_ARM_PAIRED_FIELDS.intersection(outcomes))
        if forbidden:
            errors.append(f"{prefix}.outcomes contains paired fields {forbidden}")
        required_outcomes = (
            "parameter_update_artifact",
            "T1a_arm_loss",
            "T1b_arm_nll",
            "propagation_ledgers",
            "semantic_events",
            "next_state_digests",
        )
        missing_outcomes = _missing(outcomes, required_outcomes)
        if missing_outcomes:
            errors.append(f"{prefix}.outcomes missing {missing_outcomes}")
        for name in ("T1a_arm_loss", "T1b_arm_nll"):
            if name in outcomes and not _endpoint_valid(outcomes[name]):
                errors.append(f"{prefix}.{name} is not a valid endpoint object")
        if "semantic_events" in outcomes:
            _validate_semantic_events(
                outcomes["semantic_events"], f"{prefix}.semantic_events", errors
            )
        if verify_artifacts and "parameter_update_artifact" in outcomes:
            if not _artifact_valid(outcomes["parameter_update_artifact"]):
                errors.append(f"{prefix}.parameter_update_artifact identity failed")
        observed_digest = record.get("record_digest")
        expected_digest = canonical_digest(record)
        if observed_digest != expected_digest:
            errors.append(f"{prefix}.record_digest mismatch")
        elif observed_digest in arm_by_digest:
            errors.append(f"{prefix}.record_digest duplicates another arm record")
        else:
            arm_by_digest[observed_digest] = record

    duplicate_arm_keys = [key for key, count in Counter(arm_keys).items() if count > 1]
    if duplicate_arm_keys:
        errors.append(f"duplicate arm keys: {duplicate_arm_keys[:3]}")
    drifted_repeat_realizations = {
        key: sorted(values, key=str)
        for key, values in repeat_realizations.items()
        if len(values) != 1
    }
    if drifted_repeat_realizations:
        errors.append(
            "same-state repeats changed compiler/graph realization: "
            f"{list(drifted_repeat_realizations.items())[:3]}"
        )

    pair_keys: list[tuple[Any, ...]] = []
    linked_arm_counts: Counter[str] = Counter()
    repeat_sets: dict[tuple[Any, ...], set[Any]] = defaultdict(set)
    for index, record in enumerate(pairs):
        prefix = f"paired_effect_records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} is not an object")
            continue
        identity = record.get("identity", {})
        links = record.get("links", {})
        effects = record.get("effects", {})
        if not isinstance(identity, dict):
            errors.append(f"{prefix}.identity is not an object")
            identity = {}
        missing_identity = _missing(identity, PAIR_IDENTITY_FIELDS)
        if missing_identity:
            errors.append(f"{prefix}.identity missing {missing_identity}")
        pair_key = _identity_key(identity, PAIR_IDENTITY_FIELDS)
        pair_keys.append(pair_key)
        state_key = _identity_key(identity, PAIR_IDENTITY_FIELDS[:-1])
        repeat_sets[state_key].add(identity.get("repeat_id"))
        if not isinstance(links, dict):
            errors.append(f"{prefix}.links is not an object")
            links = {}
        required_links = (
            "reference_arm_record_digest",
            "candidate_arm_record_digest",
            "coupling_protocol_digest",
        )
        missing_links = _missing(links, required_links)
        if missing_links:
            errors.append(f"{prefix}.links missing {missing_links}")
        reference_digest = links.get("reference_arm_record_digest")
        candidate_digest = links.get("candidate_arm_record_digest")
        reference = arm_by_digest.get(reference_digest)
        candidate = arm_by_digest.get(candidate_digest)
        if reference is None:
            errors.append(f"{prefix} reference arm link is unresolved")
        if candidate is None:
            errors.append(f"{prefix} candidate arm link is unresolved")
        if reference is not None and candidate is not None:
            linked_arm_counts[reference_digest] += 1
            linked_arm_counts[candidate_digest] += 1
            ref_identity = reference["identity"]
            cand_identity = candidate["identity"]
            if ref_identity.get("arm") != "reference" or cand_identity.get("arm") != "candidate":
                errors.append(f"{prefix} arm links have reversed or invalid labels")
            ref_core = {key: ref_identity.get(key) for key in PAIR_IDENTITY_FIELDS}
            cand_core = {key: cand_identity.get(key) for key in PAIR_IDENTITY_FIELDS}
            pair_core = {key: identity.get(key) for key in PAIR_IDENTITY_FIELDS}
            if ref_core != cand_core or ref_core != pair_core:
                errors.append(f"{prefix} linked identity mismatch")
            if reference.get("pre_state") != candidate.get("pre_state"):
                errors.append(f"{prefix} linked pre-state mismatch")

        if not isinstance(effects, dict):
            errors.append(f"{prefix}.effects is not an object")
            effects = {}
        required_effects = (
            "U1",
            "U2_delta",
            "T1a_shift",
            "T1b_shift",
            "paired_semantic_events",
            "paired_next_state_digests",
        )
        missing_effects = _missing(effects, required_effects)
        if missing_effects:
            errors.append(f"{prefix}.effects missing {missing_effects}")
        for name in ("U1", "U2_delta", "T1a_shift", "T1b_shift"):
            if name in effects and not _endpoint_valid(effects[name]):
                errors.append(f"{prefix}.{name} is not a valid endpoint object")

        if reference is not None and candidate is not None:
            for arm_name, effect_name in (
                ("T1a_arm_loss", "T1a_shift"),
                ("T1b_arm_nll", "T1b_shift"),
            ):
                ref_endpoint = reference["outcomes"].get(arm_name, {})
                cand_endpoint = candidate["outcomes"].get(arm_name, {})
                pair_endpoint = effects.get(effect_name, {})
                if ref_endpoint.get("status") == cand_endpoint.get("status") == "MEASURED":
                    expected = float(cand_endpoint["value"]) - float(ref_endpoint["value"])
                    if pair_endpoint.get("status") != "MEASURED" or not _close(
                        float(pair_endpoint.get("value", math.nan)), expected
                    ):
                        errors.append(f"{prefix}.{effect_name} does not equal candidate-reference")
                elif pair_endpoint.get("status") == "MEASURED":
                    errors.append(f"{prefix}.{effect_name} measured while an arm scalar is unavailable")
            _validate_paired_semantic_events(
                effects.get("paired_semantic_events"),
                reference["outcomes"].get("semantic_events"),
                candidate["outcomes"].get("semantic_events"),
                f"{prefix}.paired_semantic_events",
                errors,
            )

        if population_eligible:
            if effects.get("U1", {}).get("status") == "UNINSTANTIATED":
                errors.append(f"{prefix}.U1 cannot be uninstantiated for a population record")
            if effects.get("U2_delta", {}).get("status") != "MEASURED":
                errors.append(f"{prefix}.U2_delta must be measured for a population record")
            if verify_artifacts and effects.get("U2_delta", {}).get("status") == "MEASURED":
                if not _artifact_valid(effects["U2_delta"].get("artifact")):
                    errors.append(f"{prefix}.U2_delta artifact identity failed")
        observed_digest = record.get("record_digest")
        if observed_digest != canonical_digest(record):
            errors.append(f"{prefix}.record_digest mismatch")

    duplicate_pair_keys = [key for key, count in Counter(pair_keys).items() if count > 1]
    if duplicate_pair_keys:
        errors.append(f"duplicate paired keys: {duplicate_pair_keys[:3]}")
    unlinked_or_reused = {
        digest: linked_arm_counts.get(digest, 0)
        for digest in arm_by_digest
        if linked_arm_counts.get(digest, 0) != 1
    }
    if unlinked_or_reused:
        errors.append(f"every arm must link to exactly one pair: {unlinked_or_reused}")
    observed_repeat_sets = {tuple(sorted(values)) for values in repeat_sets.values()}
    if len(observed_repeat_sets) > 1:
        errors.append(f"states have unbalanced paired repeat sets: {sorted(observed_repeat_sets)}")
    if population_eligible and observed_repeat_sets and len(next(iter(observed_repeat_sets))) < 2:
        errors.append("population records require at least two paired repeats per state")

    return {
        "schema_version": "forkcert.bias-oracle-record-validation.v0.2",
        "valid": not errors,
        "verdict": "VALID" if not errors else "INVALID",
        "errors": errors,
        "counts": {
            "arm_records": len(arms),
            "paired_effect_records": len(pairs),
            "states": len(repeat_sets),
            "repeats_per_state": (
                list(next(iter(observed_repeat_sets))) if len(observed_repeat_sets) == 1 else None
            ),
        },
        "population_eligible": population_eligible,
        "interpretation": (
            "Record construction is internally valid; population eligibility is a separate scope field."
            if not errors
            else "Record construction failed closed; no endpoint may enter B/H/N/U estimation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-artifact-hashes", action="store_true")
    args = parser.parse_args()
    bundle_path = Path(args.bundle).resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    report = validate_record_bundle(bundle, verify_artifacts=not args.skip_artifact_hashes)
    report["bundle"] = {
        "path": str(bundle_path),
        "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
