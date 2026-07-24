#!/usr/bin/env python
"""Freeze a calibration-learned U2 direction before confirmation collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "forkcert.qwen3-u2-frozen-direction.v0.1"
SPEC_VERSION = "forkcert.qwen3-u2-direction-freeze-spec.v0.1"
DIAGNOSTIC_VERSION = "forkcert.qwen3-u2-direction-stability.v0.1"
MULTI_VERSION = "forkcert.qwen3-calibration-u2-multi-trajectory.v0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def freeze_direction(
    multi: dict[str, Any],
    multi_path: Path,
    diagnostic: dict[str, Any],
    diagnostic_path: Path,
    spec: dict[str, Any],
    spec_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if spec.get("schema_version") != SPEC_VERSION:
        errors.append("unsupported direction freeze spec")
    if spec.get("status") != "FROZEN_BEFORE_DIRECTION_DIAGNOSTIC_REVIEW":
        errors.append("direction thresholds were not frozen before diagnostic review")
    if (
        multi.get("schema_version") != MULTI_VERSION
        or multi.get("valid") is not True
        or multi.get("verdict")
        != "VALID_COMPLETE_FOUR_TRAJECTORY_VECTOR_CALIBRATION"
    ):
        errors.append("multi-trajectory U2 calibration is invalid")
    if (
        diagnostic.get("schema_version") != DIAGNOSTIC_VERSION
        or diagnostic.get("valid") is not True
        or diagnostic.get("verdict")
        != "VALID_U2_DIRECTION_STABILITY_CALIBRATION_DIAGNOSTIC"
    ):
        errors.append("U2 direction stability diagnostic is invalid")
    diagnostic_multi = diagnostic.get("inputs", {}).get("multi_summary", {})
    if diagnostic_multi.get("sha256") != sha256_file(multi_path):
        errors.append("direction diagnostic is not bound to this multi summary")
    try:
        measurement_floor = finite_number(
            spec.get("vector_measurement_floor_l2"), "vector measurement floor"
        )
        minimum_cosine = finite_number(
            spec.get("minimum_full_vs_leave_one_out_cosine"),
            "minimum leave-one-out cosine",
        )
        desired_half_width = finite_number(
            spec.get("desired_projection_half_width"),
            "desired projection half-width",
        )
        variance_floor_sd = finite_number(
            spec.get("projection_variance_floor_sd"),
            "projection variance floor",
        )
        shift_floor = finite_number(
            spec.get("projection_shift_existence_floor"),
            "projection shift-existence floor",
        )
    except ValueError as error:
        errors.append(str(error))
        measurement_floor = minimum_cosine = desired_half_width = math.nan
        variance_floor_sd = shift_floor = math.nan
    if math.isfinite(measurement_floor) and measurement_floor < 0:
        errors.append("vector measurement floor must be non-negative")
    if math.isfinite(minimum_cosine) and not -1 <= minimum_cosine <= 1:
        errors.append("minimum leave-one-out cosine must lie in [-1, 1]")
    if math.isfinite(desired_half_width) and desired_half_width <= 0:
        errors.append("desired projection half-width must be positive")
    if math.isfinite(variance_floor_sd) and variance_floor_sd < 0:
        errors.append("projection variance floor must be non-negative")
    if math.isfinite(shift_floor) and shift_floor < 0:
        errors.append("projection shift-existence floor must be non-negative")
    if not isinstance(spec.get("threshold_source"), str) or not spec.get(
        "threshold_source"
    ):
        errors.append("threshold source must be declared")
    if spec.get("claim_role") not in {
        "MEMBER_OF_JOINT_CONFIRMATION_FAMILY",
        "NAMED_SECONDARY_NO_JOINT_CLAIM",
    }:
        errors.append("U2 claim role is uninstantiated")
    if errors:
        raise ValueError("; ".join(errors))

    diagnostics = diagnostic["diagnostics"]
    observed_norm = finite_number(
        diagnostics.get("full_calibration_mean_norm"), "observed mean-field norm"
    )
    observed_cosine = diagnostics.get("minimum_full_vs_leave_one_out_cosine")
    observed_cosine = (
        finite_number(observed_cosine, "observed minimum leave-one-out cosine")
        if observed_cosine is not None
        else None
    )
    pass_norm = observed_norm > measurement_floor
    pass_stability = observed_cosine is not None and observed_cosine >= minimum_cosine
    if not pass_norm or not pass_stability:
        reason = []
        if not pass_norm:
            reason.append("mean field does not exceed independent vector floor")
        if not pass_stability:
            reason.append("leave-one-out direction stability threshold not met")
        return {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "verdict": "UNINSTANTIATED_DIRECTION",
            "reason": reason,
            "observed": {
                "full_calibration_mean_norm": observed_norm,
                "minimum_full_vs_leave_one_out_cosine": observed_cosine,
            },
            "thresholds": {
                "vector_measurement_floor_l2": measurement_floor,
                "minimum_full_vs_leave_one_out_cosine": minimum_cosine,
            },
            "population_B_claim_allowed": False,
        }

    shards = multi.get("calibration_mean_field", {}).get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("multi summary lacks calibration mean-field shards")
    observed_keys: set[str] = set()
    frozen_shards: list[dict[str, Any]] = []
    for artifact in shards:
        path = Path(artifact.get("path", "")).resolve()
        expected = artifact.get("sha256")
        key = artifact.get("tensor_key")
        if not isinstance(expected, str) or not isinstance(key, str):
            raise ValueError("direction shard lacks hash or tensor key")
        if key in observed_keys:
            raise ValueError("duplicate direction tensor key")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"direction shard identity failed: {key}")
        observed_keys.add(key)
        frozen_shards.append(
            {"path": str(path), "sha256": expected, "tensor_key": key}
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "verdict": "VALID_FROZEN_U2_CALIBRATION_DIRECTION",
        "status": "FROZEN_BEFORE_CONFIRMATION",
        "endpoint_name": "U2_calibration_direction_shift",
        "direction": {
            "role": "unit direction of the complete four-trajectory calibration mean candidate-minus-reference update field",
            "normalization_l2": observed_norm,
            "shards": frozen_shards,
        },
        "stability": {
            "minimum_full_vs_leave_one_out_cosine": observed_cosine,
            "threshold": minimum_cosine,
            "crossfit_projection_sample_variance": diagnostics.get(
                "crossfit_projection_sample_variance"
            ),
            "crossfit_projections": [
                row["held_out_projection_on_leave_one_out_direction"]
                for row in diagnostics["leave_one_out_rows"]
            ],
        },
        "precision_contract": {
            "desired_projection_half_width": desired_half_width,
            "projection_variance_floor_sd": variance_floor_sd,
            "projection_shift_existence_floor": shift_floor,
            "claim_role": spec["claim_role"],
        },
        "thresholds": {
            "vector_measurement_floor_l2": measurement_floor,
            "minimum_full_vs_leave_one_out_cosine": minimum_cosine,
            "source": spec["threshold_source"],
        },
        "inputs": {
            "multi_summary": {
                "path": str(multi_path),
                "sha256": sha256_file(multi_path),
            },
            "stability_diagnostic": {
                "path": str(diagnostic_path),
                "sha256": sha256_file(diagnostic_path),
            },
            "freeze_spec": {
                "path": str(spec_path),
                "sha256": sha256_file(spec_path),
            },
        },
        "population_B_claim_allowed": False,
        "nonclaims": [
            "freezing a direction does not confirm a population update shift",
            "directional replication is not a full-vector omnibus claim",
            "the reference implementation is not asserted to be mathematical truth",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-summary", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    multi_path = Path(args.multi_summary).resolve()
    diagnostic_path = Path(args.diagnostic).resolve()
    spec_path = Path(args.spec).resolve()
    out = Path(args.out).resolve()
    try:
        result = freeze_direction(
            load_json(multi_path),
            multi_path,
            load_json(diagnostic_path),
            diagnostic_path,
            load_json(spec_path),
            spec_path,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "verdict": "INVALID_OR_UNINSTANTIATED_DIRECTION_FREEZE",
            "errors": [str(error)],
            "population_B_claim_allowed": False,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "valid": result["valid"]}, indent=2))
    if result.get("valid") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
