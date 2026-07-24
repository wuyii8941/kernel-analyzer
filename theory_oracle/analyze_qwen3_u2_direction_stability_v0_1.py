#!/usr/bin/env python
"""Diagnose calibration U2 direction stability with leave-one-trajectory-out projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "forkcert.qwen3-u2-direction-stability.v0.1"
EXPECTED_TRAJECTORIES = {f"calibration-{index}" for index in range(4)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two values are required")
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / (
        len(values) - 1
    )


def direction_diagnostics_from_gram(gram: list[list[float]]) -> dict[str, Any]:
    count = len(gram)
    if count < 3 or any(len(row) != count for row in gram):
        raise ValueError("Gram matrix must be square with at least three trajectories")
    for left in range(count):
        for right in range(count):
            value = float(gram[left][right])
            if not math.isfinite(value):
                raise ValueError("Gram matrix values must be finite")
            if not math.isclose(
                value,
                float(gram[right][left]),
                rel_tol=1e-10,
                abs_tol=1e-18,
            ):
                raise ValueError("Gram matrix must be symmetric")
        if gram[left][left] < 0:
            raise ValueError("Gram diagonal must be non-negative")

    full_norm_sq = math.fsum(
        float(gram[left][right])
        for left in range(count)
        for right in range(count)
    ) / (count**2)
    if full_norm_sq < -1e-16:
        raise ValueError("Gram matrix implies a negative full mean norm")
    full_norm = math.sqrt(max(0.0, full_norm_sq))
    crossfit: list[float | None] = []
    full_projections: list[float | None] = []
    loo_rows: list[dict[str, Any]] = []
    for held_out in range(count):
        kept = [index for index in range(count) if index != held_out]
        loo_norm_sq = math.fsum(
            float(gram[left][right]) for left in kept for right in kept
        ) / ((count - 1) ** 2)
        if loo_norm_sq < -1e-16:
            raise ValueError("Gram matrix implies a negative leave-one-out norm")
        loo_norm = math.sqrt(max(0.0, loo_norm_sq))
        held_dot_loo = math.fsum(
            float(gram[held_out][right]) for right in kept
        ) / (count - 1)
        projection = held_dot_loo / loo_norm if loo_norm else None
        crossfit.append(projection)
        held_dot_full = math.fsum(
            float(gram[held_out][right]) for right in range(count)
        ) / count
        full_projection = held_dot_full / full_norm if full_norm else None
        full_projections.append(full_projection)
        full_dot_loo = math.fsum(
            float(gram[left][right])
            for left in range(count)
            for right in kept
        ) / (count * (count - 1))
        cosine = (
            full_dot_loo / (full_norm * loo_norm)
            if full_norm and loo_norm
            else None
        )
        loo_rows.append(
            {
                "held_out_index": held_out,
                "leave_one_out_mean_norm": loo_norm,
                "held_out_projection_on_leave_one_out_direction": projection,
                "held_out_projection_on_full_in_sample_direction": full_projection,
                "cosine_full_direction_vs_leave_one_out_direction": cosine,
            }
        )

    finite_crossfit = [float(value) for value in crossfit if value is not None]
    return {
        "trajectory_count": count,
        "full_calibration_mean_norm": full_norm,
        "direction_defined_algebraically": full_norm > 0,
        "leave_one_out_rows": loo_rows,
        "crossfit_projection_mean": (
            math.fsum(finite_crossfit) / len(finite_crossfit)
            if len(finite_crossfit) == count
            else None
        ),
        "crossfit_projection_sample_variance": (
            sample_variance(finite_crossfit)
            if len(finite_crossfit) == count
            else None
        ),
        "minimum_full_vs_leave_one_out_cosine": (
            min(
                float(row["cosine_full_direction_vs_leave_one_out_direction"])
                for row in loo_rows
                if row["cosine_full_direction_vs_leave_one_out_direction"] is not None
            )
            if all(
                row["cosine_full_direction_vs_leave_one_out_direction"] is not None
                for row in loo_rows
            )
            else None
        ),
        "in_sample_full_direction_projections": full_projections,
        "planning_rule": "use cross-fitted projection dispersion plus an independent variance floor; never use in-sample full-direction dispersion alone",
    }


def validate_inputs(
    summary_paths: list[Path], multi_path: Path
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    errors: list[str] = []
    summaries = [load_json(path) for path in summary_paths]
    multi = load_json(multi_path)
    trajectory_ids = [
        row.get("construction", {}).get("trajectory_id") for row in summaries
    ]
    if set(trajectory_ids) != EXPECTED_TRAJECTORIES or len(trajectory_ids) != 4:
        errors.append("requires exactly calibration-0..calibration-3")
    for path, summary in zip(summary_paths, summaries, strict=True):
        if (
            summary.get("valid") is not True
            or summary.get("verdict")
            != "VALID_COMPLETE_ONE_TRAJECTORY_VECTOR_DESCRIPTION"
        ):
            errors.append(f"invalid trajectory vector summary: {path}")
        if summary.get("construction", {}).get("states") != 24:
            errors.append(f"trajectory vector summary lacks 24 states: {path}")
    if (
        multi.get("valid") is not True
        or multi.get("verdict")
        != "VALID_COMPLETE_FOUR_TRAJECTORY_VECTOR_CALIBRATION"
    ):
        errors.append("multi-trajectory vector calibration is invalid")
    evidence = {
        row.get("trajectory_id"): row.get("summary_sha256")
        for row in multi.get("input_evidence", [])
    }
    for trajectory_id, path in zip(trajectory_ids, summary_paths, strict=True):
        if evidence.get(trajectory_id) != sha256_file(path):
            errors.append(f"multi summary is not bound to {trajectory_id}")
    if summaries:
        key_sets = [
            {row["parameter_name"] for row in summary.get("parameter_rows", [])}
            for summary in summaries
        ]
        if not key_sets[0] or any(keys != key_sets[0] for keys in key_sets[1:]):
            errors.append("trajectory parameter sets differ")
    return errors, summaries, multi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-summary", action="append", required=True, metavar="JSON"
    )
    parser.add_argument("--multi-summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from safetensors import safe_open

    paths = [Path(value).resolve() for value in args.trajectory_summary]
    multi_path = Path(args.multi_summary).resolve()
    out = Path(args.out).resolve()
    errors, summaries, multi = validate_inputs(paths, multi_path)
    if errors:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "verdict": "INVALID_U2_DIRECTION_SOURCES",
            "errors": errors,
            "population_B_claim_allowed": False,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)

    order = sorted(
        range(4), key=lambda index: summaries[index]["construction"]["trajectory_id"]
    )
    summaries = [summaries[index] for index in order]
    paths = [paths[index] for index in order]
    row_maps = [
        {row["parameter_name"]: row for row in summary["parameter_rows"]}
        for summary in summaries
    ]
    keys = sorted(row_maps[0])
    gram = [[0.0 for _ in range(4)] for _ in range(4)]
    source_evidence: list[dict[str, Any]] = []
    handles: dict[str, Any] = {}
    for key in keys:
        tensors = []
        for index, rows in enumerate(row_maps):
            artifact = rows[key]["trajectory_mean_delta_artifact"]
            path = Path(artifact["path"]).resolve()
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"trajectory shard identity failed: {key}")
            if str(path) not in handles:
                handles[str(path)] = safe_open(path, framework="pt", device="cpu")
            handle = handles[str(path)]
            if sorted(handle.keys()) != [key]:
                raise ValueError(f"trajectory shard tensor key mismatch: {key}")
            tensors.append(handle.get_tensor(key).double())
            source_evidence.append(
                {
                    "trajectory_id": summaries[index]["construction"]["trajectory_id"],
                    "parameter_name": key,
                    "path": str(path),
                    "sha256": artifact["sha256"],
                }
            )
        for left in range(4):
            for right in range(left, 4):
                value = float((tensors[left] * tensors[right]).sum().item())
                gram[left][right] += value
                if left != right:
                    gram[right][left] += value

    diagnostics = direction_diagnostics_from_gram(gram)
    expected_norm = float(multi["calibration_mean_field"]["l2"])
    if not math.isclose(
        diagnostics["full_calibration_mean_norm"],
        expected_norm,
        rel_tol=1e-10,
        abs_tol=1e-18,
    ):
        raise ValueError("Gram-derived mean norm disagrees with multi summary")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "verdict": "VALID_U2_DIRECTION_STABILITY_CALIBRATION_DIAGNOSTIC",
        "trajectory_order": [
            summary["construction"]["trajectory_id"] for summary in summaries
        ],
        "gram_matrix": gram,
        "diagnostics": diagnostics,
        "inputs": {
            "trajectory_summaries": [
                {"path": str(path), "sha256": sha256_file(path)} for path in paths
            ],
            "multi_summary": {
                "path": str(multi_path),
                "sha256": sha256_file(multi_path),
            },
        },
        "source_shards": source_evidence,
        "direction_freeze_allowed": False,
        "direction_freeze_reason": "requires an independently frozen vector measurement floor and stability threshold; this diagnostic does not choose them",
        "population_B_claim_allowed": False,
        "nonclaims": [
            "the norm of the calibration mean field is not a population-B estimate",
            "cross-fitted projections are precision-planning inputs, not confirmation outcomes",
            "direction stability does not establish correctness or long-run impact",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"verdict": payload["verdict"], "diagnostics": diagnostics},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
