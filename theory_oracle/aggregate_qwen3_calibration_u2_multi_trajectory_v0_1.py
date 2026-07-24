#!/usr/bin/env python
"""Combine four signed trajectory-level U2 mean fields for calibration only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.aggregate_qwen3_calibration_records_v0_1 import (  # noqa: E402
    load_json,
    sha256_file,
)


SCHEMA_VERSION = "forkcert.qwen3-calibration-u2-multi-trajectory.v0.1"
LEDGER_VERSION = "forkcert.qwen3-calibration-u2-multi-trajectory-ledger.v0.1"
EXPECTED_TRAJECTORIES = {f"calibration-{index}" for index in range(4)}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def aggregate_trajectory_tensor(tensors: list[Any]) -> tuple[Any, float]:
    import torch

    if len(tensors) < 2:
        raise ValueError("at least two independent trajectory tensors are required")
    shape = tuple(tensors[0].shape)
    if any(tuple(tensor.shape) != shape for tensor in tensors):
        raise ValueError("trajectory mean-field tensor shapes differ")
    mean = torch.zeros_like(tensors[0], dtype=torch.float64)
    converted = [tensor.double() for tensor in tensors]
    for tensor in converted:
        mean.add_(tensor)
    mean.div_(len(converted))
    between_trace = math.fsum(
        float(((tensor - mean) ** 2).sum().item()) for tensor in converted
    ) / (len(converted) - 1)
    return mean.contiguous(), between_trace


def validate_inputs(
    summaries: list[tuple[Path, dict[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    by_trajectory: dict[str, list[dict[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    for path, summary in summaries:
        construction = summary.get("construction", {})
        trajectory_id = construction.get("trajectory_id")
        if summary.get("valid") is not True:
            errors.append(f"{path}: single-trajectory vector summary is not valid")
            continue
        if construction.get("trajectory_count") != 1:
            errors.append(f"{path}: expected one trajectory")
        if construction.get("states") != 24:
            errors.append(f"{path}: expected 24 frozen states")
        if construction.get("query_id") != "Q-R":
            errors.append(f"{path}: expected query Q-R")
        if trajectory_id in by_trajectory:
            errors.append(f"duplicate trajectory summary: {trajectory_id}")
            continue
        rows = summary.get("parameter_rows")
        if not isinstance(rows, list) or not rows:
            errors.append(f"{path}: missing parameter rows")
            continue
        names = [row.get("parameter_name") for row in rows]
        if None in names or len(names) != len(set(names)):
            errors.append(f"{path}: parameter names must be present and unique")
            continue
        if construction.get("parameters") != len(rows):
            errors.append(f"{path}: construction parameter count mismatch")
        by_trajectory[str(trajectory_id)] = rows
        evidence.append(
            {
                "trajectory_id": trajectory_id,
                "summary": str(path),
                "summary_sha256": sha256_file(path),
                "source_fingerprint": summary.get("source_fingerprint"),
            }
        )
    if set(by_trajectory) != EXPECTED_TRAJECTORIES:
        errors.append("requires exactly calibration-0..calibration-3 vector summaries")
    if by_trajectory:
        key_sets = [
            {row.get("parameter_name") for row in rows}
            for rows in by_trajectory.values()
        ]
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            errors.append("trajectory parameter key sets differ")
    return by_trajectory, evidence, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-summary", action="append", required=True, metavar="JSON"
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    summary_paths = [Path(value).resolve() for value in args.trajectory_summary]
    missing_paths = [str(path) for path in summary_paths if not path.is_file()]
    summaries = [(path, load_json(path)) for path in summary_paths if path.is_file()]
    by_trajectory, evidence, errors = validate_inputs(summaries)
    errors = [f"missing trajectory summary: {path}" for path in missing_paths] + errors
    out_path = Path(args.out).resolve()
    if errors:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "verdict": "INVALID_INCOMPLETE_FROZEN_VECTOR_CALIBRATION",
            "errors": errors,
            "observed_trajectories": sorted(by_trajectory),
            "population_B_claim_allowed": False,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(out_path, payload)
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)

    trajectory_order = sorted(by_trajectory)
    row_maps = {
        trajectory: {row["parameter_name"]: row for row in rows}
        for trajectory, rows in by_trajectory.items()
    }
    keys = sorted(row_maps[trajectory_order[0]])
    artifact_dir = Path(args.artifact_dir).resolve()
    shard_dir = artifact_dir / "calibration_mean_delta_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    source_fingerprint = canonical_sha256(evidence)
    ledger_path = artifact_dir / "u2_multi_trajectory_ledger.json"
    if ledger_path.is_file():
        ledger = load_json(ledger_path)
        if (
            ledger.get("schema_version") != LEDGER_VERSION
            or ledger.get("source_fingerprint") != source_fingerprint
        ):
            raise ValueError("existing multi-trajectory U2 ledger has different sources")
    else:
        ledger = {
            "schema_version": LEDGER_VERSION,
            "source_fingerprint": source_fingerprint,
            "parameters": {},
        }

    handles: dict[str, Any] = {}
    for index, key in enumerate(keys):
        source_artifacts = []
        for trajectory in trajectory_order:
            row = row_maps[trajectory][key]
            artifact = row.get("trajectory_mean_delta_artifact", {})
            path = Path(artifact.get("path", "")).resolve()
            expected = artifact.get("sha256")
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"{trajectory}/{key} source shard identity failed")
            source_artifacts.append(
                {"trajectory_id": trajectory, "path": str(path), "sha256": expected}
            )
        existing = ledger["parameters"].get(key)
        if existing is not None:
            artifact = existing.get("calibration_mean_delta_artifact", {})
            path = Path(artifact.get("path", ""))
            if (
                existing.get("source_artifacts") == source_artifacts
                and path.is_file()
                and sha256_file(path) == artifact.get("sha256")
            ):
                continue
            raise ValueError(f"resume artifact identity failed for {key}")
        tensors = []
        for source in source_artifacts:
            path = Path(source["path"])
            if str(path) not in handles:
                handles[str(path)] = safe_open(path, framework="pt", device="cpu")
            handle = handles[str(path)]
            if sorted(handle.keys()) != [key]:
                raise ValueError(
                    f"{source['trajectory_id']}/{key} source shard tensor key mismatch"
                )
            tensors.append(handle.get_tensor(key))
        mean, between_trace = aggregate_trajectory_tensor(tensors)
        filename = f"{index:04d}_{hashlib.sha256(key.encode()).hexdigest()[:16]}.safetensors"
        shard_path = shard_dir / filename
        temporary = shard_path.with_suffix(".safetensors.tmp")
        save_file({key: mean}, temporary)
        temporary.replace(shard_path)
        reference_norm_sq_mean = math.fsum(
            row_maps[trajectory][key]["trajectory_mean_reference_update_l2"] ** 2
            for trajectory in trajectory_order
        ) / len(trajectory_order)
        row = {
            "parameter_name": key,
            "shape": list(mean.shape),
            "coordinates": mean.numel(),
            "calibration_mean_delta_l2": math.sqrt(float((mean * mean).sum().item())),
            "calibration_mean_delta_max_abs": float(mean.abs().max().item()),
            "between_trajectory_variance_trace": between_trace,
            "mean_within_trajectory_phase_variance_trace": math.fsum(
                row_maps[trajectory][key]["within_trajectory_phase_variance_trace"]
                for trajectory in trajectory_order
            )
            / len(trajectory_order),
            "mean_within_phase_state_variance_trace_repeat_corrected": math.fsum(
                row_maps[trajectory][key][
                    "mean_within_phase_state_variance_trace_repeat_corrected"
                ]
                for trajectory in trajectory_order
            )
            / len(trajectory_order),
            "mean_same_state_paired_effect_variance_trace": math.fsum(
                row_maps[trajectory][key][
                    "mean_same_state_paired_effect_variance_trace"
                ]
                for trajectory in trajectory_order
            )
            / len(trajectory_order),
            "rms_trajectory_mean_reference_update_l2": math.sqrt(
                reference_norm_sq_mean
            ),
            "calibration_mean_delta_artifact": {
                "path": str(shard_path),
                "sha256": sha256_file(shard_path),
                "tensor_key": key,
            },
            "source_artifacts": source_artifacts,
        }
        ledger["parameters"][key] = row
        atomic_json(ledger_path, ledger)
        del tensors, mean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    parameter_rows = [ledger["parameters"][key] for key in keys]
    mean_sq = math.fsum(row["calibration_mean_delta_l2"] ** 2 for row in parameter_rows)
    reference_rms_sq = math.fsum(
        row["rms_trajectory_mean_reference_update_l2"] ** 2
        for row in parameter_rows
    )
    mean_l2 = math.sqrt(mean_sq)
    reference_rms = math.sqrt(reference_rms_sq)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "verdict": "VALID_COMPLETE_FOUR_TRAJECTORY_VECTOR_CALIBRATION",
        "construction": {
            "trajectory_ids": trajectory_order,
            "trajectory_count": 4,
            "top_level_df": 3,
            "parameters": len(parameter_rows),
            "coordinates": sum(row["coordinates"] for row in parameter_rows),
            "weighting": "equal independent trajectory mean fields",
        },
        "calibration_mean_field": {
            "role": "signed coordinate-frame average of four trajectory mean candidate-minus-reference update fields",
            "l2": mean_l2,
            "max_abs_coordinate": max(
                row["calibration_mean_delta_max_abs"] for row in parameter_rows
            ),
            "relative_to_rms_trajectory_mean_reference_update_l2": (
                mean_l2 / reference_rms if reference_rms else None
            ),
            "shards": [
                row["calibration_mean_delta_artifact"] for row in parameter_rows
            ],
        },
        "H_trace": {
            "between_trajectory_variance": math.fsum(
                row["between_trajectory_variance_trace"] for row in parameter_rows
            ),
            "mean_within_trajectory_phase_variance": math.fsum(
                row["mean_within_trajectory_phase_variance_trace"]
                for row in parameter_rows
            ),
            "mean_within_phase_state_variance_repeat_corrected": math.fsum(
                row["mean_within_phase_state_variance_trace_repeat_corrected"]
                for row in parameter_rows
            ),
        },
        "N_trace": {
            "mean_same_state_paired_effect_variance": math.fsum(
                row["mean_same_state_paired_effect_variance_trace"]
                for row in parameter_rows
            )
        },
        "U": {
            "primary_unit": "independent trajectory",
            "top_level_df": 3,
            "confirmation_interval": None,
            "reason": "calibration is for scale/design and has fewer than eight trajectories",
        },
        "parameter_rows": parameter_rows,
        "input_evidence": evidence,
        "population_B_claim_allowed": False,
        "nonclaims": [
            "calibration mean field is not a confirmed population B",
            "four trajectories do not pass the independent confirmation gate",
            "vector norm has no sign and is not substituted for the signed mean field",
            "the reference denominator is an explicitly named RMS scale, not the norm of a pooled reference vector",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out_path, payload)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "construction": payload["construction"],
                "calibration_mean_l2": mean_l2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
