#!/usr/bin/env python
"""Aggregate U2 as a signed coordinate-frame vector, not as mean L2 magnitude.

The program is deliberately trajectory-level.  It emits a sharded artifact for
the calibration trajectory mean field and descriptive H/N traces.  It does not
turn one trajectory into a population-B claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.aggregate_qwen3_calibration_records_v0_1 import (
    linked_arms,
    load_complete_state_bundles,
    load_json,
    sha256_file,
)


SCHEMA_VERSION = "forkcert.qwen3-calibration-u2-vector-aggregate.v0.1"
LEDGER_VERSION = "forkcert.qwen3-calibration-u2-vector-ledger.v0.1"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _norm_sq(tensor: Any) -> float:
    return float((tensor * tensor).sum().item())


def aggregate_one_tensor(
    rows_by_phase: dict[str, list[dict[str, Any]]],
    phases: tuple[str, ...],
    key: str,
    load_tensor: Callable[[Any, str], Any],
) -> tuple[Any, dict[str, Any]]:
    """Compute one tensor's balanced trajectory mean and H/N trace summaries."""

    import torch

    if len(phases) < 2:
        raise ValueError("at least two phases are required for phase heterogeneity")
    trajectory_delta_sum = None
    trajectory_reference_sum = None
    phase_delta_norm_sq_sum = 0.0
    phase_runtime_variances: list[float] = []
    phase_state_heterogeneities: list[float] = []
    shape: tuple[int, ...] | None = None
    state_count = 0
    repeat_count: int | None = None

    for phase in phases:
        phase_rows = rows_by_phase.get(phase, [])
        if len(phase_rows) < 2:
            raise ValueError(f"phase {phase} needs at least two states to identify state H")
        phase_delta_sum = None
        phase_reference_sum = None
        state_delta_norm_sq_sum = 0.0
        state_runtime_variances: list[float] = []
        for row in phase_rows:
            repeats = sorted(row["repeats"], key=lambda item: int(item["repeat_id"]))
            if repeat_count is None:
                repeat_count = len(repeats)
            if len(repeats) != repeat_count or repeat_count < 2:
                raise ValueError("all states require one balanced repeat set of size at least two")
            deltas = [load_tensor(item["delta_source"], key).double() for item in repeats]
            current_shape = tuple(deltas[0].shape)
            if any(tuple(tensor.shape) != current_shape for tensor in deltas):
                raise ValueError(f"delta shape mismatch for {key}")
            if shape is None:
                shape = current_shape
            elif shape != current_shape:
                raise ValueError(f"cross-state delta shape mismatch for {key}")
            delta_mean = torch.zeros_like(deltas[0], dtype=torch.float64)
            for tensor in deltas:
                delta_mean.add_(tensor)
            delta_mean.div_(repeat_count)
            delta_runtime = sum(_norm_sq(tensor - delta_mean) for tensor in deltas) / (
                repeat_count - 1
            )
            if phase_delta_sum is None:
                phase_delta_sum = torch.zeros_like(delta_mean, dtype=torch.float64)
            phase_delta_sum.add_(delta_mean)
            state_delta_norm_sq_sum += _norm_sq(delta_mean)
            state_runtime_variances.append(delta_runtime)
            del deltas

            references = [
                load_tensor(item["reference_source"], key).double() for item in repeats
            ]
            if any(tuple(tensor.shape) != shape for tensor in references):
                raise ValueError(f"reference/delta shape mismatch for {key}")
            reference_mean = torch.zeros_like(references[0], dtype=torch.float64)
            for tensor in references:
                reference_mean.add_(tensor)
            reference_mean.div_(repeat_count)
            if phase_reference_sum is None:
                phase_reference_sum = torch.zeros_like(reference_mean, dtype=torch.float64)
            phase_reference_sum.add_(reference_mean)
            del references
            state_count += 1

        assert phase_delta_sum is not None and phase_reference_sum is not None
        phase_delta_mean = phase_delta_sum / len(phase_rows)
        phase_reference_mean = phase_reference_sum / len(phase_rows)
        if trajectory_delta_sum is None:
            trajectory_delta_sum = torch.zeros_like(phase_delta_mean, dtype=torch.float64)
            trajectory_reference_sum = torch.zeros_like(
                phase_reference_mean, dtype=torch.float64
            )
        trajectory_delta_sum.add_(phase_delta_mean)
        assert trajectory_reference_sum is not None
        trajectory_reference_sum.add_(phase_reference_mean)
        current_phase_norm_sq = _norm_sq(phase_delta_mean)
        phase_delta_norm_sq_sum += current_phase_norm_sq
        observed_state_variance = max(
            0.0,
            (state_delta_norm_sq_sum - len(phase_rows) * current_phase_norm_sq)
            / (len(phase_rows) - 1),
        )
        mean_runtime = math.fsum(state_runtime_variances) / len(state_runtime_variances)
        phase_runtime_variances.append(mean_runtime)
        phase_state_heterogeneities.append(
            max(0.0, observed_state_variance - mean_runtime / repeat_count)
        )

    assert trajectory_delta_sum is not None and trajectory_reference_sum is not None
    assert shape is not None and repeat_count is not None
    trajectory_delta_mean = trajectory_delta_sum / len(phases)
    trajectory_reference_mean = trajectory_reference_sum / len(phases)
    delta_mean_sq = _norm_sq(trajectory_delta_mean)
    reference_mean_sq = _norm_sq(trajectory_reference_mean)
    phase_heterogeneity = max(
        0.0,
        (phase_delta_norm_sq_sum - len(phases) * delta_mean_sq) / (len(phases) - 1),
    )
    dot_reference = float(
        (trajectory_delta_mean * trajectory_reference_mean).sum().item()
    )
    statistics = {
        "parameter_name": key,
        "shape": list(shape),
        "coordinates": trajectory_delta_mean.numel(),
        "states": state_count,
        "repeats_per_state": repeat_count,
        "trajectory_mean_delta_l2": math.sqrt(delta_mean_sq),
        "trajectory_mean_reference_update_l2": math.sqrt(reference_mean_sq),
        "trajectory_mean_delta_max_abs": float(
            trajectory_delta_mean.abs().max().item()
        ),
        "trajectory_mean_delta_dot_reference": dot_reference,
        "mean_same_state_paired_effect_variance_trace": math.fsum(
            phase_runtime_variances
        )
        / len(phase_runtime_variances),
        "mean_within_phase_state_variance_trace_repeat_corrected": math.fsum(
            phase_state_heterogeneities
        )
        / len(phase_state_heterogeneities),
        "within_trajectory_phase_variance_trace": phase_heterogeneity,
    }
    return trajectory_delta_mean.contiguous(), statistics


def build_source_rows(
    state_bundles: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[str]]:
    rows_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence: list[dict[str, Any]] = []
    errors: list[str] = []
    hash_cache: dict[tuple[str, str], bool] = {}

    def checked_artifact(value: Any, label: str) -> dict[str, str] | None:
        if not isinstance(value, dict):
            errors.append(f"{label} is not an artifact object")
            return None
        path = value.get("path")
        expected = value.get("sha256")
        if not isinstance(path, str) or not isinstance(expected, str):
            errors.append(f"{label} lacks path/sha256")
            return None
        item = Path(path).resolve()
        cache_key = (str(item), expected)
        if cache_key not in hash_cache:
            hash_cache[cache_key] = item.is_file() and sha256_file(item) == expected
        if not hash_cache[cache_key]:
            errors.append(f"{label} artifact identity failed")
            return None
        return {"path": str(item), "sha256": expected}

    for target, bundle in state_bundles:
        repeats = []
        for pair in sorted(
            bundle["paired_effect_records"],
            key=lambda value: int(value["identity"]["repeat_id"]),
        ):
            reference, _ = linked_arms(bundle, pair)
            u2 = pair["effects"]["U2_delta"]
            if u2.get("status") != "MEASURED":
                errors.append(f"U2 unavailable for {target['state_id']}")
                continue
            delta = checked_artifact(
                u2.get("artifact"),
                f"{target['state_id']} repeat {pair['identity']['repeat_id']} U2",
            )
            reference_update = checked_artifact(
                reference["outcomes"].get("parameter_update_artifact"),
                f"{target['state_id']} repeat {pair['identity']['repeat_id']} reference update",
            )
            if delta is None or reference_update is None:
                continue
            repeats.append(
                {
                    "repeat_id": int(pair["identity"]["repeat_id"]),
                    "delta_artifact": delta,
                    "reference_artifact": reference_update,
                }
            )
            evidence.extend((delta, reference_update))
        if len(repeats) != 2 or {row["repeat_id"] for row in repeats} != {1, 2}:
            errors.append(f"unbalanced U2 sources for {target['state_id']}")
            continue
        rows_by_phase[target["phase"]].append(
            {"state_id": target["state_id"], "repeats": repeats}
        )
    return dict(rows_by_phase), evidence, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    plan_path = Path(args.plan).resolve()
    results_root = Path(args.results_root).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    out_path = Path(args.out).resolve()
    plan = load_json(plan_path)
    targets = plan["targets"]
    state_bundles, state_evidence, errors = load_complete_state_bundles(
        plan, results_root
    )
    expected_phase_counts = Counter(target["phase"] for target in targets)
    observed_phase_counts = Counter(target["phase"] for target, _ in state_bundles)
    complete = (
        not errors
        and len(state_bundles) == len(targets)
        and observed_phase_counts == expected_phase_counts
    )
    if complete:
        rows_by_phase, vector_evidence, source_errors = build_source_rows(state_bundles)
        errors.extend(source_errors)
        complete = not errors
    else:
        rows_by_phase, vector_evidence = {}, []

    if not complete:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "verdict": "INVALID_INCOMPLETE_FROZEN_VECTOR_POPULATION",
            "errors": errors,
            "states": len(state_bundles),
            "expected_states": len(targets),
            "nonclaim": "partial states are not aggregated",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(out_path, payload)
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)

    phases = tuple(dict.fromkeys(target["phase"] for target in targets))
    source_fingerprint = canonical_sha256(
        {
            "plan_sha256": sha256_file(plan_path),
            "state_evidence": state_evidence,
            "vector_evidence": vector_evidence,
            "phases": phases,
        }
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = artifact_dir / "trajectory_mean_delta_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = artifact_dir / "u2_vector_aggregate_ledger.json"
    if ledger_path.is_file():
        ledger = load_json(ledger_path)
        if (
            ledger.get("schema_version") != LEDGER_VERSION
            or ledger.get("source_fingerprint") != source_fingerprint
        ):
            raise ValueError("existing U2 ledger belongs to different frozen sources")
    else:
        ledger = {
            "schema_version": LEDGER_VERSION,
            "source_fingerprint": source_fingerprint,
            "parameters": {},
        }

    handles: dict[str, Any] = {}
    all_keys: list[str] | None = None
    for phase_rows in rows_by_phase.values():
        for row in phase_rows:
            for repeat in row["repeats"]:
                for role in ("delta", "reference"):
                    artifact = repeat[f"{role}_artifact"]
                    path = artifact["path"]
                    if path not in handles:
                        handle = safe_open(path, framework="pt", device="cpu")
                        keys = sorted(handle.keys())
                        if all_keys is None:
                            all_keys = keys
                        elif keys != all_keys:
                            raise ValueError("U2/reference tensor key sets differ")
                        handles[path] = handle
                    repeat[f"{role}_source"] = handles[path]
    if not all_keys:
        raise ValueError("empty U2 vector artifacts")

    def load_tensor(handle: Any, key: str) -> Any:
        return handle.get_tensor(key)

    for index, key in enumerate(all_keys):
        existing = ledger["parameters"].get(key)
        if existing is not None:
            shard = existing.get("trajectory_mean_delta_artifact", {})
            shard_path = Path(shard.get("path", ""))
            if shard_path.is_file() and sha256_file(shard_path) == shard.get("sha256"):
                continue
            raise ValueError(f"resume shard identity failed for {key}")
        mean_delta, statistics = aggregate_one_tensor(
            rows_by_phase, phases, key, load_tensor
        )
        filename = f"{index:04d}_{hashlib.sha256(key.encode()).hexdigest()[:16]}.safetensors"
        shard_path = shard_dir / filename
        temporary = shard_path.with_suffix(".safetensors.tmp")
        save_file({key: mean_delta}, temporary)
        temporary.replace(shard_path)
        statistics["trajectory_mean_delta_artifact"] = {
            "path": str(shard_path),
            "sha256": sha256_file(shard_path),
            "tensor_key": key,
        }
        ledger["parameters"][key] = statistics
        atomic_json(ledger_path, ledger)
        del mean_delta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    parameter_rows = [ledger["parameters"][key] for key in all_keys]
    trajectory_ids = {
        bundle["paired_effect_records"][0]["identity"]["trajectory_id"]
        for _, bundle in state_bundles
    }
    query_ids = {
        bundle["paired_effect_records"][0]["identity"]["query_id"]
        for _, bundle in state_bundles
    }
    if len(trajectory_ids) != 1 or len(query_ids) != 1:
        raise ValueError("single-trajectory U2 aggregate received mixed identities")
    total_delta_sq = math.fsum(row["trajectory_mean_delta_l2"] ** 2 for row in parameter_rows)
    total_reference_sq = math.fsum(
        row["trajectory_mean_reference_update_l2"] ** 2 for row in parameter_rows
    )
    total_dot = math.fsum(
        row["trajectory_mean_delta_dot_reference"] for row in parameter_rows
    )
    total_delta_l2 = math.sqrt(total_delta_sq)
    total_reference_l2 = math.sqrt(total_reference_sq)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "verdict": "VALID_COMPLETE_ONE_TRAJECTORY_VECTOR_DESCRIPTION",
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "source_fingerprint": source_fingerprint,
        "construction": {
            "trajectory_count": 1,
            "trajectory_id": next(iter(trajectory_ids)),
            "query_id": next(iter(query_ids)),
            "states": len(state_bundles),
            "phases": list(phases),
            "phase_counts": dict(observed_phase_counts),
            "repeats_per_state": 2,
            "coordinates": sum(row["coordinates"] for row in parameter_rows),
            "parameters": len(parameter_rows),
            "weighting": "equal phase; equal state within phase; equal paired repeat within state",
        },
        "trajectory_mean_field": {
            "role": "signed coordinate-frame candidate-minus-reference update shift",
            "l2": total_delta_l2,
            "max_abs_coordinate": max(
                row["trajectory_mean_delta_max_abs"] for row in parameter_rows
            ),
            "relative_to_trajectory_mean_reference_update_l2": (
                total_delta_l2 / total_reference_l2 if total_reference_l2 else None
            ),
            "alignment_with_trajectory_mean_reference_update": (
                total_dot / (total_delta_l2 * total_reference_l2)
                if total_delta_l2 and total_reference_l2
                else None
            ),
            "shards": [
                row["trajectory_mean_delta_artifact"] for row in parameter_rows
            ],
        },
        "H_trace": {
            "mean_within_phase_state_variance_repeat_corrected": math.fsum(
                row["mean_within_phase_state_variance_trace_repeat_corrected"]
                for row in parameter_rows
            ),
            "within_trajectory_phase_variance": math.fsum(
                row["within_trajectory_phase_variance_trace"]
                for row in parameter_rows
            ),
            "between_trajectory_variance": None,
        },
        "N_trace": {
            "mean_same_state_paired_effect_variance": math.fsum(
                row["mean_same_state_paired_effect_variance_trace"]
                for row in parameter_rows
            )
        },
        "U": {
            "primary_unit": "independent trajectory",
            "population_interval": None,
            "reason": "one trajectory cannot identify population uncertainty",
        },
        "population_B_claim_allowed": False,
        "parameter_rows": parameter_rows,
        "state_evidence": state_evidence,
        "nonclaims": [
            "trajectory mean field is not confirmed population B",
            "the norm of the mean vector is not the mean of per-state delta norms",
            "H/N trace summaries do not identify harmfulness or correctness",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out_path, payload)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "construction": payload["construction"],
                "trajectory_mean_field": {
                    key: value
                    for key, value in payload["trajectory_mean_field"].items()
                    if key != "shards"
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
