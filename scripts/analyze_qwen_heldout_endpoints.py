#!/usr/bin/env python3
"""Merge held-out shards and test complete-step directional candidate bias.

This deliberately reports training endpoints only.  It never projects a
whole-step result back onto individual invocations; local invocation verdicts
must come from an exact same-input region observation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/coverage/qwen_oracle_protocol.json"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def vector(observation: dict[str, Any], order: list[str]) -> np.ndarray:
    values = [float(observation["loss"])]
    gradients = observation["parameter_gradients"]
    for name in order:
        row = gradients[name]
        if row["status"] != "SAMPLED" or not row["finite"]:
            raise ValueError(f"invalid gradient endpoint for {name}")
        values.extend(float(value) for value in row["values"])
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite endpoint vector")
    return result


def mean_vector(
    observations: list[dict[str, Any]], order: list[str], expected: int
) -> np.ndarray:
    if len(observations) != expected:
        raise ValueError(f"expected {expected} repeats, found {len(observations)}")
    return np.mean([vector(row, order) for row in observations], axis=0)


def cross_state_u(errors: np.ndarray) -> float:
    """Unbiased mean cross-state inner product, normalized per coordinate."""
    states, coordinates = errors.shape
    if states < 2:
        raise ValueError("at least two states are required")
    summed = errors.sum(axis=0)
    numerator = float(np.dot(summed, summed) - np.square(errors).sum())
    return numerator / (states * (states - 1) * coordinates)


def state_pair_cosines(errors: np.ndarray) -> np.ndarray:
    normalized = errors / np.maximum(
        np.linalg.norm(errors, axis=1, keepdims=True), np.finfo(np.float64).tiny
    )
    matrix = normalized @ normalized.T
    return matrix[np.triu_indices(len(errors), 1)]


def bootstrap(
    errors: np.ndarray, *, draws: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    states = len(errors)
    coordinates = errors.shape[1]
    gram = errors @ errors.T
    values = []
    while len(values) < draws:
        counts = np.bincount(
            rng.integers(0, states, size=states), minlength=states
        ).astype(np.float64)
        denominator = float(counts.sum() ** 2 - np.square(counts).sum())
        if denominator <= 0:
            continue
        # Repeated draws of one cluster are not independent cross-state
        # pairs. Exclude every same-original-cluster pair from numerator and
        # denominator rather than feeding duplicates to the ordinary U-stat.
        weighted = counts[:, None] * counts[None, :] * gram
        numerator = float(weighted.sum() - np.trace(weighted))
        values.append(numerator / (denominator * coordinates))
    values = np.asarray(values, dtype=np.float64)
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def summarize(
    errors: np.ndarray, *, draws: int, seed: int
) -> dict[str, Any]:
    norms = np.sqrt(np.mean(np.square(errors), axis=1))
    cosines = state_pair_cosines(errors)
    confidence = bootstrap(errors, draws=draws, seed=seed)
    return {
        "states": int(len(errors)),
        "coordinates": int(errors.shape[1]),
        "cross_state_inner_product_u": cross_state_u(errors),
        "cluster_bootstrap_95": confidence,
        "state_rms": {
            "mean": float(norms.mean()),
            "q95": float(np.quantile(norms, 0.95)),
            "maximum": float(norms.max()),
        },
        "pairwise_cosine": {
            "mean": float(cosines.mean()),
            "q05": float(np.quantile(cosines, 0.05)),
            "positive_fraction": float(np.mean(cosines > 0)),
        },
        "stable_directional_bias": confidence["lower_95"] > 0,
    }


def subset_rows(
    state_rows: list[tuple[str, dict[str, Any]]], stratum: str | None
) -> Iterable[tuple[str, dict[str, Any]]]:
    return (
        (state_id, row)
        for state_id, row in state_rows
        if stratum is None or row["stratum"] == stratum
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", type=Path, nargs="+", required=True,
        help="The complete set of held-out shard .json.gz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/coverage/qwen_heldout_endpoint_oracle.json.gz",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=9417)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text())
    shards = [load(path) for path in args.inputs]
    expected_indices = set(range(len(shards)))
    actual_indices = {int(shard["shard_index"]) for shard in shards}
    if actual_indices != expected_indices:
        raise ValueError(f"shard indices are incomplete: {sorted(actual_indices)}")
    if any(int(shard["shard_count"]) != len(shards) for shard in shards):
        raise ValueError("shard count mismatch")
    if any(shard["status"] != "COMPLETE_HELDOUT_ENDPOINT_SHARD" for shard in shards):
        raise ValueError("a held-out shard is incomplete")
    if any(shard["protocol_sha256"] != protocol["protocol_sha256"] for shard in shards):
        raise ValueError("protocol binding mismatch")

    coordinate_maps = [shard["parameter_coordinates"] for shard in shards]
    if any(value != coordinate_maps[0] for value in coordinate_maps[1:]):
        raise ValueError("parameter coordinate identity changed across shards")
    order = sorted(coordinate_maps[0])
    expected_states = {row["state_id"]: row for row in protocol["heldout_states"]}
    states: dict[str, dict[str, Any]] = {}
    for shard in shards:
        overlap = states.keys() & shard["states"].keys()
        if overlap:
            raise ValueError(f"states repeated across shards: {sorted(overlap)}")
        states.update(shard["states"])
    if states.keys() != expected_states.keys():
        raise ValueError("held-out state population is not exactly the frozen population")

    state_rows = sorted(states.items())
    deltas: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    repeat_deltas: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for state_id, row in state_rows:
        design = expected_states[state_id]
        if row["record_sha256"] != design["record_sha256"]:
            raise ValueError(f"state digest changed for {state_id}")
        eager_rows = row["bf16_eager"]
        eager = mean_vector(eager_rows, order, 2)
        fp32 = mean_vector(row["fp32_eager_strict"], order, 1)
        standard_rows = row["bf16_inductor_standard"]
        preserve_rows = row["bf16_inductor_preserve_aot_aten"]
        standard = mean_vector(standard_rows, order, 2)
        preserve = mean_vector(preserve_rows, order, 2)
        deltas[state_id] = {
            "precision": eager - fp32,
            "optimization_standard": standard - eager,
            "optimization_preserve_aot": preserve - eager,
            "compiler_structure": standard - preserve,
        }
        repeat_deltas[state_id] = {
            "runtime_eager": vector(eager_rows[1], order) - vector(eager_rows[0], order),
            "runtime_standard": vector(standard_rows[1], order) - vector(standard_rows[0], order),
            "runtime_preserve_aot": vector(preserve_rows[1], order) - vector(preserve_rows[0], order),
        }

    strata = [None, "seq64", "seq128", "seq256"]
    summaries: dict[str, Any] = {}
    for stratum_index, stratum in enumerate(strata):
        ids = [state_id for state_id, _ in subset_rows(state_rows, stratum)]
        label = "all" if stratum is None else stratum
        summaries[label] = {}
        for contrast_index, contrast in enumerate(
            ("precision", "optimization_standard", "optimization_preserve_aot", "compiler_structure")
        ):
            matrix = np.stack([deltas[state_id][contrast] for state_id in ids])
            summaries[label][contrast] = summarize(
                matrix,
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + 100 * stratum_index + contrast_index,
            )
        summaries[label]["runtime"] = {}
        for repeat_index, contrast in enumerate(
            ("runtime_eager", "runtime_standard", "runtime_preserve_aot")
        ):
            matrix = np.stack([repeat_deltas[state_id][contrast] for state_id in ids])
            summaries[label]["runtime"][contrast] = summarize(
                matrix,
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + 1000 + 100 * stratum_index + repeat_index,
            )

    payload = {
        "schema": "kernel-analyzer-qwen-heldout-endpoint-oracle-v1",
        "status": "COMPLETE_HELDOUT_TRAINING_ENDPOINT_SCREEN",
        "protocol_sha256": protocol["protocol_sha256"],
        "states": len(states),
        "state_counts_by_stratum": {
            stratum: sum(row["stratum"] == stratum for row in states.values())
            for stratum in ("seq64", "seq128", "seq256")
        },
        "endpoint_population": {
            "loss": 1,
            "parameters": len(order),
            "sampled_parameter_coordinates": sum(len(coordinate_maps[0][name]) for name in order),
            "total_vector_coordinates": 1 + sum(len(coordinate_maps[0][name]) for name in order),
        },
        "summaries": summaries,
        "gates": {
            "frozen_heldout_population_exact": True,
            "all_repeats_present": True,
            "coordinate_identity_exact": True,
            "precision_and_optimization_contrasts_separate": True,
            "candidate_values_not_used_for_calibration": protocol["candidate_values_used"] is False,
            "invocation_level_verdict_inferred_from_training_endpoint": False,
        },
        "claim_boundary": (
            "This is a complete-step loss and sampled parameter-gradient directional-bias screen. "
            "It is not an internal-op measurement and assigns no invocation-level verdict."
        ),
        "source_shards": [str(path.resolve().relative_to(ROOT)) for path in args.inputs],
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output.resolve().relative_to(ROOT)),
        "status": payload["status"],
        "states": payload["states"],
        "all_state_standard": summaries["all"]["optimization_standard"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
