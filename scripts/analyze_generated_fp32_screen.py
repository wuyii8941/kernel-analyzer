#!/usr/bin/env python3
"""Merge a complete generated FP32 screen and test cross-state direction."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/coverage/generated_fp32_protocol.json"


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def u_statistic(errors: np.ndarray) -> float:
    states, coordinates = errors.shape
    total = errors.sum(axis=0)
    return float(
        (np.dot(total, total) - np.square(errors).sum())
        / (states * (states - 1) * coordinates)
    )


def bootstrap_counts(states: int, draws: int, seed: int) -> np.ndarray:
    if states < 2:
        raise ValueError("cluster bootstrap requires at least two states")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, states, size=(draws, states))
    counts = np.zeros((draws, states), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(draws), states), samples.reshape(-1)), 1.0)
    # The cross-state U statistic has no denominator when a resample contains
    # only one distinct cluster.  Rejection sampling preserves the declared
    # number of valid bootstrap draws instead of silently dropping a
    # seed-dependent fraction (especially material for two-state pilots).
    invalid = np.count_nonzero(counts, axis=1) < 2
    while invalid.any():
        row_ids = np.flatnonzero(invalid)
        replacement = rng.integers(0, states, size=(len(row_ids), states))
        counts[row_ids] = 0.0
        np.add.at(
            counts,
            (
                np.repeat(row_ids, states),
                replacement.reshape(-1),
            ),
            1.0,
        )
        invalid = np.count_nonzero(counts, axis=1) < 2
    return counts


def bootstrap(errors: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    coordinates = errors.shape[1]
    gram = errors @ errors.T
    denominator = np.square(counts.sum(axis=1)) - np.square(counts).sum(axis=1)
    numerator = np.einsum("bi,ij,bj->b", counts, gram, counts)
    numerator -= np.einsum("bi,i,bi->b", counts, np.diag(gram), counts)
    data = numerator[denominator > 0] / (denominator[denominator > 0] * coordinates)
    if data.size != counts.shape[0]:
        raise RuntimeError("degenerate cluster-bootstrap resample")
    return {
        "lower_95": float(np.quantile(data, 0.025)),
        "median": float(np.quantile(data, 0.5)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def metric_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    scalar_keys = (
        "exact", "nonzero_elements", "signed_mean", "rms", "max_abs",
        "candidate_finite", "reference_finite", "nonfinite_mismatch",
    )
    if any(left.get(key) != right.get(key) for key in scalar_keys):
        return False
    a, b = left["directional_error_sketch"], right["directional_error_sketch"]
    if a["flat_coordinate_indices"] != b["flat_coordinate_indices"]:
        return False
    return bool(np.array_equal(
        np.asarray(a["signed_delta_values"], dtype=float),
        np.asarray(b["signed_delta_values"], dtype=float), equal_nan=True,
    ))


def records_by_key(summary: Mapping[str, Any], mode: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    occurrences: Counter[tuple[Any, ...]] = Counter()
    for record in summary["records"]:
        if mode == "triton":
            base = (record["region_id"],)
        else:
            base = (
                record["implementation_kind"], record["source_line_sha256"],
                record["region_id"],
            )
        ordinal = occurrences[base]
        occurrences[base] += 1
        for endpoint, metric in sorted(record["endpoint_metrics"].items()):
            key = (*base, ordinal, endpoint)
            if key in result:
                raise RuntimeError(f"duplicate runtime endpoint identity: {key}")
            result[key] = {"record": record, "endpoint": endpoint, "metric": metric}
    return result


def expected_state_ids(bank: Mapping[str, Any]) -> set[str]:
    rows = bank.get("states", bank.get("records"))
    return {
        str(row.get("sequence_id", row.get("state_id", index)))
        for index, row in enumerate(rows)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=14031)
    args = parser.parse_args()
    shards = [load(path) for path in args.inputs]
    protocol = json.loads(PROTOCOL.read_text())
    if any(row.get("protocol_sha256") != protocol["protocol_sha256"] for row in shards):
        raise RuntimeError("a shard lacks the frozen generated FP32 protocol binding")
    schemas = {row["schema"] for row in shards}
    if schemas in (
        {"kernel-analyzer-generated-fp32-screen-v1"},
        {"kernel-analyzer-generated-typed-fp32-screen-v2"},
    ):
        mode, complete = "triton", "COMPLETE_SHARD_ALL_TRITON_FP32_REPLAY"
        binding_key = "campaign_sha256"
    elif schemas == {"kernel-analyzer-generated-nontriton-fp32-screen-v1"}:
        mode, complete = "nontriton", "COMPLETE_SHARD_ALL_NONTRITON_FP32_REPLAY"
        binding_key = "inventory_sha256"
    else:
        raise RuntimeError(f"mixed or unsupported screen schemas: {schemas}")
    count = len(shards)
    if {row["shard_index"] for row in shards} != set(range(count)):
        raise RuntimeError("shard set is incomplete")
    if any(row["shard_count"] != count or row["status"] != complete for row in shards):
        raise RuntimeError("shard count/status mismatch")
    bindings = {row[binding_key] for row in shards}
    if len(bindings) != 1:
        raise RuntimeError("candidate artifact changed across shards")
    origin_certificates = [row.get("inductor_buffer_origins") for row in shards]
    if any(origin_certificates):
        if not all(origin_certificates):
            raise RuntimeError("IR-buffer origin capture is missing from a subset of shards")
        origin_hashes = {
            row["result_sha256"] for row in origin_certificates if row is not None
        }
        if len(origin_hashes) != 1:
            raise RuntimeError("IR-buffer origins changed across shards")
        buffer_origin_binding = {
            "result_sha256": next(iter(origin_hashes)),
            "status": origin_certificates[0]["status"],
            "denominator": origin_certificates[0]["denominator"],
        }
    else:
        buffer_origin_binding = None
    if any(row["input_bank_sha256"] != file_digest(args.input_bank) for row in shards):
        raise RuntimeError("input bank identity mismatch")
    states: dict[str, Any] = {}
    for shard in shards:
        overlap = states.keys() & shard["states"].keys()
        if overlap:
            raise RuntimeError(f"states repeated across shards: {sorted(overlap)}")
        states.update(shard["states"])
    typed_triton = schemas == {"kernel-analyzer-generated-typed-fp32-screen-v2"}
    bank = json.loads(args.input_bank.read_text())
    expected_states = expected_state_ids(bank)
    if states.keys() != expected_states or len(states) != 32:
        raise RuntimeError("frozen 32-state population is incomplete")

    observations: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    all_repeats_exact = True
    static_missing_by_state: dict[str, list[dict[str, Any]]] = {}
    invocation_counts_by_state: dict[str, int] = {}
    for state_id, state in sorted(states.items()):
        repeats = state["repeats"]
        if len(repeats) != 2:
            raise RuntimeError(f"repeat denominator changed for {state_id}")
        left = records_by_key(repeats[0]["summary"], mode)
        right = records_by_key(repeats[1]["summary"], mode)
        if typed_triton:
            for repeat in repeats:
                summary = repeat["summary"]
                if any(
                    row.get("reference_abi")
                    != "INDEPENDENT_RECOMPILED_FLOATING_POINTER_FP32"
                    for row in summary["records"]
                ):
                    raise RuntimeError(f"typed Triton shard contains a stale pointer ABI: {state_id}")
                programs = summary.get("typed_reference_programs", {})
                if not programs or any(
                    not row.get("only_pointer_abi_literals_changed")
                    or any(change.get("to") != "*fp32" for change in row.get("changed_float_pointers", {}).values())
                    for row in programs.values()
                ):
                    raise RuntimeError(f"typed Triton program provenance is incomplete: {state_id}")
        if left.keys() != right.keys():
            raise RuntimeError(f"runtime endpoint identity changed for {state_id}")
        invocation_counts_by_state[state_id] = len({key[:-1] for key in left})
        if mode == "nontriton":
            missing = repeats[0]["summary"]["missing_rows"]
            if missing != repeats[1]["summary"]["missing_rows"]:
                raise RuntimeError(f"static disposition changed across repeats: {state_id}")
            # Graph-break candidates may execute a state-dependent subset of
            # the frozen generated callsites.  Every actual invocation remains
            # in the denominator; non-execution is retained per state instead
            # of pretending that the runtime census is invariant.
            static_missing_by_state[state_id] = missing
        for key in left:
            stable = metric_equal(left[key]["metric"], right[key]["metric"])
            all_repeats_exact &= stable
            observations[key].append({
                "state_id": state_id, "record": left[key]["record"],
                "endpoint": left[key]["endpoint"],
                "metrics": [left[key]["metric"], right[key]["metric"]],
                "repeat_stable": stable,
            })

    rows = []
    for key, values in sorted(observations.items(), key=lambda item: repr(item[0])):
        sketches = [
            metric["directional_error_sketch"]
            for row in values for metric in row["metrics"]
        ]
        coordinates = sketches[0]["flat_coordinate_indices"]
        if any(sketch["flat_coordinate_indices"] != coordinates for sketch in sketches[1:]):
            raise RuntimeError(f"coordinate identity changed: {key}")
        repeat_errors = np.asarray(
            [
                [metric["directional_error_sketch"]["signed_delta_values"] for metric in row["metrics"]]
                for row in values
            ], dtype=float,
        )
        errors = repeat_errors.mean(axis=1)
        metrics = [metric for row in values for metric in row["metrics"]]
        matching_nonfinite = sum(
            metric.get("matching_nan", 0)
            + metric.get("matching_posinf", 0)
            + metric.get("matching_neginf", 0)
            for metric in metrics
        )
        nonfinite_mismatch = sum(
            metric.get("nonfinite_mismatch", 0) for metric in metrics
        )
        nonfinite = nonfinite_mismatch > 0 or not np.isfinite(errors).all()
        exact = all(metric["exact"] for metric in metrics)
        runtime_unstable_states = sum(not row["repeat_stable"] for row in values)
        if nonfinite:
            confidence = None
            statistic = None
            verdict = "NONFINITE_RISK"
        elif exact and runtime_unstable_states == 0:
            confidence = {"lower_95": 0.0, "median": 0.0, "upper_95": 0.0}
            statistic = 0.0
            verdict = (
                "EQUIVALENT_EXACT_INCLUDING_MATCHING_NONFINITE_GEOMETRY"
                if matching_nonfinite else "EQUIVALENT_EXACT_ON_HELDOUT_STATES"
            )
        elif len(values) < 2:
            confidence = None
            statistic = None
            verdict = "INSUFFICIENT_STATE_SUPPORT"
        else:
            endpoint_resamples = bootstrap_counts(
                len(values), args.bootstrap_draws, args.bootstrap_seed
            )
            confidence = bootstrap(errors, endpoint_resamples)
            statistic = u_statistic(errors)
            if confidence["lower_95"] > 0:
                verdict = (
                    "DIRECTIONAL_BIAS_SCREEN_POSITIVE_WITH_RUNTIME_VARIANCE"
                    if runtime_unstable_states else "DIRECTIONAL_BIAS_SCREEN_POSITIVE"
                )
            elif runtime_unstable_states:
                verdict = "RUNTIME_VARIANCE_RISK"
            else:
                verdict = "FINITE_NONEXACT_WITHOUT_STABLE_DIRECTION"
        record = values[0]["record"]
        row = {
            "runtime_identity": list(key), "region_id": record["region_id"],
            "phase": record["phase"], "implementation_kind": (
                "TRITON" if mode == "triton" else record["implementation_kind"]
            ),
            "function": record.get("symbol", record.get("function")),
            "endpoint": values[0]["endpoint"], "states": len(values),
            "sampled_coordinates": len(coordinates), "all_states_exact": exact,
            "nonexact_state_repeats": sum(not metric["exact"] for metric in metrics),
            "runtime_unstable_states": runtime_unstable_states,
            "matching_nonfinite_count": matching_nonfinite,
            "nonfinite_mismatch_count": nonfinite_mismatch,
            "max_abs_over_state_repeats": max(metric["max_abs"] for metric in metrics),
            "rms_mean_over_state_repeats": float(np.mean([metric["rms"] for metric in metrics])),
            "cross_state_inner_product_u": statistic,
            "cluster_bootstrap_95": confidence, "verdict": verdict,
        }
        row["row_sha256"] = digest(row)
        rows.append(row)
    verdicts = Counter(row["verdict"] for row in rows)
    static_missing_union = {
        digest(row): row
        for missing in static_missing_by_state.values() for row in missing
    }
    payload = {
        "schema": "kernel-analyzer-generated-fp32-oracle-v1",
        "status": "COMPLETE_PRECISION_ONLY_RUNTIME_DENOMINATOR_ORACLE",
        "mode": mode, binding_key: next(iter(bindings)),
        "inductor_buffer_origin_binding": buffer_origin_binding,
        "protocol_sha256": protocol["protocol_sha256"],
        "input_bank_sha256": file_digest(args.input_bank),
        "denominator": {
            "states": len(states), "repeats_per_state": 2,
            "actual_runtime_invocations_total": sum(invocation_counts_by_state.values()),
            "actual_runtime_invocations_per_state": {
                "min": min(invocation_counts_by_state.values()),
                "max": max(invocation_counts_by_state.values()),
                "by_state": invocation_counts_by_state,
            },
            "measured_runtime_endpoints": len(rows),
            "static_generated_calls_not_executed_in_at_least_one_state": len(static_missing_union),
        },
        "static_not_executed_rows_by_state": static_missing_by_state,
        "verdict_counts": dict(sorted(verdicts.items())),
        "gates": {
            "all_32_frozen_states_present": True,
            "runtime_identity_repeat_stable": True,
            "state_varying_actual_invocations_retained": True,
            "same_state_metrics_repeat_exact": all_repeats_exact,
            "candidate_values_used_to_select_coordinates": False,
            "static_nonexecuted_calls_retained_with_disposition": True,
            "typed_triton_pointer_abi_valid": typed_triton if mode == "triton" else None,
        },
        "cluster_bootstrap": {
            "unit": "STATE", "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed, "common_resamples_across_endpoints": False,
        },
        "rows": rows,
        "claim_boundary": (
            "Precision-only typed generated-program/declared-op screen. Triton floating pointer "
            "ABIs are independently recompiled as FP32 and structurally bound to the frozen "
            "program. It does not by itself prove "
            "eager semantic equivalence or a complete forward+backward carrier mechanism."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output), "denominator": payload["denominator"],
        "verdict_counts": payload["verdict_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
