#!/usr/bin/env python3
"""Merge current Triton shards and test cross-state directional local error."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/coverage/qwen_oracle_protocol.json"


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def u_statistic(errors: np.ndarray) -> float:
    states, coordinates = errors.shape
    total = errors.sum(axis=0)
    return float(
        (np.dot(total, total) - np.square(errors).sum())
        / (states * (states - 1) * coordinates)
    )


def bootstrap(errors: np.ndarray, draws: int, seed: int) -> dict[str, float]:
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
        weighted = counts[:, None] * counts[None, :] * gram
        numerator = float(weighted.sum() - np.trace(weighted))
        values.append(numerator / (denominator * coordinates))
    values = np.asarray(values, dtype=np.float64)
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/coverage/qwen_current_triton_oracle.json.gz",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=12831)
    parser.add_argument("--length-bucket", default="seq64")
    args = parser.parse_args()
    shards = [load(path) for path in args.inputs]
    protocol = json.loads(PROTOCOL.read_text())
    if {int(row["shard_index"]) for row in shards} != set(range(len(shards))):
        raise ValueError("held-out shard set is incomplete")
    if any(row["shard_count"] != len(shards) for row in shards):
        raise ValueError("shard count mismatch")
    complete_statuses = {
        "COMPLETE_CURRENT_SEQ64_TRITON_HELDOUT_SHARD",
        "COMPLETE_CURRENT_TRITON_HELDOUT_SHARD",
    }
    if any(row["status"] not in complete_statuses for row in shards):
        raise ValueError("a Triton shard is incomplete")
    if any(row["protocol_sha256"] != protocol["protocol_sha256"] for row in shards):
        raise ValueError("protocol binding mismatch")
    campaign_hashes = {row["campaign_sha256"] for row in shards}
    if len(campaign_hashes) != 1:
        raise ValueError("campaign changed across shards")
    campaign_path = Path(shards[0]["campaign_path"])
    campaign = load(campaign_path)
    if campaign["result_sha256"] != next(iter(campaign_hashes)):
        raise ValueError("campaign artifact identity changed")
    specialized_by_symbol = {
        row["symbol"]: row for row in campaign["rows"]
        if row["boundary_capture_mode"].startswith("SPECIALIZED_")
    }
    states = {}
    for shard in shards:
        if states.keys() & shard["states"].keys():
            raise ValueError("state repeated across shards")
        states.update(shard["states"])
    expected = {
        row["state_id"] for row in protocol["heldout_states"]
        if row["stratum"] == args.length_bucket
    }
    if states.keys() != expected or len(states) != 32:
        raise ValueError(f"{args.length_bucket} held-out population is not exact")

    expected_regions = campaign["denominator"]["triton_invocations"]
    expected_measured = campaign["denominator"]["reference_adapter_exact"]
    expected_unresolved = campaign["denominator"]["reference_adapter_unresolved"]

    observations: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    repeat_exact = True
    for state_id, state in sorted(states.items()):
        repeats = state["repeats"]
        if len(repeats) != 2 or not all(row["observation_stable"] for row in repeats):
            raise ValueError(f"invalid repeats for {state_id}")
        by_repeat = []
        for repeat in repeats:
            rows = {}
            for record in repeat["summary"]["records"]:
                for endpoint, metric in record["endpoint_metrics"].items():
                    rows[(record["region_id"], endpoint)] = {
                        "record": record,
                        "metric": metric,
                    }
            for specialized in repeat["specialized_embedding"]["records"]:
                campaign_row = specialized_by_symbol.get(specialized["symbol"])
                if campaign_row is None:
                    raise ValueError("specialized symbol is absent from campaign")
                record = {
                    "region_id": campaign_row["region_id"],
                    "phase": campaign_row["phase"],
                    "symbol": specialized["symbol"],
                    "reference_symbol": specialized["symbol"],
                    "reference_role": specialized.get(
                        "reference_role", "BOUNDED_EMBEDDING_GENERATED_BOUNDARY"
                    ),
                }
                rows[(record["region_id"], "out_ptr0")] = {
                    "record": record, "metric": specialized["metrics"],
                }
            by_repeat.append(rows)
        if (
            by_repeat[0].keys() != by_repeat[1].keys()
            or len({key[0] for key in by_repeat[0]}) != expected_measured
        ):
            raise ValueError(f"region/endpoint population changed for {state_id}")
        for key in sorted(by_repeat[0]):
            left, right = by_repeat[0][key], by_repeat[1][key]
            left_sketch = left["metric"]["directional_error_sketch"]
            right_sketch = right["metric"]["directional_error_sketch"]
            same = (
                left_sketch["flat_coordinate_indices"] == right_sketch["flat_coordinate_indices"]
                and left_sketch["signed_delta_values"] == right_sketch["signed_delta_values"]
                and left["metric"]["exact"] == right["metric"]["exact"]
                and left["metric"]["max_abs"] == right["metric"]["max_abs"]
            )
            repeat_exact = repeat_exact and same
            observations[key].append({
                "state_id": state_id,
                "record": left["record"],
                "metric": left["metric"],
            })

    rows = []
    for row_index, (key, values) in enumerate(sorted(observations.items())):
        region_id, endpoint = key
        sketches = [row["metric"]["directional_error_sketch"] for row in values]
        indices = sketches[0]["flat_coordinate_indices"]
        if any(sketch["flat_coordinate_indices"] != indices for sketch in sketches[1:]):
            raise ValueError(f"coordinate identity changed for {region_id}/{endpoint}")
        errors = np.asarray(
            [sketch["signed_delta_values"] for sketch in sketches], dtype=np.float64
        )
        finite = all(
            row["metric"]["candidate_finite"] and row["metric"]["reference_finite"]
            for row in values
        )
        exact = all(row["metric"]["exact"] for row in values)
        confidence = bootstrap(
            errors, args.bootstrap_draws, args.bootstrap_seed + row_index
        )
        directional = finite and not exact and confidence["lower_95"] > 0
        if not finite:
            verdict = "NONFINITE_RISK"
        elif exact:
            verdict = "EQUIVALENT_EXACT_ON_HELDOUT_STATES"
        elif directional:
            verdict = "DIRECTIONAL_BIAS_SCREEN_POSITIVE"
        else:
            verdict = "FINITE_NONEXACT_WITHOUT_STABLE_DIRECTION"
        record = values[0]["record"]
        result = {
            "region_id": region_id,
            "phase": record["phase"],
            "symbol": record["symbol"],
            "reference_symbol": record["reference_symbol"],
            "reference_role": record["reference_role"],
            "endpoint": endpoint,
            "states": len(values),
            "sampled_coordinates": int(errors.shape[1]),
            "all_repeats_exact": repeat_exact,
            "all_values_finite": finite,
            "all_states_exact": exact,
            "nonexact_states": sum(not row["metric"]["exact"] for row in values),
            "max_abs_over_states": max(row["metric"]["max_abs"] for row in values),
            "rms_mean": float(np.mean([row["metric"]["rms"] for row in values])),
            "cross_state_inner_product_u": u_statistic(errors),
            "cluster_bootstrap_95": confidence,
            "verdict": verdict,
        }
        result["row_sha256"] = digest(result)
        rows.append(result)

    verdicts = Counter(row["verdict"] for row in rows)
    measured_region_ids = {row["region_id"] for row in rows}
    if len(measured_region_ids) != expected_measured:
        raise ValueError("measured ordinary-region denominator changed")
    payload = {
        "schema": "kernel-analyzer-current-qwen-triton-oracle-v1",
        "status": (
            "COMPLETE_TRITON_DENOMINATOR_HELDOUT_SCREEN"
            if expected_unresolved == 0
            else "PARTIAL_TRITON_DENOMINATOR_HELDOUT_SCREEN"
        ),
        "protocol_sha256": protocol["protocol_sha256"],
        "campaign_sha256": next(iter(campaign_hashes)),
        "denominator": {
            "current_triton_invocations": expected_regions,
            "invocations_with_exact_adapter_and_measurement": len(measured_region_ids),
            "measured_region_endpoints": len(rows),
            "specialized_bounded_invocations": len(specialized_by_symbol),
            "unresolved_reference_adapter": expected_unresolved,
            "heldout_states": len(states),
        },
        "verdict_counts": dict(sorted(verdicts.items())),
        "gates": {
            "all_32_frozen_shape_states_present": True,
            "all_online_observers_nonperturbing": True,
            "all_repeat_local_metrics_exact": repeat_exact,
            "candidate_values_used_to_select_regions_or_coordinates": False,
            "unmeasured_rows_retained_in_denominator": True,
            "full_current_triton_denominator_screened": expected_unresolved == 0,
        },
        "rows": rows,
        "claim_boundary": (
            f"Verdicts screen {expected_measured}/{expected_regions} {args.length_bucket} "
            "Triton invocations with exact local references. "
            "A directional screen is not yet a complete F+B carrier/accumulation bias case."
        ),
    }
    payload["result_sha256"] = digest(payload)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output.resolve().relative_to(ROOT)),
        "denominator": payload["denominator"],
        "verdict_counts": payload["verdict_counts"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
