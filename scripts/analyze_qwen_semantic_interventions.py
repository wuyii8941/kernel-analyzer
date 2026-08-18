#!/usr/bin/env python3
"""Analyze frozen-coordinate causal carriers for Qwen semantic interventions."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "backward:106": "model.layers.24.mlp.up_proj.weight",
    "backward:145": "model.layers.23.post_attention_layernorm.weight",
}


def u_statistic(values: np.ndarray) -> float:
    count = len(values)
    return float(
        (np.square(values.sum(axis=0)).sum() - np.square(values).sum())
        / (count * (count - 1))
    )


def distinct_cluster_bootstrap(
    values: np.ndarray, *, draws: int = 5000, seed: int = 260145
) -> dict[str, float | int]:
    """Bootstrap clusters while excluding pairs copied from one source state."""
    rng = np.random.default_rng(seed)
    count = len(values)
    samples = []
    gram = values @ values.T
    for _ in range(draws):
        selected = rng.integers(0, count, size=count)
        total = 0.0
        pairs = 0
        for left in range(count):
            for right in range(left + 1, count):
                if selected[left] == selected[right]:
                    continue
                total += gram[selected[left], selected[right]]
                pairs += 1
        if pairs:
            samples.append(total / pairs)
    lower, median, upper = np.quantile(samples, [0.025, 0.5, 0.975])
    return {
        "draws": len(samples),
        "seed": seed,
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
    }


def main() -> None:
    result_rows = []
    for region_id, parameter in CONFIGS.items():
        suffix = region_id.split(":")[1]
        paths = sorted((ROOT / "results/coverage").glob(
            f"qwen_semantic_bias_{suffix}_shard*.json.gz"
        ))
        states = {}
        local_maxima = []
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload["status"] != "COMPLETE_CURRENT_SEQ64_TRITON_HELDOUT_SHARD":
                raise RuntimeError(f"incomplete shard: {path}")
            for state_id, state in payload["states"].items():
                if state_id in states:
                    raise RuntimeError(f"duplicate state: {state_id}")
                repeats = state["repeats"]
                if len(repeats) != 2:
                    raise RuntimeError(f"incomplete repeats: {state_id}")
                vectors = []
                for repeat in repeats:
                    carrier = repeat["carrier_deltas"][parameter]
                    vectors.append(np.asarray(
                        carrier["intervened_minus_candidate"], dtype=np.float64
                    ))
                    record = next(
                        row for row in repeat["summary"]["records"]
                        if row["region_id"] == region_id
                    )
                    if record["intervened_endpoints"] != ["out_ptr0"]:
                        raise RuntimeError(f"intervention missed: {state_id}")
                    local_maxima.append(record["endpoint_metrics"]["out_ptr0"]["max_abs"])
                if not np.array_equal(vectors[0], vectors[1]):
                    raise RuntimeError(f"runtime-unstable carrier: {state_id}")
                states[state_id] = vectors[0]
        if len(states) != 32:
            raise RuntimeError(f"held-out population incomplete for {region_id}: {len(states)}")
        values = np.stack([states[key] for key in sorted(states)])
        bootstrap = distinct_cluster_bootstrap(values)
        result_rows.append({
            "region_id": region_id,
            "carrier_parameter": parameter,
            "states": len(values),
            "repeats_per_state": 2,
            "repeat_exact": True,
            "sampled_coordinates": int(values.shape[1]),
            "nonzero_states": int(np.count_nonzero(np.any(values != 0, axis=1))),
            "carrier_cross_state_inner_product_u": u_statistic(values),
            "carrier_cluster_bootstrap_95": bootstrap,
            "carrier_rms": float(np.sqrt(np.mean(np.square(values)))),
            "mean_carrier_l2": float(np.linalg.norm(values.mean(axis=0))),
            "local_max_abs_over_states_and_repeats": float(max(local_maxima)),
            "verdict": (
                "CAUSAL_COHERENT_WEIGHT_GRADIENT_BIAS"
                if bootstrap["lower_95"] > 0
                else "CAUSAL_PROPAGATION_WITHOUT_COHERENT_CARRIER"
            ),
        })
    payload = {
        "schema": "kernel-analyzer-qwen-semantic-intervention-analysis-v1",
        "status": "COMPLETE_32_STATE_CAUSAL_CARRIER_ANALYSIS",
        "rows": result_rows,
        "claim_boundary": (
            "Exact semantic replacement establishes local causal reach. A positive "
            "distinct-state bootstrap lower bound establishes coherence only for the "
            "frozen sampled parameter-gradient coordinates; it does not establish a "
            "multi-step optimizer trajectory or generalize to unresolved invocations."
        ),
    }
    payload["result_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    output = ROOT / "results/coverage/qwen_semantic_intervention_analysis.json.gz"
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
