#!/usr/bin/env python3
"""Apply the frozen backward:145 confirmation gate to its 64 new states."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_qwen_semantic_interventions import (
    distinct_cluster_bootstrap,
    u_statistic,
)


ROOT = Path(__file__).resolve().parents[1]
PARAMETER = "model.layers.23.post_attention_layernorm.weight"


def main() -> None:
    design = json.loads((ROOT / "results/coverage/qwen145_confirmation_design.json").read_text())
    states = {}
    local_maxima = []
    for path in sorted((ROOT / "results/coverage").glob("qwen145_confirmation_shard*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            shard = json.load(handle)
        if shard["status"] != "COMPLETE_CURRENT_SEQ64_TRITON_HELDOUT_SHARD":
            raise RuntimeError(f"incomplete shard: {path}")
        if shard.get("state_design_sha256") != design["design_sha256"]:
            raise RuntimeError(f"state-design mismatch: {path}")
        for state_id, state in shard["states"].items():
            repeats = state["repeats"]
            if len(repeats) != 2:
                raise RuntimeError(f"incomplete repeats: {state_id}")
            vectors = []
            for repeat in repeats:
                vectors.append(np.asarray(
                    repeat["carrier_deltas"][PARAMETER]["intervened_minus_candidate"],
                    dtype=np.float64,
                ))
                record = next(
                    row for row in repeat["summary"]["records"]
                    if row["region_id"] == "backward:145"
                )
                if record["intervened_endpoints"] != ["out_ptr0"]:
                    raise RuntimeError(f"intervention missed: {state_id}")
                local_maxima.append(record["endpoint_metrics"]["out_ptr0"]["max_abs"])
            if not np.array_equal(vectors[0], vectors[1]):
                raise RuntimeError(f"runtime-unstable carrier: {state_id}")
            states[state_id] = vectors[0]
    if len(states) != 64 or set(states) != {row["sequence_id"] for row in design["records"]}:
        raise RuntimeError("confirmation population is not exactly the frozen 64 states")
    values = np.stack([states[key] for key in sorted(states)])
    bootstrap = distinct_cluster_bootstrap(values, draws=10_000, seed=145640)
    passed = bootstrap["lower_95"] > 0
    payload = {
        "schema": "kernel-analyzer-qwen145-independent-confirmation-v1",
        "status": "COMPLETE_FROZEN_CONFIRMATION",
        "design_sha256": design["design_sha256"],
        "region_id": "backward:145",
        "carrier_parameter": PARAMETER,
        "states": 64,
        "repeats_per_state": 2,
        "repeat_exact": True,
        "coordinates": int(values.shape[1]),
        "nonzero_states": int(np.count_nonzero(np.any(values != 0, axis=1))),
        "carrier_cross_state_inner_product_u": u_statistic(values),
        "carrier_cluster_bootstrap_95": bootstrap,
        "local_max_abs_over_states_and_repeats": float(max(local_maxima)),
        "frozen_success_gate_passed": bool(passed),
        "verdict": (
            "INDEPENDENTLY_CONFIRMED_COHERENT_WEIGHT_GRADIENT_BIAS"
            if passed else "CAUSAL_NUMERICAL_DIFFERENCE_NOT_COHERENTLY_CONFIRMED"
        ),
        "claim_boundary": (
            "The candidate-blind continuation bank, full 2048-coordinate carrier, and "
            "success gate were frozen before these candidate executions. Failure of the "
            "gate rejects promotion to a coherent-bias case; it does not prove zero local error."
        ),
    }
    payload["result_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    output = ROOT / "results/coverage/qwen145_confirmation_analysis.json.gz"
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
