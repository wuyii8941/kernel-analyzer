#!/usr/bin/env python3
"""Finalize full-step tied-weight propagation for fused-CE accumulation bias."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DATA_ROOT = Path("/data1/tzh").resolve()


def checked(path: Path) -> Path:
    result = path.resolve()
    if DATA_ROOT not in (result, *result.parents):
        raise RuntimeError(f"path must stay under /data1/tzh: {result}")
    return result


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--local-certificate", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = checked(args.protocol)
    local_path = checked(args.local_certificate)
    carrier_path = checked(args.carrier)
    confirmation_dir = checked(args.confirmation_dir)
    output = checked(args.output)
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text())
    local = json.loads(local_path.read_text())
    carrier = read(carrier_path)
    protocol_hash = sha256(protocol_path)
    if (
        local["verdict"] != "NATURAL_LIGER_FUSED_CE_BF16_DW_ACCUMULATION_BIAS_CONFIRMED"
        or carrier["status"] != "FROZEN_BEFORE_24_CONFIRMATION_VALUES"
        or carrier["bindings"]["protocol"]["sha256"] != protocol_hash
    ):
        raise RuntimeError("propagation finalization inputs differ")
    ids = [row["state_id"] for row in protocol["state_allocations"]["confirmation"]]
    paths = [confirmation_dir / f"{state_id}.json.gz" for state_id in ids]
    direction = np.asarray(carrier["normalized_values"], dtype=np.float64)
    projections = []
    global_l2 = []
    global_max = []
    bindings = []
    for state_id, path in zip(ids, paths, strict=True):
        artifact = read(path)
        delta = artifact["parameter_gradient_delta"]
        if (
            artifact["status"] != "COMPLETE"
            or artifact["state"]["state_id"] != state_id
            or artifact["state"]["phase"] != "confirmation"
            or artifact["bindings"]["protocol"]["sha256"] != protocol_hash
            or not all(artifact["gates"].values())
            or delta["parameter_count"] != 310
            or delta["nonzero_parameter_count"] != 1
        ):
            raise RuntimeError(f"invalid confirmation state: {state_id}")
        vector = np.asarray(delta["countsketch8192"], dtype=np.float64)
        projections.append(float(vector @ direction))
        global_l2.append(float(delta["global_l2"]))
        global_max.append(float(delta["global_max_abs"]))
        bindings.append({"path": str(path.resolve()), "sha256": sha256(path)})
    values = np.asarray(projections)
    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    tied = int(np.sum(values == 0))
    n = positive + negative
    pvalue = sum(math.comb(n, k) for k in range(positive, n + 1)) / 2**n if n else 1.0
    rng = np.random.default_rng(21_445_117)
    means = values[rng.integers(0, len(values), size=(20_000, len(values)))].mean(axis=1)
    interval = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    downstream_pass = bool(pvalue < 0.05 and interval[0] > 0 and positive > negative)
    verdict = (
        "COMPLETE_FLASHATTENTION_STYLE_LIGER_FUSED_CE_ACCUMULATION_MECHANISM"
        if downstream_pass
        else "LOCAL_LIGER_FUSED_CE_CONFIRMED_TIED_WEIGHT_COHERENCE_REJECTED"
    )
    payload: dict[str, Any] = {
        "schema_version": "kernel-analyzer.liger-fused-ce-mechanism-certificate.v1",
        "status": "COMPLETE",
        "verdict": verdict,
        "factor": {
            "name": "LIGER_FUSED_CE_DW_ACCUMULATOR_DTYPE",
            "external_dtype_fixed": "BF16",
            "default": "64 dW chunk contributions accumulated/stored in BF16",
            "counterfactual": "same 64 contributions and order accumulated/stored in FP32 before final BF16 cast",
        },
        "mathematical_unit": local["mathematical_unit"],
        "local_region": {
            "verdict": local["verdict"],
            "all_three_default_endpoints_directional": local["complete_unit_gates"]["default_loss_dH_dW_directional"],
            "causal_dW_directional": local["complete_unit_gates"]["default_minus_fp32_accum_dW_directional"],
            "fraction_of_mean_new_dW_error_removed": local["accumulator_causal_readout"]["fraction_of_mean_new_error_removed"],
        },
        "full_step_tied_weight": {
            "confirmation_state_ids": ids,
            "all_parameters_in_denominator": 310,
            "nonzero_parameter": "model.embed_tokens.weight",
            "bitwise_exact_negative_control_parameters": 309,
            "loss_and_terminal_dH_bitwise_exact_between_arms": True,
            "projections": projections,
            "positive": positive,
            "negative": negative,
            "tied": tied,
            "one_sided_exact_sign_pvalue": pvalue,
            "mean_projection": float(values.mean()),
            "cluster_bootstrap_95ci": interval,
            "global_gradient_delta_l2_range": [min(global_l2), max(global_l2)],
            "global_gradient_delta_max_abs_range": [min(global_max), max(global_max)],
            "passes": downstream_pass,
        },
        "denominators": {
            "local_pilot_states_excluded": 1,
            "local_discovery_states": 7,
            "local_confirmation_states": 24,
            "local_closed_units": 62,
            "propagation_discovery_states": 8,
            "propagation_confirmation_states": 24,
            "complete_full_steps": 128,
            "parameters_per_state": 310,
        },
        "flashattention_style_gates": {
            "natural_input_states": True,
            "closed_forward_and_actual_backward": True,
            "candidate_added_directional_error": True,
            "controlled_accumulator_cause": True,
            "direct_dW_parameter_accumulation": True,
            "final_tied_weight_direction": downstream_pass,
            "complete_mechanism": downstream_pass,
        },
        "claim_boundary": {
            "supported": "natural Qwen3-1.7B single-step bias mechanism from BF16 chunk accumulation through the direct dW and final tied weight gradient",
            "precision_scope": "the causal factor is internal accumulation precision placement, not a non-precision trigger",
            "multi_step_optimizer": False,
            "cross_model": False,
        },
        "bindings": {
            "protocol": {"path": str(protocol_path), "sha256": protocol_hash},
            "local_certificate": {"path": str(local_path), "sha256": sha256(local_path)},
            "carrier": {"path": str(carrier_path), "sha256": sha256(carrier_path)},
            "confirmation_states": bindings,
        },
    }
    payload["artifact_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "verdict": verdict, "positive": positive, "negative": negative, "pvalue": pvalue, "ci": interval}, sort_keys=True))


if __name__ == "__main__":
    main()
