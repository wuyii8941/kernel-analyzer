#!/usr/bin/env python3
"""Freeze a disjoint-state full-step tied-weight propagation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


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
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-certificate", type=Path, required=True)
    parser.add_argument("--state-design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_path = checked(args.local_certificate)
    design_path = checked(args.state_design)
    output = checked(args.output)
    if output.exists():
        raise FileExistsError(output)
    local = json.loads(local_path.read_text())
    design = json.loads(design_path.read_text())
    if (
        local["verdict"] != "NATURAL_LIGER_FUSED_CE_BF16_DW_ACCUMULATION_BIAS_CONFIRMED"
        or not local["flashattention_style_gates"]["complete_region_mechanism"]
        or local["flashattention_style_gates"]["full_tied_weight_gradient"]
    ):
        raise RuntimeError("local fused-CE certificate differs")
    states = [
        row
        for row in design["records"]
        if row["split"] == "heldout" and row["length_bucket"] == "seq128"
    ]
    if len(states) != 32 or len({row["cluster_id"] for row in states}) != 32:
        raise RuntimeError("independent heldout seq128 denominator differs")
    allocations = [
        {
            "state_id": row["sequence_id"],
            "cluster_id": row["cluster_id"],
            "record_sha256": row["record_sha256"],
        }
        for row in states
    ]
    payload: dict[str, Any] = {
        "schema_version": "kernel-analyzer.liger-fused-ce-propagation-protocol.v1",
        "status": "FROZEN_BEFORE_ANY_FULL_STEP_TIED_WEIGHT_VALUES",
        "controlled_full_step": {
            "backbone": "complete Qwen3-1.7B eager BF16 transformer and final norm",
            "terminal_region": "Liger fused-linear cross entropy",
            "default_arm": "64 dW chunks accumulated in BF16",
            "counterfactual_arm": "same implementation and chunks with only dW accum_dtype=FP32",
            "common": "model weights, hidden computation, labels, loss, dH, external dtype, TF32 setting and chunk order",
        },
        "readout": {
            "endpoints": ["loss", "terminal dH", "all 310 final named-parameter gradients"],
            "residual_orientation": "default-BF16-accumulator minus FP32-accumulator",
            "expected_reach": "only the tied model.embed_tokens.weight final gradient may differ; all upstream gradients must be exact because dH is invariant",
            "discovery": "freeze one named-parameter CountSketch8192 carrier on eight states",
            "confirmation": "24 untouched states with exact sign test and cluster bootstrap",
            "repeats": "both arms repeat twice per state and fail closed on any instability",
        },
        "state_allocations": {
            "discovery": allocations[:8],
            "confirmation": allocations[8:],
        },
        "denominators": {
            "disjoint_states": 32,
            "discovery_states": 8,
            "confirmation_states": 24,
            "parameters_per_state": 310,
            "complete_full_steps": 32 * 4,
        },
        "claim_gate": {
            "complete_flashattention_style": "the local complete region mechanism plus a discovery-frozen final tied-parameter carrier with positive bootstrap lower bound",
            "negative_control": "loss, terminal dH, and all 309 other named-parameter gradients remain bitwise exact",
        },
        "bindings": {
            "local_certificate": {"path": str(local_path), "sha256": sha256(local_path)},
            "state_design": {"path": str(design_path), "sha256": sha256(design_path)},
        },
    }
    payload["artifact_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
