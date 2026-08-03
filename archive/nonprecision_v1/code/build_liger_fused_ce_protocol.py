#!/usr/bin/env python3
"""Freeze a natural BF16 Liger fused-linear cross-entropy experiment."""

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
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design_path = checked(args.design)
    output = checked(args.output)
    if output.exists():
        raise FileExistsError(output)
    design = json.loads(design_path.read_text())
    if design["status"] != "FROZEN_SUPPLEMENTARY_STATE_EXTENSION" or design["candidate_data_used"]:
        raise RuntimeError("supplementary design differs")
    rows = sorted(
        (row for row in design["records"] if row["length_bucket"] == "seq128"),
        key=lambda row: row["sequence_id"],
    )
    if len(rows) != 32 or len({row["cluster_id"] for row in rows}) != 32:
        raise RuntimeError("seq128 independent-state denominator differs")

    def compact(row: Mapping[str, Any]) -> dict[str, str]:
        return {
            "state_id": str(row["sequence_id"]),
            "cluster_id": str(row["cluster_id"]),
            "record_sha256": str(row["record_sha256"]),
        }

    payload: dict[str, Any] = {
        "schema_version": "kernel-analyzer.liger-fused-ce-protocol.v1",
        "status": "FROZEN_BEFORE_ANY_LIGER_FUSED_CE_VALUES",
        "scope": {
            "model": "Qwen3-1.7B",
            "sequence_length": 128,
            "hidden_width": 2048,
            "vocabulary": 151936,
            "dtype": "BF16",
            "tf32": False,
            "semantic_region": "lm_head linear plus causal cross entropy",
            "state_source": "natural supplementary held-out text",
        },
        "mathematical_unit": {
            "symbols": "H in R^(T x D), W in R^(V x D), targets a_t, scalar upstream q=1",
            "forward": [
                "Z=H W^T",
                "L=-(1/N) sum_t log(exp(Z[t,a_t])/sum_v exp(Z[t,v]))",
            ],
            "actual_vjp": [
                "G[t,v]=(softmax(Z)[t,v]-1[v=a_t])/N",
                "dH=G W",
                "dW=G^T H",
            ],
            "closed_endpoints": ["loss", "dH", "dW"],
            "same_precision_reference": "BF16 eager lm_head followed by Transformers FP32 cross entropy and actual autograd",
            "candidate": "LigerFusedLinearCrossEntropyLoss custom forward and saved actual backward",
            "interpretive_reference": "FP32 evaluation of the displayed region using the exact BF16 H and W values",
        },
        "controlled_implementations": {
            "default_accum": "Liger accum_dtype=None; chunk contributions to dW are stored/added in BF16",
            "fp32_accum": "the same Liger implementation with only accum_dtype=torch.float32",
            "fixed": "H, W, labels, loss reduction, chunk schedule, external dtype and kernels other than dW accumulator storage",
            "predicted_chunk_schedule": "V/D gives inc_factor=75; T=128 gives chunk_size=2 and 64 sequential dW additions",
        },
        "runtime_gates": [
            "baseline and hooked full-step loss and sentinel gradient are bitwise exact",
            "the eager standalone region exactly reproduces the full-model loss",
            "captured lm_head x and q are bound to the reference region and labels",
            "all implementations receive identical H, W and shifted labels",
            "loss, dH and dW shape/dtype boundaries match and remain finite",
            "two repeats of every arm are bitwise exact",
            "all six implementation-by-endpoint cells remain in the denominator",
        ],
        "statistics": {
            "residuals": ["default_accum minus eager", "fp32_accum minus eager", "default_accum minus fp32_accum"],
            "discovery": "freeze CountSketch8192 directions for every nonzero implementation-by-endpoint cell over seven states",
            "confirmation": "24 untouched states; exact sign test, cluster bootstrap 95% lower bound, and BH-FDR across all discovered cells",
            "pilot": "one execution audit state excluded from carrier construction and confirmation",
            "causal_accumulator_gate": "default-minus-fp32 dW is directional and FP32 accumulation reduces error against the FP32 region reference",
        },
        "state_allocations": {
            "pilot": [compact(rows[0])],
            "discovery": [compact(row) for row in rows[1:8]],
            "confirmation": [compact(row) for row in rows[8:]],
        },
        "denominators": {
            "pilot_states_excluded": 1,
            "discovery_states": 7,
            "confirmation_states": 24,
            "closed_forward_actual_vjp_units": 31 * 2,
            "implementation_endpoint_state_cells": 31 * 2 * 3,
            "directional_hypotheses_maximum": 6,
        },
        "claim_gate": {
            "local_directional_case": "at least one endpoint passes the frozen direction test",
            "complete_accumulation_mechanism": "the accumulator intervention passes on dW, the direct parameter-gradient endpoint, while loss/dH controls and every boundary gate are retained",
            "full_tied_parameter_claim": "requires a later full-step shared-weight carrier because embedding and lm_head gradients are added",
        },
        "bindings": {
            "state_design": {"path": str(design_path), "sha256": sha256(design_path)},
        },
    }
    payload["artifact_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
