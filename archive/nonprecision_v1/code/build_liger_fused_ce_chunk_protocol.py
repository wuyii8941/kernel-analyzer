#!/usr/bin/env python3
"""Freeze a same-semantics fused-CE physical-token/chunk-geometry experiment."""

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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--prior-certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design_path = checked(args.design)
    prior_path = checked(args.prior_certificate)
    output = checked(args.output)
    if output.exists():
        raise FileExistsError(output)
    design = json.loads(design_path.read_text())
    prior = json.loads(prior_path.read_text())
    if design["status"] != "FROZEN_SUPPLEMENTARY_STATE_EXTENSION" or design["candidate_data_used"]:
        raise RuntimeError("supplementary design differs")
    if prior["verdict"] != "NATURAL_LIGER_FUSED_CE_BF16_DW_ACCUMULATION_BIAS_CONFIRMED":
        raise RuntimeError("prior fused-CE certificate differs")
    rows = sorted(
        (row for row in design["records"] if row["length_bucket"] == "seq128"),
        key=lambda row: row["sequence_id"],
    )
    if len(rows) != 32:
        raise RuntimeError("seq128 state denominator differs")

    def compact(row: Mapping[str, Any]) -> dict[str, str]:
        return {"state_id": str(row["sequence_id"]), "cluster_id": str(row["cluster_id"]), "record_sha256": str(row["record_sha256"])}

    payload: dict[str, Any] = {
        "schema_version": "kernel-analyzer.liger-fused-ce-chunk-protocol.v1",
        "status": "FROZEN_BEFORE_ANY_PADDED_CHUNK_VALUES_AFTER_BASELINE_CASE",
        "selection_disclosure": "the known natural base-128 accumulator case motivated this factor test; no padded candidate value was read before freezing",
        "mathematical_unit": prior["mathematical_unit"],
        "same_semantics_intervention": {
            "base": "128 actual hidden rows with 127 active shifted labels; physical BT=128, chunk_size=2",
            "padded": "append 128 exactly-zero hidden rows and 128 ignore_index labels; physical BT=256, chunk_size=4",
            "real_number_invariant": "zero ignored rows contribute exactly zero loss, dH and dW, so active loss, active dH and dW are unchanged",
            "controlled_factor": "physical token cardinality, chunk width, and grouping/count of active dW additions",
            "fixed": "actual H/W/labels, active support, Liger source, external BF16 dtype, accumulator dtype within each stratum, TF32=false",
        },
        "accumulator_strata": {
            "bf16": "Liger accum_dtype=None",
            "fp32": "Liger accum_dtype=torch.float32",
        },
        "endpoints": ["loss", "active_dH", "dW"],
        "runtime_gates": [
            "the natural hidden state is produced by the unmodified eager Qwen backbone",
            "base and padded arms share bitwise-identical active H/W/labels",
            "all padded dH rows are exactly zero",
            "both arms and accumulator strata repeat bitwise exactly",
            "all six padding-contrast endpoint cells remain in the denominator",
        ],
        "statistics": {
            "primary_residual": "padded-256 minus base-128 within each accumulator stratum",
            "discovery": "freeze every nonzero CountSketch8192 endpoint direction on seven states",
            "confirmation": "24 untouched states; exact sign, cluster bootstrap, and BH-FDR across all discovered cells",
            "pilot": "one execution-only state excluded",
            "causal_gate": "BF16 dW padding contrast is directional; FP32 stratum quantifies whether higher accumulation precision suppresses the geometry effect",
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
            "accumulator_strata": 2,
            "closed_forward_actual_vjp_units": 31 * 2 * 2,
            "padding_contrast_endpoint_cells": 6,
        },
        "claim_boundary": {
            "supported_if_passed": "physical token/chunk geometry is a non-dtype trigger of a same-semantics dW residual",
            "not_independent_operator": "this refines the existing fused-CE mechanism and cannot be counted as a new operator case",
        },
        "bindings": {
            "state_design": {"path": str(design_path), "sha256": sha256(design_path)},
            "prior_certificate": {"path": str(prior_path), "sha256": sha256(prior_path)},
        },
    }
    payload["artifact_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
