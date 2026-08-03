#!/usr/bin/env python3
"""Record a denominator-only fused-CE protocol amendment before confirmation."""

from __future__ import annotations

import argparse
import gzip
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


def read(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = checked(args.protocol)
    discovery_dir = checked(args.discovery_dir)
    output = checked(args.output)
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text())
    if protocol["status"] != "FROZEN_BEFORE_ANY_LIGER_FUSED_CE_VALUES":
        raise RuntimeError("original protocol differs")
    protocol_hash = sha256(protocol_path)
    state_bindings = []
    for row in protocol["state_allocations"]["discovery"]:
        path = discovery_dir / f"{row['state_id']}.json.gz"
        artifact = read(path)
        if (
            artifact["status"] != "COMPLETE"
            or artifact["state"]["phase"] != "discovery"
            or artifact["bindings"]["protocol"]["sha256"] != protocol_hash
            or not all(artifact["gates"].values())
        ):
            raise RuntimeError(f"invalid discovery artifact: {path}")
        state_bindings.append({"path": str(path.resolve()), "sha256": sha256(path)})
    payload: dict[str, Any] = {
        "schema_version": "kernel-analyzer.liger-fused-ce-protocol-amendment.v1",
        "status": "FROZEN_BEFORE_CONFIRMATION",
        "change_type": "DENOMINATOR_ARITHMETIC_ONLY",
        "original_text": {
            "residuals": protocol["statistics"]["residuals"],
            "directional_hypotheses_maximum": protocol["denominators"]["directional_hypotheses_maximum"],
        },
        "correction": {
            "primary_candidate_vs_eager_cells": 6,
            "causal_accumulator_cells": 3,
            "structural_zero_controls": ["default_minus_fp32_accum.loss", "default_minus_fp32_accum.dH"],
            "maximum_nonzero_directional_hypotheses": 7,
            "multiplicity_rule": "BH-FDR jointly over every nonzero discovery carrier among all nine residual cells",
        },
        "selection_unchanged": (
            "The original protocol already required all six implementation-by-endpoint cells and the "
            "default-minus-fp32 accumulator residual. No endpoint, sign, carrier, threshold, state, or claim gate changes."
        ),
        "timing_disclosure": (
            "This correction was written after the seven discovery executions exposed the bookkeeping mismatch "
            "and before any of the 24 confirmation states were executed."
        ),
        "bindings": {
            "original_protocol": {"path": str(protocol_path), "sha256": protocol_hash},
            "discovery_states": state_bindings,
        },
    }
    payload["artifact_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
