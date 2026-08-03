#!/usr/bin/env python3
"""Freeze all fused-CE residual carriers before confirmation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DATA_ROOT = Path("/data1/tzh").resolve()
ENDPOINTS = ("loss", "dH", "dW")


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


def cell(artifact: Mapping[str, Any], contrast: str, endpoint: str) -> Mapping[str, Any]:
    if contrast == "default_minus_fp32_accum":
        return artifact[contrast][endpoint]
    return artifact["implementations"][contrast][endpoint]["candidate_minus_eager"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = checked(args.protocol)
    amendment_path = checked(args.amendment)
    input_dir = checked(args.input_dir)
    output = checked(args.output)
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    protocol_hash = sha256(protocol_path)
    if (
        amendment["status"] != "FROZEN_BEFORE_CONFIRMATION"
        or amendment["bindings"]["original_protocol"]["sha256"] != protocol_hash
        or amendment["correction"]["maximum_nonzero_directional_hypotheses"] != 7
    ):
        raise RuntimeError("fused-CE amendment differs")
    ids = [row["state_id"] for row in protocol["state_allocations"]["discovery"]]
    paths = [input_dir / f"{state_id}.json.gz" for state_id in ids]
    artifacts = [read(path) for path in paths]
    for state_id, path, artifact, binding in zip(
        ids, paths, artifacts, amendment["bindings"]["discovery_states"], strict=True
    ):
        if (
            artifact["status"] != "COMPLETE"
            or artifact["state"]["state_id"] != state_id
            or artifact["state"]["phase"] != "discovery"
            or artifact["bindings"]["protocol"]["sha256"] != protocol_hash
            or not all(artifact["gates"].values())
            or binding["sha256"] != sha256(path)
        ):
            raise RuntimeError(f"invalid discovery state: {state_id}")
    candidates = []
    zero_controls = []
    for contrast in ("default_accum", "fp32_accum", "default_minus_fp32_accum"):
        for endpoint in ENDPOINTS:
            rows = [cell(artifact, contrast, endpoint) for artifact in artifacts]
            matrix = np.asarray([row["residual_countsketch8192"] for row in rows], dtype=np.float64)
            carrier = matrix.mean(axis=0)
            norm = float(np.linalg.norm(carrier))
            common = {"contrast": contrast, "endpoint": endpoint}
            if norm == 0.0:
                zero_controls.append(
                    {**common, "discovery_nonzero_states": sum(not row["exact"] for row in rows)}
                )
                continue
            direction = carrier / norm
            candidates.append(
                {
                    **common,
                    "normalized_values": direction.tolist(),
                    "discovery_mean_norm": norm,
                    "discovery_nonzero_states": sum(not row["exact"] for row in rows),
                    "discovery_projections": [float(value) for value in matrix @ direction],
                    "discovery_signed_means": [float(row["mean_signed"]) for row in rows],
                }
            )
    if len(candidates) != 7 or len(zero_controls) != 2:
        raise RuntimeError(f"unexpected carrier denominator: {len(candidates)=} {len(zero_controls)=}")
    if {(row["contrast"], row["endpoint"]) for row in zero_controls} != {
        ("default_minus_fp32_accum", "loss"),
        ("default_minus_fp32_accum", "dH"),
    }:
        raise RuntimeError("structural zero controls differ")
    payload: dict[str, Any] = {
        "schema_version": "kernel-analyzer.liger-fused-ce-carriers.v1",
        "status": "FROZEN_BEFORE_24_CONFIRMATION_VALUES",
        "representation": "CountSketch8192(seed=3407)",
        "selection_rule": "all nine cells retained; every nonzero seven-state mean sketch tested jointly",
        "state_ids": ids,
        "candidates": candidates,
        "zero_controls": zero_controls,
        "denominators": {
            "states": 7,
            "residual_cells": 9,
            "candidate_cells": 7,
            "zero_control_cells": 2,
        },
        "bindings": {
            "protocol": {"path": str(protocol_path), "sha256": protocol_hash},
            "amendment": {"path": str(amendment_path), "sha256": sha256(amendment_path)},
            "states": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths],
        },
    }
    payload["artifact_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"output": str(output), **payload["denominators"]}, sort_keys=True))


if __name__ == "__main__":
    main()
