#!/usr/bin/env python3
"""Freeze the eight-state full-step tied-parameter carrier."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
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
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = checked(args.protocol)
    local_path = checked(args.local_certificate)
    input_dir = checked(args.input_dir)
    output = checked(args.output)
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(protocol_path.read_text())
    local = json.loads(local_path.read_text())
    if (
        protocol["status"] != "FROZEN_BEFORE_ANY_FULL_STEP_TIED_WEIGHT_VALUES"
        or local["verdict"] != "NATURAL_LIGER_FUSED_CE_BF16_DW_ACCUMULATION_BIAS_CONFIRMED"
    ):
        raise RuntimeError("propagation inputs differ")
    ids = [row["state_id"] for row in protocol["state_allocations"]["discovery"]]
    paths = [input_dir / f"{state_id}.json.gz" for state_id in ids]
    artifacts = [read(path) for path in paths]
    protocol_hash = sha256(protocol_path)
    for state_id, artifact in zip(ids, artifacts, strict=True):
        delta = artifact["parameter_gradient_delta"]
        if (
            artifact["status"] != "COMPLETE"
            or artifact["state"]["state_id"] != state_id
            or artifact["state"]["phase"] != "discovery"
            or artifact["bindings"]["protocol"]["sha256"] != protocol_hash
            or not all(artifact["gates"].values())
            or delta["parameter_count"] != 310
            or delta["nonzero_parameter_count"] != 1
        ):
            raise RuntimeError(f"invalid propagation discovery state: {state_id}")
    vectors = np.asarray(
        [artifact["parameter_gradient_delta"]["countsketch8192"] for artifact in artifacts],
        dtype=np.float64,
    )
    carrier = vectors.mean(axis=0)
    norm = float(np.linalg.norm(carrier))
    if norm == 0.0:
        raise RuntimeError("full-step carrier is zero")
    direction = carrier / norm
    projections = vectors @ direction
    cosines = projections / np.linalg.norm(vectors, axis=1)
    payload: dict[str, Any] = {
        "schema_version": "kernel-analyzer.liger-fused-ce-parameter-carrier.v1",
        "status": "FROZEN_BEFORE_24_CONFIRMATION_VALUES",
        "representation": "parameter-name-keyed CountSketch8192(seed=3407) over all 310 default-minus-FP32-accum gradients",
        "state_ids": ids,
        "normalized_values": direction.tolist(),
        "discovery_mean_norm": norm,
        "discovery_projections": [float(value) for value in projections],
        "discovery_cosines": [float(value) for value in cosines],
        "discovery_global_l2": [artifact["parameter_gradient_delta"]["global_l2"] for artifact in artifacts],
        "bindings": {
            "protocol": {"path": str(protocol_path), "sha256": protocol_hash},
            "local_certificate": {"path": str(local_path), "sha256": sha256(local_path)},
            "states": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths],
        },
    }
    payload["artifact_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"output": str(output), "projection_min": float(projections.min()), "cosine_min": float(cosines.min()), "cosine_max": float(cosines.max())}, sort_keys=True))


if __name__ == "__main__":
    main()
