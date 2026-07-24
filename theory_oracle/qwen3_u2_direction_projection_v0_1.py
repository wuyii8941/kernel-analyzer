"""Project signed U2 delta artifacts onto one frozen calibration direction."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


DIRECTION_VERSION = "forkcert.qwen3-u2-frozen-direction.v0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_direction(link: dict[str, Any]) -> dict[str, Any]:
    path_value = link.get("path") if isinstance(link, dict) else None
    expected = link.get("sha256") if isinstance(link, dict) else None
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError("U2 direction link lacks path/sha256")
    path = Path(path_value).resolve()
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError("U2 direction manifest identity failed")
    direction = json.loads(path.read_text(encoding="utf-8"))
    if (
        direction.get("schema_version") != DIRECTION_VERSION
        or direction.get("valid") is not True
        or direction.get("verdict") != "VALID_FROZEN_U2_CALIBRATION_DIRECTION"
        or direction.get("status") != "FROZEN_BEFORE_CONFIRMATION"
        or direction.get("endpoint_name") != "U2_calibration_direction_shift"
    ):
        raise ValueError("U2 direction manifest is not valid/frozen")
    norm = float(direction.get("direction", {}).get("normalization_l2", 0.0))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("U2 direction normalization must be positive and finite")
    shards = direction.get("direction", {}).get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("U2 direction has no shards")
    by_key: dict[str, dict[str, str]] = {}
    for artifact in shards:
        key = artifact.get("tensor_key")
        shard_path = Path(artifact.get("path", "")).resolve()
        shard_sha = artifact.get("sha256")
        if not isinstance(key, str) or not isinstance(shard_sha, str):
            raise ValueError("U2 direction shard lacks tensor key/hash")
        if key in by_key:
            raise ValueError("U2 direction contains duplicate tensor keys")
        if not shard_path.is_file() or sha256_file(shard_path) != shard_sha:
            raise ValueError(f"U2 direction shard identity failed: {key}")
        by_key[key] = {"path": str(shard_path), "sha256": shard_sha}
    return {
        "path": str(path),
        "sha256": expected,
        "normalization_l2": norm,
        "by_key": by_key,
        "manifest": direction,
    }


def project_delta_artifact(
    artifact: dict[str, Any], frozen_direction: dict[str, Any]
) -> float:
    from safetensors import safe_open

    path_value = artifact.get("path") if isinstance(artifact, dict) else None
    expected = artifact.get("sha256") if isinstance(artifact, dict) else None
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError("U2 delta artifact lacks path/sha256")
    path = Path(path_value).resolve()
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError("U2 delta artifact identity failed")
    delta = safe_open(path, framework="pt", device="cpu")
    keys = sorted(delta.keys())
    direction_by_key = frozen_direction["by_key"]
    if keys != sorted(direction_by_key):
        raise ValueError("U2 delta and frozen direction tensor keys differ")
    direction_handles: dict[str, Any] = {}
    dot = 0.0
    for key in keys:
        source = direction_by_key[key]
        shard_path = source["path"]
        if shard_path not in direction_handles:
            direction_handles[shard_path] = safe_open(
                shard_path, framework="pt", device="cpu"
            )
        direction = direction_handles[shard_path]
        if sorted(direction.keys()) != [key]:
            raise ValueError(f"U2 direction shard key mismatch: {key}")
        delta_tensor = delta.get_tensor(key)
        direction_tensor = direction.get_tensor(key)
        if tuple(delta_tensor.shape) != tuple(direction_tensor.shape):
            raise ValueError(f"U2 direction/delta shape mismatch: {key}")
        dot += float((delta_tensor.double() * direction_tensor.double()).sum().item())
    return dot / float(frozen_direction["normalization_l2"])

