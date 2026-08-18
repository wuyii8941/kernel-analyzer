"""State-atomic checkpoint journal for frozen candidate screens."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_gzip(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def state_checkpoint_path(directory: Path, state_id: str) -> Path:
    """Return a filesystem-safe, collision-resistant path for one state."""
    key = hashlib.sha256(state_id.encode()).hexdigest()
    return directory / f"{key}.json.gz"


def load_state_checkpoints(
    *, directory: Path, release_capture_sha256: str,
    triton_payload: dict[str, Any], nontriton_payload: dict[str, Any],
) -> None:
    """Merge state-atomic journal entries into the in-memory screen payloads."""
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            row = json.load(handle)
        if row.get("schema") != "kernel-analyzer-joint-frozen-candidate-state-v1":
            raise RuntimeError(f"unknown joint state checkpoint schema: {path}")
        if row.get("release_capture_sha256") != release_capture_sha256:
            raise RuntimeError(f"joint state checkpoint binds another release: {path}")
        state_id = row["state_id"]
        expected_path = state_checkpoint_path(directory, state_id)
        if path != expected_path:
            raise RuntimeError(f"joint state checkpoint filename mismatch: {path}")
        for payload, key in (
            (triton_payload, "triton_state"),
            (nontriton_payload, "nontriton_state"),
        ):
            existing = payload["states"].get(state_id)
            incoming = row[key]
            if existing is not None and _canonical_hash(existing) != _canonical_hash(incoming):
                raise RuntimeError(f"conflicting checkpoint for state {state_id}")
            payload["states"][state_id] = incoming
