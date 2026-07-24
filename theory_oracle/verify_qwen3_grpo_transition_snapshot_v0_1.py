#!/usr/bin/env python
"""Independently audit a full pre-minibatch Qwen GRPO transition snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensor_manifest(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if torch.is_tensor(value):
        rows.append(
            {
                "path": prefix or "<root>",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": tensor_sha256(value),
            }
        )
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            rows.extend(tensor_manifest(value[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            rows.extend(tensor_manifest(item, f"{prefix}[{index}]"))
    return rows


def trusted_torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def audit(snapshot_dir: Path) -> dict[str, Any]:
    required = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "forkcert_transition_snapshot.json",
    ]
    missing = [name for name in required if not (snapshot_dir / name).is_file()]
    if missing:
        return {"valid": False, "verdict": "INVALID", "missing": missing}

    metadata = json.loads(
        (snapshot_dir / "forkcert_transition_snapshot.json").read_text(encoding="utf-8")
    )
    trainer_state = json.loads((snapshot_dir / "trainer_state.json").read_text(encoding="utf-8"))
    history = metadata.get("compiler_history") or []
    history_steps = [int(row.get("optimizer_step", -1)) for row in history]
    target_step = int(metadata.get("optimizer_step", -1))
    num_iterations = int(metadata.get("num_iterations", 3))
    history_selection = metadata.get("history_selection", "FINAL_POLICY_ITERATION_ONLY")
    if history_selection == "EVERY_OPTIMIZER_PRE_STEP":
        expected_steps = list(range(0, target_step + 1)) if target_step >= 0 else []
    elif history_selection == "FINAL_POLICY_ITERATION_ONLY":
        expected_steps = (
            list(range(num_iterations - 1, target_step + 1, num_iterations))
            if target_step >= 0 and num_iterations > 0
            else []
        )
    else:
        expected_steps = []
    history_records = []
    history_valid = history_steps == expected_steps
    for row in history:
        path = Path(str(row.get("path", ""))).resolve()
        try:
            path.relative_to(snapshot_dir.resolve())
            inside_snapshot = True
        except ValueError:
            inside_snapshot = False
        exists = path.is_file()
        manifest_matches = False
        tensor_count = 0
        if inside_snapshot and exists:
            value = trusted_torch_load(path)
            observed = tensor_manifest(value)
            tensor_count = len(observed)
            manifest_matches = observed == row.get("tensor_manifest")
            del value
        valid = (
            inside_snapshot
            and exists
            and manifest_matches
            and bool(row.get("torch_rng_preserved"))
            and bool(row.get("python_rng_preserved"))
            and bool(row.get("numpy_rng_preserved"))
            and bool(row.get("gradients_preserved"))
            and bool(row.get("tensor_versions_preserved"))
        )
        history_valid = history_valid and valid
        history_records.append(
            {
                "optimizer_step": int(row.get("optimizer_step", -1)),
                "path": str(path),
                "inside_snapshot": inside_snapshot,
                "exists": exists,
                "tensor_count": tensor_count,
                "tensor_manifest_matches": manifest_matches,
                "preservation_flags": {
                    "torch_rng": bool(row.get("torch_rng_preserved")),
                    "python_rng": bool(row.get("python_rng_preserved")),
                    "numpy_rng": bool(row.get("numpy_rng_preserved")),
                    "gradients": bool(row.get("gradients_preserved")),
                    "tensor_versions": bool(row.get("tensor_versions_preserved")),
                },
                "valid": valid,
            }
        )

    optimizer = trusted_torch_load(snapshot_dir / "optimizer.pt")
    scheduler = trusted_torch_load(snapshot_dir / "scheduler.pt")
    scaler = trusted_torch_load(snapshot_dir / "scaler.pt")
    rng = trusted_torch_load(snapshot_dir / "rng_state.pth")
    optimizer_valid = (
        isinstance(optimizer, dict)
        and isinstance(optimizer.get("state"), dict)
        and bool(optimizer["state"])
        and isinstance(optimizer.get("param_groups"), list)
        and bool(optimizer["param_groups"])
    )
    scheduler_valid = isinstance(scheduler, dict) and bool(scheduler)
    scaler_valid = isinstance(scaler, dict) and "scale" in scaler
    rng_valid = (
        isinstance(rng, dict)
        and set(rng) == {"torch", "python", "numpy"}
        and isinstance(rng["torch"], dict)
        and torch.is_tensor(rng["torch"].get("cpu"))
        and isinstance(rng["torch"].get("cuda"), list)
        and len(rng["torch"]["cuda"]) == 1
    )
    del optimizer, scheduler, scaler, rng

    model_path = snapshot_dir / "model.safetensors"
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        model_tensor_count = len(handle.keys())
    model_valid = model_tensor_count > 0
    target_path = Path(str(metadata.get("target_minibatch_path", ""))).resolve()
    target_valid = bool(history_records) and target_path == Path(history_records[-1]["path"])
    preservation_valid = (
        bool(metadata.get("torch_rng_preserved"))
        and bool(metadata.get("python_rng_preserved"))
        and bool(metadata.get("numpy_rng_preserved"))
        and bool(metadata.get("gradients_preserved"))
        and bool(metadata.get("tensor_versions_preserved"))
        and metadata.get("gradient_signature_before") == metadata.get("gradient_signature_after")
        and metadata.get("tensor_versions_before") == metadata.get("tensor_versions_after")
    )
    identity_valid = (
        metadata.get("schema_version") == "forkcert.full-pre-minibatch-transition-state.v0.1"
        and metadata.get("state") == "pre_minibatch"
        and target_step >= 0
        and (
            history_selection == "EVERY_OPTIMIZER_PRE_STEP"
            or int(metadata.get("policy_iteration", -1)) == num_iterations - 1
        )
        and int(trainer_state.get("global_step", -1)) == target_step
    )
    valid = all(
        (
            history_valid,
            optimizer_valid,
            scheduler_valid,
            scaler_valid,
            rng_valid,
            model_valid,
            target_valid,
            preservation_valid,
            identity_valid,
        )
    )
    paths = [snapshot_dir / name for name in required]
    paths.extend(Path(row["path"]) for row in history_records)
    files = {
        str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    }
    return {
        "schema_version": "forkcert.transition-snapshot-audit.v0.1",
        "snapshot_dir": str(snapshot_dir.resolve()),
        "valid": valid,
        "verdict": "VALID" if valid else "INVALID",
        "identity_valid": identity_valid,
        "history_steps": history_steps,
        "expected_history_steps": expected_steps,
        "history_selection": history_selection,
        "history_valid": history_valid,
        "history_records": history_records,
        "optimizer_valid": optimizer_valid,
        "scheduler_valid": scheduler_valid,
        "scaler_valid": scaler_valid,
        "rng_valid": rng_valid,
        "model_tensor_count": model_tensor_count,
        "model_valid": model_valid,
        "target_minibatch_valid": target_valid,
        "capture_preservation_valid": preservation_valid,
        "files": files,
        "nonclaims": [
            "snapshot validity is not replay identity",
            "snapshot validity is not transition impact",
            "snapshot validity is not numerical correctness",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = audit(Path(args.snapshot_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["valid"] else 1)


if __name__ == "__main__":
    main()
