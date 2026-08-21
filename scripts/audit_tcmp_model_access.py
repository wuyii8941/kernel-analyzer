#!/usr/bin/env python3
"""Audit exact-revision local model availability without reading tensor data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import get_token


def local_path(root: Path, model_id: str) -> Path:
    owner, name = model_id.split("/", 1)
    return root / owner / name


def audit_model(path: Path, revision: str) -> dict:
    config = path / "config.json"
    metadata = path / ".cache/huggingface/download/config.json.metadata"
    observed_revision = metadata.read_text().splitlines()[0] if metadata.exists() else None
    missing_weights = []
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        missing_weights = sorted({name for name in weight_map.values() if not (path / name).is_file()})
    elif not any(path.glob("*.safetensors")):
        missing_weights = ["NO_SAFETENSORS_OR_INDEX"]
    complete = config.is_file() and observed_revision == revision and not missing_weights
    return {
        "path": str(path), "expected_revision": revision,
        "observed_revision": observed_revision, "config_present": config.is_file(),
        "missing_weight_files": missing_weights,
        "status": "READY" if complete else (
            "DOWNLOAD_IN_PROGRESS" if path.exists() and observed_revision == revision
            else "MISSING_OR_UNAUTHORIZED"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/data1/tzh/models"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roster = json.loads(args.roster.read_text())
    unique = {}
    for cell in roster["cells"]:
        unique[cell["model_id"]] = cell["model_revision"]
    rows = {
        model_id: audit_model(local_path(args.model_root, model_id), revision)
        for model_id, revision in sorted(unique.items())
    }
    payload = {
        "schema": "kernel-analyzer-tcmp-model-access-v1",
        "status": "READY" if all(row["status"] == "READY" for row in rows.values()) else "BLOCKED_MODEL_ACCESS",
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or get_token()),
        "models": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "counts": {
        status: sum(row["status"] == status for row in rows.values())
        for status in sorted({row["status"] for row in rows.values()})
    }}, sort_keys=True))


if __name__ == "__main__":
    main()
