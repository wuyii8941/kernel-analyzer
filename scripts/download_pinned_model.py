#!/usr/bin/env python3
"""Download one immutable Hugging Face snapshot below /data1/tzh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DATA_ROOT = Path("/data1/tzh").resolve()


def file_sha256(path: Path, chunk_bytes: int = 16 * 2**20) -> str:
    """Hash large model shards without materializing them in host RAM."""
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--min-free-gib", type=int, default=500)
    args = parser.parse_args()

    output = args.output.resolve()
    manifest = args.manifest.resolve()
    if not output.is_relative_to(DATA_ROOT) or not manifest.is_relative_to(DATA_ROOT):
        raise RuntimeError("model and manifest must live below /data1/tzh")
    free_gib = shutil.disk_usage(DATA_ROOT).free / 2**30
    if free_gib < args.min_free_gib:
        raise RuntimeError(f"only {free_gib:.1f} GiB free below required floor")
    hf_home = Path(os.environ.get("HF_HOME", "")).resolve()
    if not hf_home.is_relative_to(DATA_ROOT):
        raise RuntimeError("HF_HOME must be below /data1/tzh")

    info = HfApi().model_info(args.model_id, revision=args.revision)
    revision = info.sha
    snapshot_download(
        repo_id=args.model_id,
        revision=revision,
        local_dir=output,
        cache_dir=Path(os.environ["HUGGINGFACE_HUB_CACHE"]),
    )
    files = []
    for path in sorted(
        p for p in output.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(output).parts
    ):
        files.append({
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    payload = {
        "schema": "kernel-analyzer-pinned-model-v1",
        "model_id": args.model_id,
        "revision": revision,
        "local_path": str(output),
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files),
        "tensor_values_copied_into_repository": False,
    }
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "model": args.model_id,
        "revision": revision,
        "bytes": payload["total_bytes"],
        "manifest": str(manifest),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
