#!/usr/bin/env python3
"""Download exact-revision TCMP checkpoints into /data1/tzh/models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/data1/tzh/models"))
    parser.add_argument("--model-id", action="append", default=[])
    args = parser.parse_args()
    root = args.model_root.resolve()
    if Path("/data1/tzh") not in (root, *root.parents):
        raise ValueError("model root must remain under /data1/tzh")
    roster = json.loads(args.roster.read_text())
    models = {}
    for cell in roster["cells"]:
        models[cell["model_id"]] = cell["model_revision"]
    selected = args.model_id or sorted(models)
    unknown = sorted(set(selected) - set(models))
    if unknown:
        raise ValueError(f"models are absent from frozen roster: {unknown}")
    for model_id in selected:
        owner, name = model_id.split("/", 1)
        target = root / owner / name
        target.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"event": "DOWNLOAD_START", "model_id": model_id, "revision": models[model_id]}), flush=True)
        snapshot_download(
            repo_id=model_id, revision=models[model_id], local_dir=target,
            token=os.environ.get("HF_TOKEN"),
        )
        print(json.dumps({"event": "DOWNLOAD_COMPLETE", "model_id": model_id, "path": str(target)}), flush=True)


if __name__ == "__main__":
    main()
