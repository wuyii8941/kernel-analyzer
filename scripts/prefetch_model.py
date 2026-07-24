#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch a Hugging Face model into ForkCert data-disk cache.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default="/data1/tzh/forkcert/cache/huggingface")
    parser.add_argument("--out", default="results/model_prefetch.json")
    args = parser.parse_args()

    hub_cache = Path(args.cache_dir) / "hub"
    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_cache))
    hub_cache.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    path = snapshot_download(args.model, cache_dir=str(hub_cache))
    payload = {
        "model": args.model,
        "cache_dir": args.cache_dir,
        "hub_cache": str(hub_cache),
        "snapshot_path": path,
        "ok": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
