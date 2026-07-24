#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an audited Transformers-4-compatible vLLM checkpoint view.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    dest = Path(args.dest).resolve()
    if source == dest:
        raise ValueError("source and destination must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)
    dest.mkdir(parents=True, exist_ok=True)

    config_name = "tokenizer_config.json"
    files = [path for path in source.iterdir() if path.is_file()]
    for src in files:
        if src.name in {config_name, "vllm_compat_manifest.json"}:
            continue
        target = dest / src.name
        if target.exists():
            target.unlink()
        os.link(src, target)

    tokenizer_config = json.loads((source / config_name).read_text(encoding="utf-8"))
    original_extra = tokenizer_config.get("extra_special_tokens")
    transformation = "none"
    if isinstance(original_extra, list):
        # Transformers 4.53 expects this metadata field to be a mapping. The
        # actual special-token vocabulary remains unchanged in tokenizer.json.
        tokenizer_config["extra_special_tokens"] = {}
        transformation = "extra_special_tokens:list_to_empty_mapping"
    elif original_extra is not None and not isinstance(original_extra, dict):
        raise TypeError(f"unsupported extra_special_tokens type: {type(original_extra).__name__}")
    (dest / config_name).write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": "forkcert.vllm_checkpoint_compat.v1",
        "source": str(source),
        "destination": str(dest),
        "transformation": transformation,
        "preserved_extra_special_tokens": original_extra,
        "model_sha256_source": sha256(source / "model.safetensors"),
        "model_sha256_destination": sha256(dest / "model.safetensors"),
        "tokenizer_sha256_source": sha256(source / "tokenizer.json"),
        "tokenizer_sha256_destination": sha256(dest / "tokenizer.json"),
    }
    if manifest["model_sha256_source"] != manifest["model_sha256_destination"]:
        raise RuntimeError("model hash changed in compatibility view")
    if manifest["tokenizer_sha256_source"] != manifest["tokenizer_sha256_destination"]:
        raise RuntimeError("tokenizer hash changed in compatibility view")
    (dest / "vllm_compat_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
