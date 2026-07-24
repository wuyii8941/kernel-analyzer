#!/usr/bin/env python
"""Prepare the predeclared 32-prompt Qwen3 greedy-impact confirmation bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_SHA256 = "8822a63fa26b73605d37380c7fe792d97f606776b59ad7eba52d7be99e599862"
EXPECTED_DISCOVERY_SHA256 = "9dbaa3d5940e4aa29e89529ffc0fd68fd70ab22341ee9843d48f04560a5906bd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--discovery-source", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-manifest", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_key(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in row.get("prompt_ids", []))


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    discovery = Path(args.discovery_source)
    source_hash = sha256_file(source)
    discovery_hash = sha256_file(discovery)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"confirmation source hash mismatch: {source_hash}")
    if discovery_hash != EXPECTED_DISCOVERY_SHA256:
        raise ValueError(f"discovery source hash mismatch: {discovery_hash}")

    discovery_prompts = {prompt_key(row) for row in read_jsonl(discovery)}
    selected = []
    selected_prompts: set[tuple[int, ...]] = set()
    for row in read_jsonl(source):
        key = prompt_key(row)
        if (
            key in discovery_prompts
            or key in selected_prompts
            or len(key) >= 64
            or not row.get("response_ids")
        ):
            continue
        selected.append(row)
        selected_prompts.add(key)
        if len(selected) == 32:
            break
    if len(selected) != 32:
        raise ValueError(f"selection produced {len(selected)} rows, expected 32")
    rollout_batches = {
        (row.get("metadata") or {}).get("rollout_batch") for row in selected
    }
    if len(selected_prompts) != 32 or len(rollout_batches) != 32:
        raise ValueError("selected bank lacks 32 unique prompts and rollout batches")

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=False)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    output_hash = sha256_file(out_jsonl)
    manifest = {
        "schema_version": "forkcert.qwen3-impact-confirmation-bank.v0.1",
        "contract": str(
            (Path(__file__).parent / "QWEN3_GREEDY_IMPACT_CONFIRMATION_CONTRACT_V0_1_2026-07-17.md").resolve()
        ),
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "discovery_source": str(discovery.resolve()),
        "discovery_source_sha256": discovery_hash,
        "output": str(out_jsonl.resolve()),
        "output_sha256": output_hash,
        "rows": len(selected),
        "unique_prompts": len(selected_prompts),
        "unique_rollout_batches": len(rollout_batches),
        "case_ids": [str(row["case_id"]) for row in selected],
        "selection": (
            "file order; exclude discovery prompts, duplicate prompts, prompt length >= 64, "
            "or empty response; first 32 eligible"
        ),
    }
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

