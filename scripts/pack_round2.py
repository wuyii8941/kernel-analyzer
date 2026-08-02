#!/usr/bin/env python3
"""Pack the compact round-2 evidence into the tracked final-result bundle."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "inventory": ROOT / "results/round2/vl_inventory.json.gz",
    "bf16_math": ROOT / "results/round2/vl_math_ledger.json.gz",
    "fp32_math": ROOT / "results/round2/vl_math_ledger_fp32.json.gz",
    "bf16_cause": ROOT / "results/round2/vl_silu_cause.json",
    "fp32_cause": ROOT / "results/round2/vl_silu_cause_fp32.json",
    "bias": ROOT / "results/round2/vl_bias.json",
    "smoke": ROOT / "results/round2/vl_smoke.json",
}


def _read(path: Path) -> object:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    results = {name: _read(path) for name, path in SOURCES.items()}
    payload = {
        "schema": "kernel-analyzer-qwen3-vl-round2-v1",
        "results": results,
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    output = ROOT / "results/final/vl.json.gz"
    with output.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as handle:
            handle.write(encoded)

    manifest_path = ROOT / "results/final/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compressed = output.read_bytes()
    manifest["archives"][output.name] = {
        "compressed_bytes": len(compressed),
        "uncompressed_bytes": len(encoded),
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "results": {
            name: {
                "source": str(path.relative_to(ROOT)),
                "canonical_sha256": _canonical_sha256(results[name]),
            }
            for name, path in SOURCES.items()
        },
    }
    summary = (ROOT / "results/final/summary.json").read_bytes()
    manifest["summary"] = {
        "bytes": len(summary),
        "sha256": hashlib.sha256(summary).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "compressed_bytes": len(compressed),
                "uncompressed_bytes": len(encoded),
                "results": sorted(results),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
