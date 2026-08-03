#!/usr/bin/env python3
"""Pack the complete Liger paired-trajectory evidence into final results."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/trajectory"
OUTPUT = ROOT / "results/final/trajectory.json.gz"


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    protocol = read(SOURCE / "liger_protocol.json")
    campaign = read(SOURCE / "liger_trajectory.json")
    steps = {
        "default": [read(SOURCE / "liger_steps" / f"default_step{step:02d}.json") for step in range(32)],
        "repair": [read(SOURCE / "liger_steps" / f"repair_step{step:02d}.json") for step in range(32)],
        "pairs": [read(SOURCE / "liger_steps" / f"pair_step{step:02d}.json") for step in range(32)],
    }
    results = {"protocol": protocol, "campaign": campaign, "steps": steps}
    payload = {"schema": "kernel-analyzer-liger-trajectory-v1", "results": results}
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as handle:
            handle.write(encoded)

    manifest_path = ROOT / "results/final/manifest.json"
    manifest = read(manifest_path)
    compressed = OUTPUT.read_bytes()
    manifest["archives"][OUTPUT.name] = {
        "compressed_bytes": len(compressed),
        "uncompressed_bytes": len(encoded),
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "results": {
            "protocol": {
                "source": "results/trajectory/liger_protocol.json",
                "canonical_sha256": canonical_sha256(protocol),
            },
            "campaign": {
                "source": "results/trajectory/liger_trajectory.json",
                "canonical_sha256": canonical_sha256(campaign),
            },
            "steps": {
                "source": "results/trajectory/liger_steps/{default,repair,pair}_step00..31.json",
                "canonical_sha256": canonical_sha256(steps),
            },
        },
    }
    summary = (ROOT / "results/final/summary.json").read_bytes()
    manifest["summary"] = {
        "bytes": len(summary),
        "sha256": hashlib.sha256(summary).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "compressed_bytes": len(compressed),
        "uncompressed_bytes": len(encoded),
        "steps": 32,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
