#!/usr/bin/env python3
"""Verify the compact final-result package."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results" / "final"


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    summary = (ROOT / "summary.json").read_bytes()
    if hashlib.sha256(summary).hexdigest() != manifest["summary"]["sha256"]:
        raise ValueError("summary digest differs")

    for name, row in manifest["archives"].items():
        compressed = (ROOT / name).read_bytes()
        if hashlib.sha256(compressed).hexdigest() != row["sha256"]:
            raise ValueError(f"archive digest differs: {name}")

        payload = gzip.decompress(compressed)
        if len(payload) != row["uncompressed_bytes"]:
            raise ValueError(f"archive size differs: {name}")

        result = json.loads(payload)
        if set(result["results"]) != set(row["results"]):
            raise ValueError(f"archive contents differ: {name}")

    print("results/final: valid")


if __name__ == "__main__":
    main()
