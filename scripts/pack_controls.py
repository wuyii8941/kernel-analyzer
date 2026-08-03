#!/usr/bin/env python3
"""Register compact next-round controls in the final-result manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest_path = ROOT / "results/final/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("flash_control.json", "implementation_atlas.json"):
        path = ROOT / "results/final" / name
        data = path.read_bytes()
        manifest.setdefault("files", {})[name] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    summary = ROOT / "results/final/summary.json"
    manifest["summary"] = {
        "bytes": summary.stat().st_size,
        "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"registered": ["flash_control.json", "implementation_atlas.json"]}))


if __name__ == "__main__":
    main()
