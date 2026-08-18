#!/usr/bin/env python3
"""Freeze compact tensor-size manifests for all 12 large-candidate cells."""

from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_large_coordinate_case import frozen_large_candidates  # noqa: E402


def main() -> None:
    for model in ("qwen", "phi4", "mamba", "deepseek8b"):
        for length in (64, 128, 256):
            cell = f"{model}_seq{length}_r1"
            release = ROOT / "results/coverage/runtime_releases" / cell
            rows = frozen_large_candidates(release, 4096)
            print(json.dumps({"event": "MANIFEST_COMPLETE", "cell": cell,
                              "large_candidates": len(rows),
                              "total_coordinates": sum(size for _, size in rows),
                              "max_tensor_coordinates": max((size for _, size in rows), default=0),
                              "max_single_spool_bytes": max((size for _, size in rows), default=0) * 32 * 4}),
                  flush=True)
            del rows
            gc.collect()


if __name__ == "__main__":
    main()
