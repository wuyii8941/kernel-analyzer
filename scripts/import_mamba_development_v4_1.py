#!/usr/bin/env python3
"""Register an existing complete Mamba consequence as development evidence.

This is deliberately an import, not a new held-out label.  The source result
was produced before v4.1 and is kept separate from the frozen v4.1 roster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/joint_bias_formation_v1/consequence/mamba_seq64_multishape-backward-cell-0450.json"
DEFAULT_OUT = ROOT / "results/property/direct_persistence_v4_1/development/mamba_0450_summary.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    required = ("local", "feedback", "actual")
    levels = source.get("statistics", {}).get("levels", {})
    if source.get("status") != "COMPLETE" or any(name not in levels for name in required):
        raise SystemExit("source Mamba consequence is incomplete")
    result = {
        "schema": "kernel-analyzer-direct-persistence-v4_1-development-summary-v1",
        "status": "COMPLETE_DERIVED_DEVELOPMENT",
        "role": "DEVELOPMENT_CROSS_ARCHITECTURE_NOT_HELDOUT",
        "case_id": source.get("case_id"),
        "model": source.get("model"),
        "carrier": source.get("carrier"),
        "sequence_length": 64,
        "steps": source.get("steps"),
        "optimizer": source.get("optimizer"),
        "levels": {
            name: {
                "A32": levels[name].get("coherence_amplification"),
                "resultant_l2": levels[name].get("resultant_l2"),
                "path_l2": levels[name].get("path_l2"),
            }
            for name in required
        },
        "source": {
            "path": str(args.source.relative_to(ROOT)),
            "sha256": digest(args.source),
            "not_heldout": True,
        },
        "claim_boundary": "This is a compact import of an existing complete Mamba consequence. It is cross-architecture development evidence, not a new prospective held-out result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "case_id": result["case_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
