#!/usr/bin/env python3
"""Apply the frozen 18-test Holm family to Liger/Phi anchor profiles."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/anchor_unified_profiles_v1"
BRANCHES = ("additive_heldout_effect", "aligned_effect", "orthogonal_heldout_effect")


def main() -> None:
    cases = {name: json.loads((BASE / f"{name}.json").read_text()) for name in ("liger", "phi")}
    rows = []
    for case, data in cases.items():
        for stage, profile in data["stages"].items():
            for branch in BRANCHES:
                rows.append((float(profile[branch]["signflip_p"]), case, stage, branch))
    running = 0.0
    corrected = {}
    for rank, (p, case, stage, branch) in enumerate(sorted(rows)):
        running = max(running, min(1.0, (len(rows) - rank) * p))
        corrected[(case, stage, branch)] = running
    for case, data in cases.items():
        for stage, profile in data["stages"].items():
            for branch in BRANCHES:
                profile[branch]["holm_p_frozen_18_test_family"] = corrected[(case, stage, branch)]
    payload = {
        "schema": "kernel-analyzer-anchor-unified-profile-summary-v1",
        "status": "COMPLETE_FROZEN_18_TEST_FAMILY",
        "protocol": "results/property/anchor_unified_profiles_v1/protocol.json",
        "cases": cases,
        "multiple_testing": "Holm across 2 cases x 3 stages x 3 effect types",
    }
    (BASE / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
