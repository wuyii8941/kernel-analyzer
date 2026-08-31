#!/usr/bin/env python3
"""Apply the two frozen Holm families and emit a compact result table."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/extended_unified_profiles_v1"
BRANCHES = ("additive_heldout_effect", "aligned_effect", "orthogonal_heldout_effect")


def load(name: str) -> dict:
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def holm(items: list[tuple[float, tuple[str, ...]]]) -> dict[tuple[str, ...], float]:
    running = 0.0
    output = {}
    for rank, (p, key) in enumerate(sorted(items)):
        running = max(running, min(1.0, (len(items) - rank) * p))
        output[key] = running
    return output


def compact(effect: dict, adjusted: float, branch: str) -> dict:
    interval = effect["bootstrap_95"]
    interval_excludes_zero = interval[2] < 0.0 or interval[0] > 0.0
    # The additive and residual-direction branches learn a direction on the
    # calibration half.  A negative confirmation effect means reversal, not
    # successful replication.  The aligned coefficient has an intrinsic sign,
    # so either nonzero direction is a valid scaling effect.
    direction_matches = branch == "aligned_effect" or effect["estimate"] > 0.0
    return {
        "estimate": effect["estimate"],
        "confidence_interval_95": [interval[0], interval[2]],
        "raw_signflip_p": effect["signflip_p"],
        "holm_adjusted_p": adjusted,
        "confirmation_direction_matches_calibration": direction_matches,
        "confirmed_after_holm": adjusted <= 0.05 and interval_excludes_zero and direction_matches,
    }


def main() -> None:
    protocol = load("protocol.json")
    ordinary = {
        name: load(f"{name}.json")
        for name in ("qwen_lmhead", "qwen_vproj", "mamba_inproj")
    }
    response = {
        name: load(f"{name}_response.json")
        for name in ("saved_p", "silu")
    }
    ordinary_tests = []
    for case, data in ordinary.items():
        for stage, profile in data["stages"].items():
            for branch in BRANCHES:
                ordinary_tests.append((float(profile[branch]["signflip_p"]), (case, stage, branch)))
    response_tests = []
    for case, data in response.items():
        for branch in BRANCHES:
            response_tests.append((float(data["profile"][branch]["signflip_p"]), (case, branch)))
    if len(ordinary_tests) != protocol["ordinary_implementation_family"]["holm_family_size"]:
        raise RuntimeError("ordinary test family differs from frozen protocol")
    if len(response_tests) != protocol["response_family"]["holm_family_size"]:
        raise RuntimeError("response test family differs from frozen protocol")
    ordinary_adjusted = holm(ordinary_tests)
    response_adjusted = holm(response_tests)

    result = {
        "schema": "kernel-analyzer-extended-unified-profile-summary-v1",
        "status": "COMPLETE_TWO_FROZEN_HOLM_FAMILIES",
        "protocol": "results/property/extended_unified_profiles_v1/protocol.json",
        "ordinary_implementation_family": {},
        "response_family": {},
        "claim_boundary": (
            "Effects are short-state matched measurements. Response remainders are a "
            "separate contrast family and are not counted as ordinary implementation cases."
        ),
    }
    for case, data in ordinary.items():
        result["ordinary_implementation_family"][case] = {
            "case_id": data["case_id"],
            "optimizer": data["optimizer"],
            "stages": {
                stage: {
                    branch: compact(profile[branch], ordinary_adjusted[(case, stage, branch)], branch)
                    for branch in BRANCHES
                }
                for stage, profile in data["stages"].items()
            },
        }
    for case, data in response.items():
        result["response_family"][case] = {
            "case_id": data["case_id"],
            "contrast_id": data["contrast_id"],
            "effects": {
                branch: compact(data["profile"][branch], response_adjusted[(case, branch)], branch)
                for branch in BRANCHES
            },
        }
    (BASE / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "ordinary_tests": 27, "response_tests": 6}))


if __name__ == "__main__":
    main()
