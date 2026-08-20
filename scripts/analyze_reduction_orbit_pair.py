#!/usr/bin/env python3
"""Compare the preregistered Phi/Qwen semantic-orbit pair."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/persistence_v1/orbits/comparison.json"


def load(name: str) -> dict:
    return json.loads((OUT.parent / name).read_text())


def row(case_id: str, artifact: dict, effective_update_a: float) -> dict:
    stats = artifact["statistics"]
    return {
        "case_id": case_id,
        "orbit_mean_energy_fraction": stats["orbit_mean_energy_fraction"],
        "orbit_mean_temporal_amplification": stats["orbit_mean"]["coherence_amplification"],
        "orbit_mean_sign_flip_p": stats["orbit_mean"]["sign_flip_null"]["one_sided_p"],
        "schedule_residual_temporal_amplification": stats["default_minus_orbit_mean"]["coherence_amplification"],
        "default_source_temporal_amplification": stats["default_schedule"]["coherence_amplification"],
        "effective_update_temporal_amplification": effective_update_a,
        "source_to_update_amplification_ratio": (
            effective_update_a / stats["default_schedule"]["coherence_amplification"]
        ),
    }


def main() -> None:
    rows = [
        row("phi4_seq64_lmhead_dx", load("phi_mm.json"), 3.2890916370209418),
        row("qwen128_vproj_mm", load("qwen128_vproj.json"), 0.9990920610983055),
    ]
    payload = {
        "schema": "kernel-analyzer-reduction-orbit-paired-analysis-v1",
        "status": "DEVELOPMENT_PAIR_COMPLETE",
        "cases": rows,
        "hypothesis_verdicts": {
            "ORBIT_MEAN_ENERGY_FRACTION_ALONE": "COUNTEREXAMPLE_FOUND",
            "reason": (
                "Qwen has a larger orbit-mean energy fraction than Phi but no local-effective-update "
                "persistence. Magnitude of the orbit mean is therefore not the property."
            ),
            "TEMPORAL_ORBIT_MEAN_PLUS_TRANSPORT": "SURVIVES_TWO_CASE_DEVELOPMENT_ONLY",
        },
        "next_decisive_experiment": (
            "Deliver a different real reduction-orbit member at each Phi step. If the schedule-specific "
            "anchor matters, local-effective-update persistence must fall without changing real semantics."
        ),
        "claim_boundary": "Two development cases cannot establish a predictive cross-kernel property.",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
