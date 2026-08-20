#!/usr/bin/env python3
"""Compare feedback amplification to the preregistered bmm matched floor."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/persistence_v1/development_baseline.json"
OUT = ROOT / "results/property/persistence_v1/feedback_matched_floor.json"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = {row["case_id"]: row for row in source["cases"]}
    floor = rows["qwen_bmm_seq64"]["levels"]["feedback"]["coherence_amplification"]
    comparisons = []
    for case_id in ("qwen_saved_p_seq128", "qwen128_vproj_mm", "qwen3vl_silu_layer0"):
        value = rows[case_id]["levels"]["feedback"]["coherence_amplification"]
        comparisons.append({
            "case_id": case_id,
            "feedback_amplification": value,
            "bmm_floor": floor,
            "ratio_to_bmm_floor": value / floor,
            "exceeds_bmm_floor": value > floor,
            "mechanism_verdict": "UNRESOLVED_WITHOUT_MATCHED_PERTURBATION_NULL",
        })
    result = {
        "schema": "kernel-analyzer-feedback-matched-floor-v1",
        "status": "DEVELOPMENT_DIAGNOSTIC",
        "null_case": "qwen_bmm_seq64",
        "comparisons": comparisons,
        "scientific_conclusion": (
            "A_B>1 is not sufficient evidence of feedback-sustained bias: the bmm hard "
            "control itself has A_B=%.6f. Saved-P does not exceed that floor; Qwen128 "
            "and SiLU require an RMS-matched random-injection null before attribution."
        ) % floor,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
