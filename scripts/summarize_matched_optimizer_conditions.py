#!/usr/bin/env python3
"""Compare cold, warm, and warm-parameter/reset-moment profile captures.

The three inputs must measure the same case and the same ordered states.  All
reported RMS and aligned values are recomputed from original-coordinate
statistics; directional sketches are deliberately not used as norm bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


STAGES = ("LOCAL", "PARAMETER_GRADIENT", "PARAMETER_WRITE")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_summary(document: dict, stage: str) -> dict:
    rows = document.get("original_coordinate_statistics", {}).get(stage, [])
    confirmation = rows[16:32]
    if len(rows) != 32 or len(confirmation) != 16:
        raise ValueError(f"{stage} requires exactly 32 original-coordinate rows")
    effect = math.fsum(float(row["effect_energy"]) for row in confirmation)
    repair = math.fsum(float(row["repair_energy"]) for row in confirmation)
    inner = math.fsum(
        float(row["effect_repair_inner_product"]) for row in confirmation
    )
    if effect < 0 or repair <= 0 or not all(
        math.isfinite(value) for value in (effect, repair, inner)
    ):
        raise ValueError(f"invalid {stage} original-coordinate statistics")
    return {
        "confirmation_total_rms": math.sqrt(effect / repair),
        "confirmation_aligned_ratio_of_sums": inner / repair,
        "confirmation_nonzero_state_count": sum(
            int(row.get("nonzero_effect_coordinates", 0)) > 0
            for row in confirmation
        ),
    }


def summarize(cold: dict, warm: dict, reset: dict) -> dict:
    documents = {"COLD_ZERO_MOMENTS": cold, "WARM_MOMENTS": warm,
                 "WARM_PARAMETERS_RESET_MOMENTS": reset}
    case_ids = {str(value.get("case_id")) for value in documents.values()}
    state_orders = {tuple(value.get("state_ids", [])) for value in documents.values()}
    if len(case_ids) != 1 or len(state_orders) != 1 or len(next(iter(state_orders))) != 32:
        raise ValueError("conditions must use one case and the same 32 ordered states")
    if any(value.get("status") != "COMPLETE" for value in documents.values()):
        raise ValueError("all condition captures must be complete")
    condition_summaries = {
        name: {stage: stage_summary(value, stage) for stage in STAGES}
        for name, value in documents.items()
    }
    cold_write = condition_summaries["COLD_ZERO_MOMENTS"]["PARAMETER_WRITE"][
        "confirmation_total_rms"
    ]
    warm_write = condition_summaries["WARM_MOMENTS"]["PARAMETER_WRITE"][
        "confirmation_total_rms"
    ]
    reset_write = condition_summaries[
        "WARM_PARAMETERS_RESET_MOMENTS"
    ]["PARAMETER_WRITE"]["confirmation_total_rms"]
    warm_gradient_equal_after_reset = (
        warm["original_coordinate_statistics"]["PARAMETER_GRADIENT"]
        == reset["original_coordinate_statistics"]["PARAMETER_GRADIENT"]
    )
    return {
        "schema": "kernel-analyzer-matched-optimizer-condition-summary-v1",
        "status": "COMPLETE_FIXED_SUITE_DESCRIPTIVE_COMPARISON",
        "case_id": next(iter(case_ids)),
        "state_ids": list(next(iter(state_orders))),
        "conditions": condition_summaries,
        "comparisons": {
            "warm_to_cold_parameter_write_rms_ratio": warm_write / cold_write,
            "reset_to_warm_parameter_write_rms_ratio": reset_write / warm_write,
            "reset_to_cold_parameter_write_rms_ratio": reset_write / cold_write,
            "warm_and_reset_gradient_statistics_exactly_equal": (
                warm_gradient_equal_after_reset
            ),
        },
        "interpretation_boundary": (
            "Same fixed 32-state suite and one target parameter. This isolates "
            "the measured AdamW-state condition but is not a random-state, "
            "full-model-training, or loss-outcome claim."
        ),
        "prediction_status": "POST_OUTCOME_DIAGNOSTIC_NOT_PREREGISTERED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--reset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    inputs = [args.cold, args.warm, args.reset]
    result = summarize(*(json.loads(path.read_text()) for path in inputs))
    result["input_sha256"] = {str(path.resolve()): sha(path) for path in inputs}
    result["summarizer_sha256"] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
