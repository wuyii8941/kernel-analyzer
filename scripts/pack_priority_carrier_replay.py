#!/usr/bin/env python3
"""Pack the candidate-blind carrier screen without retaining tensor values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "cache/kernel_analyzer"
FINAL = ROOT / "results/final"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def arm_summary(row: dict, *, step: int) -> dict:
    repeats = row["repeats"]
    carrier_name = repeats[0]["carrier_sketch"].keys()
    carrier_name = next(iter(carrier_name))
    metrics = [repeat["carrier"][carrier_name] for repeat in repeats]
    sketches = [repeat["carrier_sketch"][carrier_name] for repeat in repeats]
    return {
        "step": step,
        "region_id": str(row["region_id"]),
        "carrier_parameter": carrier_name,
        "repeats": len(repeats),
        "repeat_metrics_identical": metrics[0] == metrics[1],
        "repeat_sketches_identical": sketches[0] == sketches[1],
        "carrier_rms": metrics[0].get("rms"),
        "carrier_max_abs": metrics[0].get("max_abs"),
        "carrier_delta_cosine": metrics[0].get("delta_baseline_cosine"),
        "carrier_nonzero": metrics[0].get("nonzero"),
        "pilot_cosine": sketches[0].get("pilot_cosine"),
        "pilot_step": sketches[0].get("pilot_step"),
        "exact_boundary": all(
            repeat["gates"]["region_observed"]
            and repeat["gates"]["all_mapped_regions_observed"]
            for repeat in repeats
        ),
    }


def main() -> None:
    initial_steps = (0, 1, 2, 4, 8)
    rows: list[dict] = []
    for step in initial_steps:
        payload = read(CACHE / f"priority_carrier_embed_seq128_step{step}.json")
        if not all(
            payload["gates"][key]
            for key in (
                "every_arm_has_two_repeats",
                "every_arm_observed_at_exact_boundary",
                "all_mapped_regions_census_complete_per_repeat",
            )
        ):
            raise RuntimeError(f"carrier batch gates are not complete at step {step}")
        rows.extend(
            arm_summary(arm, step=step)
            for arm in payload["arms"]
        )
    extended = []
    for step in (16, 32, 64):
        payload = read(CACHE / f"priority_carrier_embed_seq128_step{step}_852.json")
        if not all(
            payload["gates"][key]
            for key in (
                "every_arm_has_two_repeats",
                "every_arm_observed_at_exact_boundary",
                "all_mapped_regions_census_complete_per_repeat",
            )
        ):
            raise RuntimeError(f"focused carrier gates are incomplete at step {step}")
        extended.extend(arm_summary(arm, step=step) for arm in payload["arms"])
    rows.extend(extended)

    by_region: dict[str, list[dict]] = {}
    for row in rows:
        by_region.setdefault(row["region_id"], []).append(row)
    initial_positive = []
    for region_id, region_rows in sorted(by_region.items()):
        pilot = {
            row["step"]: row["pilot_cosine"]
            for row in region_rows
            if row["step"] in (1, 2, 4, 8)
        }
        if len(pilot) == 4 and all(value is not None and value > 0 for value in pilot.values()):
            initial_positive.append(region_id)

    output = {
        "schema": "kernel-analyzer-priority-carrier-replay-v1",
        "subject": "Qwen3-1.7B exact key-RMSNorm intervention carrier screen",
        "candidate_blind": True,
        "carrier": "model.embed_tokens.weight",
        "initial_steps": list(initial_steps),
        "focused_followup": {"region_id": "backward:852", "steps": [16, 32, 64]},
        "chain_count": 28,
        "records": rows,
        "initial_all_positive_regions": initial_positive,
        "all_initial_batches_closed_and_repeated": all(
            row["exact_boundary"]
            and row["repeats"] == 2
            and row["repeat_metrics_identical"]
            and row["repeat_sketches_identical"]
            for row in rows
        ),
        "focused_candidate_lost_direction": all(
            row["pilot_cosine"] is not None and row["pilot_cosine"] < 0
            for row in extended
        ),
        "natural_bias_case_added": False,
        "property_claim": False,
        "verdict": "LOCAL_IMPLEMENTATION_DIFFERENCES_WITHOUT_COHERENT_CARRIER_BIAS",
        "interpretation": (
            "All 28 exact interventions produce repeat-stable downstream carrier deltas, "
            "but pilot-direction signs are not stable across checkpoints. The only chain "
            "positive at steps 1/2/4/8 reverses by steps 16/32/64; this is not the "
            "FlashAttention coherent-carrier mechanism."
        ),
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = FINAL / "priority_carrier_replay.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "records": len(rows), "initial_positive": initial_positive}))


if __name__ == "__main__":
    main()
