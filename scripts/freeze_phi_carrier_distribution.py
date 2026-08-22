#!/usr/bin/env python3
"""Freeze an outcome-blind, depth-stratified Phi carrier sample.

The sample is derived only from model depth and parameter names.  No candidate
measurement, persistence score, historical case label, or trajectory result is
read.  The resulting manifest is the input to the carrier-distribution run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "results/property/joint_bias_formation_v1/carrier_distribution/manifest.json"
)


def evenly_spaced_indices(last: int, count: int) -> list[int]:
    if count < 2 or last < 1:
        raise ValueError("the frozen stratification requires at least two depth points")
    values = [round(index * last / (count - 1)) for index in range(count)]
    if len(set(values)) != count:
        raise ValueError("depth grid contains duplicates")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_depths = evenly_spaced_indices(31, 6)
    post_depths = evenly_spaced_indices(31, 5)
    carriers = ["model.norm.weight"]
    carriers += [f"model.layers.{depth}.input_layernorm.weight" for depth in input_depths]
    carriers += [
        f"model.layers.{depth}.post_attention_layernorm.weight" for depth in post_depths
    ]
    if len(carriers) != 12 or len(set(carriers)) != 12:
        raise RuntimeError("the frozen sample must contain 12 distinct carriers")

    rows = []
    for index, name in enumerate(carriers):
        if name == "model.norm.weight":
            stratum, depth = "FINAL_NORM", None
        elif ".input_layernorm." in name:
            stratum, depth = "INPUT_LAYERNORM", int(name.split(".")[2])
        else:
            stratum, depth = "POST_ATTENTION_LAYERNORM", int(name.split(".")[2])
        rows.append({
            "index": index,
            "carrier": name,
            "stratum": stratum,
            "layer_index": depth,
            "shard": index % 4,
        })

    selection = {
        "model": "microsoft/Phi-4-mini-instruct",
        "endpoint": "phi4_seq64_lmhead_dx_mm",
        "rule": (
            "final norm plus six input-layernorm depths round(linspace(0,31,6)) "
            "and five post-attention-layernorm depths round(linspace(0,31,5))"
        ),
        "input_layernorm_depths": input_depths,
        "post_attention_layernorm_depths": post_depths,
        # The final norm is a previously studied positive anchor.  It is kept
        # in the distribution for calibration, but must never be described as
        # outcome-blind.  The other eleven locations are selected without
        # reading candidate measurements or historical verdicts.
        "known_anchor_carriers": ["model.norm.weight"],
        "outcome_blind_carriers": [row["carrier"] for row in rows if row["carrier"] != "model.norm.weight"],
        "known_anchor_count": 1,
        "outcome_blind_count": 11,
        "uses_candidate_values": False,
        "uses_historical_persistence": False,
        "uses_case_names_to_select_depth": False,
    }
    digest = hashlib.sha256(
        json.dumps({"selection": selection, "carriers": rows}, sort_keys=True).encode()
    ).hexdigest()
    payload = {
        "schema": "kernel-analyzer-phi-carrier-distribution-manifest-v1",
        "status": "FROZEN_BEFORE_GPU_MEASUREMENT_WITH_ANCHOR_DECLARED",
        "selection": selection,
        "carrier_count": len(rows),
        "carriers": rows,
        "selection_sha256": digest,
        "premeasurement_correction": {
            "supersedes": "carrier_distribution/manifest.json",
            "reason": "The previous manifest called all twelve rows outcome-blind even though model.norm.weight was a known historical anchor.",
            "gpu_science_results_before_correction": False,
        },
        "protocol": {
            "steps": 32,
            "optimizer": "SGD_FP32_MASTER",
            "learning_rate": 1e-3,
            "contrast": "candidate lm_head dX MM versus matched FP32-accumulation repair",
            "only_one_declared_carrier_evolves_per_trajectory": True,
            "all_other_parameters_frozen": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "carriers": len(rows), "sha256": digest}))


if __name__ == "__main__":
    main()
