#!/usr/bin/env python3
"""Combine four complete boundary diagnostics at the trajectory level."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)


SCHEMA_VERSION = "forkcert.qwen3-boundary-conditioned-four-trajectory-calibration.v0.1"
SCRIPT_PATH = Path(__file__).resolve()
ONE_TRAJECTORY_AGGREGATOR_PATH = (
    SCRIPT_PATH.parent / "aggregate_qwen3_boundary_conditioned_calibration_v0_1.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_complete(path: Path) -> tuple[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE_CALIBRATION_BOUNDARY_DESCRIPTION":
        raise ValueError(f"incomplete boundary summary: {path}")
    states = value.get("states", [])
    identities = {row.get("trajectory_id") for row in states}
    if len(states) != 24 or len(identities) != 1 or None in identities:
        raise ValueError(f"boundary summary lacks one complete 24-state trajectory: {path}")
    return str(next(iter(identities))), value


def combine(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 4:
        raise ValueError("exactly four trajectory summaries are required")
    loaded = [load_complete(path) for path in paths]
    trajectory_ids = [item[0] for item in loaded]
    if len(set(trajectory_ids)) != 4:
        raise ValueError("trajectory summaries must have four distinct identities")
    tau_grids = [item[1]["construction"]["tau_grid"] for item in loaded]
    if any(grid != tau_grids[0] for grid in tau_grids[1:]):
        raise ValueError("trajectory boundary tau grids differ")
    if any(
        item[1].get("construction", {}).get("weighting_contract_id")
        != WEIGHTING_CONTRACT_ID
        for item in loaded
    ):
        raise ValueError("trajectory boundary weighting contract drifted")
    reference_anchor_rows = []
    for trajectory_id, value in loaded:
        anchor = value.get("construction", {}).get(
            "reference_anchor_identification", {}
        )
        exact = anchor.get(
            "observed_states_with_exact_reference_scorer_logps_across_repeats"
        )
        total = anchor.get("observed_states_total")
        if exact != 24 or total != 24 or anchor.get("all_observed_states_exact") is not True:
            raise ValueError(
                f"trajectory reference anchor is not exact for all states: {trajectory_id}"
            )
        reference_anchor_rows.append(
            {
                "trajectory_id": trajectory_id,
                "exact_states": int(exact),
                "observed_states": int(total),
            }
        )
    expected_one_trajectory_hash = sha256_file(ONE_TRAJECTORY_AGGREGATOR_PATH)
    for path, (_, value) in zip(paths, loaded, strict=True):
        analysis = value.get("analysis_code", {}).get("boundary_aggregator", {})
        if (
            Path(analysis.get("path", "")).resolve()
            != ONE_TRAJECTORY_AGGREGATOR_PATH
            or analysis.get("sha256") != expected_one_trajectory_hash
        ):
            raise ValueError(f"trajectory boundary analysis provenance drifted: {path}")

    tau_rows: dict[str, Any] = {}
    for tau in tau_grids[0]:
        key = str(tau)
        rows = []
        for trajectory_id, value in loaded:
            profile = value["aggregate"]["tau_profiles"][key]
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "states_with_exposure": profile["states_with_exposure"],
                    "mean_margin_shift": profile[
                        "phase_balanced_state_weighted_mean_margin_shift"
                    ],
                    "directional_event_shift": profile[
                        "phase_balanced_directional_event_shift"
                    ],
                    "semantic_disagreement": profile[
                        "phase_balanced_semantic_disagreement"
                    ],
                    "all_phase_conditionals_identified": profile[
                        "all_phase_conditionals_identified"
                    ],
                    "reference_side_phase_support": {
                        phase: {
                            "sampled_states": phase_profile["sampled_states"],
                            "states_with_exposure": phase_profile[
                                "states_with_exposure"
                            ],
                            "total_exposures_descriptive_only": phase_profile[
                                "total_exposures_descriptive_only"
                            ],
                        }
                        for phase, phase_profile in profile[
                            "phase_profiles"
                        ].items()
                    },
                }
            )
        available = [row for row in rows if row["mean_margin_shift"] is not None]
        tau_rows[key] = {
            "trajectory_rows": rows,
            "trajectories_with_exposure": len(available),
            "calibration_mean_of_trajectory_margin_shifts": (
                sum(float(row["mean_margin_shift"]) for row in available)
                / len(available)
                if available
                else None
            ),
            "trajectory_effect_sign_counts": {
                "positive": sum(float(row["mean_margin_shift"]) > 0 for row in available),
                "zero": sum(float(row["mean_margin_shift"]) == 0 for row in available),
                "negative": sum(float(row["mean_margin_shift"]) < 0 for row in available),
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "status": "VALID_COMPLETE_FOUR_TRAJECTORY_BOUNDARY_CALIBRATION_DESCRIPTION",
        "construction": {
            "trajectories": 4,
            "states": 96,
            "top_level_df_for_future_inference": 3,
            "tau_grid": tau_grids[0],
            "tau_selected": False,
            "weighting": "equal trajectory; equal phase; equal exposed state within phase; equal near-boundary token within state",
            "weighting_contract_id": WEIGHTING_CONTRACT_ID,
            "reference_anchor_identification": {
                "all_four_trajectories_exact": True,
                "trajectories_with_all_24_exact": 4,
                "exact_states": 96,
                "observed_states": 96,
                "trajectory_rows": reference_anchor_rows,
            },
        },
        "inputs": [
            {
                "trajectory_id": trajectory_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "plan": value["plan"],
            }
            for path, (trajectory_id, value) in zip(paths, loaded, strict=True)
        ],
        "analysis_code": {
            "multi_trajectory_aggregator": {
                "path": str(SCRIPT_PATH),
                "sha256": sha256_file(SCRIPT_PATH),
            },
            "one_trajectory_aggregator": {
                "path": str(ONE_TRAJECTORY_AGGREGATOR_PATH),
                "sha256": expected_one_trajectory_hash,
            },
        },
        "tau_profiles": tau_rows,
        "claims_allowed": {
            "calibration_description": True,
            "population_conditional_B": False,
            "operator_attribution": False,
            "correctness": False,
        },
        "next_gate": "freeze any retained tau and endpoint family before independent confirmation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-summary", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = combine([Path(path).resolve() for path in args.trajectory_summary])
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "construction": result["construction"]}, indent=2))


if __name__ == "__main__":
    main()
