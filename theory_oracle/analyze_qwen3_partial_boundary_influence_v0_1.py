#!/usr/bin/env python3
"""Leave-one-state-out influence diagnostic for a partial boundary summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    ALL_ELIGIBLE_WEIGHTING_CONTRACT_ID,
    REQUIRED_PHASES,
    SCHEMA_VERSION as BOUNDARY_SCHEMA_VERSION,
    WEIGHTING_CONTRACT_ID,
)


SCHEMA_VERSION = "forkcert.qwen3-partial-boundary-influence.v0.1"
SCRIPT_PATH = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def influence_profile(rows: list[tuple[str, float]]) -> dict[str, Any]:
    if any(not math.isfinite(value) for _, value in rows):
        raise ValueError("influence profile received a nonfinite effect")
    if not rows:
        return {
            "states": 0,
            "full_sample_mean": None,
            "leave_one_out_identified": False,
            "leave_one_out_rows": [],
            "leave_one_out_mean_range": [None, None],
            "leave_one_out_signs": [],
            "minimum_states_for_stability_interpretation": 3,
            "stability_interpretable": False,
            "sign_stable_to_any_single_state_deletion": None,
        }
    full = sum(value for _, value in rows) / len(rows)
    loo = []
    if len(rows) >= 2:
        for state_id, _ in rows:
            remaining = [value for current, value in rows if current != state_id]
            current_mean = sum(remaining) / len(remaining)
            loo.append(
                {
                    "deleted_state_id": state_id,
                    "mean": current_mean,
                    "sign": sign(current_mean),
                    "change_from_full_sample_mean": current_mean - full,
                }
            )
    loo_signs = sorted({row["sign"] for row in loo})
    return {
        "states": len(rows),
        "full_sample_mean": full,
        "full_sample_sign": sign(full),
        "leave_one_out_identified": bool(loo),
        "leave_one_out_rows": loo,
        "leave_one_out_mean_range": (
            [min(row["mean"] for row in loo), max(row["mean"] for row in loo)]
            if loo
            else [None, None]
        ),
        "leave_one_out_signs": loo_signs,
        "minimum_states_for_stability_interpretation": 3,
        "stability_interpretable": len(rows) >= 3,
        "sign_stable_to_any_single_state_deletion": (
            len(loo_signs) == 1 and loo_signs[0] == sign(full) if loo else None
        ),
    }


def analyze(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    if summary.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
        errors.append("unsupported boundary-summary schema")
    construction = summary.get("construction", {})
    if construction.get("weighting_contract_id") != WEIGHTING_CONTRACT_ID:
        errors.append("boundary-summary weighting contract drifted")
    if (
        construction.get("all_eligible_weighting_contract_id")
        != ALL_ELIGIBLE_WEIGHTING_CONTRACT_ID
    ):
        errors.append("all-eligible weighting contract drifted")
    states = summary.get("states")
    if not isinstance(states, list) or not states:
        errors.append("boundary summary has no states")
        states = []
    phase_profiles = {}
    for phase in REQUIRED_PHASES:
        current = [row for row in states if row.get("phase") == phase]
        for row in current:
            value = row.get("all_eligible_mean_margin_shift")
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                errors.append(
                    f"{row.get('state_id')}: nonfinite all-eligible margin shift"
                )
        identified_all_eligible = [
            row
            for row in current
            if row.get("all_eligible_endpoint_status") in (None, "IDENTIFIED")
            and isinstance(row.get("all_eligible_mean_margin_shift"), (int, float))
            and math.isfinite(float(row["all_eligible_mean_margin_shift"]))
        ]
        phase_profiles[phase] = {
            "sampled_states": len(current),
            "states_with_identified_all_eligible_endpoint": len(
                identified_all_eligible
            ),
            "uninstantiated_all_eligible_state_ids": [
                str(row.get("state_id"))
                for row in current
                if row not in identified_all_eligible
            ],
            "all_eligible_mean_margin_shift": influence_profile(
                [
                    (str(row["state_id"]), float(row["all_eligible_mean_margin_shift"]))
                    for row in identified_all_eligible
                ]
            ),
            "tau_profiles": {},
        }
        for tau in construction.get("tau_grid", []):
            key = str(tau)
            metrics = {}
            for metric in (
                "mean_margin_shift",
                "directional_event_shift",
                "semantic_disagreement",
            ):
                metric_rows = []
                for row in current:
                    profile = row.get("tau_profiles", {}).get(key, {})
                    value = profile.get(metric)
                    if (
                        profile.get("exposures", 0) > 0
                        and isinstance(value, (int, float))
                        and not math.isfinite(float(value))
                    ):
                        errors.append(
                            f"{row.get('state_id')}: nonfinite {metric} at tau={key}"
                        )
                    if (
                        profile.get("exposures", 0) > 0
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                    ):
                        metric_rows.append((str(row["state_id"]), float(value)))
                metrics[metric] = influence_profile(metric_rows)
            phase_profiles[phase]["tau_profiles"][key] = metrics
    valid = not errors
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "status": (
            "VALID_PARTIAL_CALIBRATION_INFLUENCE_DESCRIPTION"
            if valid
            else "INVALID_PARTIAL_CALIBRATION_INFLUENCE"
        ),
        "input": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "construction": {
            "observed_states": len(states),
            "weighting_contract_id": WEIGHTING_CONTRACT_ID,
            "all_eligible_weighting_contract_id": ALL_ELIGIBLE_WEIGHTING_CONTRACT_ID,
            "unit_deleted": "one matched state within its declared phase",
            "phases_pooled": False,
            "population_inference_allowed": False,
        },
        "phase_profiles": phase_profiles,
        "analysis_code": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "errors": errors,
        "nonclaims": [
            "leave-one-state-out stability is not population-B confirmation",
            "sign instability is finite-sample influence, not proof that population B is zero",
            "leave-one-out sign stability with fewer than three states is mechanical and not interpreted",
            "semantic disagreement remains nonnegative impact and is not Bias",
            "missing phases are not imputed or pooled",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = analyze(summary, summary_path)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "construction": result["construction"], "errors": result["errors"]}, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
