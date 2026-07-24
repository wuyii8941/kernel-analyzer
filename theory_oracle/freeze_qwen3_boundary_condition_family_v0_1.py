#!/usr/bin/env python3
"""Freeze support-complete boundary endpoints without inspecting candidate effects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)


SCHEMA_VERSION = "forkcert.qwen3-boundary-condition-family.v0.1"
CALIBRATION_SCHEMA = (
    "forkcert.qwen3-boundary-conditioned-four-trajectory-calibration.v0.1"
)
CALIBRATION_STATUS = (
    "VALID_COMPLETE_FOUR_TRAJECTORY_BOUNDARY_CALIBRATION_DESCRIPTION"
)
CANDIDATE_TAUS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
REQUIRED_PHASES = ("early", "middle", "late")
MIN_EXPOSED_STATES_PER_TRAJECTORY_PHASE = 2
SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
EXPECTED_ANALYSIS = {
    "multi_trajectory_aggregator": ROOT
    / "theory_oracle"
    / "aggregate_qwen3_boundary_conditioned_multi_v0_1.py",
    "one_trajectory_aggregator": ROOT
    / "theory_oracle"
    / "aggregate_qwen3_boundary_conditioned_calibration_v0_1.py",
}
CONTRACT_PATH = (
    ROOT
    / "theory_oracle"
    / "QWEN3_BOUNDARY_CONDITIONAL_B_CONFIRMATION_CONTRACT_V0_1_2026-07-20.md"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confirmation_outcome_exists(root: Path) -> bool:
    if not root.exists():
        return False
    names = {
        "record_validation.json",
        "record_bundle.json",
        "confirmation_result.json",
    }
    return any(path.name in names for path in root.rglob("*.json"))


def freeze_family(
    calibration: dict[str, Any], calibration_path: Path
) -> dict[str, Any]:
    errors: list[str] = []
    if calibration.get("schema_version") != CALIBRATION_SCHEMA:
        errors.append("unsupported boundary calibration schema")
    if calibration.get("valid") is not True or calibration.get("status") != CALIBRATION_STATUS:
        errors.append("boundary calibration is not complete and valid")
    construction = calibration.get("construction", {})
    if construction.get("trajectories") != 4 or construction.get("states") != 96:
        errors.append("boundary family freeze requires four complete 24-state trajectories")
    if construction.get("tau_grid") != CANDIDATE_TAUS:
        errors.append("boundary candidate tau grid drifted")
    if construction.get("weighting_contract_id") != WEIGHTING_CONTRACT_ID:
        errors.append("boundary calibration weighting contract drifted")
    anchor = construction.get("reference_anchor_identification", {})
    if not (
        anchor.get("all_four_trajectories_exact") is True
        and anchor.get("trajectories_with_all_24_exact") == 4
        and anchor.get("exact_states") == 96
        and anchor.get("observed_states") == 96
    ):
        errors.append("boundary calibration reference anchor is not exact")
    analysis = calibration.get("analysis_code", {})
    for name, path in EXPECTED_ANALYSIS.items():
        link = analysis.get(name, {})
        if (
            Path(link.get("path", "")).resolve() != path.resolve()
            or not path.is_file()
            or link.get("sha256") != sha256_file(path)
        ):
            errors.append(f"boundary calibration analysis provenance failed: {name}")

    retained: list[float] = []
    support_audit: dict[str, Any] = {}
    profiles = calibration.get("tau_profiles", {})
    for tau in CANDIDATE_TAUS:
        key = str(tau)
        profile = profiles.get(key, {})
        trajectory_rows = profile.get("trajectory_rows", [])
        eligible = len(trajectory_rows) == 4
        rows = []
        identities = set()
        for row in trajectory_rows:
            trajectory_id = row.get("trajectory_id")
            identities.add(trajectory_id)
            phase_support = row.get("reference_side_phase_support", {})
            phase_counts = {
                phase: int(phase_support.get(phase, {}).get("states_with_exposure", -1))
                for phase in REQUIRED_PHASES
            }
            current_eligible = all(
                phase_counts[phase] >= MIN_EXPOSED_STATES_PER_TRAJECTORY_PHASE
                for phase in REQUIRED_PHASES
            )
            eligible = eligible and current_eligible
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "states_with_exposure_by_phase": phase_counts,
                    "support_complete": current_eligible,
                }
            )
        if len(identities) != 4 or None in identities:
            eligible = False
        support_audit[key] = {
            "retained": eligible,
            "trajectory_rows": rows,
            "selection_fields_read": [
                "trajectory_id",
                "reference_side_phase_support.*.states_with_exposure",
            ],
            "candidate_effect_fields_read": [],
        }
        if eligible:
            retained.append(tau)

    valid = not errors
    status = (
        "FROZEN_SUPPORT_COMPLETE_BOUNDARY_FAMILY"
        if valid and retained
        else "UNINSTANTIATED_BOUNDARY_FAMILY_NO_SUPPORT"
        if valid
        else "INVALID_BOUNDARY_FAMILY_FREEZE"
    )
    endpoint_family = [
        endpoint
        for tau in retained
        for endpoint in (
            f"boundary_margin_shift::tau={tau}",
            f"boundary_clip_directional_shift::tau={tau}",
            f"boundary_semantic_disagreement::tau={tau}",
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "status": status,
        "calibration": {
            "path": str(calibration_path),
            "sha256": sha256_file(calibration_path),
        },
        "contract": {"path": str(CONTRACT_PATH), "sha256": sha256_file(CONTRACT_PATH)},
        "analysis_code": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "candidate_tau_grid": CANDIDATE_TAUS,
        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
        "minimum_exposed_states_per_trajectory_phase": MIN_EXPOSED_STATES_PER_TRAJECTORY_PHASE,
        "retained_taus": retained,
        "support_audit": support_audit,
        "endpoint_family": endpoint_family,
        "confirmatory_comparisons": len(endpoint_family),
        "selection_rule": "retain every candidate tau with at least two reference-side exposed states in every calibration trajectory x phase; inspect no candidate effect field",
        "uses_candidate_margin_shift_or_event_effect_for_selection": False,
        "disagreement_endpoint_role": "CONFIRMABLE_NONNEGATIVE_SEMANTIC_IMPACT_NOT_B",
        "operator_attribution_allowed": False,
        "next_gate": (
            "bind all endpoints to confirmation multiplicity and independent trajectory bank"
            if retained
            else "start no boundary confirmation in v0.1"
        ),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--confirmation-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    calibration_path = Path(args.calibration).resolve()
    confirmation_root = Path(args.confirmation_root).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit("boundary family manifest already exists; freeze is write-once")
    if confirmation_outcome_exists(confirmation_root):
        raise SystemExit("confirmation outcome already exists; refusing post-outcome freeze")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    result = freeze_family(calibration, calibration_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "retained_taus": result["retained_taus"], "errors": result["errors"]}, indent=2))
    if result["valid"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
