#!/usr/bin/env python3
"""Freeze a disjoint reference-mask bank for boundary operator confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    REQUIRED_PHASES,
    WEIGHTING_CONTRACT_ID,
)
from theory_oracle.freeze_qwen3_boundary_attribution_bridge_v0_1 import (
    SCHEMA_VERSION as BRIDGE_SCHEMA_VERSION,
)


INPUT_SCHEMA_VERSION = (
    "forkcert.qwen3-boundary-independent-contributor-bank-input.v0.1"
)
SCHEMA_VERSION = "forkcert.qwen3-boundary-independent-contributor-bank.v0.1"
SCRIPT_PATH = Path(__file__).resolve()
TRAJECTORY_FIELDS = ("trajectory_id", "trajectory_seed", "data_slice_id")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_link(
    link: Any, base: Path, *, path_key: str = "path", hash_key: str = "sha256"
) -> tuple[dict[str, Any] | None, Path | None]:
    if not isinstance(link, dict) or not isinstance(link.get(path_key), str):
        return None, None
    path = resolve_from(base, link[path_key])
    if not path.is_file() or link.get(hash_key) != sha256_file(path):
        return None, path
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except (OSError, json.JSONDecodeError):
        return None, path


def forbidden_identities_from_bridge(
    bridge: dict[str, Any], bridge_path: Path
) -> tuple[dict[str, set[Any]], list[str]]:
    errors: list[str] = []
    confirmation, confirmation_path = load_link(
        bridge.get("endpoint_confirmation"), bridge_path.parent
    )
    if confirmation is None or confirmation_path is None:
        return {field: set() for field in TRAJECTORY_FIELDS}, [
            "bridge endpoint-confirmation identity failed"
        ]
    manifest, _ = load_link(
        confirmation.get("confirmation_manifest"), confirmation_path.parent
    )
    if manifest is None:
        return {field: set() for field in TRAJECTORY_FIELDS}, [
            "endpoint confirmation manifest identity failed"
        ]
    forbidden = {field: set() for field in TRAJECTORY_FIELDS}
    for row in manifest.get("trajectory_inputs", []):
        if isinstance(row, dict):
            for field in TRAJECTORY_FIELDS:
                if row.get(field) is not None:
                    forbidden[field].add(row[field])
    exclusion = manifest.get("calibration_exclusion", {})
    mapping = {
        "trajectory_id": "trajectory_ids",
        "trajectory_seed": "trajectory_seeds",
        "data_slice_id": "data_slice_ids",
    }
    for field, plural in mapping.items():
        values = exclusion.get(plural, []) if isinstance(exclusion, dict) else []
        if not isinstance(values, list):
            errors.append(f"confirmation manifest {plural} exclusion is invalid")
        else:
            forbidden[field].update(values)
    return forbidden, errors


def freeze_independent_bank(
    bank_input: dict[str, Any],
    bank_input_path: Path,
    bridge: dict[str, Any],
    bridge_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    if bank_input.get("schema_version") != INPUT_SCHEMA_VERSION:
        errors.append("unsupported independent contributor-bank input schema")
    if bank_input.get("status") != "REFERENCE_MASKS_FROZEN_BEFORE_OPERATOR_INTERVENTIONS":
        errors.append("reference masks were not frozen before operator interventions")
    if bank_input.get("operator_intervention_outcomes_exist") is not False:
        errors.append("operator outcomes already exist or their absence is not declared")
    if (
        bridge.get("schema_version") != BRIDGE_SCHEMA_VERSION
        or bridge.get("valid") is not True
        or bridge.get("status") != "FROZEN_BEFORE_BOUNDARY_ATTRIBUTION_PILOT"
        or bridge.get("state_bank_role")
        != "ENDPOINT_CONFIRMATION_BANK_REUSED_FOR_MECHANISM_PILOT"
        or bridge.get("weighting_contract_id") != WEIGHTING_CONTRACT_ID
        or bridge.get("population_operator_contribution_claim_allowed") is not False
        or bridge.get("frozen_effect_direction") not in {-1, 1}
        or bridge.get("target_effect_role")
        not in {
            "SIGNED_BOUNDARY_CONDITIONAL_EFFECT",
            "NONNEGATIVE_SEMANTIC_IMPACT_NOT_B",
        }
    ):
        errors.append("endpoint-definition bridge is not valid")
    bridge_link = bank_input.get("endpoint_definition_bridge", {})
    if (
        not isinstance(bridge_link, dict)
        or resolve_from(bank_input_path.parent, str(bridge_link.get("path", "")))
        != bridge_path.resolve()
        or bridge_link.get("sha256") != sha256_file(bridge_path)
    ):
        errors.append("independent bank is not bound to the exact endpoint bridge")
    if bank_input.get("target_endpoint") != bridge.get("target_endpoint"):
        errors.append("independent bank endpoint differs from confirmed endpoint")
    if bank_input.get("target_effect_role") != bridge.get("target_effect_role"):
        errors.append("independent bank target effect role differs from endpoint bridge")
    if bank_input.get("weighting_contract_id") != WEIGHTING_CONTRACT_ID:
        errors.append("independent bank weighting contract drifted")

    forbidden, identity_errors = forbidden_identities_from_bridge(bridge, bridge_path)
    errors.extend(identity_errors)
    trajectories = bank_input.get("trajectory_inputs")
    if not isinstance(trajectories, list) or len(trajectories) < 8:
        errors.append("at least eight independent contributor trajectories are required")
        trajectories = []
    trajectory_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(trajectories):
        if not isinstance(row, dict) or any(row.get(field) is None for field in TRAJECTORY_FIELDS):
            errors.append(f"trajectory_inputs[{index}] identity is incomplete")
            continue
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in trajectory_by_id:
            errors.append("independent contributor trajectory IDs must be unique")
        trajectory_by_id[trajectory_id] = row
        for field in TRAJECTORY_FIELDS:
            if row[field] in forbidden[field]:
                errors.append(f"trajectory_inputs[{index}] reuses forbidden {field}")
    for field in TRAJECTORY_FIELDS[1:]:
        values = [row.get(field) for row in trajectories if isinstance(row, dict)]
        if len(values) != len(set(values)):
            errors.append(f"independent contributor {field} values must be unique")

    rows = bank_input.get("state_rows")
    if not isinstance(rows, list):
        errors.append("independent contributor state_rows must be a list")
        rows = []
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    state_ids: list[str] = []
    masks: dict[str, str] = {}
    exposed_groups: dict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"state_rows[{index}] is not an object")
            continue
        trajectory = str(row.get("trajectory_id"))
        phase = row.get("phase")
        state_id = row.get("state_id")
        exposures = row.get("reference_exposures")
        mask_hash = row.get("reference_condition_mask_sha256")
        if trajectory not in trajectory_by_id or phase not in REQUIRED_PHASES:
            errors.append(f"state_rows[{index}] trajectory/phase is outside frozen bank")
            continue
        if not isinstance(state_id, str) or not state_id:
            errors.append(f"state_rows[{index}] lacks state_id")
            continue
        state_ids.append(state_id)
        by_trajectory[trajectory].append(row)
        if (
            not isinstance(exposures, int)
            or exposures < 0
            or not isinstance(mask_hash, str)
            or len(mask_hash) != 64
            or row.get("reference_scorer_logps_exact_across_repeats") is not True
            or row.get("mask_frozen_before_operator_interventions") is not True
        ):
            errors.append(f"state_rows[{index}] reference-mask evidence is invalid")
            continue
        if exposures > 0:
            masks[state_id] = mask_hash
            exposed_groups[f"{trajectory}::{phase}"].append(state_id)
    if len(state_ids) != len(set(state_ids)):
        errors.append("independent contributor state IDs must be unique")
    for trajectory in trajectory_by_id:
        current = by_trajectory.get(trajectory, [])
        if len(current) != 24 or Counter(row.get("phase") for row in current) != Counter(
            {phase: 8 for phase in REQUIRED_PHASES}
        ):
            errors.append(f"{trajectory}: independent bank requires frozen 8x3 state census")
        for phase in REQUIRED_PHASES:
            if len(exposed_groups.get(f"{trajectory}::{phase}", [])) < 2:
                errors.append(f"{trajectory}/{phase}: fewer than two exposed states")
    if len(rows) != len(trajectory_by_id) * 24:
        errors.append("independent contributor state census is incomplete")

    valid = not errors
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "status": (
            "FROZEN_BEFORE_BOUNDARY_OPERATOR_INTERVENTIONS"
            if valid
            else "INVALID_INDEPENDENT_BOUNDARY_CONTRIBUTOR_BANK"
        ),
        "input": {"path": str(bank_input_path), "sha256": sha256_file(bank_input_path)},
        "endpoint_definition_bridge": {
            "path": str(bridge_path),
            "sha256": sha256_file(bridge_path),
        },
        "target_endpoint": bridge.get("target_endpoint"),
        "tau": bridge.get("tau"),
        "frozen_effect_direction": bridge.get("frozen_effect_direction"),
        "target_effect_role": bridge.get("target_effect_role"),
        "attribution_objective": bridge.get("attribution_objective"),
        "weighting_contract_id": WEIGHTING_CONTRACT_ID,
        "state_bank_role": "INDEPENDENT_CONTRIBUTOR_CONFIRMATION_BANK",
        "trajectory_inputs": trajectories,
        "full_state_count": len(rows),
        "exposed_state_ids_by_trajectory_phase": {
            key: sorted(value) for key, value in sorted(exposed_groups.items())
        },
        "reference_anchor_condition_mask_sha256_by_state": dict(sorted(masks.items())),
        "operator_intervention_outcomes_used_for_bank_freeze": False,
        "eligible_for_independent_contributor_measurement": valid,
        "population_operator_contribution_claim_allowed": False,
        "next_gate": (
            "run prospectively frozen repair/injection family and independently "
            "estimate its contribution with trajectory-level inference"
        ),
        "analysis_code": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "errors": errors,
        "nonclaims": [
            "freezing an independent bank does not confirm an operator effect",
            "reference masks do not make eager a correctness authority",
            "repair and injection remain separate intervention estimands",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    bridge_path = Path(args.bridge).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit("independent contributor bank already exists; freeze is write-once")
    bank_input = json.loads(input_path.read_text(encoding="utf-8"))
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    result = freeze_independent_bank(bank_input, input_path, bridge, bridge_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": result["errors"]}, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
