#!/usr/bin/env python3
"""Plan the minimum boundary-confirmation trajectory count from family size only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from theory_oracle.bias_oracle_confirmation_precision_v0_1 import (
    minimum_attainable_signflip_p,
    signflip_resolution_requirement,
)
from theory_oracle.freeze_qwen3_boundary_condition_family_v0_1 import (
    SCHEMA_VERSION as FAMILY_SCHEMA_VERSION,
)
from theory_oracle.aggregate_qwen3_boundary_conditioned_calibration_v0_1 import (
    WEIGHTING_CONTRACT_ID,
)


SCHEMA_VERSION = "forkcert.qwen3-boundary-confirmation-resource-plan.v0.1"
FAMILY_STATUS = "FROZEN_SUPPORT_COMPLETE_BOUNDARY_FAMILY"
SCRIPT_PATH = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_resource(
    family: dict[str, Any],
    family_path: Path,
    *,
    family_alpha: float,
    minimum_trajectories: int,
    resource_cap: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if (
        family.get("schema_version") != FAMILY_SCHEMA_VERSION
        or family.get("valid") is not True
        or family.get("status") != FAMILY_STATUS
    ):
        errors.append("boundary family is not frozen and support-complete")
    if family.get("weighting_contract_id") != WEIGHTING_CONTRACT_ID:
        errors.append("boundary family weighting contract drifted")
    endpoints = family.get("endpoint_family")
    comparisons = family.get("confirmatory_comparisons")
    if (
        not isinstance(endpoints, list)
        or not endpoints
        or comparisons != len(endpoints)
    ):
        errors.append("boundary family comparison count is invalid")
        comparisons = 0
    if not math.isfinite(family_alpha) or not 0 < family_alpha < 1:
        errors.append("family alpha must lie between zero and one")
    if minimum_trajectories < 8 or resource_cap < minimum_trajectories:
        errors.append("boundary resource range is invalid")
    interval_alpha = family_alpha / comparisons if comparisons else math.nan
    required = (
        signflip_resolution_requirement(
            interval_alpha, minimum_trajectories, resource_cap
        )
        if not errors
        else None
    )
    if required is None and not errors:
        errors.append("boundary sign-flip resolution exceeds resource cap")
    valid = not errors and required is not None
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "status": (
            "VALID_FROZEN_BOUNDARY_RESOURCE_REQUIREMENT"
            if valid
            else "INVALID_OR_INFEASIBLE_BOUNDARY_RESOURCE_REQUIREMENT"
        ),
        "boundary_family": {
            "path": str(family_path),
            "sha256": sha256_file(family_path),
        },
        "multiplicity": "BONFERRONI_SIMULTANEOUS_TWO_SIDED",
        "family_alpha": family_alpha,
        "confirmatory_comparisons": comparisons,
        "per_interval_alpha": interval_alpha if math.isfinite(interval_alpha) else None,
        "minimum_declared_trajectories": minimum_trajectories,
        "resource_cap": resource_cap,
        "minimum_trajectories_for_signflip_resolution": required,
        "minimum_attainable_p_at_required_count": (
            minimum_attainable_signflip_p(required) if required is not None else None
        ),
        "planning_inputs": [
            "family endpoint count",
            "family alpha",
            "minimum trajectories",
            "resource cap",
        ],
        "candidate_effect_mean_sign_or_variance_used": False,
        "analysis_code": {"path": str(SCRIPT_PATH), "sha256": sha256_file(SCRIPT_PATH)},
        "errors": errors,
        "nonclaims": [
            "this is a discrete p-value resolution requirement, not a power calculation",
            "meeting this minimum does not confirm a shift or guarantee interval precision",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-family", required=True)
    parser.add_argument("--family-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-trajectories", type=int, default=8)
    parser.add_argument("--resource-cap", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    family_path = Path(args.boundary_family).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit("boundary resource plan already exists; freeze is write-once")
    family = json.loads(family_path.read_text(encoding="utf-8"))
    result = plan_resource(
        family,
        family_path,
        family_alpha=args.family_alpha,
        minimum_trajectories=args.minimum_trajectories,
        resource_cap=args.resource_cap,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "minimum_trajectories_for_signflip_resolution": result["minimum_trajectories_for_signflip_resolution"], "errors": result["errors"]}, indent=2))
    if result["valid"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
