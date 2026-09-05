#!/usr/bin/env python3
"""Recompute a scoped analysis result from one retained raw case artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from kernel_analyzer.analysis_result import AnalysisResult
from kernel_analyzer.training_equivalence import classify_fixed_suite_update_equivalence
import kernel_analyzer.training_equivalence as training_equivalence_module


MARGINS = {"additive": 0.001, "repair_aligned": 0.01, "residual_direction": 0.001}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preferred_profile(stage: dict, *, allow_corrected_seed_names: bool) -> tuple[str, dict] | None:
    if "EXACT" in stage:
        return "FULL_VECTOR", stage["EXACT"]["profile"]
    corrected = sorted(name for name in stage if name.startswith("COUNT_SKETCH_V2"))
    if corrected:
        return corrected[0], stage[corrected[0]]["profile"]
    legacy_names = sorted(name for name in stage if name.startswith("SKETCH_SEED_"))
    if allow_corrected_seed_names and legacy_names:
        return "COUNT_SKETCH_V2_WITH_LEGACY_VIEW_NAME", stage[legacy_names[0]]["profile"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rms-margin", type=float, default=0.01)
    parser.add_argument("--endpoint", choices=("PARAMETER_WRITE", "ADAMW_UPDATE"))
    parser.add_argument(
        "--corrected-sketch-with-legacy-name", action="store_true",
        help="Use only for new recaptures whose external protocol pins the corrected hash mapping.",
    )
    args = parser.parse_args()
    payload = json.loads(args.raw.read_text())
    endpoint = args.endpoint or payload.get("primary_update_endpoint")
    statistics = payload.get("original_coordinate_statistics", {}).get(endpoint or "", [])
    stage = payload.get("stages", {}).get(endpoint or "", {})
    profile = _preferred_profile(
        stage, allow_corrected_seed_names=args.corrected_sketch_with_legacy_name
    )
    status = "VALID"
    reason = None
    decision = "NOT_ASSESSED"
    bias = {}
    if endpoint is None:
        status, reason = "PARTIAL", "PRIMARY_UPDATE_ENDPOINT_NOT_DECLARED"
    elif endpoint == "ADAMW_UPDATE" and args.endpoint is None:
        status, reason = "PARTIAL", "ACTUAL_PARAMETER_WRITE_NOT_CAPTURED"
    elif len(statistics) < 32:
        status, reason = "PARTIAL", "ORIGINAL_COORDINATE_STATISTICS_INCOMPLETE"
    else:
        confirmation = statistics[16:32]
        effect = sum(row["effect_energy"] for row in confirmation)
        repair = sum(row["repair_energy"] for row in confirmation)
        if repair <= 0.0:
            status, reason = "PARTIAL", "CONFIRMATION_REPAIR_ENERGY_IS_ZERO"
        else:
            total_rms = math.sqrt(effect / repair)
            exact_identity = effect == 0.0
            bias["fixed_suite_total_rms"] = total_rms
            if profile is None:
                classified = classify_fixed_suite_update_equivalence(
                    None, MARGINS, total_rms=total_rms,
                    total_rms_margin=args.rms_margin,
                    exact_identity_verified=exact_identity,
                )
            else:
                geometry, result = profile
                bias["profile_geometry"] = geometry
                intervals = {
                    name: branch["confidence_interval_95"]
                    for name, branch in result["population_inference"]["branches"].items()
                    if "confidence_interval_95" in branch
                }
                classified = classify_fixed_suite_update_equivalence(
                    intervals if set(intervals) == set(MARGINS) else None,
                    MARGINS, total_rms=total_rms,
                    total_rms_margin=args.rms_margin,
                    exact_identity_verified=exact_identity,
                )
            bias["fixed_suite_classification"] = classified
            decision = (
                "EQUIVALENT" if classified["decision"] in {
                    "FIXED_SUITE_UPDATE_EQUIVALENT",
                    "EXACT_UPDATE_IDENTITY_ON_FIXED_SUITE",
                }
                else "INCONCLUSIVE" if classified["decision"] == "INCONCLUSIVE"
                else "NON_EQUIVALENT"
            )
    if reason:
        bias["not_assessed_reason"] = reason
    result = AnalysisResult(
        case_id=str(payload.get("case_id", args.raw.stem)),
        contrast_id="NATURAL_IMPLEMENTATION",
        measurement_status=status,
        claim_scope=(
            "FIXED_SUITE_PARAMETER_WRITE" if endpoint == "PARAMETER_WRITE"
            else "FIXED_SUITE_PROPOSED_OPTIMIZER_UPDATE"
        ),
        bias_analysis=bias,
        equivalence_decision=decision,
        training_outcome="NOT_MEASURED",
        mandatory_endpoints=("PARAMETER_WRITE_TOTAL_RMS",),
        provenance={
            "raw_artifact": str(args.raw), "raw_schema": payload.get("schema"),
            "corrected_sketch_with_legacy_name": args.corrected_sketch_with_legacy_name,
            "raw_artifact_sha256": _sha(args.raw),
            "recompute_script_sha256": _sha(Path(__file__)),
            "training_equivalence_sha256": _sha(Path(training_equivalence_module.__file__)),
        },
    ).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
