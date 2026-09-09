"""One fixed-suite analysis path for capture artifacts and synthetic validation.

Original-coordinate energy controls the decision. Gram-based projections are
descriptive diagnostics only, never full-space or population certificates.
Historical artifacts without optimizer readback provenance remain unassessed.
"""
from __future__ import annotations

import math
import numpy as np

from .analysis_result import AnalysisResult
from .training_equivalence import (
    bounded_population_aligned_equivalence,
    bounded_population_total_energy_equivalence,
    simultaneous_intervals_from_joint_gram,
)


def analyze_artifact(payload: dict, protocol: dict, *, endpoint: str | None = None) -> dict:
    endpoint = endpoint or protocol["primary_stage"]
    result = AnalysisResult(
        case_id=payload.get("case_id", "UNDECLARED"),
        contrast_id=payload.get("contrast_id", "UNDECLARED"),
        measurement_status="PARTIAL", claim_scope="FIXED_SUITE_UPDATE",
        mandatory_endpoints=(endpoint + "_TOTAL_RMS",),
        provenance={"protocol_version": protocol["schema"],
                    "write_protocol": payload.get("parameter_write_protocol"),
                    "data_use": "REANALYSIS_UNLESS_SEPARATELY_FROZEN"},
    ).to_dict()
    def unavailable(reason: str, invalid: bool = False) -> dict:
        result["measurement_status"] = "INVALID" if invalid else "PARTIAL"
        result["bias_analysis"] = {"not_assessed_reason": reason}
        return result

    if protocol.get("claim_scope") != "FIXED_SUITE_UPDATE":
        return unavailable("POPULATION_REQUIRES_A_SEPARATE_SAMPLING_PROTOCOL")
    if payload.get("status") != "COMPLETE":
        return unavailable("CAPTURE_NOT_COMPLETE")
    if endpoint == "PARAMETER_WRITE":
        write_protocol = payload.get("parameter_write_protocol", {})
        if write_protocol.get("version") not in {
            "adamw-readback-v2", "optimizer-implementation-readback-v1",
        } or write_protocol.get("measurement") != (
            "parameter_after_step_minus_parameter_before_step"
        ):
            return unavailable("ACTUAL_OPTIMIZER_READBACK_NOT_VERIFIED")
    if not payload.get("contrast_id") or not payload.get("runtime_boundary"):
        return unavailable("EXECUTION_OR_CONTRAST_NOT_DECLARED")
    if payload.get("determinism", {}).get("all_exact") is False:
        return unavailable("DETERMINISM_CHECK_FAILED", True)
    cal = list(payload.get("calibration_state_ids", []))
    conf = list(payload.get("confirmation_state_ids", []))
    ids = list(payload.get("state_ids", cal + conf))
    if not cal or not conf or len(set(ids)) != len(ids) or set(cal) & set(conf) or set(ids) != set(cal + conf):
        return unavailable("INVALID_STATE_PARTITIONS", True)
    rows = payload.get("original_coordinate_statistics", {}).get(endpoint, [])
    if len(rows) != len(ids):
        return unavailable("ORIGINAL_COORDINATE_STATISTICS_INCOMPLETE")
    try:
        x = np.array([r["effect_energy"] for r in rows], dtype=float)
        b = np.array([r["repair_energy"] for r in rows], dtype=float)
        a = np.array([r["effect_repair_inner_product"] for r in rows], dtype=float)
        margin = float(protocol["fixed_suite_margins"]["full_update_rms"])
    except (KeyError, TypeError, ValueError):
        return unavailable("MALFORMED_STATISTICS_OR_MARGIN", True)
    if not all(np.isfinite(v).all() for v in (x,b,a)) or np.any(x < 0) or np.any(b < 0) or not math.isfinite(margin) or margin <= 0:
        return unavailable("NONFINITE_OR_NEGATIVE_STATISTICS", True)
    bound = np.sqrt(x) * np.sqrt(b)
    if np.any(np.abs(a) > bound + 1e-10 * np.maximum(bound, 1e-300)):
        return unavailable("INNER_PRODUCT_EXCEEDS_ENERGY_BOUND", True)
    indices = [ids.index(s) for s in conf]
    energy = math.fsum(x[indices]); repair = math.fsum(b[indices])
    if repair <= 0:
        return unavailable("ZERO_REPAIR_ENERGY")
    rms = math.sqrt(energy/repair)
    aligned = math.fsum(a[indices])/repair
    result["measurement_status"] = "VALID"
    result["equivalence_decision"] = "EQUIVALENT" if rms < margin else "NON_EQUIVALENT" if rms > margin else "INCONCLUSIVE"
    result["bias_analysis"] = {
        "fixed_suite_total_rms": rms,
        "fixed_suite_aligned_ratio_of_sums": aligned,
        "full_update_rms_margin": margin,
        "confirmation_state_ids": conf,
        "decision_rule": "ORIGINAL_COORDINATE_Q_ONLY_FIXED_SUITE",
        "population_guarantee": False,
        "identity_verified_by_coordinate_count": all(r.get("nonzero_effect_coordinates") == 0 for r in (rows[i] for i in indices)),
    }
    stage = payload.get("stages", {}).get(endpoint, {})
    view = "EXACT" if "EXACT" in stage else next((
        k for k in sorted(stage)
        if k.startswith(("COUNT_SKETCH_V3_FLOAT64", "COUNT_SKETCH_V2"))
    ), None)
    if view is not None:
        profile = stage[view].get("profile", {})
        gram = profile.get("suite", profile).get("joint_gram")
        if gram is not None:
            order = [ids.index(s) for s in cal+conf]
            try:
                ordered = {}
                for key in ("effect_effect", "repair_repair", "effect_repair"):
                    matrix = np.asarray(gram[key], dtype=float)
                    if matrix.shape != (len(ids),len(ids)) or not np.isfinite(matrix).all():
                        raise ValueError("invalid Gram shape or entries")
                    ordered[key] = matrix[np.ix_(order,order)]
                intervals = simultaneous_intervals_from_joint_gram(ordered, calibration_count=len(cal))
                result["bias_analysis"]["direction_diagnostics"] = {
                    "geometry": view, "empirical_intervals": intervals,
                    "role": "DESCRIPTIVE_ONLY_NOT_USED_FOR_EQUIVALENCE",
                }
            except (ValueError, KeyError, TypeError) as error:
                result["bias_analysis"]["direction_diagnostic_unavailable"] = str(error)
    return result


def analyze_bounded_population_artifact(
    payload: dict, protocol: dict, *, endpoint: str | None = None
) -> dict:
    """Analyze independent training units under predeclared energy bounds.

    This is intentionally separate from :func:`analyze_artifact`: a frozen
    finite suite and a sampled state population are different claims.  The
    caller must provide explicit, unique inference-unit identifiers and a
    protocol-level argument for finite energy bounds.  State IDs are never
    silently treated as independent units.
    """

    endpoint = endpoint or protocol["primary_stage"]
    result = AnalysisResult(
        case_id=payload.get("case_id", "UNDECLARED"),
        contrast_id=payload.get("contrast_id", "UNDECLARED"),
        measurement_status="PARTIAL",
        claim_scope="DECLARED_STATE_POPULATION_UPDATE",
        mandatory_endpoints=tuple(protocol.get("mandatory_population_endpoints", ("TOTAL_RMS",))),
        provenance={
            "protocol_version": protocol.get("schema", "UNDECLARED"),
            "write_protocol": payload.get("parameter_write_protocol"),
            "data_use": protocol.get("data_use", "UNDECLARED"),
        },
    ).to_dict()

    def unavailable(reason: str, invalid: bool = False) -> dict:
        result["measurement_status"] = "INVALID" if invalid else "PARTIAL"
        result["bias_analysis"] = {"not_assessed_reason": reason}
        return result

    if protocol.get("claim_scope") != "DECLARED_STATE_POPULATION_UPDATE":
        return unavailable("POPULATION_CLAIM_SCOPE_NOT_DECLARED")
    if payload.get("status") != "COMPLETE":
        return unavailable("CAPTURE_NOT_COMPLETE")
    if endpoint == "PARAMETER_WRITE":
        write_protocol = payload.get("parameter_write_protocol", {})
        if write_protocol.get("version") not in {
            "adamw-readback-v2", "optimizer-implementation-readback-v1",
        } or write_protocol.get("measurement") != (
            "parameter_after_step_minus_parameter_before_step"
        ):
            return unavailable("ACTUAL_OPTIMIZER_READBACK_NOT_VERIFIED")
    if not payload.get("contrast_id") or not payload.get("runtime_boundary"):
        return unavailable("EXECUTION_OR_CONTRAST_NOT_DECLARED")
    if payload.get("determinism", {}).get("all_exact") is False:
        return unavailable("DETERMINISM_CHECK_FAILED", True)

    ids = list(payload.get("state_ids", []))
    units = list(payload.get("inference_unit_ids", []))
    if not ids or len(set(ids)) != len(ids) or len(units) != len(ids):
        return unavailable("EXPLICIT_INFERENCE_UNITS_REQUIRED", True)
    if len(set(units)) != len(units):
        return unavailable("ONE_ROW_PER_INDEPENDENT_UNIT_REQUIRED", True)
    rows = payload.get("original_coordinate_statistics", {}).get(endpoint, [])
    if len(rows) != len(ids):
        return unavailable("ORIGINAL_COORDINATE_STATISTICS_INCOMPLETE")

    bounds = protocol.get("population_energy_bounds", {})
    if bounds.get("fixed_before_observation") is not True:
        return unavailable("PREOBSERVATION_ENERGY_BOUNDS_REQUIRED")
    try:
        x = np.asarray([row["effect_energy"] for row in rows], dtype=np.float64)
        b = np.asarray([row["repair_energy"] for row in rows], dtype=np.float64)
        a = np.asarray([row["effect_repair_inner_product"] for row in rows], dtype=np.float64)
        rms_margin = float(protocol["population_margins"]["full_update_rms"])
        x_max = float(bounds["effect_energy_upper_bound"])
        b_max = float(bounds["repair_energy_upper_bound"])
        bound_provenance = str(bounds["provenance"])
        family_alpha = float(protocol.get("family_alpha", 0.05))
    except (KeyError, TypeError, ValueError):
        return unavailable("MALFORMED_POPULATION_STATISTICS_OR_PROTOCOL", True)

    mandatory = tuple(protocol.get("mandatory_population_endpoints", ("TOTAL_RMS",)))
    if not mandatory or any(name not in {"TOTAL_RMS", "REPAIR_ALIGNED"} for name in mandatory):
        return unavailable("UNKNOWN_MANDATORY_POPULATION_ENDPOINT", True)
    # Bonferroni is used because the same bounds also support a claim that any
    # endpoint is outside its range. TOTAL_RMS has one upper-side failure
    # hypothesis; REPAIR_ALIGNED has distinct positive and negative sides.
    # Counting both sides is conservative for the IUT equivalence pass.
    failure_hypothesis_count = (
        (1 if "TOTAL_RMS" in mandatory else 0)
        + (2 if "REPAIR_ALIGNED" in mandatory else 0)
    )
    endpoint_alpha = family_alpha / failure_hypothesis_count
    analyses = {}
    try:
        if "TOTAL_RMS" in mandatory:
            analyses["TOTAL_RMS"] = bounded_population_total_energy_equivalence(
                x,
                b,
                rms_margin=rms_margin,
                effect_energy_upper_bound=x_max,
                repair_energy_upper_bound=b_max,
                bound_provenance=bound_provenance,
                alpha=endpoint_alpha,
            )
        if "REPAIR_ALIGNED" in mandatory:
            aligned_margin = float(protocol["population_margins"]["repair_aligned"])
            analyses["REPAIR_ALIGNED"] = bounded_population_aligned_equivalence(
                a,
                b,
                margin=aligned_margin,
                effect_energy_upper_bound=x_max,
                repair_energy_upper_bound=b_max,
                bound_provenance=bound_provenance,
                alpha=endpoint_alpha,
            )
    except (ValueError, TypeError) as error:
        return unavailable(f"BOUNDED_POPULATION_ASSUMPTION_FAILED: {error}", True)

    decisions = [entry["decision"] for entry in analyses.values()]
    overall = (
        "EQUIVALENT"
        if decisions and all(value == "EQUIVALENT" for value in decisions)
        else "NON_EQUIVALENT"
        if any(value == "NON_EQUIVALENT" for value in decisions)
        else "INCONCLUSIVE"
    )
    result["measurement_status"] = "VALID"
    result["equivalence_decision"] = overall
    result["bias_analysis"] = {
        "population_endpoints": analyses,
        "inference_unit_count": len(units),
        "inference_unit_ids": units,
        "family_alpha": family_alpha,
        "per_endpoint_alpha": endpoint_alpha,
        "outside_failure_hypothesis_count": failure_hypothesis_count,
        "multiple_endpoint_rule": "BONFERRONI_OVER_TOTAL_AND_BOTH_ALIGNED_SIDES;_CONSERVATIVE_FOR_IUT_PASS",
        "population_guarantee": True,
        "guarantee_conditions": (
            "independent declared units and valid finite energy bounds fixed before observation"
        ),
    }
    return result
