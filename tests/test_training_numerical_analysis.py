import copy
import numpy as np
import pytest

from kernel_analyzer.training_numerical_analysis import (
    analyze_artifact,
    analyze_bounded_population_artifact,
    analyze_population_exceedance_artifact,
)


PROTOCOL = {"schema": "test-v2", "claim_scope": "FIXED_SUITE_UPDATE",
            "primary_stage": "PARAMETER_WRITE", "fixed_suite_margins": {"full_update_rms": .01}}


def artifact(u, r):
    n = len(u)
    return {
        "case_id": "synthetic", "contrast_id": "CONTROLLED_UPDATE", "status": "COMPLETE",
        "runtime_boundary": {"kind": "SYNTHETIC"},
        "parameter_write_protocol": {
            "version": "adamw-readback-v2",
            "measurement": "parameter_after_step_minus_parameter_before_step",
        },
        "state_ids": list(range(n)), "calibration_state_ids": list(range(n//2)),
        "confirmation_state_ids": list(range(n//2,n)),
        "original_coordinate_statistics": {"PARAMETER_WRITE": [
            {"effect_energy": float(x@x), "repair_energy": float(y@y),
             "effect_repair_inner_product": float(x@y), "nonzero_effect_coordinates": int(np.count_nonzero(x))}
            for x,y in zip(u,r)]},
        "stages": {"PARAMETER_WRITE": {"EXACT": {"profile": {"suite": {"joint_gram": {
            "effect_effect": (u@u.T).tolist(), "repair_repair": (r@r.T).tolist(),
            "effect_repair": (u@r.T).tolist()}}}}}},
    }


@pytest.mark.parametrize("scale,decision", [(.0095,"EQUIVALENT"),(.0105,"NON_EQUIVALENT"),(0.,"EQUIVALENT")])
def test_original_energy_boundary(scale, decision):
    r=np.ones((12,3)); u=scale*r
    out=analyze_artifact(artifact(u,r), PROTOCOL)
    assert out["equivalence_decision"]==decision
    assert len(out["bias_analysis"]["confirmation_state_ids"])==6


def test_orthogonal_drift_does_not_pass():
    r=np.zeros((32,3)); r[:,0]=1
    u=np.zeros_like(r); u[:16,1]=1e-6; u[16:,2]=1
    assert analyze_artifact(artifact(u,r),PROTOCOL)["equivalence_decision"]=="NON_EQUIVALENT"


def test_cached_verdict_and_intervals_are_ignored():
    r=np.ones((32,3)); raw=artifact(.02*r,r)
    reference=analyze_artifact(raw, PROTOCOL)
    raw["stages"]["PARAMETER_WRITE"]["EXACT"]["profile"]["population_inference"]={"branches": {"repair_aligned": {"confidence_interval_95": [0,0]}}}
    assert analyze_artifact(raw,PROTOCOL)==reference


def test_legacy_write_cannot_be_recertified():
    raw=artifact(np.zeros((32,3)),np.ones((32,3)))
    del raw["parameter_write_protocol"]
    out=analyze_artifact(raw,PROTOCOL)
    assert out["measurement_status"]=="PARTIAL"
    assert out["equivalence_decision"]=="NOT_ASSESSED"


def test_optimizer_implementation_readback_is_accepted_but_wrong_measurement_is_not():
    raw = artifact(np.zeros((32, 3)), np.ones((32, 3)))
    raw["parameter_write_protocol"]["version"] = "optimizer-implementation-readback-v1"
    assert analyze_artifact(raw, PROTOCOL)["equivalence_decision"] == "EQUIVALENT"
    raw["parameter_write_protocol"]["measurement"] = "optimizer_proposal_before_write"
    out = analyze_artifact(raw, PROTOCOL)
    assert out["measurement_status"] == "PARTIAL"
    assert out["bias_analysis"]["not_assessed_reason"] == (
        "ACTUAL_OPTIMIZER_READBACK_NOT_VERIFIED"
    )


def test_nonfinite_and_false_complete_fail_closed():
    raw=artifact(np.zeros((32,3)),np.ones((32,3)))
    raw["original_coordinate_statistics"]["PARAMETER_WRITE"][0]["effect_energy"]=float('nan')
    assert analyze_artifact(raw,PROTOCOL)["measurement_status"]=="INVALID"
    raw["status"]="ABSTAIN"
    assert analyze_artifact(raw,PROTOCOL)["equivalence_decision"]=="NOT_ASSESSED"


def test_state_order_is_declared_not_hardcoded():
    r=np.ones((12,3)); raw=artifact(.005*r,r)
    raw["original_coordinate_statistics"]["PARAMETER_WRITE"][0]["effect_energy"]=3
    raw["calibration_state_ids"],raw["confirmation_state_ids"]=raw["confirmation_state_ids"],raw["calibration_state_ids"]
    assert analyze_artifact(raw,PROTOCOL)["equivalence_decision"]=="NON_EQUIVALENT"


def test_population_is_not_inferred_from_state_ids():
    p=copy.deepcopy(PROTOCOL); p["claim_scope"]="STATE_POPULATION"
    raw=artifact(np.zeros((32,3)),np.ones((32,3)))
    assert analyze_artifact(raw,p)["equivalence_decision"]=="NOT_ASSESSED"


def _population_protocol():
    return {
        "schema": "test-bounded-population-v1",
        "claim_scope": "DECLARED_STATE_POPULATION_UPDATE",
        "primary_stage": "PARAMETER_WRITE",
        "data_use": "CONTROLLED_TEST",
        "mandatory_population_endpoints": ["TOTAL_RMS", "REPAIR_ALIGNED"],
        "population_margins": {"full_update_rms": 0.01, "repair_aligned": 0.01},
        "population_energy_bounds": {
            "effect_energy_upper_bound": 0.0,
            "repair_energy_upper_bound": 3.0,
            "provenance": "protocol-enforced exact synthetic identity",
            "fixed_before_observation": True,
        },
        "family_alpha": 0.05,
    }


def test_bounded_population_artifact_uses_explicit_independent_units():
    raw = artifact(np.zeros((128, 3)), np.ones((128, 3)))
    raw["inference_unit_ids"] = [f"unit-{i}" for i in range(128)]
    out = analyze_bounded_population_artifact(raw, _population_protocol())
    assert out["measurement_status"] == "VALID"
    assert out["equivalence_decision"] == "EQUIVALENT"
    assert out["claim_scope"] == "DECLARED_STATE_POPULATION_UPDATE"
    assert out["bias_analysis"]["population_guarantee"] is True
    assert out["bias_analysis"]["outside_failure_hypothesis_count"] == 3
    assert out["bias_analysis"]["per_endpoint_alpha"] == pytest.approx(0.05 / 3.0)


def test_bounded_population_artifact_does_not_infer_units_from_states():
    raw = artifact(np.zeros((32, 3)), np.ones((32, 3)))
    out = analyze_bounded_population_artifact(raw, _population_protocol())
    assert out["measurement_status"] == "INVALID"
    assert out["equivalence_decision"] == "NOT_ASSESSED"
    assert out["bias_analysis"]["not_assessed_reason"] == "EXPLICIT_INFERENCE_UNITS_REQUIRED"


def test_bounded_population_artifact_requires_preobservation_bounds():
    raw = artifact(np.zeros((32, 3)), np.ones((32, 3)))
    raw["inference_unit_ids"] = [f"unit-{i}" for i in range(32)]
    protocol = _population_protocol()
    protocol["population_energy_bounds"]["fixed_before_observation"] = False
    out = analyze_bounded_population_artifact(raw, protocol)
    assert out["measurement_status"] == "PARTIAL"
    assert out["bias_analysis"]["not_assessed_reason"] == "PREOBSERVATION_ENERGY_BOUNDS_REQUIRED"


def test_bounded_population_artifact_rejects_observed_bound_violation():
    raw = artifact(np.full((32, 3), 0.1), np.ones((32, 3)))
    raw["inference_unit_ids"] = [f"unit-{i}" for i in range(32)]
    out = analyze_bounded_population_artifact(raw, _population_protocol())
    assert out["measurement_status"] == "INVALID"
    assert out["equivalence_decision"] == "NOT_ASSESSED"
    assert "BOUNDED_POPULATION_ASSUMPTION_FAILED" in out["bias_analysis"]["not_assessed_reason"]


def _exceedance_protocol():
    return {
        "schema": "population-statewise-exceedance-v1",
        "primary_stage": "PARAMETER_WRITE",
        "claim_scope": "DECLARED_STATE_POPULATION_UPDATE",
        "population_estimand": "STATEWISE_RMS_EXCEEDANCE_PROBABILITY",
        "statewise_rms_margin": 0.1,
        "maximum_exceedance_probability": 0.1,
        "repair_energy_floor": 0.0,
        "alpha": 0.05,
        "data_use": "SYNTHETIC_VALIDATION",
    }


def test_population_exceedance_artifact_uses_original_coordinate_statistics():
    raw = artifact(np.zeros((64, 3)), np.ones((64, 3)))
    raw["inference_unit_ids"] = [f"unit-{index}" for index in range(64)]
    out = analyze_population_exceedance_artifact(raw, _exceedance_protocol())
    assert out["measurement_status"] == "VALID"
    assert out["equivalence_decision"] == "EQUIVALENT"
    assert out["bias_analysis"]["not_a_mean_energy_certificate"] is True


def test_population_exceedance_artifact_rejects_duplicate_units():
    raw = artifact(np.zeros((64, 3)), np.ones((64, 3)))
    raw["inference_unit_ids"] = ["same-unit"] * 64
    out = analyze_population_exceedance_artifact(raw, _exceedance_protocol())
    assert out["measurement_status"] == "INVALID"
    assert out["bias_analysis"]["not_assessed_reason"] == (
        "ONE_ROW_PER_INDEPENDENT_UNIT_REQUIRED"
    )
