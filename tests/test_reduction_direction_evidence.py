import math
import json
from pathlib import Path

from scripts.analyze_reduction_direction_evidence import analyze
from scripts.build_current_root_cause_closure import heldout_direction_from_gram


def profile_for(vectors, nc):
    gram = [[sum(a * b for a, b in zip(u, v)) for v in vectors] for u in vectors]
    return {"suite": {"calibration_state_count": nc,
                      "confirmation_state_count": len(vectors) - nc,
                      "joint_gram": {"effect_effect": gram}}}


def test_positive_sign_frequency_does_not_establish_mean_bias():
    data = profile_for([[1.0], [1.0]] + [[1.0]] * 9 + [[-9.0]], 2)
    result = heldout_direction_from_gram(data)
    assert result["confirmation_positive_count"] == 9
    assert result["confirmation_projection_mean"] == 0
    assert result["population_mean_bias_decision"] == "NOT_ASSESSED_FIXED_SUITE"


def test_orthogonal_or_missing_direction_does_not_establish_zero_mean():
    result = heldout_direction_from_gram(profile_for([[1, 0], [1, 0], [0, 100], [0, 100]], 2))
    assert result["confirmation_zero_count"] == 2
    assert result["confirmation_mean_energy"] == 10000
    assert result["zero_population_mean_proven"] is False
    missing = heldout_direction_from_gram(profile_for([[1], [-1], [5], [5]], 2))
    assert missing["direction_status"] == "NOT_IDENTIFIABLE"
    assert missing["confirmation_projections"] is None


def test_reduction_audit_uses_actual_grams_and_rejects_old_population_labels():
    report = analyze()
    granite = report["granite"]
    direction = granite["stages"]["PARAMETER_WRITE"]
    assert granite["state_count"] == 32
    assert granite["confirmation_nonzero_write_states"] == 3
    assert direction["confirmation_zero_count"] == 16
    assert direction["confirmation_nonzero_cross_state_inner_products"] == 0
    assert direction["confirmation_mean_energy"] > 0
    assert math.isclose(granite["confirmation_original_coordinate_write_rms"], 2.281040966351542e-5)
    write = report["liger"]["actual_parameter_write"]
    assert write["confirmation_positive_count"] == 12
    assert write["confirmation_projection_mean"] > 0
    assert report["liger"]["legacy_analysis_errors"]["same_bank_gradient_effect_gram_matches_correct_reference_run"]
    assert report["natural_population_mean_bias_established"] is False


def test_liger_population_mean_runs_are_scoped_and_positive():
    root = Path(__file__).resolve().parents[1]
    for length, expected_positive, expected_lower in (
        (64, 44, 1.762303937507884e-7),
        (256, 50, 1.5508373531989473e-7),
    ):
        payload = json.loads(
            (root / f"results/property/liger_fp32_chunk_order_v1/length{length}_population_mean_v1.json").read_text()
        )
        assert payload["dtype"] == "FP32_FOR_BOTH_IMPLEMENTATIONS"
        assert payload["confirmation"]["positive_count"] == expected_positive
        assert payload["confirmation"]["one_sided_95_lower_bound"] >= expected_lower
        assert payload["confirmation"]["implies_vector_mean_nonzero_if_one_sided_bound_positive"] is True
        assert "natural LLM data" in payload["claim_boundary"]
