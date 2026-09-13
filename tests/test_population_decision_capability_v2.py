import numpy as np

from kernel_analyzer.training_numerical_analysis import analyze_bounded_population_artifact
from scripts.validate_population_decision_capability_v2 import artifact, protocol


def test_production_bounded_path_passes_nonzero_small_effect_and_rejects_large_effect():
    repair = np.ones((4096, 1), dtype=np.float64)
    small = np.full_like(repair, 0.005)
    large = np.full_like(repair, 0.02)
    passed = analyze_bounded_population_artifact(
        artifact(small, repair), protocol(0.005 ** 2, 1.0),
    )
    rejected = analyze_bounded_population_artifact(
        artifact(large, repair), protocol(0.02 ** 2, 1.0),
    )
    assert passed["measurement_status"] == "VALID"
    assert passed["equivalence_decision"] == "EQUIVALENT"
    assert rejected["measurement_status"] == "VALID"
    assert rejected["equivalence_decision"] == "NON_EQUIVALENT"


def test_all_zero_sample_abstains_when_predeclared_support_allows_rare_boundary_event():
    repair = np.ones((64, 1), dtype=np.float64)
    effect = np.zeros_like(repair)
    rare_probability = 1.0 / (20.0 * len(repair))
    result = analyze_bounded_population_artifact(
        artifact(effect, repair), protocol((0.01 / rare_probability) ** 2, 1.0),
    )
    assert result["measurement_status"] == "VALID"
    assert result["equivalence_decision"] == "INCONCLUSIVE"
