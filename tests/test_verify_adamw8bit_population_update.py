import math

from scripts.verify_adamw8bit_population_update import verify


def test_real_population_artifact_recomputes_when_available():
    from pathlib import Path
    root = Path("results/property/numerical_coverage_v1/adamw8bit_population_update_v1")
    if not (root / "analysis.json").exists():
        return
    result = verify(root)
    assert result["status"] == "VERIFIED"
    assert result["decision"] == "NON_EQUIVALENT"
    assert result["exceedance_count"] == 32
    assert math.isclose(result["one_sided_probability_bounds"][0], 0.9106318010137313)
