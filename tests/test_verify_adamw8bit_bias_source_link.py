from pathlib import Path

from scripts.verify_adamw8bit_bias_source_link import recompute


def test_saved_source_link_recomputes():
    root = Path("results/property/bias_proof_plan_v1/adamw8bit_source_link")
    if not (root / "result.json").exists():
        return
    result = recompute(root)
    assert result["status"] == "VERIFIED"
    assert result["unit_count"] == 16
    assert result["default_positive_count"] == 16
    assert result["compensated_positive_count"] == 8
    assert result["absolute_gain_reduction_count"] == 16
