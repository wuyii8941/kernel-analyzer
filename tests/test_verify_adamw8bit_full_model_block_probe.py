from pathlib import Path

from scripts.verify_adamw8bit_full_model_block_probe import verify


def test_saved_full_model_probe_recomputes_when_available():
    root = Path("results/property/numerical_coverage_v1/adamw8bit_full_model_block_probe_v1")
    if not (root / "result.json").exists():
        return
    result = verify(root)
    assert result["status"] == "VERIFIED"
    assert result["prediction_result"] == "CONFIRMED"
    assert result["block64_lower_rms_count"] == 16
    assert result["negative_final_gradient_inner_product_counts"] == {
        "block64": 16, "block256": 16,
    }
