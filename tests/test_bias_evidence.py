import pytest

from kernel_analyzer.bias_evidence import binary_prevalence_summary, statewise_aligned_summary, total_rms_from_gram


def test_aligned_summary_separates_frequency_from_vector_mean():
    result = statewise_aligned_summary([1, 2, 3, 4], [10, 10, 10, 10])
    assert result["positive_direction_frequency"]["positive_count"] == 4
    assert result["ratio_of_sums_descriptive"] == pytest.approx(.25)
    assert result["vector_mean_nonzero_established"] is False


def test_aligned_summary_rejects_invalid_energy():
    with pytest.raises(ValueError):
        statewise_aligned_summary([1, 2], [1, 0])


def test_total_rms_uses_requested_gram_diagonal():
    gram = {
        "effect_effect": [[1, 0], [0, 9]],
        "repair_repair": [[4, 0], [0, 9]],
        "effect_repair": [[0, 0], [0, 0]],
    }
    assert total_rms_from_gram(gram, range(1, 2)) == 1


def test_binary_prevalence_has_generic_semantics():
    result = binary_prevalence_summary([True] * 16, estimand="P(TEST)")
    assert result["decision"] == "PREVALENCE_ABOVE_NULL"
    assert result["success_count"] == 16
    assert "DIRECTION" not in result["estimand"]
