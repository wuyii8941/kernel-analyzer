import torch

from scripts.fx_replay import _tensor_comparison


def test_tensor_comparison_matches_exact_tensor_predicates_across_chunks():
    size = 1_048_576 + 3
    actual = torch.zeros(size, dtype=torch.float32)
    replay = actual.clone()
    replay[-1] = 2.5

    result = _tensor_comparison(actual, replay)

    assert result == {
        "kind": "tensor",
        "metadata_equal": True,
        "bitwise_equal": False,
        "all_finite": True,
        "max_abs_error": 2.5,
    }


def test_tensor_comparison_reports_nonfinite_without_changing_max_semantics():
    actual = torch.tensor([1.0, float("inf")])
    replay = actual.clone()

    result = _tensor_comparison(actual, replay)

    assert result["bitwise_equal"] is True
    assert result["all_finite"] is False
    assert result["max_abs_error"] != result["max_abs_error"]  # NaN
