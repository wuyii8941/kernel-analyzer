import torch

from scripts.analyze_direct_persistence_raw_tolerance import aggregate_rows, magnitude_row, metric_row


def test_raw_tolerance_metrics_keep_exact_gradient_pair_and_ulp():
    candidate = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float32)
    reference = torch.tensor([1.0, 2.0 + 2 ** -20, -3.0], dtype=torch.float32)
    row = metric_row(candidate, reference)
    assert row["l2"] > 0
    assert row["ulp"]["status"] == "COMPLETE"
    assert row["ulp"]["max"] > 0


def test_mixed_dtype_ulp_fails_closed():
    row = metric_row(torch.tensor([1.0], dtype=torch.bfloat16), torch.tensor([1.0], dtype=torch.float32))
    assert row["ulp"]["status"] == "ABSTAIN_MIXED_DTYPES"


def test_update_magnitude_does_not_invent_relative_or_ulp():
    row = magnitude_row(torch.tensor([3.0, 4.0], dtype=torch.float32))
    summary = aggregate_rows([row], "local_effective_update_magnitude")
    assert summary["relative_l2_mean"] is None
    assert summary["ulp_mean"] is None
