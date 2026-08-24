import torch

from scripts.analyze_direct_persistence_raw_tolerance import (
    _coherence_summary,
    aggregate_rows,
    magnitude_row,
    metric_row,
)


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


def test_coherence_summary_is_streamable_and_uses_path_energy():
    total = torch.tensor([3.0, 4.0])
    result = _coherence_summary(total, 25.0, 4, "toy")
    assert result["status"] == "COMPLETE"
    assert result["resultant_l2"] == 5.0
    assert result["path_l2"] == 5.0
    assert result["A"] == 1.0


def test_coherence_summary_missing_sequence_fails_closed():
    result = _coherence_summary(torch.tensor([0.0]), 0.0, 0, "toy")
    assert result["status"] == "ABSTAIN_MISSING_RAW_VECTORS"
