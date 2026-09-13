import torch

from kernel_analyzer.moment_reconstruction import (
    reconstruction_terms, update_component_terms,
)


def test_recurrence_bookkeeping_includes_floating_remainder():
    ref0 = torch.tensor([1.0, -2.0], dtype=torch.float64)
    cand0 = torch.tensor([1.1, -1.8], dtype=torch.float64)
    ref1 = torch.tensor([0.7, -1.0], dtype=torch.float64)
    unstored = torch.tensor([0.85, -0.75], dtype=torch.float64)
    read = torch.tensor([0.8, -0.8], dtype=torch.float64)
    row = reconstruction_terms(
        reference_previous=ref0, reference_current=ref1,
        candidate_previous_read=cand0, candidate_current_unstored=unstored,
        candidate_current_read=read, beta=.9,
    )
    assert row["relative_reconstruction_residual"] < 1e-14


def test_update_components_close_by_definition():
    reference = torch.tensor([1., 2.])
    candidate = torch.tensor([1.4, 1.7])
    first = torch.tensor([1.2, 1.9])
    second = torch.tensor([1.1, 1.8])
    row = update_component_terms(
        reference=reference, candidate=candidate,
        first_only=first, second_only=second,
    )
    assert row["component_reconstruction_residual"] == 0.0
