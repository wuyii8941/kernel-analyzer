from __future__ import annotations

import torch

from kernel_analyzer.update_write import parameter_write_delta


def test_small_proposal_can_exist_without_a_bfloat16_parameter_write() -> None:
    base = torch.tensor([1.0], dtype=torch.bfloat16)
    proposed = torch.tensor([1e-4], dtype=torch.float32)
    assert proposed.item() != 0.0
    assert parameter_write_delta(base, proposed).item() == 0.0


def test_float32_parameter_write_retains_the_same_proposal() -> None:
    base = torch.tensor([1.0], dtype=torch.float32)
    proposed = torch.tensor([0.125], dtype=torch.float32)
    assert parameter_write_delta(base, proposed).item() == proposed.item()
