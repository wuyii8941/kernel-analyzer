"""Measure optimizer proposals separately from stored-parameter writes."""

from __future__ import annotations

import torch


def parameter_write_delta(base: torch.Tensor, proposed: torch.Tensor) -> torch.Tensor:
    """Return the change produced by adding ``proposed`` in ``base``'s dtype."""

    stored = base.detach()
    return ((stored + proposed.to(stored.dtype)) - stored).float()
