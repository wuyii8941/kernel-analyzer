"""Single frozen loss-forward/backward compilation boundary for Qwen."""

from __future__ import annotations

import torch


def configure_candidate_runtime(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False


class LossStep(torch.nn.Module):
    def __init__(self, subject: torch.nn.Module) -> None:
        super().__init__()
        self.subject = subject

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.subject(
            input_ids=values, labels=values, use_cache=False, return_dict=False
        )[0]
