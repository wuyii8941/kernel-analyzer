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

    def forward(
        self,
        values: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.subject(
            input_ids=values,
            labels=values,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
        )[0]


def text_step_values(
    state: dict[str, object], device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Build the declared text inputs, including optional absolute positions.

    Most existing banks contain only token IDs.  Long-context mechanisms can
    be exercised with a short token window and explicit legal ``position_ids``;
    keeping this in the shared step avoids a family-specific model runner.
    """

    tokens = state.get("token_ids", state.get("input_ids"))
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("text state must contain nonempty token IDs")
    values = torch.tensor([tokens], dtype=torch.long, device=device)
    positions = state.get("position_ids")
    if positions is None:
        return (values,)
    if (not isinstance(positions, list)
            or len(positions) != len(tokens)
            or any(type(value) is not int or value < 0 for value in positions)):
        raise ValueError("position_ids must be nonnegative integers matching token length")
    return (
        values,
        torch.tensor([positions], dtype=torch.long, device=device),
    )
