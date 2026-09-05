"""Pure helpers for the single-implementation-boundary collapse experiment."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np


INJECTION_MODES = (
    "ZERO",
    "COHERENT_DIRECTION",
    "BALANCED_DIRECTION",
    "LIGER_SHAPED",
    "BALANCED_LIGER_SHAPED",
)


def balanced_signs(*, block_size: int, seed: int, block_index: int) -> list[int]:
    """Return a deterministic block with exactly half positive signs."""

    if block_size <= 0 or block_size % 2:
        raise ValueError("block_size must be a positive even number")
    digest = hashlib.sha256(f"{seed}:{block_index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    values = np.array([1] * (block_size // 2) + [-1] * (block_size // 2))
    rng.shuffle(values)
    return [int(value) for value in values]


def balanced_sign(step_index: int, *, block_size: int = 32, seed: int = 20260905) -> int:
    if step_index < 0:
        raise ValueError("step_index must be nonnegative")
    block_index, offset = divmod(step_index, block_size)
    return balanced_signs(block_size=block_size, seed=seed, block_index=block_index)[offset]


def injection_scale_sign(mode: str, step_index: int, *, sign_seed: int) -> int:
    if mode not in INJECTION_MODES:
        raise ValueError(f"unknown injection mode: {mode}")
    if mode.startswith("BALANCED_"):
        return balanced_sign(step_index, seed=sign_seed)
    return 1


def prefix_statistics(
    injection_norms: Sequence[float],
    repair_update_norms: Sequence[float],
    signed_projections: Sequence[float],
) -> dict[str, float]:
    if not injection_norms or not (
        len(injection_norms) == len(repair_update_norms) == len(signed_projections)
    ):
        raise ValueError("prefix sequences must have one common nonzero length")
    injected_energy = math.fsum(float(value) ** 2 for value in injection_norms)
    repair_energy = math.fsum(float(value) ** 2 for value in repair_update_norms)
    if repair_energy <= 0.0:
        raise ValueError("repair update energy must be positive")
    projection = math.fsum(float(value) for value in signed_projections)
    return {
        "relative_injection_energy": injected_energy / repair_energy,
        "relative_mean_direction_energy": projection**2 / (len(injection_norms) * repair_energy),
        "cumulative_signed_projection": projection,
    }


def first_sustained_true(values: Sequence[bool], *, required: int) -> int | None:
    """Return the zero-based start of the first sustained true interval."""

    if required <= 0:
        raise ValueError("required must be positive")
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if value else 0
        if run >= required:
            return index - required + 1
    return None


def classify_paired_loss_collapse(
    candidate_losses: Sequence[float],
    repair_losses: Sequence[float],
    *,
    ema_window: int = 128,
    ratio_threshold: float = 1.5,
    sustained_steps: int = 256,
) -> dict[str, Any]:
    """Apply the frozen sustained-loss rule to one paired run."""

    if len(candidate_losses) != len(repair_losses) or len(candidate_losses) == 0:
        raise ValueError("candidate and repair loss sequences must share one nonzero length")
    if ema_window <= 0 or sustained_steps <= 0:
        raise ValueError("window sizes must be positive")
    candidate = np.asarray(candidate_losses, dtype=np.float64)
    repair = np.asarray(repair_losses, dtype=np.float64)
    if not np.all(np.isfinite(repair)):
        return {"collapsed": False, "reason": "REPAIR_NONFINITE", "first_step": None}
    if not np.all(np.isfinite(candidate)):
        first = int(np.flatnonzero(~np.isfinite(candidate))[0]) + 1
        return {"collapsed": True, "reason": "CANDIDATE_NONFINITE", "first_step": first}
    if candidate.size < ema_window + sustained_steps - 1:
        return {"collapsed": False, "reason": "INSUFFICIENT_STEPS", "first_step": None}
    kernel = np.ones(ema_window, dtype=np.float64) / ema_window
    candidate_ema = np.convolve(candidate, kernel, mode="valid")
    repair_ema = np.convolve(repair, kernel, mode="valid")
    above = candidate_ema > ratio_threshold * repair_ema
    start = first_sustained_true(above.tolist(), required=sustained_steps)
    return {
        "collapsed": start is not None,
        "reason": "SUSTAINED_LOSS_RATIO" if start is not None else "NO_SUSTAINED_LOSS_RATIO",
        "first_step": None if start is None else start + ema_window,
        "maximum_ema_ratio": float(np.max(candidate_ema / np.maximum(repair_ema, 1e-30))),
    }
