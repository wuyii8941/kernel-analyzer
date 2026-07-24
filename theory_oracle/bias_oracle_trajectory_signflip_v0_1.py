"""Trajectory-level sign-flip sensitivity for a frozen scalar mean claim."""

from __future__ import annotations

import itertools
import math
import random
from typing import Any


SCHEMA_VERSION = "forkcert.bias-oracle-trajectory-signflip.v0.1"


def studentized_mean(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two trajectory effects are required")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (
        len(values) - 1
    )
    if variance == 0.0:
        if mean == 0.0:
            return 0.0
        return math.copysign(math.inf, mean)
    return mean / math.sqrt(variance / len(values))


def trajectory_signflip_test(
    effects: list[float],
    *,
    null_center: float,
    exact_max_trajectories: int = 16,
    monte_carlo_draws: int = 99999,
    monte_carlo_seed: int = 172904,
) -> dict[str, Any]:
    if len(effects) < 2 or any(not math.isfinite(value) for value in effects):
        raise ValueError("effects must contain at least two finite trajectory values")
    if not math.isfinite(null_center):
        raise ValueError("null center must be finite")
    if exact_max_trajectories < 2 or monte_carlo_draws < 999:
        raise ValueError("invalid exact/Monte Carlo budget")
    residuals = [value - null_center for value in effects]
    observed = abs(studentized_mean(residuals))

    def exceeds(signs: tuple[int, ...] | list[int]) -> bool:
        statistic = abs(
            studentized_mean(
                [sign * residual for sign, residual in zip(signs, residuals, strict=True)]
            )
        )
        return statistic >= observed - 1e-15

    if len(effects) <= exact_max_trajectories:
        total = 2 ** len(effects)
        extreme = sum(
            exceeds(signs)
            for signs in itertools.product((-1, 1), repeat=len(effects))
        )
        p_value = extreme / total
        method = "EXACT_RADEMACHER_SIGN_FLIP_STUDENTIZED"
        seed = None
    else:
        rng = random.Random(monte_carlo_seed)
        total = monte_carlo_draws
        extreme = sum(
            exceeds([1 if rng.getrandbits(1) else -1 for _ in effects])
            for _ in range(total)
        )
        p_value = (extreme + 1) / (total + 1)
        method = "MONTE_CARLO_RADEMACHER_SIGN_FLIP_STUDENTIZED"
        seed = monte_carlo_seed
    return {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "assumption": "independent trajectory effects are sign-symmetric around the tested null center",
        "trajectories": len(effects),
        "null_center": null_center,
        "observed_abs_studentized_mean": observed,
        "extreme_draws": extreme,
        "total_draws": total,
        "two_sided_p_value": p_value,
        "monte_carlo_seed": seed,
        "role": "sensitivity only; cannot promote a shift rejected by the primary interval",
    }
