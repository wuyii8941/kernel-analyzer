"""Protocol-relative statistics for matched training implementation effects.

The module deliberately separates two questions:

* what happened on one fixed suite of matched states; and
* what can be inferred beyond that suite from independent training units.

One training unit can contain several consecutive states.  Such states are
kept together for uncertainty calculations.  If calibration and confirmation
share a training unit, or if too few independent units are supplied, the
module returns descriptive effects but abstains from population inference.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


BRANCHES = ("additive", "repair_aligned", "residual_direction")


def _as_matrix(values: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"{name} must be a nonempty state-by-coordinate matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains nonfinite values")
    return matrix


def _t_critical_95(df: int) -> float:
    """Accurate dependency-free approximation to the two-sided 95% t critical."""

    if df < 1:
        raise ValueError("degrees of freedom must be positive")
    z = 1.959963984540054
    f = float(df)
    return (
        z
        + (z**3 + z) / (4.0 * f)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * f**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * f**3)
    )


def _mean_interval(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        raise ValueError("an interval requires at least two independent units")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    half_width = _t_critical_95(values.size - 1) * standard_error
    return mean - half_width, mean + half_width


def _signflip_p(values: np.ndarray, *, draws: int, seed: int) -> float:
    if draws < 999:
        raise ValueError("sign-flip calibration requires at least 999 draws")
    rng = np.random.default_rng(seed)
    observed_scale = float(values.std(ddof=1) / math.sqrt(values.size))
    observed = (
        abs(float(values.mean())) / observed_scale
        if observed_scale > 0.0
        else (math.inf if values.mean() != 0.0 else 0.0)
    )
    signs = rng.choice((-1.0, 1.0), size=(draws, values.size))
    randomized_values = signs * values[None, :]
    randomized_scale = randomized_values.std(axis=1, ddof=1) / math.sqrt(values.size)
    randomized = np.divide(
        np.abs(randomized_values.mean(axis=1)),
        randomized_scale,
        out=np.full(draws, math.inf),
        where=randomized_scale > 0.0,
    )
    return float((1.0 + np.count_nonzero(randomized >= observed)) / (draws + 1.0))


def _groups(indices: np.ndarray, unit_ids: Sequence[str]) -> list[np.ndarray]:
    ordered: list[str] = []
    members: dict[str, list[int]] = {}
    for index in indices.tolist():
        unit = str(unit_ids[index])
        if unit not in members:
            ordered.append(unit)
            members[unit] = []
        members[unit].append(index)
    return [np.asarray(members[unit], dtype=np.int64) for unit in ordered]


def _branch_result(
    unit_values: np.ndarray,
    *,
    direction_must_repeat: bool,
    draws: int,
    seed: int,
) -> dict:
    estimate = float(unit_values.mean())
    lower, upper = _mean_interval(unit_values)
    p_value = _signflip_p(unit_values, draws=draws, seed=seed)
    direction_repeats = not direction_must_repeat or estimate > 0.0
    return {
        "estimate": estimate,
        "confidence_interval_95": [lower, upper],
        "raw_studentized_signflip_p": p_value,
        "independent_unit_count": int(unit_values.size),
        "confirmation_direction_matches_calibration": direction_repeats,
        "raw_confirmed": bool(
            p_value <= 0.05
            and (lower > 0.0 or upper < 0.0)
            and direction_repeats
        ),
    }


def matched_training_bias_profile(
    effects: Sequence[Sequence[float]] | np.ndarray,
    repairs: Sequence[Sequence[float]] | np.ndarray,
    *,
    calibration_indices: Sequence[int],
    confirmation_indices: Sequence[int],
    inference_unit_ids: Sequence[str] | None,
    repair_rms_floor: float = 1e-12,
    minimum_independent_units: int = 8,
    signflip_draws: int = 4000,
    seed: int = 0,
) -> dict:
    """Measure one matched effect at one training stage.

    ``effects[i]`` is candidate minus repair and ``repairs[i]`` is the normal
    repair-side signal at the same state.  The additive and residual directions
    are learned only on calibration states.  Population intervals are computed
    over explicitly supplied independent units, never over individual steps by
    default.
    """

    u = _as_matrix(effects, "effects")
    r = _as_matrix(repairs, "repairs")
    if u.shape != r.shape:
        raise ValueError("effects and repairs must share one coordinate system")
    cal = np.asarray(calibration_indices, dtype=np.int64)
    conf = np.asarray(confirmation_indices, dtype=np.int64)
    if cal.size < 2 or conf.size < 2:
        raise ValueError("calibration and confirmation each require at least two states")
    if np.intersect1d(cal, conf).size:
        raise ValueError("calibration and confirmation states overlap")
    if min(cal.min(), conf.min()) < 0 or max(cal.max(), conf.max()) >= len(u):
        raise ValueError("split index is outside the state matrix")

    repair_scale = math.sqrt(float(np.mean(np.sum(r[conf] * r[conf], axis=1))))
    if repair_scale <= repair_rms_floor:
        return {
            "status": "ABSTAIN_REPAIR_SCALE",
            "repair_rms": repair_scale,
            "repair_rms_floor": repair_rms_floor,
        }

    calibration_mean = u[cal].mean(axis=0)
    calibration_norm = float(np.linalg.norm(calibration_mean))
    additive_direction = (
        calibration_mean / calibration_norm
        if calibration_norm > 0.0
        else np.zeros(u.shape[1], dtype=np.float64)
    )
    additive_values = u[conf] @ additive_direction / repair_scale

    repair_energy = np.sum(r * r, axis=1)
    relative_floor = max(float(np.median(repair_energy[conf])) * 1e-12, repair_rms_floor**2)
    usable = repair_energy > relative_floor
    if np.count_nonzero(usable[conf]) != conf.size or np.count_nonzero(usable[cal]) != cal.size:
        return {
            "status": "ABSTAIN_PER_STATE_REPAIR_SCALE",
            "repair_rms": repair_scale,
            "usable_calibration_states": int(np.count_nonzero(usable[cal])),
            "usable_confirmation_states": int(np.count_nonzero(usable[conf])),
        }

    per_state_gain = np.sum(u * r, axis=1) / repair_energy
    residual = u - per_state_gain[:, None] * r
    residual_mean = residual[cal].mean(axis=0)
    residual_norm = float(np.linalg.norm(residual_mean))
    residual_direction = (
        residual_mean / residual_norm
        if residual_norm > 0.0
        else np.zeros(u.shape[1], dtype=np.float64)
    )
    residual_values = residual[conf] @ residual_direction / repair_scale

    # Fixed-suite aligned effect: the declared energy-weighted coefficient.
    aligned_suite = float(
        np.sum(np.sum(u[conf] * r[conf], axis=1))
        / np.sum(repair_energy[conf])
    )
    suite = {
        "state_count": int(len(u)),
        "calibration_state_count": int(cal.size),
        "confirmation_state_count": int(conf.size),
        "coordinate_count": int(u.shape[1]),
        "repair_rms": repair_scale,
        "total_effect_rms": float(np.sqrt(np.mean(np.sum(u[conf] * u[conf], axis=1)))),
        "mean_effect_over_repair_rms": float(np.linalg.norm(u[conf].mean(axis=0)) / repair_scale),
        "additive_heldout_effect": float(additive_values.mean()),
        "repair_aligned_effect": aligned_suite,
        "residual_direction_heldout_effect": float(residual_values.mean()),
    }

    if inference_unit_ids is None:
        return {
            "status": "DESCRIPTIVE_FIXED_SUITE_ONLY",
            "suite": suite,
            "population_inference": None,
            "abstention_reason": "INDEPENDENT_TRAINING_UNITS_NOT_DECLARED",
        }
    if len(inference_unit_ids) != len(u):
        raise ValueError("one inference-unit id is required per state")
    calibration_units = {str(inference_unit_ids[index]) for index in cal.tolist()}
    confirmation_units = {str(inference_unit_ids[index]) for index in conf.tolist()}
    if calibration_units & confirmation_units:
        return {
            "status": "DESCRIPTIVE_FIXED_SUITE_ONLY",
            "suite": suite,
            "population_inference": None,
            "abstention_reason": "CALIBRATION_AND_CONFIRMATION_SHARE_TRAINING_UNITS",
        }
    cal_groups = _groups(cal, inference_unit_ids)
    conf_groups = _groups(conf, inference_unit_ids)
    if min(len(cal_groups), len(conf_groups)) < minimum_independent_units:
        return {
            "status": "DESCRIPTIVE_FIXED_SUITE_ONLY",
            "suite": suite,
            "population_inference": None,
            "abstention_reason": "TOO_FEW_INDEPENDENT_TRAINING_UNITS",
            "calibration_unit_count": len(cal_groups),
            "confirmation_unit_count": len(conf_groups),
            "minimum_independent_units": minimum_independent_units,
        }

    additive_by_unit = np.asarray([
        float((u[group] @ additive_direction).mean() / repair_scale)
        for group in conf_groups
    ])
    residual_by_unit = np.asarray([
        float((residual[group] @ residual_direction).mean() / repair_scale)
        for group in conf_groups
    ])
    # Each independent run contributes one energy-weighted gain estimate.
    aligned_by_unit = np.asarray([
        float(np.sum(np.sum(u[group] * r[group], axis=1)) / np.sum(repair_energy[group]))
        for group in conf_groups
    ])
    branches = {
        "additive": _branch_result(
            additive_by_unit, direction_must_repeat=True, draws=signflip_draws, seed=seed
        ),
        "repair_aligned": _branch_result(
            aligned_by_unit, direction_must_repeat=False, draws=signflip_draws, seed=seed + 1
        ),
        "residual_direction": _branch_result(
            residual_by_unit, direction_must_repeat=True, draws=signflip_draws, seed=seed + 2
        ),
    }
    return {
        "status": "POPULATION_INFERENCE_COMPLETE",
        "suite": suite,
        "population_inference": {
            "unit": "DECLARED_INDEPENDENT_TRAINING_UNIT",
            "calibration_unit_count": len(cal_groups),
            "confirmation_unit_count": len(conf_groups),
            "branches": branches,
        },
    }


def holm_adjusted_p(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm family-wise adjusted p-values for one declared family."""

    if not p_values:
        raise ValueError("Holm adjustment requires at least one p-value")
    ordered = sorted((float(value), key) for key, value in p_values.items())
    if ordered[0][0] < 0.0 or ordered[-1][0] > 1.0:
        raise ValueError("p-values must lie in [0, 1]")
    result: dict[str, float] = {}
    running = 0.0
    for rank, (value, key) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        result[key] = running
    return result
