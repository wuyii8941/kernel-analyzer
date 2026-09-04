"""Protocol-relative training numerical equivalence decisions.

The decision uses simultaneous confidence intervals for the three frozen
update effects.  Absence of statistical significance is never treated as
equivalence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from statistics import NormalDist

import numpy as np


BRANCHES = ("additive", "repair_aligned", "residual_direction")


def _t_critical(df: int, probability: float) -> float:
    z = NormalDist().inv_cdf(probability)
    f = float(df)
    return (
        z
        + (z**3 + z) / (4.0 * f)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * f**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * f**3)
    )


def simultaneous_intervals_from_joint_gram(
    joint_gram: Mapping[str, Sequence[Sequence[float]]],
    *,
    calibration_count: int = 16,
    family_alpha: float = 0.05,
) -> dict[str, list[float]]:
    """Recover three simultaneous held-out intervals from saved inner products.

    The saved matrices are sufficient to reproduce the fixed-direction,
    repair-aligned, and scaling-removed confirmation values without retaining
    the original high-dimensional parameter vectors.
    """

    uu = np.asarray(joint_gram["effect_effect"], dtype=np.float64)
    rr = np.asarray(joint_gram["repair_repair"], dtype=np.float64)
    ur = np.asarray(joint_gram["effect_repair"], dtype=np.float64)
    if uu.shape != rr.shape or uu.shape != ur.shape or uu.ndim != 2 or uu.shape[0] != uu.shape[1]:
        raise ValueError("joint Gram matrices must be square and share one shape")
    count = uu.shape[0]
    if not 1 < calibration_count < count - 1:
        raise ValueError("calibration split leaves too few confirmation states")
    cal = np.arange(calibration_count)
    conf = np.arange(calibration_count, count)
    repair_energy = np.diag(rr)
    if np.any(repair_energy <= 0.0):
        raise ValueError("repair energy must be positive for all states")
    repair_scale = math.sqrt(float(repair_energy[conf].mean()))

    calibration_norm = math.sqrt(max(float(uu[np.ix_(cal, cal)].mean()), 0.0))
    if calibration_norm <= 0.0:
        # Exact identity needs no learned direction: every implementation
        # effect is zero.  Other degenerate cases remain fail-closed.
        if np.count_nonzero(uu) == 0 and np.count_nonzero(ur) == 0:
            return {name: [0.0, 0.0] for name in BRANCHES}
        raise ValueError("calibration mean direction is not identifiable")
    additive = uu[np.ix_(conf, cal)].mean(axis=1) / calibration_norm / repair_scale

    gain = np.diag(ur) / repair_energy
    aligned = gain[conf]
    residual_gram = (
        uu
        - ur * gain[None, :]
        - ur.T * gain[:, None]
        + rr * gain[:, None] * gain[None, :]
    )
    residual_norm = math.sqrt(max(float(residual_gram[np.ix_(cal, cal)].mean()), 0.0))
    if residual_norm <= 0.0:
        residual = np.zeros(conf.size, dtype=np.float64)
    else:
        residual = residual_gram[np.ix_(conf, cal)].mean(axis=1) / residual_norm / repair_scale

    values = {"additive": additive, "repair_aligned": aligned, "residual_direction": residual}
    critical = _t_critical(conf.size - 1, 1.0 - family_alpha / (2.0 * len(BRANCHES)))
    intervals = {}
    for name, samples in values.items():
        center = float(samples.mean())
        half = critical * float(samples.std(ddof=1) / math.sqrt(samples.size))
        intervals[name] = [center - half, center + half]
    return intervals


def _interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} interval must have two bounds")
    lower, upper = float(value[0]), float(value[1])
    if lower > upper:
        raise ValueError(f"{name} interval bounds are reversed")
    return lower, upper


def classify_training_equivalence(
    intervals: Mapping[str, Sequence[float]],
    margins: Mapping[str, float],
    *,
    material_consequence_failed: bool = False,
    valid_protocol: bool = True,
) -> dict:
    """Classify one declared model/repair/optimizer/state protocol."""

    if not valid_protocol:
        return {"decision": "ABSTAIN", "reason": "PROTOCOL_OR_REPAIR_INVALID"}
    if set(intervals) != set(BRANCHES) or set(margins) != set(BRANCHES):
        raise ValueError("all and only the three frozen update branches are required")
    checked = {name: _interval(intervals[name], name) for name in BRANCHES}
    limits = {name: float(margins[name]) for name in BRANCHES}
    if any(value <= 0.0 for value in limits.values()):
        raise ValueError("equivalence margins must be positive")

    details = {}
    all_inside = True
    any_material = False
    any_small = False
    any_crosses_margin = False
    for name in BRANCHES:
        lower, upper = checked[name]
        margin = limits[name]
        inside = lower > -margin and upper < margin
        material = lower > margin or upper < -margin
        excludes_zero = lower > 0.0 or upper < 0.0
        crosses_margin = not inside and not material
        small = excludes_zero and inside
        details[name] = {
            "simultaneous_confidence_interval": [lower, upper],
            "equivalence_margin": margin,
            "inside_equivalence_region": inside,
            "entirely_beyond_materiality_margin": material,
            "detectable_inside_margin": small,
            "crosses_margin": crosses_margin,
        }
        all_inside &= inside
        any_material |= material
        any_small |= small
        any_crosses_margin |= crosses_margin

    if material_consequence_failed:
        decision = "MATERIAL_CONSEQUENCE"
        reason = "PREDECLARED_TRAINING_CONSEQUENCE_FAILED"
    elif any_material:
        decision = "MATERIAL_EFFECT"
        reason = "AT_LEAST_ONE_UPDATE_EFFECT_INTERVAL_IS_BEYOND_ITS_MARGIN"
    elif any_crosses_margin:
        decision = "INCONCLUSIVE"
        reason = "AT_LEAST_ONE_INTERVAL_CROSSES_AN_EQUIVALENCE_MARGIN"
    elif all_inside and any_small:
        decision = "DETECTABLE_BUT_SMALL"
        reason = "ALL_INTERVALS_ARE_INSIDE_MARGINS_AND_AT_LEAST_ONE_EXCLUDES_ZERO"
    elif all_inside:
        decision = "EQUIVALENT_UNDER_PROTOCOL"
        reason = "ALL_SIMULTANEOUS_INTERVALS_ARE_INSIDE_THE_FROZEN_EQUIVALENCE_REGION"
    else:
        decision = "INCONCLUSIVE"
        reason = "AVAILABLE_INTERVALS_DO_NOT_SUPPORT_EQUIVALENCE_OR_MATERIALITY"
    return {
        "decision": decision,
        "reason": reason,
        "material_consequence_failed": bool(material_consequence_failed),
        "branches": details,
    }
