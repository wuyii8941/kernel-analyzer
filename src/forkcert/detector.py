from __future__ import annotations

import math
from typing import Any

from .schema import ForkCertificate

REGION_STABLE = "stable"
REGION_FRAGILE = "fragile"
REGION_BUG = "bug"
REGION_UNKNOWN = "unknown"


def advantage_sign(advantage: float) -> int:
    if advantage > 0:
        return 1
    if advantage < 0:
        return -1
    return 0


def clip_boundary(sign: int, eps: float) -> float:
    if sign > 0:
        return math.log1p(eps)
    if sign < 0:
        return math.log1p(-eps)
    raise ValueError("advantage sign must be non-zero for PPO clipping branch")


def clip_active(logp_new: float, old_logp: float, sign: int, eps: float) -> bool:
    boundary = clip_boundary(sign, eps)
    log_ratio = logp_new - old_logp
    if sign > 0:
        return log_ratio > boundary
    return log_ratio < boundary


def classify_region(margin: float, delta: float, bound: float | None) -> str:
    if bound is None:
        return REGION_UNKNOWN
    if delta > bound:
        return REGION_BUG
    if margin > bound:
        return REGION_STABLE
    return REGION_FRAGILE


def detect_clipping_fork(
    *,
    case_id: str,
    token_index: int,
    logp_ref: float,
    logp_alt: float,
    old_logp: float,
    advantage: float | None = None,
    advantage_sign_value: int | None = None,
    eps: float = 0.2,
    token_id: int | None = None,
    token_text: str | None = None,
    path_ref: str = "path_ref",
    path_alt: str = "path_alt",
    delta_self_ref: float | None = None,
    delta_self_alt: float | None = None,
    delta_bound_legal: float | None = None,
    grad_contribution_ref: float | None = None,
    grad_contribution_alt: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> ForkCertificate:
    sign = advantage_sign_value if advantage_sign_value is not None else advantage_sign(float(advantage))
    if sign == 0:
        raise ValueError("zero advantage has no PPO clipping direction")

    boundary = clip_boundary(sign, eps)
    delta = abs(logp_alt - logp_ref)
    margin = abs((logp_ref - old_logp) - boundary)
    ref_branch = clip_active(logp_ref, old_logp, sign, eps)
    alt_branch = clip_active(logp_alt, old_logp, sign, eps)
    grad_diff = None
    if grad_contribution_ref is not None and grad_contribution_alt is not None:
        grad_diff = abs(grad_contribution_alt - grad_contribution_ref)

    return ForkCertificate(
        case_id=case_id,
        token_index=token_index,
        token_id=token_id,
        token_text=token_text,
        path_ref=path_ref,
        path_alt=path_alt,
        logp_ref=float(logp_ref),
        logp_alt=float(logp_alt),
        old_logp=float(old_logp),
        advantage_sign=sign,
        eps=float(eps),
        logprob_delta=float(delta),
        delta_self_ref=delta_self_ref,
        delta_self_alt=delta_self_alt,
        clip_boundary=float(boundary),
        clip_margin=float(margin),
        clip_ref=ref_branch,
        clip_alt=alt_branch,
        delta_bound_legal=delta_bound_legal,
        region=classify_region(margin, delta, delta_bound_legal),
        fork_possible=delta >= margin,
        actual_fork=ref_branch != alt_branch,
        grad_contribution_ref=grad_contribution_ref,
        grad_contribution_alt=grad_contribution_alt,
        grad_contribution_diff=grad_diff,
        metadata=metadata or {},
    )

