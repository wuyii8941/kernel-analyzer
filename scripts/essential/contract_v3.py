"""Numeric contract v3 (specs/phase2/numeric_contract_v3.md; parameters in docs/protocol_closure_v3_20261008.md section 2).

Three questions stay apart: how large the error is (reference enclosure), whether the mean effect is nonzero (the frozen
statistics layer plus the premise rules here), whether a requirement is violated (the contract).  This module adds, on top of
the frozen layer and without changing its fields:

- ``statistical_judgment``: the three "cannot judge" triggers (degenerate sample, distribution premise, sample size) with
  S0 = 2, N0 = 64, n_min = 16 read from the calibration (results/reference_eval/calibration_equivalence.json);
- ``precision_invariance``: the black-box semantic / numerical classification from the float32 / float64 deviation ratio;
- ``four_column``: the four report columns for a classified condition.
"""
from __future__ import annotations

import numpy as np

S0 = 2.0          # |unit skewness| above which the t approximation is not trusted (calibration: coverage < 0.93 at 3.3)
N0 = 64           # ... for n <= N0 confirmation units
N_MIN = 16        # fewest confirmation units for any statistical judgment
BOOTSTRAP_OK_N = 64   # bootstrap-t calibrated error <= 0.07 at skewness 3.3 only for n = 64 (0.068); n = 16 gives 0.095

R_NUMERICAL = 2.0 ** -16
R_SEMANTIC = 2.0 ** -4
F64_NOISE = 2.0 ** -40


def statistical_judgment(rec: dict) -> dict:
    """Map one frozen rule record (analysis._summarize / apply_direction_rules output) to the v3 judgment.

    Returns {"judgment", "basis"}: judgment is one of "nonzero (positive)", "nonzero (negative)", "not confirmed",
    "cannot judge (degenerate)", "cannot judge (distribution)", "cannot judge (sample)", "cannot judge (numerical)",
    "not established"."""
    v = rec.get("verdict")
    n = int(rec.get("n") or len(rec.get("per_unit_bounds") or []))
    if v is None:
        return {"judgment": "not established", "basis": "no rule record"}
    if v in ("UNRESOLVED_REFERENCE", "UNRESOLVED_MEASUREMENT"):
        return {"judgment": "not established", "basis": f"{v}: {rec.get('reason', '')}"}
    if v == "UNRESOLVED_NUMERICAL":
        return {"judgment": "cannot judge (numerical)", "basis": rec.get("reason", "")}
    if v == "UNRESOLVED_SAMPLE":
        reason = rec.get("reason", "")
        if "zero sample variance" in reason:
            return {"judgment": "cannot judge (degenerate)", "basis": reason}
        return {"judgment": "cannot judge (sample)", "basis": reason}
    bounds = rec.get("per_unit_bounds")
    if bounds is not None and len(bounds) >= 1:
        b = np.asarray(bounds, dtype=np.float64)
        if np.all(b == b[0]):
            return {"judgment": "cannot judge (degenerate)", "basis": "all per-unit interval endpoints are identical"}
    if n < N_MIN:
        return {"judgment": "cannot judge (sample)", "basis": f"n = {n} < n_min = {N_MIN}"}
    g = rec.get("unit_skewness")
    if g is not None and abs(g) > S0 and n <= N0:
        if n >= BOOTSTRAP_OK_N and rec.get("robust"):
            r = rec["robust"]["verdict"]
            return {"judgment": _verdict(r), "basis": f"|skewness| = {abs(g):.2f} > {S0}, n = {n}: bootstrap-t companion"}
        return {"judgment": "cannot judge (distribution)",
                "basis": f"|skewness| = {abs(g):.2f} > {S0} with n = {n} <= {N0}; bootstrap-t not calibrated <= 0.07 here"}
    return {"judgment": _verdict(v), "basis": "endpoint-conservative t (frozen layer)"}


def _verdict(v):
    return {"DETECTED_POSITIVE": "nonzero (positive)", "DETECTED_NEGATIVE": "nonzero (negative)",
            "NOT_CONFIRMED": "not confirmed"}.get(v, f"unmapped: {v}")


def precision_invariance(k32, k64, f_mid, f_mid64=None) -> dict:
    """Black-box semantic / numerical classification (numeric_contract_v3 section 2).

    k32, k64: outputs of the same algorithm in float32 and float64 (same inputs, or each on its own received inputs with
    the spec evaluated on those: then f_mid is the spec on the float32 candidate's inputs and f_mid64 on the float64
    candidate's); f_mid: spec interval midpoints.  Non-finite candidate values go to column 4 and are not ratioed.
    Anomalies (float32 exact, float64 off) are reported with how many lie within the float64 evaluation noise."""
    k32, k64, f = (np.asarray(a, dtype=np.float64).ravel() for a in (k32, k64, f_mid))
    f64 = f if f_mid64 is None else np.asarray(f_mid64, dtype=np.float64).ravel()
    nonfin32, nonfin64 = ~np.isfinite(k32), ~np.isfinite(k64)
    nonfinite = nonfin32 | nonfin64 | ~np.isfinite(f) | ~np.isfinite(f64)
    d32 = np.abs(k32 - f)
    d64 = np.abs(k64 - f64)
    fin = ~nonfinite
    agree = fin & (d32 == 0) & (d64 == 0)
    anomaly = fin & (d32 == 0) & (d64 != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(fin & (d32 > 0), d64 / np.where(d32 > 0, d32, 1.0), np.nan)
    numerical = fin & (d32 > 0) & (r < R_NUMERICAL)
    semantic = fin & (d32 > 0) & (r > R_SEMANTIC)
    undecided = fin & (d32 > 0) & ~numerical & ~semantic
    noise64 = F64_NOISE * (1 + np.abs(f64))
    semantic_real = semantic & (d64 > noise64)
    anomaly_noise = anomaly & (d64 <= noise64)
    out = {"elements": int(f.size), "agree": int(agree.sum()), "numerical": int(numerical.sum()),
           "semantic": int(semantic.sum()), "semantic_above_f64_noise": int(semantic_real.sum()),
           "undecided": int(undecided.sum()), "anomaly_f32_exact_f64_off": int(anomaly.sum()),
           "anomaly_within_f64_noise": int(anomaly_noise.sum()),
           "nonfinite_f32_only": int((nonfin32 & ~nonfin64).sum()), "nonfinite_f64_only": int((nonfin64 & ~nonfin32).sum()),
           "nonfinite_both": int((nonfin32 & nonfin64).sum())}
    # condition level as the contract states it: "semantic deviation present" iff one semantic element lies above the
    # float64 evaluation noise; otherwise no semantic element (the numerical / undecided / agree counts stay in the record).
    # Anomalies beyond the float64 noise are flagged for review apart.
    if semantic_real.any():
        cond = "semantic deviation present"
    elif (anomaly & ~anomaly_noise).any():
        cond = "no semantic element; anomaly beyond float64 noise: review"
    else:
        cond = "no semantic element"
    out["semantic_fraction"] = float(semantic.sum() / max(1, int(fin.sum())))
    out["condition"] = cond
    out["column4_nonfinite"] = bool(nonfinite.any() and np.isfinite(f).all())
    return out


def four_column(kind: str) -> str:
    """kind -> report column: 'doc_violation' 1, 'reading_difference' 2, 'undefined_convention' 3, 'numerical_failure' 4."""
    return {"doc_violation": "1 文档明确条款的违反", "reading_difference": "2 声明解释下的差异",
            "undefined_convention": "3 未定义情形的实现约定", "numerical_failure": "4 数值失败"}[kind]
