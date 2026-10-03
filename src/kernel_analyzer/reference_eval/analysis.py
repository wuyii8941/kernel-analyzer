"""Unified entry: captured launches + a measurement declaration -> reference, residual, classes, statistics.

The declaration is a JSON document (see ``results/reference_eval/declarations``)::

    {
      "location": "...",                       # free text, copied into the report
      "capture_root": "...",                   # unitNNN/<variant>/launchNNN packages (+ arrays.npz)
      "variants": ["original", "reverse"],     # implementations captured in every unit
      "target_buffer": "ACC",                  # TTIR argument whose final value is measured
      "programs": {"rows": [...]} | {"all": true},
      "mode": "numerical_difference",
      "measurement": {"type": "adamw_write", "lr": ..., "betas": [...], "eps": ...,
                      "weight_decay": 0.0, "parameter_values": "weight"} | {"type": "output"},
      "comparisons": [{"candidate": "original", "reference": "K_R"},
                      {"candidate": "original", "reference": "reverse"}],
      "direction_rules": ["fixed_direction", "aligned_reference_update"],
                                               # also "cross_fit" (folds: "cross_fit_folds", default 2) and
                                               # "grouped_<rule>" (aggregate first: "groups": {"size": s})
      "population": {"description": "...", "calibration": 32, "confirmation": 64},
      "alpha": 0.05,
      "factors": {...}                         # treatment of the five factors, copied into the report
    }

Stage "reference" evaluates every unit and variant (composed reference over
the unit's launches) and stores compact arrays; stage "statistics" builds u =
measure(K) - measure(RN(K_R)) (or variant - variant), projects it with the
declared direction rules and applies the endpoint-conservative t test with
Holm correction.  Binding (where the kernel is and how it is captured) stays
outside this module.

Direction rules (the projection of an interval vector [l, h] on w is
[sum min(l w, h w), sum max(l w, h w)]):

* fixed_direction: w = normalized mean of the calibration units; the
  confirmation units are tested;
* aligned_reference_update: w_i = the unit's own reference measurement, a
  direction given by the mechanism (uniform scaling) rather than learned;
* cross_fit: K folds over all units; fold k is projected on the direction
  learned from the other folds and tested at alpha / K; the rule rejects when
  any fold does (Bonferroni, valid under the dependence between folds);
* grouped_<rule>: coordinates are first summed over declared contiguous
  groups (rows, tensors), which lowers the dimension the direction is learned
  in; the group sums of [l, h] enclose the group sums of u.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import intervals as iv
from .capture import load_launch
from .ttir_eval import ST_OK, ST_UNDEF, NumericMode, evaluate_sequence
from .ttir_mapping import kernel_coverage
from .ttir_parser import parse_ttir

STORAGE_FORMAT = {"float32": "f32", "float16": "f16", "bfloat16": "bf16", "float64": "f64"}


def load_declaration(path) -> dict:
    decl = json.loads(Path(path).read_text())
    decl.setdefault("mode", NumericMode.NUMERICAL_DIFFERENCE)
    decl.setdefault("alpha", 0.05)
    decl.setdefault("direction_rules", ["fixed_direction"])
    return decl


def _units(root: Path) -> list:
    return sorted(p for p in Path(root).iterdir() if p.is_dir() and p.name.startswith("unit"))


# ---------------------------------------------------------------------------
# Stage 1: reference
# ---------------------------------------------------------------------------


def reference_stage(decl: dict, out_dir: Path, units: Optional[list] = None, delete_captures: bool = False) -> list:
    """Composed reference of the target buffer for every unit and variant."""

    import shutil

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(decl["capture_root"])
    selected = [u for u in _units(root) if units is None or u.name in units]
    summaries = []
    for unit_dir in selected:
        arrays, summary = {}, {"unit": unit_dir.name}
        for variant in decl["variants"]:
            vdir = unit_dir / variant
            launches = [load_launch(p) for p in sorted(vdir.glob("launch*"))]
            coverage = kernel_coverage(parse_ttir(launches[0].asm["ttir"]))
            rows = decl["programs"].get("rows")
            if "rows_from" in decl["programs"]:
                rows = json.loads((root / decl["programs"]["rows_from"]).read_text())["rows"]
            if rows is not None:
                pids = [(int(r), 0, 0) for r in rows]
                programs_for = lambda launch, pids=pids: pids  # noqa: E731
            else:
                programs_for = None
            t0 = time.time()
            seq = evaluate_sequence(launches, mode=decl["mode"], programs_for=programs_for)
            seconds = time.time() - t0
            arg = next(a for a in launches[0].args if a.name == decl["target_buffer"])
            buf = seq.memory[arg.storage_ptr]
            m = buf.written
            st, cond = buf.st[m], buf.cond[m]
            extra = np.load(vdir / "arrays.npz") if (vdir / "arrays.npz").exists() else {}
            arrays.update({f"{variant}__actual": buf.actual_after[m], f"{variant}__ref_lo": buf.lo[m],
                           f"{variant}__ref_hi": buf.hi[m], f"{variant}__st": st, f"{variant}__cond": cond,
                           f"{variant}__index": buf.global_indices()[m]})
            for key in getattr(extra, "files", []):
                # Binding arrays (e.g. parameter values) must follow the sorted window order of the target.
                if extra[key].shape[0] == int(m.sum()):
                    arrays[f"{variant}__extra__{key}"] = extra[key]
            r_lo = iv.add_bounds(buf.actual_after[m], -buf.hi[m])[0]
            r_hi = iv.add_bounds(buf.actual_after[m], -buf.lo[m])[1]
            finite = st == ST_OK
            summary[variant] = {
                "launches": len(launches), "kernel": launches[0].kernel_name,
                "kernel_hash": launches[0].kernel_hash, "libtriton_sha256": launches[0].libtriton_sha256,
                "triton": launches[0].environment.get("triton"), "coverage_complete": coverage["complete"],
                "aborted_programs": sum(len(r.aborted) for r in seq.launches),
                "target_rewritten_outside": any(e["buffer"] == decl["target_buffer"] for e in seq.external_writes),
                "external_inputs": len(seq.external_writes),
                "elements": int(m.sum()),
                "classes": {"complete_composed": int((finite & ~cond).sum()),
                            "conditional_local": int((finite & cond).sum()),
                            "special_value": int(((st > ST_OK) & (st < ST_UNDEF)).sum()),
                            "not_established": int((st >= ST_UNDEF).sum())},
                "reference_width_max": float(np.max(buf.hi[m][finite] - buf.lo[m][finite])) if finite.any() else None,
                "residual_positive": int((finite & (r_lo > 0)).sum()),
                "residual_negative": int((finite & (r_hi < 0)).sum()),
                "residual_contains_zero": int((finite & (r_lo <= 0) & (r_hi >= 0)).sum()),
                "storage_dtype": arg.dtype, "seconds": round(seconds, 2),
            }
        np.savez_compressed(out_dir / f"{unit_dir.name}.npz", **arrays)
        (out_dir / f"{unit_dir.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        if delete_captures:
            shutil.rmtree(unit_dir)
    return summaries


# ---------------------------------------------------------------------------
# Stage 2: statistics
# ---------------------------------------------------------------------------


def _measure(decl: dict, arrays, variant: str, values: np.ndarray, device: str) -> np.ndarray:
    meas = decl["measurement"]
    if meas["type"] == "output":
        return np.asarray(values, dtype=np.float64)
    if meas["type"] == "adamw_write":
        import torch

        params = arrays[f"{variant}__extra__{meas['parameter_values']}"]
        w = torch.nn.Parameter(torch.as_tensor(params, dtype=torch.float32, device=device).clone())
        before = w.detach().clone()
        opt = torch.optim.AdamW([w], lr=meas["lr"], betas=tuple(meas["betas"]), eps=meas["eps"],
                                weight_decay=meas.get("weight_decay", 0.0), foreach=False, fused=False)
        w.grad = torch.as_tensor(values, dtype=torch.float32, device=device)
        opt.step()
        return (w.detach() - before).double().cpu().numpy()
    raise ValueError(f"unknown measurement {meas['type']}")


def _reference_interval(decl, arrays, variant):
    """The reference interval: automatic K_R, or a manual reference with its declared error bound.

    ``reference_source = {"type": "manual", "value": <extra key>, "abs_sum": <extra key>, "terms": n}``
    takes a hand-written FP64 accumulation x_m with |x_m - x| <= gamma_n * sum |terms|.
    """

    src = decl.get("reference_source", {"type": "automatic"})
    if src.get("type", "automatic") == "automatic":
        return arrays[f"{variant}__ref_lo"], arrays[f"{variant}__ref_hi"]
    manual = arrays[f"{variant}__extra__{src['value']}"]
    bound = iv.up(arrays[f"{variant}__extra__{src['abs_sum']}"] * iv.gamma(src["terms"]) * (1 + 8 * iv.U))
    return iv.down(manual - bound), iv.up(manual + bound)


def _reference_bounds(decl, arrays, variant, device, valid=None):
    """measure(RN(reference)) bounds; RN to the target's storage format at both interval ends.
    Coordinates outside ``valid`` (reference not of a declared class) are measured at the actual value
    and dropped by the caller."""

    fmt = STORAGE_FORMAT[decl.get("_storage_dtype", "float32")]
    ref_lo, ref_hi = _reference_interval(decl, arrays, variant)
    if valid is not None:
        actual = arrays[f"{variant}__actual"]
        ref_lo, ref_hi = np.where(valid, ref_lo, actual), np.where(valid, ref_hi, actual)
    lo, of_lo = iv.round_nearest_even(ref_lo, fmt)
    hi, of_hi = iv.round_nearest_even(ref_hi, fmt)
    if of_lo.any() or of_hi.any():
        raise ValueError("reference overflows the storage format")
    ambiguous = lo != hi
    m_lo = _measure(decl, arrays, variant, lo, device)
    m_hi = _measure(decl, arrays, variant, hi, device) if ambiguous.any() else m_lo
    return np.minimum(m_lo, m_hi), np.maximum(m_lo, m_hi), int(ambiguous.sum())


def _t_stats(values: np.ndarray, alpha: float):
    from scipy.stats import t

    n = values.size
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    half = float(t.ppf(1 - alpha / 2, n - 1)) * sd / math.sqrt(n) if n > 1 else math.inf
    if sd == 0:
        p = 0.0 if mean != 0 else 1.0
    else:
        p = float(2 * t.sf(abs(mean) / (sd / math.sqrt(n)), n - 1))
    return mean, sd, (mean - half, mean + half), p


def _holm(pvalues, alpha):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    reject = [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (len(pvalues) - rank):
            reject[i] = True
        else:
            break
    return reject


def _one_sided_p(values: np.ndarray, greater: bool) -> float:
    """t-test p-value for E[values] > 0 (greater) or E[values] < 0."""

    from scipy.stats import t

    n = values.size
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    if sd == 0 or n < 2:
        return 0.0 if (mean > 0 if greater else mean < 0) else 1.0
    stat = mean / (sd / math.sqrt(n))
    return float(t.sf(stat, n - 1) if greater else t.cdf(stat, n - 1))


def _summarize(name, rule, l, h, alpha):
    """Endpoint-conservative t inference for the mean projection mu, known only to lie in [E[l], E[h]].

    A positive mean needs E[l] > 0 and a negative mean E[h] < 0, so the conservative two-sided p-value is
    2 min(p(E[l] > 0), p(E[h] < 0)), capped at 1 -- the same decision as the interval [lower bound of E[l],
    upper bound of E[h]] at level alpha.  (An earlier form, max of the two two-sided p-values, came out small
    for boxes straddling zero with E[l] < 0 < E[h]; Holm then counted such tests as rejections.)"""

    mean, sd, interval, p_mid = _t_stats(0.5 * (l + h), alpha)
    _, _, ci_l, _ = _t_stats(l, alpha)
    _, _, ci_h, _ = _t_stats(h, alpha)
    lower, upper = ci_l[0], ci_h[1]  # endpoint-conservative: lower bound of E[l], upper bound of E[h]
    verdict = "DETECTED_POSITIVE" if lower > 0 else ("DETECTED_NEGATIVE" if upper < 0 else "NOT_CONFIRMED")
    p = min(1.0, 2.0 * min(_one_sided_p(l, True), _one_sided_p(h, False)))
    return {"comparison": name, "rule": rule, "n": int(l.size), "mean_projection": mean, "sd": sd,
            "t_interval": list(interval), "lower_bound_of_E_l": lower, "upper_bound_of_E_h": upper,
            "positive": int((l > 0).sum()), "negative": int((h < 0).sum()),
            "zero_or_ambiguous": int(((l <= 0) & (h >= 0)).sum()), "verdict": verdict,
            "p_value_two_sided_conservative": p}


def _project(lows: np.ndarray, highs: np.ndarray, w: np.ndarray):
    """Interval projection over the last axis: a strict outward enclosure of sum(e w) for every e in the box
    (directed products, exact sum, one ulp outward), so the endpoints stay conservative."""

    return iv.project_bounds(lows, highs, w)


def _learned_direction(lows, highs):
    direction = (0.5 * (lows + highs)).mean(axis=0)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else None


# ---------------------------------------------------------------------------
# Decision layer: direction rules, endpoint-conservative inference, multiplicity
# ---------------------------------------------------------------------------

# Every rule projects the residual interval of a unit on a direction w and tests the mean projection mu with
# the endpoint-conservative t inference.  "positive" / "negative" say what the sign of mu means for the raw
# residual e (several directions carry a minus sign, so a positive mu can mean a negative residual).
RULES = {
    "negative_ones": {"alias": "R1", "question": "fixed_direction_mean", "definition": "w = -1 / sqrt(n)",
                      "positive": "the coordinate mean of the residual is negative",
                      "negative": "the coordinate mean of the residual is positive"},
    "toward_zero": {"alias": "R2", "question": "reference_alignment",
                    "definition": "w = -sign(reference) / sqrt(n), per unit",
                    "positive": "the residual has the opposite sign of the reference: magnitudes pulled toward zero",
                    "negative": "the residual has the sign of the reference: magnitudes pushed away from zero"},
    "scale_down": {"alias": "R3", "question": "reference_alignment",
                   "definition": "w = -reference / |reference|, per unit",
                   "positive": "the residual is anti-aligned with the reference: the output is scaled down",
                   "negative": "the residual is aligned with the reference: the output is scaled up"},
    "aligned_reference_update": {"question": "reference_alignment",
                                 "definition": "w = reference / |reference|, per unit",
                                 "positive": "the residual is aligned with the reference: the output is scaled up",
                                 "negative": "the residual is anti-aligned with the reference: scaled down"},
    "declared_vector": {"alias": "R4", "question": "fixed_direction_mean",
                        "definition": "a declared fixed vector (normalized)",
                        "positive": "the residual points along the declared vector",
                        "negative": "the residual points against the declared vector"},
    "fixed_direction": {"alias": "R5", "question": "fixed_direction_mean",
                        "definition": "w = normalized mean of the development midpoints (saved)",
                        "positive": "the residual points along the direction learned on the development units",
                        "negative": "the residual points against the learned direction"},
    "cross_fit": {"question": "fixed_direction_mean",
                  "definition": "K folds; each fold tested on the direction learned from the others, Bonferroni",
                  "positive": "the residual points along the learned direction of the selected fold",
                  "negative": "the residual points against the learned direction of the selected fold"},
}
ALIASES = {info["alias"]: name for name, info in RULES.items() if "alias" in info}


def rule_base(rule: str) -> str:
    base = rule[len("grouped_"):] if rule.startswith("grouped_") else rule
    return ALIASES.get(base, base)


def interpret(rule: str, verdict: str) -> Optional[str]:
    info = RULES[rule_base(rule)]
    return {"DETECTED_POSITIVE": info["positive"], "DETECTED_NEGATIVE": info["negative"]}.get(verdict)


def holm_adjusted(pvalues) -> list:
    """Holm step-down adjusted p-values: reject at level alpha exactly when the adjusted p is <= alpha."""

    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [1.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def apply_holm(tests: list, alpha: float, p_key: str = "p_value_two_sided_conservative") -> None:
    """Holm over ``tests`` (dicts with ``p_key`` and an interval verdict): adds the adjusted p-value, the
    rejection and the final verdict (the interval verdict when rejected, NOT_CONFIRMED otherwise)."""

    for t, adj in zip(tests, holm_adjusted([t[p_key] for t in tests])):
        t["holm_adjusted_p"] = adj
        t["holm_reject"] = bool(adj <= alpha)
        t["final_verdict"] = t["verdict"] if t["holm_reject"] else "NOT_CONFIRMED"
        t["final_interpretation"] = interpret(t["rule"], t["final_verdict"]) if "rule" in t else None


def _with_units(result, l, h, direction_record):
    result["per_unit_bounds"] = [[float(a), float(b)] for a, b in zip(l, h)]
    result["direction"] = direction_record
    return result


def _cross_fit(name, rule, lows, highs, folds, alpha, arrays=None):
    n = lows.shape[0]
    edges = np.linspace(0, n, folds + 1).astype(int)
    per_fold, directions = [], []
    for k in range(folds):
        a, b = edges[k], edges[k + 1]
        rest = np.r_[0:a, b:n]
        w = _learned_direction(lows[rest], highs[rest])
        if w is None:
            return {"comparison": name, "rule": rule, "verdict": "UNRESOLVED_MEASUREMENT",
                    "reason": f"direction learned without fold {k} is zero"}
        directions.append(w)
        l, h = _project(lows[a:b], highs[a:b], w)
        per_fold.append(_with_units(_summarize(name, f"{rule}_fold{k}", l, h, alpha / folds), l, h,
                                    {"kind": "learned_without_fold", "saved_as": f"{rule}__fold{k}"}))
        if arrays is not None:
            arrays[f"{name}__{rule}__fold{k}"] = w
    best_k = min(range(folds), key=lambda k: per_fold[k]["p_value_two_sided_conservative"])
    best = per_fold[best_k]
    # The verdict is the Bonferroni decision over folds; the projection summary belongs to the selected
    # fold only (its own direction and units), not to a pooled estimate over all units.
    return {"comparison": name, "rule": rule, "n_total": n, "n_fold": [r["n"] for r in per_fold],
            "folds": folds, "fold_level": alpha / folds, "selected_fold": best_k,
            "verdict": best["verdict"],
            "p_value_two_sided_conservative": min(1.0, folds * best["p_value_two_sided_conservative"]),
            "selected_fold_summary": best, "fold_results": per_fold,
            "fold_directions_saved": [w.tolist() for w in directions] if directions[0].size <= 4096 else None,
            "fold_direction_cosines": [float(directions[i] @ directions[j])
                                       for i in range(folds) for j in range(i + 1, folds)],
            "note": "selected_fold_summary is the selected fold's projection, not a common effect estimate"}


def _direction(base, lo, hi, ref, n_cal, conf, declared_vectors, rule):
    """(w for the confirmation units, a JSON record of the direction actually used, an array to save or None).
    Unit-dependent directions are rebuilt exactly from the saved reference midpoints and the recorded
    normalization scalars."""

    n = lo.shape[1]
    if base == "negative_ones":
        return np.full(n, -1.0 / math.sqrt(n)), {"kind": "constant", "value": -1.0 / math.sqrt(n)}, None
    if base == "toward_zero":
        scale = 1.0 / math.sqrt(n)
        return -np.sign(ref[conf]) * scale, {"kind": "per_unit", "formula": "-sign(reference) * scale",
                                              "scale": scale}, None
    if base in ("scale_down", "aligned_reference_update"):
        sign = -1.0 if base == "scale_down" else 1.0
        norms = np.linalg.norm(ref[conf], axis=1, keepdims=True)
        w = np.divide(sign * ref[conf], norms, out=np.zeros_like(ref[conf]), where=norms > 0)
        return w, {"kind": "per_unit", "formula": f"{'-' if sign < 0 else ''}reference / norm",
                   "norms": [float(x) for x in norms[:, 0]]}, None
    if base == "declared_vector":
        if not declared_vectors or rule not in declared_vectors:
            raise ValueError(f"rule {rule} needs a declared vector")
        v = np.asarray(declared_vectors[rule], dtype=np.float64)
        w = v / np.linalg.norm(v)
        return w, {"kind": "declared", "saved_as": rule}, w
    if base == "fixed_direction":
        w = _learned_direction(lo[:n_cal], hi[:n_cal])
        return w, {"kind": "learned_on_development_units", "saved_as": rule}, w
    raise ValueError(f"unknown direction rule {rule}")


def apply_direction_rules(name, lows, highs, ref_measure, decl, n_cal, n_conf, alpha, valid=None,
                          arrays=None) -> list:
    """Project the unit residual intervals [lows, highs] (units x coordinates) with the declared rules.

    ``valid`` marks the coordinates of the declared set (original coordinate order).  Ungrouped rules use
    the valid coordinates; grouped rules keep the declared groups (contiguous blocks of ``groups.size`` in
    the original order) and use only complete groups -- remaining coordinates are never regrouped.
    Every result carries the per-unit projection bounds, the direction actually used (vectors are put into
    ``arrays`` when given) and the reading of its sign.  ``decl["declared_vectors"]`` maps a declared_vector
    rule name to its vector over the valid coordinates.
    """

    valid = np.ones(lows.shape[1], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    results = []
    conf = slice(n_cal, n_cal + n_conf)
    for rule in decl["direction_rules"]:
        base, lo, hi, ref = rule_base(rule), lows[:, valid], highs[:, valid], ref_measure[:, valid]
        extra = {"coordinates_used": int(valid.sum()), "question": RULES[base]["question"],
                 "rule_definition": RULES[base]["definition"]}
        if rule.startswith("grouped_"):
            size = decl["groups"]["size"]
            if lows.shape[1] % size:
                raise ValueError(f"group size {size} does not divide {lows.shape[1]} coordinates")
            complete = valid.reshape(-1, size).all(axis=1)
            extra.update({"groups_used": int(complete.sum()), "groups_declared": int(complete.size),
                          "groups_dropped_incomplete": int((~complete).sum())})
            extra.pop("coordinates_used")
            if not complete.any():
                results.append({"comparison": name, "rule": rule, "verdict": "UNRESOLVED_REFERENCE",
                                "reason": "no complete declared group", **extra})
                continue
            # Group sums divided by sqrt(size): a unit direction over groups pulls back to a unit direction
            # over the original coordinates, so projections stay in the units of the ungrouped rules.
            # Sums and the division are directed, so the grouped endpoints stay conservative.
            norm = math.sqrt(size)
            s_lo, s_hi = iv.fsum_bounds(lows.reshape(lows.shape[0], -1, size)[:, complete],
                                        highs.reshape(highs.shape[0], -1, size)[:, complete], axis=2)
            lo, hi = iv.div_bounds(s_lo, norm)[0], iv.div_bounds(s_hi, norm)[1]
            ref = ref_measure.reshape(ref_measure.shape[0], -1, size)[:, complete].sum(axis=2) / norm
        if base == "cross_fit":
            results.append(_cross_fit(name, rule, lo[:n_cal + n_conf], hi[:n_cal + n_conf],
                                      decl.get("cross_fit_folds", 2), alpha, arrays))
            results[-1].update(extra)
            continue
        declared = decl.get("declared_vectors") or {}
        if rule in declared and np.size(declared[rule]) == valid.size and not rule.startswith("grouped_"):
            declared = {**declared, rule: np.asarray(declared[rule], dtype=np.float64)[valid]}  # full length given
        w, record, saved = _direction(base, lo, hi, ref, n_cal, conf, declared, rule)
        if w is None:
            results.append({"comparison": name, "rule": rule, "verdict": "UNRESOLVED_MEASUREMENT",
                            "reason": "calibration direction is zero", **extra})
            continue
        if saved is not None and arrays is not None:
            arrays[f"{name}__{rule}"] = saved
        l, h = _project(lo[conf], hi[conf], w)
        results.append(_with_units(_summarize(name, rule, l, h, alpha), l, h, record))
        results[-1]["interpretation"] = interpret(rule, results[-1]["verdict"])
        results[-1].update(extra)
    return results


def residual_summary(lows, highs, valid) -> dict:
    """Numerical inconsistency at the measurement point, over the coordinate set and all units: where the
    residual interval lies entirely above / below zero, or contains zero."""

    lo, hi = lows[:, valid], highs[:, valid]
    total = lo.size
    return {"positive_frac": float((lo > 0).sum() / total), "negative_frac": float((hi < 0).sum() / total),
            "contains_zero_frac": float(((lo <= 0) & (hi >= 0)).sum() / total),
            "mean_midpoint": float((0.5 * (lo + hi)).mean()), "max_width": float((hi - lo).max())}


def assess_units(name, lows, highs, ref_mid, ok, n_dev, rules, alpha=0.05, groups=None, declared_vectors=None,
                 alignment_reference=None, detector_shape=None, run_detector=True, cross_fit_folds=2,
                 confirmation_invalid="unresolved", measurement=None, unit_ids=None, seed=0):
    """The decision layer for one measured quantity: units x coordinates residual intervals -> record.

    lows, highs: residual interval endpoints; ref_mid: the reference midpoints that define the
    reference-dependent directions; ok: coordinates whose reference is of a declared class, per unit.  The
    coordinate set is fixed on the first ``n_dev`` (development) units; a confirmation unit whose reference is
    not valid on that set follows ``confirmation_invalid`` ("unresolved" or "drop_unit").  Rules are tested on
    the confirmation units with strict projection bounds and the endpoint-conservative t inference; the
    default detector (interval version) runs on the same coordinates.  Multiplicity across tests is applied by
    the caller (:func:`apply_holm`), because its scope (all comparisons, all programs) is wider than one call.

    Returns (record, arrays): a JSON-ready record with everything needed to recompute the decisions offline
    (coordinate set size, dev/confirmation unit ids, per-unit projection bounds, directions and normalization
    scalars, raw p-values, exclusions) and the vectors to save (coordinate set, learned / declared
    directions, detector directions)."""

    lows, highs = np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)
    ok = np.asarray(ok, dtype=bool)
    n_units = lows.shape[0]
    unit_ids = list(range(n_units)) if unit_ids is None else list(unit_ids)
    record = {"comparison": name, "measurement": measurement, "alpha": alpha,
              "units": {"development": unit_ids[:n_dev], "confirmation": unit_ids[n_dev:]},
              "coordinate_set": "fixed on the development units (reference of a declared class in every one)"}
    arrays = {}
    valid = ok[:n_dev].all(axis=0)
    record["coordinates"] = {"total": int(valid.size), "used": int(valid.sum()),
                             "excluded_on_development": int((~valid).sum())}
    if not valid.any():
        record.update(verdict="UNRESOLVED_REFERENCE", reason="no coordinate valid on all development units")
        return record, arrays
    bad = [i for i in range(n_dev, n_units) if (valid & ~ok[i]).any()]
    if bad and confirmation_invalid == "unresolved":
        record.update(verdict="UNRESOLVED_REFERENCE",
                      reason="confirmation units with references not valid on the coordinate set",
                      confirmation_units_invalid=[unit_ids[i] for i in bad])
        return record, arrays
    keep = np.array([i not in bad for i in range(n_units)])
    record["confirmation_units_dropped"] = [unit_ids[i] for i in bad]
    lows, highs, ref_mid = lows[keep], highs[keep], np.asarray(ref_mid, dtype=np.float64)[keep]
    n_conf = int(keep.sum()) - n_dev
    arrays[f"{name}__coordinate_set"] = valid
    record["residual"] = residual_summary(lows, highs, valid)
    decl = {"direction_rules": list(rules), "groups": groups, "cross_fit_folds": cross_fit_folds,
            "declared_vectors": declared_vectors}
    record["rules"] = apply_direction_rules(name, lows, highs, ref_mid, decl, n_dev, n_conf, alpha, valid=valid,
                                            arrays=arrays)
    if run_detector:
        from .detect import detect

        align = ref_mid if alignment_reference is None else np.asarray(alignment_reference, dtype=np.float64)[keep]
        shape = detector_shape if valid.all() else None
        det = detect(lows[:, valid], highs[:, valid], align[:, valid], n_dev, shape=shape, seed=seed,
                     return_directions=True)
        for key, vec in det.pop("directions").items():
            if np.ndim(vec) == 1:  # unit-dependent directions are rebuilt from the saved references
                arrays[f"{name}__detector__{key}"] = vec
        record["default_detector"] = det
    return record, arrays


def statistics_stage(decl: dict, ref_dir: Path, device: str = "cuda:0", arrays_out=None) -> dict:
    """Residuals per comparison -> :func:`assess_units` (rules, default detector) -> Holm over all
    comparison x rule tests.  ``arrays_out`` (an .npz path) receives the coordinate set and the directions
    actually used, so the decisions can be recomputed offline from the report and the saved references."""

    ref_dir = Path(ref_dir)
    paths = sorted(ref_dir.glob("unit*.npz"))
    pop = decl["population"]
    n_cal, n_conf = pop["calibration"], pop["confirmation"]
    if len(paths) != n_cal + n_conf:
        raise ValueError(f"declaration expects {n_cal + n_conf} units, found {len(paths)}")
    summaries = [json.loads(p.with_suffix(".json").read_text()) for p in paths]
    decl = dict(decl)
    decl["_storage_dtype"] = summaries[0][decl["variants"][0]]["storage_dtype"]
    alpha = decl["alpha"]
    # The coordinate set is fixed on the calibration (development) units: a coordinate enters only if its
    # reference is of a declared class (default: complete composed) in every calibration unit and variant.
    # Confirmation data never re-selects coordinates; a confirmation unit whose reference is not valid on
    # that set follows the declared policy ("unresolved", the default, or "drop_unit").
    classes = set(decl.get("reference_classes", ["complete_composed"]))
    policy = decl.get("confirmation_invalid", "unresolved")
    if policy not in ("unresolved", "drop_unit"):
        raise ValueError("confirmation_invalid must be 'unresolved' or 'drop_unit'")
    excluded = {"conditional_local": 0, "special_value": 0, "not_established": 0}
    unit_ok = []
    for path in paths:
        arrays = np.load(path)
        ok_all = None
        for v in decl["variants"]:
            st, cond = arrays[f"{v}__st"], arrays[f"{v}__cond"].astype(bool)
            ok = (st == ST_OK) & (~cond | ("conditional_local" in classes))
            excluded["conditional_local"] += int(((st == ST_OK) & cond & ("conditional_local" not in classes)).sum())
            excluded["special_value"] += int(((st > ST_OK) & (st < ST_UNDEF)).sum())
            excluded["not_established"] += int((st >= ST_UNDEF).sum())
            ok_all = ok if ok_all is None else (ok_all & ok)
        unit_ok.append(ok_all)
    valid = np.logical_and.reduce(unit_ok[:n_cal])
    header = {"schema": "kernel-analyzer-reference-bias-analysis-v1",
              "declaration": {k: v for k, v in decl.items() if not k.startswith("_")},
              "coordinate_set": "fixed on the calibration units", "confirmation_invalid_policy": policy,
              "excluded_reference_elements": excluded}
    if not valid.any():
        return {**header, "verdict": "UNRESOLVED_REFERENCE", "reason": "no coordinate valid in the calibration units",
                "results": []}
    bad_conf = [i for i in range(n_cal, n_cal + n_conf) if (valid & ~unit_ok[i]).any()]
    if bad_conf and policy == "unresolved":
        return {**header, "verdict": "UNRESOLVED_REFERENCE",
                "reason": "confirmation units with references not valid on the declared coordinate set",
                "confirmation_units_invalid": [paths[i].stem for i in bad_conf], "results": []}
    used_units = [i for i in range(n_cal + n_conf) if i not in bad_conf]
    n_conf_used = n_conf - len(bad_conf)
    pairs = {f"{c['candidate']}_vs_{c['reference']}": [] for c in decl["comparisons"]}
    ref_measure, ambiguous_total = [], 0
    for i in used_units:
        arrays = np.load(paths[i])
        measured = {v: _measure(decl, arrays, v, arrays[f"{v}__actual"], device) for v in decl["variants"]}
        reference = {}
        for v in decl["variants"]:
            lo, hi, amb = _reference_bounds(decl, arrays, v, device, valid)
            reference[v] = (lo, hi)
            ambiguous_total += amb
        base = decl["variants"][0]
        ref_measure.append(0.5 * (reference[base][0] + reference[base][1]))
        for c in decl["comparisons"]:
            cand = measured[c["candidate"]]
            if c["reference"] == "K_R":
                r_lo, r_hi = reference[c["candidate"]]
                pairs[f"{c['candidate']}_vs_K_R"].append((cand - r_hi, cand - r_lo))
            else:
                d = cand - measured[c["reference"]]
                pairs[f"{c['candidate']}_vs_{c['reference']}"].append((d, d))
    results, assessments, arrays = [], [], {}
    for name, items in pairs.items():
        lows = np.stack([a for a, _ in items])
        highs = np.stack([b for _, b in items])
        # bad confirmation units are already handled above, so every remaining unit is valid on the set
        record, vecs = assess_units(name, lows, highs, np.stack(ref_measure), np.broadcast_to(valid, lows.shape),
                                    n_cal, decl["direction_rules"], alpha=alpha, groups=decl.get("groups"),
                                    declared_vectors=decl.get("declared_vectors"),
                                    run_detector=decl.get("default_detector", True),
                                    cross_fit_folds=decl.get("cross_fit_folds", 2),
                                    measurement=decl["measurement"], unit_ids=[paths[i].stem for i in used_units])
        results.extend(record.pop("rules"))
        assessments.append(record)
        arrays.update(vecs)
    tested = [r for r in results if "p_value_two_sided_conservative" in r]
    apply_holm(tested, alpha)
    if arrays_out is not None:
        np.savez_compressed(arrays_out, **arrays)
    reference_summary = {}
    for v in decl["variants"]:
        rows = [s[v] for s in summaries]
        reference_summary[v] = {
            "units": len(rows), "launches_per_unit": sorted({r["launches"] for r in rows}),
            "kernels": sorted({r["kernel"] for r in rows}), "kernel_hashes": sorted({r["kernel_hash"] for r in rows}),
            "triton": sorted({str(r["triton"]) for r in rows}),
            "libtriton_sha256": sorted({str(r["libtriton_sha256"]) for r in rows}),
            "coverage_complete": all(r["coverage_complete"] for r in rows),
            "aborted_programs": sum(r["aborted_programs"] for r in rows),
            "target_rewritten_outside": any(r["target_rewritten_outside"] for r in rows),
            "elements_total": sum(r["elements"] for r in rows),
            "classes_total": {k: sum(r["classes"][k] for r in rows) for k in rows[0]["classes"]},
            "reference_width_max": max((r["reference_width_max"] or 0.0) for r in rows),
            "residual_positive_total": sum(r["residual_positive"] for r in rows),
            "residual_negative_total": sum(r["residual_negative"] for r in rows),
            "residual_contains_zero_total": sum(r["residual_contains_zero"] for r in rows),
            "seconds_mean": float(np.mean([r["seconds"] for r in rows])),
        }
    n_total = sum(sum(s[v]["elements"] for v in decl["variants"]) for s in summaries)
    n_ne = sum(sum(s[v]["classes"]["not_established"] for v in decl["variants"]) for s in summaries)
    return {
        "schema": "kernel-analyzer-reference-bias-analysis-v1",
        "declaration": {k: v for k, v in decl.items() if not k.startswith("_")},
        "reference": reference_summary,
        "not_established_fraction_removed": n_ne / n_total if n_total else 0.0,
        "reference_classes_used": sorted(classes),
        "coordinate_set": "fixed on the calibration units", "confirmation_invalid_policy": policy,
        "confirmation_units_dropped": [paths[i].stem for i in bad_conf],
        "coordinates_used": int(valid.sum()), "coordinates_total": int(valid.size),
        "excluded_reference_elements": excluded,
        "ambiguous_rounding_elements": ambiguous_total,
        "multiplicity": "Holm over the declared comparison x rule pairs (holm_adjusted_p); the default detector "
                        "applies Holm within each of its two families per comparison",
        "results": results,
        "assessments": assessments,
    }
