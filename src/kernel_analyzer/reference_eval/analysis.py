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


def _reference_bounds(decl, arrays, variant, device):
    """measure(RN(reference)) bounds; RN to the target's storage format at both interval ends."""

    fmt = STORAGE_FORMAT[decl.get("_storage_dtype", "float32")]
    ref_lo, ref_hi = _reference_interval(decl, arrays, variant)
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


def _summarize(name, rule, l, h, alpha):
    mean, sd, interval, p_mid = _t_stats(0.5 * (l + h), alpha)
    _, _, ci_l, p_l = _t_stats(l, alpha)
    _, _, ci_h, p_h = _t_stats(h, alpha)
    lower, upper = ci_l[0], ci_h[1]  # endpoint-conservative: lower bound of E[l], upper bound of E[h]
    verdict = "DETECTED_POSITIVE" if lower > 0 else ("DETECTED_NEGATIVE" if upper < 0 else "NOT_CONFIRMED")
    return {"comparison": name, "rule": rule, "n": int(l.size), "mean_projection": mean, "sd": sd,
            "t_interval": list(interval), "lower_bound_of_E_l": lower, "upper_bound_of_E_h": upper,
            "positive": int((l > 0).sum()), "negative": int((h < 0).sum()),
            "zero_or_ambiguous": int(((l <= 0) & (h >= 0)).sum()), "verdict": verdict,
            "p_value_two_sided_conservative": max(p_l, p_h)}


def statistics_stage(decl: dict, ref_dir: Path, device: str = "cuda:0") -> dict:
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
    pairs = {f"{c['candidate']}_vs_{c['reference']}": [] for c in decl["comparisons"]}
    ref_measure, ambiguous_total = [], 0
    for path in paths:
        arrays = np.load(path)
        measured = {v: _measure(decl, arrays, v, arrays[f"{v}__actual"], device) for v in decl["variants"]}
        reference = {}
        for v in decl["variants"]:
            lo, hi, amb = _reference_bounds(decl, arrays, v, device)
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
    results = []
    for name, items in pairs.items():
        lows = np.stack([a for a, _ in items])
        highs = np.stack([b for _, b in items])
        if "fixed_direction" in decl["direction_rules"]:
            direction = (0.5 * (lows[:n_cal] + highs[:n_cal])).mean(axis=0)
            norm = np.linalg.norm(direction)
            if norm == 0:
                results.append({"comparison": name, "rule": "fixed_direction", "verdict": "UNRESOLVED_MEASUREMENT",
                                "reason": "calibration direction is zero"})
            else:
                w = direction / norm
                proj = np.array([(np.minimum(lows[i] * w, highs[i] * w).sum(), np.maximum(lows[i] * w, highs[i] * w).sum())
                                 for i in range(n_cal, n_cal + n_conf)])
                results.append(_summarize(name, "fixed_direction", proj[:, 0], proj[:, 1], alpha))
        if "aligned_reference_update" in decl["direction_rules"]:
            proj = []
            for i in range(n_cal, n_cal + n_conf):
                r = ref_measure[i]
                rn = np.linalg.norm(r)
                w = r / rn if rn > 0 else r
                proj.append((np.minimum(lows[i] * w, highs[i] * w).sum(), np.maximum(lows[i] * w, highs[i] * w).sum()))
            proj = np.array(proj)
            results.append(_summarize(name, "aligned_reference_update", proj[:, 0], proj[:, 1], alpha))
    tested = [r for r in results if "p_value_two_sided_conservative" in r]
    for r, rej in zip(tested, _holm([r["p_value_two_sided_conservative"] for r in tested], alpha)):
        r["holm_reject"] = bool(rej)
        r["final_verdict"] = r["verdict"] if rej else "NOT_CONFIRMED"
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
        "ambiguous_rounding_elements": ambiguous_total,
        "multiplicity": "Holm over the declared comparison x rule pairs",
        "results": results,
    }
