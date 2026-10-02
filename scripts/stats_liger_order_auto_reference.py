#!/usr/bin/env python3
"""Step 5: controlled replay with the automatic K_R and the endpoint-conservative test.

For every unit, the parameter write of the real torch AdamW step (the same
zero-moment configuration as run_liger_fp32_order_population_mean.py) is
computed on the declared rows D for:

* candidate gradient K (the captured FP32 accumulation, original or reverse order);
* reference gradient RN32(K_R), the automatic reference rounded back to the
  storage format.  Where RN32(lo) != RN32(hi) the write is evaluated at both
  endpoints and the projection gets bounds [l, h].

u = write(K) - write(RN32(K_R)).  Two pre-declared direction rules:

* fixed direction: mean u over the 32 calibration draws, normalized, frozen,
  then scored on the 64 confirmation draws (same split as the original run);
* aligned direction: w = r / ||r||, r = reference write on D (no calibration).

The mean projection is tested with the t interval; the endpoint-conservative
version uses the lower confidence bound of E[l] and the upper bound of E[h].
The known comparison original - reverse is recomputed on D with the same
rules, and decomposed as (original - K_R) - (reverse - K_R).

Runs in the mainline environment on the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402

LEARNING_RATE, BETAS, EPSILON = 1e-4, (0.9, 0.95), 1e-8


def adamw_write(weight: np.ndarray, grad: np.ndarray, device: str) -> np.ndarray:
    """Write of one real torch AdamW step (FP32, single-tensor path), as in the original run."""

    w = torch.nn.Parameter(torch.as_tensor(weight, dtype=torch.float32, device=device).clone())
    before = w.detach().clone()
    opt = torch.optim.AdamW([w], lr=LEARNING_RATE, betas=BETAS, eps=EPSILON, weight_decay=0.0,
                            foreach=False, fused=False)
    w.grad = torch.as_tensor(grad, dtype=torch.float32, device=device)
    opt.step()
    return (w.detach() - before).double().cpu().numpy()


def t_interval(values: np.ndarray, alpha: float = 0.05):
    from scipy.stats import t

    n = values.size
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    half = float(t.ppf(1 - alpha / 2, n - 1)) * sd / math.sqrt(n) if n > 1 else math.inf
    return mean, sd, (mean - half, mean + half)


def one_sided(values: np.ndarray, side: str, alpha: float = 0.05):
    from scipy.stats import t

    n = values.size
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(t.ppf(1 - alpha, n - 1)) * sd / math.sqrt(n)
    return mean - half if side == "lower" else mean + half


def decide(l: np.ndarray, h: np.ndarray, alpha: float = 0.05) -> dict:
    """Endpoint-conservative decision (stage summary section 4)."""

    lower = one_sided(l, "lower", alpha / 2)
    upper = one_sided(h, "upper", alpha / 2)
    if lower > 0:
        verdict = "DETECTED_POSITIVE"
    elif upper < 0:
        verdict = "DETECTED_NEGATIVE"
    else:
        verdict = "NOT_CONFIRMED"
    return {"lower_bound_of_E_l": lower, "upper_bound_of_E_h": upper, "verdict": verdict}


def t_pvalue(values: np.ndarray) -> float:
    from scipy.stats import t

    n = values.size
    sd = float(values.std(ddof=1))
    if sd == 0:
        return 0.0 if values.mean() != 0 else 1.0
    stat = float(values.mean()) / (sd / math.sqrt(n))
    return float(2 * t.sf(abs(stat), n - 1))


def holm(pvalues: list, alpha: float = 0.05) -> list:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    reject = [False] * len(pvalues)
    m = len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def summarize(name: str, l: np.ndarray, h: np.ndarray, rule: str) -> dict:
    mid = 0.5 * (l + h)
    mean, sd, interval = t_interval(mid)
    out = {
        "comparison": name, "rule": rule, "n": int(mid.size), "mean_projection": mean, "sd": sd,
        "t_interval_95": list(interval), "positive": int((l > 0).sum()), "negative": int((h < 0).sum()),
        "zero_or_ambiguous": int(((l <= 0) & (h >= 0)).sum()),
        "max_projection_width": float(np.max(h - l)),
    }
    out.update(decide(l, h))
    # Conservative p-value: the endpoint closer to zero carries the evidence.
    out["p_value_two_sided_conservative"] = max(t_pvalue(l), t_pvalue(h))
    return out


def project(u_lo, u_hi, w):
    """Bounds of <u, w> for u in [u_lo, u_hi] coordinatewise."""

    a = u_lo * w
    b = u_hi * w
    return float(np.minimum(a, b).sum()), float(np.maximum(a, b).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-partial", action="store_true", help="debug only: run on the units present")
    args = parser.parse_args()
    design = json.loads((args.compact / "design.json").read_text())
    n_cal = design["calibration"]
    units = sorted(p for p in args.compact.glob("unit*.npz"))
    if len(units) != design["calibration"] + design["confirmation"] and not args.allow_partial:
        raise RuntimeError(f"expected {design['calibration'] + design['confirmation']} units, found {len(units)}")

    per_unit = []
    u = {"original_vs_KR": [], "reverse_vs_KR": [], "original_vs_reverse": []}
    ref_writes = []
    ambiguous_total = 0
    reference_consistency = []
    for path in units:
        a = np.load(path)
        weight = a["original_weight"]
        writes = {}
        for variant in ("original", "reverse"):
            writes[variant] = adamw_write(weight, a[f"{variant}_grad"], args.device)
        # The exact sum is order independent: both chains must enclose the same value.
        lo = np.maximum(a["original_ref_lo"], a["reverse_ref_lo"])
        hi = np.minimum(a["original_ref_hi"], a["reverse_ref_hi"])
        intersect = bool((lo <= hi).all())
        reference_consistency.append(intersect)
        g_lo, of_lo = iv.round_nearest_even(lo, "f32")
        g_hi, of_hi = iv.round_nearest_even(hi, "f32")
        if of_lo.any() or of_hi.any():
            raise RuntimeError("reference overflows FP32")
        ambiguous = g_lo != g_hi
        ambiguous_total += int(ambiguous.sum())
        w_ref_lo = adamw_write(weight, g_lo, args.device)
        w_ref_hi = adamw_write(weight, g_hi, args.device) if ambiguous.any() else w_ref_lo
        r_lo, r_hi = np.minimum(w_ref_lo, w_ref_hi), np.maximum(w_ref_lo, w_ref_hi)
        ref_writes.append(0.5 * (r_lo + r_hi))
        u["original_vs_KR"].append((writes["original"] - r_hi, writes["original"] - r_lo))
        u["reverse_vs_KR"].append((writes["reverse"] - r_hi, writes["reverse"] - r_lo))
        d = writes["original"] - writes["reverse"]
        u["original_vs_reverse"].append((d, d))
        per_unit.append({"unit": path.stem, "references_intersect": intersect, "ambiguous_coordinates": int(ambiguous.sum()),
                         "original_differs_from_KR": int((writes["original"] != r_lo).sum()),
                         "reverse_differs_from_KR": int((writes["reverse"] != r_lo).sum())})

    results = []
    for name, pairs in u.items():
        lows = np.stack([p[0] for p in pairs])
        highs = np.stack([p[1] for p in pairs])
        mids = 0.5 * (lows + highs)
        direction = mids[:n_cal].mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm == 0:
            results.append({"comparison": name, "rule": "fixed_direction", "verdict": "UNRESOLVED_MEASUREMENT",
                            "reason": "calibration direction is zero"})
        else:
            nu = direction / norm
            proj = np.array([project(lows[i], highs[i], nu) for i in range(n_cal, len(pairs))])
            results.append(summarize(name, proj[:, 0], proj[:, 1], "fixed_direction"))
        aligned = []
        for i in range(n_cal, len(pairs)):
            r = ref_writes[i]
            rn = np.linalg.norm(r)
            aligned.append(project(lows[i], highs[i], r / rn) if rn > 0 else (0.0, 0.0))
        aligned = np.array(aligned)
        results.append(summarize(name, aligned[:, 0], aligned[:, 1], "aligned_reference_update"))

    step4 = {"variants": {}}
    for variant in ("original", "reverse"):
        rows = [json.loads(p.with_suffix(".json").read_text())[variant] for p in units]
        step4["variants"][variant] = {
            "units": len(rows),
            "launches_per_unit": sorted({r["launches"] for r in rows}),
            "coverage_complete": all(r["coverage_complete"] for r in rows),
            "accumulator_rewritten_outside": any(r["accumulator_rewritten_outside"] for r in rows),
            "bitwise_equal_to_torch_path": all(all(r["bitwise_equal_to_torch_path"].values()) for r in rows),
            "coordinates_total": sum(r["coordinates"] for r in rows),
            "classes_total": {c: sum(r["classes"][c] for r in rows) for c in rows[0]["classes"]},
            "manual_agrees_total": sum(r["manual_agrees"] for r in rows),
            "manual_disagrees_total": sum(r["manual_disagrees"] for r in rows),
            "reference_width_max": max(r["reference_width_max"] for r in rows),
            "manual_bound_max": max(r["manual_bound_max"] for r in rows),
            "residual_positive_total": sum(r["residual_positive"] for r in rows),
            "residual_negative_total": sum(r["residual_negative"] for r in rows),
            "residual_contains_zero_total": sum(r["residual_contains_zero"] for r in rows),
            "evaluation_seconds_mean": float(np.mean([r["evaluation_seconds"] for r in rows])),
        }
    tested = [r for r in results if "p_value_two_sided_conservative" in r]
    rejections = holm([r["p_value_two_sided_conservative"] for r in tested])
    for r, rej in zip(tested, rejections):
        r["holm_reject_at_0.05"] = bool(rej)
        r["final_verdict"] = r["verdict"] if rej else "NOT_CONFIRMED"
    payload = {
        "schema": "kernel-analyzer-liger-order-auto-reference-v1",
        "multiplicity": "Holm over the six pre-declared comparison/rule pairs",
        "step4_automatic_vs_manual_reference": step4,
        "location": "Liger 0.7 fused linear CE: FP32 dW chunk accumulation (64 chunks, sequence length 64)",
        "reference": "automatic K_R: TTIR composed reference over the 64 accumulate launches, rounded to FP32",
        "reference_mode": "numerical_difference",
        "task_semantics": "f taken as the declared semantics (real sum of the chunk contributions); not separately checked",
        "measurement": "lm_head parameter write of one real torch AdamW step, zero moments, on the declared rows D",
        "coordinates": {"rows": len(design["rows"]), "row_seed": design["row_seed"],
                        "coordinates": len(design["rows"]) * design["block"]},
        "population": {"bank": "32 declared Qwen3-1.7B length-64 states", "draw_seed": design["draw_seed"],
                       "calibration": n_cal, "confirmation": design["confirmation"],
                       "sampling": "with replacement from the empirical bank"},
        "factors": {
            "instruction_stream": "fixed: accumulate kernel, Triton 3.6.0, sm_86, BLOCK=2048, default options",
            "memory_contents": "sampled: declared bank, with-replacement draws",
            "launch_configuration": "fixed: grid = ceil(V*H / 2048)",
            "concurrency_order": "fixed: no atomics; one writer per element",
            "hardware": "fixed: NVIDIA RTX A6000",
        },
        "alpha": 0.05,
        "comparison_set": [r["comparison"] + "/" + r["rule"] for r in results],
        "reference_consistency": {"both_order_references_intersect_in_all_units": all(reference_consistency)},
        "ambiguous_rounding_coordinates": ambiguous_total,
        "results": results,
        "per_unit": per_unit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    for r in results:
        print(json.dumps({k: r.get(k) for k in ("comparison", "rule", "mean_projection", "t_interval_95",
                                                  "positive", "negative", "verdict", "final_verdict")}))


if __name__ == "__main__":
    main()
