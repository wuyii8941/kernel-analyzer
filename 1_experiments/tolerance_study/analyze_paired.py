#!/usr/bin/env python3
"""Paired short sequences and the training confirmation (protocol 5A for A-2 / A-3, appendix B for B-2).

A unit is a seed (one data order); the variants of a seed start from the same pretrained checkpoint and see the same
batches.  Short-sequence quantities (within a unit, over the captured coordinates; protocol section 2):
  displacement D_v(t) = ||theta_t - theta_0|| (theta_0: the capture before step 1), its ratio to the reference variant,
  the trajectory distance ||theta_t^v - theta_t^ref|| / D_ref(t), the median exp_avg_sq ratio, and for the EMAs of the
  pure-bf16 runs the EMA displacement ratio against the fp32 EMA.  Coordinate groups: matrices (q_proj, down_proj
  weights) and vectors (biases, norm weights), reported apart.
Training confirmation: validation loss at the last step, paired differences across seeds; nonzero axis (paired t,
two-sided, alpha 0.05) and equivalence axis (TOST against delta = 0.01 nats, alpha 0.05), Holm over the declared family.
The inference is the tool's (``analysis.unit_axes`` and ``analysis.holm_adjusted``), not a copy.

    python analyze_paired.py A --out results/A23/summary.json
    python analyze_paired.py B --out results/B2/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kernel_analyzer.reference_eval import analysis as A

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_TENSORS = [f"model.layers.{i}.{n}" for i in (0, 11, 23) for n in
                   ("self_attn.q_proj.weight", "self_attn.q_proj.bias", "mlp.down_proj.weight", "input_layernorm.weight")]
MATRICES = [n for n in CAPTURE_TENSORS if n.endswith(("q_proj.weight", "down_proj.weight"))]
VECTORS = [n for n in CAPTURE_TENSORS if n not in MATRICES]
DELTA = 0.01
ALPHA = 0.05


def load(path):
    return torch.load(path, weights_only=False)


def vec(cap, names, key="param", source="tensors"):
    if source == "tensors":
        return torch.cat([cap["tensors"][n][key].double().reshape(-1) for n in names])
    return torch.cat([cap["ema"][source][n].double().reshape(-1) for n in names])


def paired_axes(diffs, delta=DELTA, alpha=ALPHA):
    """The frozen two axes on the paired differences, through the tool's per-unit inference (with exact values the
    endpoint-conservative t test and TOST are the paired t test and TOST); field names as frozen in the protocol."""
    ax = A.unit_axes(diffs, alpha, delta)
    nz, eq = ax["nonzero"], ax["equivalence"]
    if "mean_projection" not in nz or "p_value_tost" not in eq:
        return {"n": len(diffs), "delta": delta, "unresolved": {"nonzero": nz, "equivalence": eq}}
    return {"n": nz["n"], "mean_difference": nz["mean_projection"], "sd": nz["sd"],
            "ci95": [nz["lower_bound_of_E_l"], nz["upper_bound_of_E_h"]],
            "p_nonzero": nz["p_value_two_sided_conservative"], "nonzero": nz["verdict"] != "NOT_CONFIRMED",
            "tost_ci90": [eq["lower_one_sided_bound_of_E_l"], eq["upper_one_sided_bound_of_E_h"]],
            "p_tost": eq["p_value_tost"], "equivalent_within_delta": eq["verdict"] == "WITHIN_DELTA", "delta": delta,
            "unit_skewness": nz["unit_skewness"], "t_approximation": nz["t_approximation"],
            **({"robust": nz["robust"]} if "robust" in nz else {})}


def group_a():
    base = ROOT / ".cache/round2/paired"
    seeds = list(range(3000, 3012))
    steps = (10, 50, 100, 300, 600)
    units, eval_rows = [], []
    for s in seeds:
        caps = {v: {t: load(base / v / f"seed{s}" / f"capture_step{t:05d}.pt") for t in (1,) + steps}
                for v in ("bf16", "master", "sr")}
        logs = {v: json.loads((base / v / f"seed{s}" / "log.json").read_text())["log"] for v in ("bf16", "master", "sr")}
        u = {"seed": s, "short_sequence": {}}
        for grp, names in (("matrices", MATRICES), ("vectors", VECTORS)):
            th0 = vec(caps["master"][1], names)
            rows = {}
            for t in steps:
                ref = vec(caps["master"][t], names)
                d_ref = float((ref - th0).norm())
                row = {"reference_displacement": d_ref}
                for v in ("bf16", "sr"):
                    th = vec(caps[v][t], names)
                    row[f"{v}_displacement_ratio"] = float((th - th0).norm()) / d_ref if d_ref else None
                    row[f"{v}_distance_over_reference_displacement"] = float((th - ref).norm()) / d_ref if d_ref else None
                vr = vec(caps["master"][t], names, "exp_avg_sq")
                for v in ("bf16", "sr"):
                    vb = vec(caps[v][t], names, "exp_avg_sq")
                    ok = vr > 0
                    row[f"{v}_median_exp_avg_sq_ratio"] = float((vb[ok] / vr[ok]).median()) if ok.any() else None
                if caps["bf16"][t].get("ema"):
                    e32_0 = vec(caps["bf16"][1], names, source="fp32")
                    e32 = vec(caps["bf16"][t], names, source="fp32")
                    d32 = float((e32 - e32_0).norm())
                    for ename in ("averaged_model", "timm_1_0_30"):
                        e0 = vec(caps["bf16"][1], names, source=ename)
                        e = vec(caps["bf16"][t], names, source=ename)
                        row[f"ema_{ename}_displacement_ratio_vs_fp32_ema"] = float((e - e0).norm()) / d32 if d32 else None
                rows[str(t)] = row
            u["short_sequence"][grp] = rows
        last = {v: logs[v][-1] for v in logs}
        u["eval"] = {v: {str(r["step"]): r.get("eval_loss") for r in logs[v] if "eval_loss" in r} for v in logs}
        u["eval_ema"] = {k: last["bf16"].get(k) for k in ("eval_loss_ema_averaged_model", "eval_loss_ema_timm_1_0_30",
                                                          "eval_loss_ema_fp32")}
        units.append(u)
        eval_rows.append({"seed": s, "bf16": last["bf16"].get("eval_loss"), "master": last["master"].get("eval_loss"),
                          "sr": last["sr"].get("eval_loss"), **u["eval_ema"]})
    comps = {"pure_bf16 - fp32_master": [r["bf16"] - r["master"] for r in eval_rows],
             "torchao_sr - fp32_master": [r["sr"] - r["master"] for r in eval_rows],
             "ema_averaged_model_bf16 - ema_fp32": [r["eval_loss_ema_averaged_model"] - r["eval_loss_ema_fp32"]
                                                    for r in eval_rows],
             "ema_timm_1_0_30_bf16 - ema_fp32": [r["eval_loss_ema_timm_1_0_30"] - r["eval_loss_ema_fp32"]
                                                 for r in eval_rows]}
    tc = {k: paired_axes(v) for k, v in comps.items()}
    adj = A.holm_adjusted([x["p_nonzero"] for x in tc.values()])
    for (k, x), a_ in zip(tc.items(), adj):
        x["holm_adjusted_p_nonzero"] = a_
        x["nonzero_after_holm"] = a_ < ALPHA
    agg = {}
    for grp in ("matrices", "vectors"):
        agg[grp] = {}
        for t in steps:
            keys = units[0]["short_sequence"][grp][str(t)].keys()
            agg[grp][str(t)] = {k: _dist([u["short_sequence"][grp][str(t)][k] for u in units]) for k in keys}
    return {"group": "A", "seeds": seeds, "steps": steps, "short_sequence": agg, "training_confirmation": tc,
            "eval_per_seed": eval_rows, "units": units}


def group_b():
    base = ROOT / ".cache/round2/paired_b"
    seeds = list(range(5000, 5008))
    steps = (10, 50, 100, 200)
    units, rows = [], []
    for s in seeds:
        caps = {v: {t: load(base / v / f"seed{s}" / f"capture_step{t:05d}.pt") for t in (1,) + steps}
                for v in ("compiled", "eager")}
        logs = {v: json.loads((base / v / f"seed{s}" / "log.json").read_text())["log"] for v in ("compiled", "eager")}
        u = {"seed": s, "short_sequence": {}}
        for grp, names in (("matrices", MATRICES), ("vectors", VECTORS)):
            th0 = vec(caps["eager"][1], names)
            u["short_sequence"][grp] = {}
            for t in steps:
                ref = vec(caps["eager"][t], names)
                th = vec(caps["compiled"][t], names)
                d_ref = float((ref - th0).norm())
                u["short_sequence"][grp][str(t)] = {
                    "reference_displacement": d_ref,
                    "compiled_displacement_ratio": float((th - th0).norm()) / d_ref if d_ref else None,
                    "compiled_distance_over_reference_displacement": float((th - ref).norm()) / d_ref if d_ref else None}
        rows.append({"seed": s, "compiled": logs["compiled"][-1].get("eval_loss"),
                     "eager": logs["eager"][-1].get("eval_loss")})
        units.append(u)
    tc = {"compiled - eager": paired_axes([r["compiled"] - r["eager"] for r in rows])}
    agg = {grp: {str(t): {k: _dist([u["short_sequence"][grp][str(t)][k] for u in units])
                          for k in units[0]["short_sequence"][grp][str(t)]} for t in steps}
           for grp in ("matrices", "vectors")}
    return {"group": "B", "seeds": seeds, "steps": steps, "short_sequence": agg, "training_confirmation": tc,
            "eval_per_seed": rows, "units": units}


def _dist(values):
    v = np.array([x for x in values if x is not None], np.float64)
    if not v.size:
        return None
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else None, "min": float(v.min()),
            "max": float(v.max()), "n": int(v.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=("A", "B"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = group_a() if a.group == "A" else group_b()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
