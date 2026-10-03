#!/usr/bin/env python3
"""Real GPU interventions against the emulator's predictions (held-out inputs).

For a node j of a captured kernel, the model counterfactual "node j correctly
rounded" (emulate(..., exact_nodes=[j], substitution="rn")) is a prediction of
what a real kernel change would output: replacing the node by a correctly
rounded implementation.  Each such change is run on the device on the same
inputs and compared with the prediction bit for bit; predicted and actual
changes are compared in sign, size and ranking.  The model's float64
substitution ("mid") contribution of the same node is reported next to it.

Own kernels (mutation_kernels.py, seeds 3000..3007, not used in development):
directed-rounding node -> round to nearest (the original kernel), softmax
division -> div_rn, softmax exp -> exp in float64, layernorm rsqrt -> rsqrt in
float64.  Liger cross-entropy (held-out capture, capture_exp_intervention.py):
division by n_non_ignore -> div_rn, division by the row sum -> div_rn, the
second-pass exp -> exp in float64.

    python scripts/validate_localization_interventions.py --liger .cache/heldout/exp_intervention \
        --out results/reference_eval/localization_interventions.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, load_launch  # noqa: E402
from kernel_analyzer.reference_eval.emulate import HardwareOracle, _outputs, emulate, localize, verify  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

SEEDS = range(3000, 3008)
OWN = [  # kernel, base MUT, node selector (op name, text fragment), intervention MUT, description
    ("scale", 1, ("tt.extern_elementwise", "__nv_fmul_rd"), 0, "mul_rd -> round to nearest"),
    ("scale", 2, ("tt.extern_elementwise", "__nv_fmul_ru"), 0, "mul_ru -> round to nearest"),
    ("sum4", 2, ("tt.extern_elementwise", "__nv_fadd_rd"), 0, "add_rd -> round to nearest"),
    ("accumulate", 1, ("tt.extern_elementwise", "__nv_fadd_rd"), 0, "add_rd -> round to nearest"),
    ("accumulate", 2, ("tt.extern_elementwise", "__nv_fadd_ru"), 0, "add_ru -> round to nearest"),
    ("softmax", 0, ("arith.divf", ""), 1, "div.full -> div_rn"),
    ("softmax", 0, ("math.exp", ""), 4, "ex2.approx exp -> exp in float64, rounded once"),
    ("layernorm", 0, ("math.rsqrt", ""), 4, "rsqrt.approx -> rsqrt in float64, rounded once"),
]
LIGER = [  # result name of the node in the TTIR, variant, description
    ("%X_block_44", "div_n_rn", "division by n_non_ignore: div.full -> div_rn"),
    ("%X_block_29", "div_d_rn", "division by the row sum d: div.full -> div_rn"),
    ("%X_block_27", "exp2nd_f64", "second-pass exp: ex2.approx -> exp in float64"),
]


def node_of(launch, op_name, fragment="", result=None):
    nodes = [o for o in parse_ttir(launch.asm["ttir"]).entry().walk()
             if o.name == op_name and fragment in o.text and (result is None or o.results[0] == result)]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one {op_name} {fragment or result}, found {len(nodes)}")
    return nodes[0].node_id


def compare(base_launch, node, real_values, oracle, buffer_name):
    """Prediction (rn substitution) vs real change on the target buffer's written, bit-identical elements."""

    v = verify(base_launch, oracle=oracle, ungrouped=[], swapped=[])
    res, _ = emulate(base_launch, exact_nodes=[node], oracle=oracle, substitution="rn")
    w, k, _, st, actual, _, _ = _outputs(res, base_launch)[buffer_name]
    base_res, _ = emulate(base_launch, oracle=oracle)
    _, k0, _, st0, act0, _, _ = _outputs(base_res, base_launch)[buffer_name]
    ok = w & (st == 0) & (st0 == 0) & (k0 == act0) & np.isfinite(real_values)
    pred, real, base = k[ok], real_values[ok], act0[ok]
    dp, dr = pred - base, real - base
    nz = base != 0
    return {"base_status": v["status"], "elements": int(ok.sum()),
            "prediction_bit_identical_to_real": int((pred.astype(np.float32).view(np.uint32)
                                                     == real.astype(np.float32).view(np.uint32)).sum()),
            "elements_changed_real": int((dr != 0).sum()), "elements_changed_predicted": int((dp != 0).sum()),
            "mean_change_real": float(dr.mean()), "mean_change_predicted": float(dp.mean()),
            "mean_relative_change_real": float((dr[nz] / base[nz]).mean()) if nz.any() else None,
            "mean_relative_change_predicted": float((dp[nz] / base[nz]).mean()) if nz.any() else None,
            "sign_agreement_on_changed": float(np.mean(np.sign(dp[dr != 0]) == np.sign(dr[dr != 0])))
            if (dr != 0).any() else None}


def mid_contribution(base_launch, node, oracle, buffer_name):
    loc = localize(base_launch, nodes=[node], oracle=oracle, endpoint_nodes=[node])
    b = loc["nodes"][0]["buffers"][buffer_name]
    return {"mean": b["mean"], "mean_relative": b["mean_relative"], "aligned": b["aligned"],
            "endpoint_check": b.get("endpoint_check")}


def _mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else None


def own_part(oracle) -> list:
    import torch

    from scripts.run_method_comparison import specs

    table = specs()
    rows = []
    for name, base_mut, (op_name, frag), int_mut, desc in OWN:
        make = table[name][0]
        per_draw = []
        for seed in SEEDS:
            launches, values = {}, {}
            for mut in (base_mut, int_mut):
                tensors, launch_fn, shape = make(seed)
                tensors = dict(tensors)
                tensors["Y"] = torch.empty(shape, device="cuda")
                rec = TritonLaunchRecorder(select=lambda n, i: n.startswith("m_"))
                with rec:
                    launch_fn(tensors, mut)
                    torch.cuda.synchronize()
                launches[mut] = rec.launches[-1]
                values[mut] = tensors["Y"].reshape(-1).double().cpu().numpy()
            base = launches[base_mut]
            node = node_of(base, op_name, frag)
            buf = "ACC" if name == "accumulate" else ("C" if name == "matmul" else "Y")
            real = values[int_mut]
            out = compare(base, node, real, oracle, buf)
            out["mid_contribution"] = mid_contribution(base, node, oracle, buf)
            per_draw.append(out)
        agg = {k: sum(d[k] for d in per_draw) for k in ("elements", "prediction_bit_identical_to_real",
                                                         "elements_changed_real", "elements_changed_predicted")}
        agg.update({k: _mean([d[k] for d in per_draw]) for k in
                    ("mean_relative_change_real", "mean_relative_change_predicted", "mean_change_real",
                     "mean_change_predicted")})
        agg["mid_contribution_mean"] = _mean([d["mid_contribution"]["mean"] for d in per_draw])
        agg["mid_contribution_mean_relative"] = _mean([d["mid_contribution"]["mean_relative"] for d in per_draw])
        agg["endpoint_outputs_differ"] = sum((d["mid_contribution"]["endpoint_check"] or {}).get("outputs_differ_by_half_ulp32_or_more", 0)
                                             for d in per_draw)
        row = {"kernel": name, "base_mutation": base_mut, "intervention_mutation": int_mut, "node": f"{op_name} {frag}",
               "intervention": desc, "draws": len(per_draw), **agg, "per_draw": per_draw}
        rows.append(row)
        print(name, desc, f"bit-identical {agg['prediction_bit_identical_to_real']}/{agg['elements']}",
              f"rel real {agg['mean_relative_change_real']:.3e} pred {agg['mean_relative_change_predicted']:.3e}",
              f"mid {agg['mid_contribution_mean_relative']:.3e}", flush=True)
    return rows


def liger_part(root: Path, oracle, launches: int) -> list:
    rows = []
    for result, variant, desc in LIGER:
        per = []
        for j in range(launches):
            base = load_launch(root / "original" / f"launch{j:03d}")
            real_launch = load_launch(root / variant / f"launch{j:03d}")
            node = node_of(base, *(("arith.divf",) if "div" in variant else ("math.exp",)), result=result)
            arg = next(a for a in real_launch.args if a.name == "X_ptr")
            real = arg.after.view(__import__("torch").float32).double().numpy()
            out = compare(base, node, real, oracle, "X_ptr")
            out["mid_contribution"] = mid_contribution(base, node, oracle, "X_ptr")
            per.append(out)
        agg = {k: sum(d[k] for d in per) for k in ("elements", "prediction_bit_identical_to_real",
                                                    "elements_changed_real", "elements_changed_predicted")}
        active = [d for d in per if d["elements_changed_real"] or d["elements_changed_predicted"]]
        agg["launches_with_changes"] = len(active)  # a launch of an ignored row writes zeros only
        agg.update({k: _mean([d[k] for d in active]) for k in
                    ("mean_relative_change_real", "mean_relative_change_predicted", "mean_change_real",
                     "mean_change_predicted")})
        agg["mid_contribution_mean_relative"] = _mean([d["mid_contribution"]["mean_relative"] for d in active])
        agg["mid_contribution_mean"] = _mean([d["mid_contribution"]["mean"] for d in active])
        agg["mid_contribution_aligned"] = _mean([d["mid_contribution"]["aligned"] for d in active])
        agg["endpoint_outputs_differ"] = sum((d["mid_contribution"]["endpoint_check"] or {}).get("outputs_differ_by_half_ulp32_or_more", 0)
                                             for d in per)
        rows.append({"node": result, "variant": variant, "intervention": desc, "launches": len(per), **agg,
                     "per_launch": per})
        print(variant, f"bit-identical {agg['prediction_bit_identical_to_real']}/{agg['elements']}",
              f"rel real {agg['mean_relative_change_real']:.3e} pred {agg['mean_relative_change_predicted']:.3e}",
              f"mid {agg['mid_contribution_mean_relative']:.3e}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--liger", type=Path, required=True)
    parser.add_argument("--liger-launches", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    oracle = HardwareOracle(options={"num_warps": 4, "enable_fp_fusion": True})
    report = {"schema": "kernel-analyzer-localization-interventions-v1",
              "prediction": "emulate with the node correctly rounded (substitution rn), frozen lowering choices",
              "own": own_part(oracle), "liger_cross_entropy": liger_part(args.liger, oracle, args.liger_launches)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
