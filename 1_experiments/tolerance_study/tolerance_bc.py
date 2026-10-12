#!/usr/bin/env python3
"""Question 1 for groups B and C (protocol appendices B, C), on the same captured units as the unified-entry runs.

B (RAdam, fp32 parameters): implementations compiled (official recipe) and eager foreach; references exact (strict
  enclosure midpoint), fp64 (the same algorithm on float64 copies) and eager fp32 (what PyTorch's compiled-optimizer
  test compares against).  Checks: fp32 assert_close defaults (rtol 1.3e-6, atol 1e-5), rtol only, 1 fp32 ulp; at the
  parameter value and the parameter increment.  Descriptive: the update scale ratio <dK, dR> / <dR, dR>.
C (TF32 matrix multiplication): implementations triton_default, inductor_high, cublas_high and the controls
  triton_ieee, cublas_highest; references exact and cublas_highest (IEEE fp32).  Checks: fp32 assert_close defaults,
  PyTorch's TF32 convention (tf32_on_and_off(0.005): atol 5e-3 with the fp32 default rtol), 1 fp32 ulp.  Descriptive:
  alignment <e, r> / <r, r> and mean(e) / rms(r).  The descriptive frame quantities are the tool's
  (``observation.output_unit``), not a copy.

    python tolerance_bc.py B --step 7 --seeds 4000-4031 --out results/B1/s7/tolerance.json
    python tolerance_bc.py C --layer 12 --mat q_proj --units 0-31 --out results/C1/l12_q_proj/tolerance.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from kernel_analyzer.reference_eval.observation import output_unit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FP32 = dict(rtol=1.3e-6, atol=1e-5)
TF32_CONVENTION = dict(rtol=1.3e-6, atol=5e-3)


def ulp_fp32(x):
    a = np.maximum(np.abs(x), 2.0 ** -126)
    return np.exp2(np.floor(np.log2(a)) - 23)


def check(a, b, tol):
    d = np.abs(a - b)
    ok = d <= tol["atol"] + tol["rtol"] * np.abs(b)
    return {"pass": bool(ok.all()), "fail_fraction": float(1 - ok.mean()), "max_abs_diff": float(d.max()),
            "relative_rms": float(np.sqrt((d ** 2).mean() / max((b ** 2).mean(), 1e-300)))}


def strict(a, b):
    ok = np.abs(a - b) <= ulp_fp32(b)
    return {"pass": bool(ok.all()), "fail_fraction": float(1 - ok.mean())}


def summarize(rows):
    out = {}
    for key in rows[0]["results"]:
        cells = [r["results"][key] for r in rows]
        out[key] = {}
        for sub in cells[0]:
            if isinstance(cells[0][sub], dict) and "pass" in cells[0][sub]:
                out[key][sub] = {"pass_units": sum(c[sub]["pass"] for c in cells), "units": len(cells),
                                 "mean_fail_fraction": float(np.mean([c[sub]["fail_fraction"] for c in cells]))}
            elif isinstance(cells[0][sub], (int, float)):
                v = np.array([c[sub] for c in cells], np.float64)
                out[key][sub] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else None,
                                 "min": float(v.min()), "max": float(v.max())}
    return out


def group_b(a):
    import os
    import calls_radam as C
    os.environ.setdefault("KA_TOL_CAPTURES_B", a.captures)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    rows = []
    for seed in range(lo, hi + 1):
        cap = C._capture(seed, a.step)
        inp = {q: cap[q].float().cuda().clone() for q in C.QUANTITIES}
        inp["step"] = torch.tensor([cap["state_step"]], dtype=torch.float32)
        old = inp["param"].double().cpu().numpy()
        t = int(inp["step"].item()) + 1
        pn, _, _ = C.radam_exact(*(inp[q].double().cpu().numpy() for q in C.QUANTITIES), t)
        exact = 0.5 * (pn[0] + pn[1])

        def ref(dtype):
            p = torch.nn.Parameter(inp["param"].to(dtype).clone())
            p.grad = inp["grad"].to(dtype).clone()
            opt = torch.optim.RAdam([p], lr=float(C.LR), foreach=True)
            st = opt.state[p]
            st["step"] = torch.tensor(float(inp["step"].item()))
            st["exp_avg"], st["exp_avg_sq"] = inp["exp_avg"].to(dtype).clone(), inp["exp_avg_sq"].to(dtype).clone()
            opt.step()
            return p.detach().double().cpu().numpy()
        refs = {"exact": exact, "fp64": ref(torch.float64), "eager_fp32": ref(torch.float32)}
        impls = {"compiled": C.radam_compiled({k: v.clone() for k, v in inp.items()})["param"].double().cpu().numpy(),
                 "eager": C.radam_eager({k: v.clone() for k, v in inp.items()})["param"].double().cpu().numpy()}
        row = {"seed": seed, "step": a.step, "results": {}}
        for iname, k in impls.items():
            for rname, r in refs.items():
                if iname == "eager" and rname == "eager_fp32":
                    continue
                dk, dr = k - old, r - old
                row["results"][f"{iname}|{rname}"] = {
                    "value_default": check(k, r, FP32), "value_rtol_only": check(k, r, dict(rtol=FP32["rtol"], atol=0)),
                    "value_strict": strict(k, r),
                    "increment_default": check(dk, dr, FP32),
                    "increment_rtol_only": check(dk, dr, dict(rtol=FP32["rtol"], atol=0)),
                    "update_scale_ratio": output_unit(dk, dr)["scale_ratio"]}
        rows.append(row)
        print(seed, flush=True)
    return {"group": "B", "step": a.step, "seeds": [lo, hi], "checks": {"default": FP32, "strict": "1 fp32 ulp"},
            "summary": summarize(rows), "units": rows}


def group_c(a):
    import os
    import calls_tf32 as C
    os.environ.setdefault("KA_TOL_CAPTURES_C", a.captures)
    lo, hi = (int(x) for x in a.units.split("-"))
    rows = []
    calls = {n: getattr(C, f"{n}_{a.layer}_{a.mat}") for n in
             ("triton_default", "inductor_high", "cublas_high", "triton_ieee", "cublas_highest")}
    spec = getattr(C, f"spec_{a.layer}_{a.mat}")
    for u in range(lo, hi + 1):
        x = getattr(C, f"x_{a.layer}_{a.mat}")(u, None, None)
        inp = {"x": x}
        f_lo, f_hi = spec(inp)["y"]
        exact = 0.5 * (f_lo + f_hi)
        outs = {n: c({"x": x.clone()})["y"].double().cpu().numpy() for n, c in calls.items()}
        refs = {"exact": exact, "cublas_highest": outs["cublas_highest"]}
        row = {"unit": u, "results": {}}
        for iname in ("triton_default", "inductor_high", "cublas_high", "triton_ieee", "cublas_highest"):
            for rname, r in refs.items():
                if iname == rname:
                    continue
                k = outs[iname]
                fr = output_unit(k, r)
                row["results"][f"{iname}|{rname}"] = {
                    "default": check(k, r, FP32), "tf32_convention": check(k, r, TF32_CONVENTION),
                    "strict": strict(k, r), "alignment": fr["alignment"],
                    "mean_error_over_rms_reference": fr["mean_error_over_rms_reference"],
                    "relative_rms": fr["relative_rms"]}
        rows.append(row)
        print(u, flush=True)
    return {"group": "C", "layer": a.layer, "mat": a.mat, "units": [lo, hi],
            "checks": {"default": FP32, "tf32_convention": TF32_CONVENTION, "strict": "1 fp32 ulp"},
            "summary": summarize(rows), "per_unit": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=("B", "C"))
    ap.add_argument("--step", type=int)
    ap.add_argument("--seeds")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--mat")
    ap.add_argument("--units")
    ap.add_argument("--captures", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = group_b(a) if a.group == "B" else group_c(a)
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
