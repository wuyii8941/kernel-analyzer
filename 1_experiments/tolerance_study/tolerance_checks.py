#!/usr/bin/env python3
"""Group A, question 1 and the criterion / reference / position comparisons of question 3 (protocol sections 3-4),
on the captured units of one stratum.  For every unit (one fine-tuning run's captured state before step t):

implementations K (bf16, as configured):  fused (torchtune default), compiled (official recipe), eager foreach
references:  eager_bf16  -- the same algorithm at the same precision (eager foreach AdamW on the same bf16 inputs; the
                            reference PyTorch's own compiled-vs-eager tests use)
             fp32        -- the same algorithm on fp32 copies of the inputs (higher precision, same formula)
             exact       -- the exact AdamW step (strict enclosure midpoint; calls_adamw.adamw_exact)
checks (frozen, protocol section 3):
  default   torch.testing.assert_close with the dtype's default tolerances (bf16: rtol 1.6e-2, atol 1e-5)
  strict    |K - R| <= 1 ulp of R in bf16, elementwise
  rtol_only the default rtol with atol = 0
at two observation positions: the written value itself and the increment (written value - old value).  The
correct-rounding control: the exact value rounded to nearest bf16, put through the same checks against exact.

    python tolerance_checks.py --captures DIR --step 100 --seeds 1000-1031 --out results/A1_tolerance_s100.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calls_adamw as C  # noqa: E402

BF16 = dict(rtol=1.6e-2, atol=1e-5)          # torch.testing default for bfloat16


def ulp_bf16(x):
    a = np.maximum(np.abs(x), 2.0 ** -126)
    return np.exp2(np.floor(np.log2(a)) - 7)


def checks(k, r, old):
    """The frozen checks of k against r (float64 arrays) at the value and the increment positions."""
    out = {}
    for pos, (a, b) in (("value", (k, r)), ("increment", (k - old, r - old))):
        d = np.abs(a - b)
        default_ok = d <= BF16["atol"] + BF16["rtol"] * np.abs(b)
        rtol_ok = d <= BF16["rtol"] * np.abs(b)
        u = ulp_bf16(r) if pos == "value" else ulp_bf16(r)          # one ulp of the reference value, both positions
        strict_ok = d <= u
        out[pos] = {"default_pass": bool(default_ok.all()), "default_fail_fraction": float(1 - default_ok.mean()),
                    "rtol_only_pass": bool(rtol_ok.all()), "rtol_only_fail_fraction": float(1 - rtol_ok.mean()),
                    "strict_1ulp_pass": bool(strict_ok.all()), "strict_fail_fraction": float(1 - strict_ok.mean()),
                    "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                    "relative_rms": float(np.sqrt((d ** 2).mean() / max((b ** 2).mean(), 1e-300)))}
    return out


def run_impl(name, inp):
    """Written (param, exp_avg, exp_avg_sq) of one implementation on the unit's inputs, as float64 arrays."""
    if name in ("fused", "compiled"):
        out = (C.adamw_fused if name == "fused" else C.adamw_compiled)(inp)
        return {q: out[q].detach().double().cpu().numpy() for q in ("param", "exp_avg", "exp_avg_sq")}
    dtype = torch.float32 if name == "fp32" else torch.bfloat16
    p = torch.nn.Parameter(inp["param"].to(dtype).clone())
    p.grad = inp["grad"].to(dtype).clone()
    opt = torch.optim.AdamW([p], lr=float(C.LR), foreach=True)
    st = opt.state[p]
    st["step"] = torch.tensor(float(inp["step"].item()))
    st["exp_avg"] = inp["exp_avg"].to(dtype).clone()
    st["exp_avg_sq"] = inp["exp_avg_sq"].to(dtype).clone()
    opt.step()
    return {"param": p.detach().double().cpu().numpy(), "exp_avg": st["exp_avg"].double().cpu().numpy(),
            "exp_avg_sq": st["exp_avg_sq"].double().cpu().numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import os
    os.environ["KA_TOL_CAPTURES"] = a.captures
    lo, hi = (int(x) for x in a.seeds.split("-"))
    rows = []
    t0 = time.time()
    for seed in range(lo, hi + 1):
        n = None
        inp = {}
        for q in C.QUANTITIES:
            v = C._capture(seed, a.step)[q]
            n = v.numel()
            inp[q] = v.to(torch.bfloat16).cuda().clone()
        inp["step"] = torch.tensor([C._capture(seed, a.step)["state_step"]], dtype=torch.float32)
        old = {q: inp[q].double().cpu().numpy() for q in C.QUANTITIES}
        t = int(inp["step"].item()) + 1
        pn, mp, vp = C.adamw_exact(old["param"], old["grad"], old["exp_avg"], old["exp_avg_sq"], t)
        exact = {"param": 0.5 * (pn[0] + pn[1]), "exp_avg": 0.5 * (mp[0] + mp[1]), "exp_avg_sq": 0.5 * (vp[0] + vp[1])}
        refs = {"exact": exact, "fp32": run_impl("fp32", inp), "eager_bf16": run_impl("eager_bf16", inp)}
        impls = {k: run_impl(k, {q: v.clone() for q, v in inp.items()}) for k in ("fused", "compiled")}
        impls["eager_bf16"] = refs["eager_bf16"]
        row = {"seed": seed, "step": a.step, "coordinates": n, "results": {}}
        for iname, kv in impls.items():
            for rname, rv in refs.items():
                if iname == rname:
                    continue
                row["results"][f"{iname}|{rname}"] = {q: checks(kv[q], rv[q], old[q])
                                                      for q in ("param", "exp_avg", "exp_avg_sq")}
        # control: the correctly rounded implementation (exact value rounded to nearest bf16) against exact
        rn = {q: torch.tensor(exact[q]).to(torch.bfloat16).double().numpy() for q in exact}
        row["results"]["correctly_rounded|exact"] = {q: checks(rn[q], exact[q], old[q]) for q in exact}
        rows.append(row)
        print(seed, f"{time.time() - t0:.0f}s", flush=True)
    summary = {}
    for key in rows[0]["results"]:
        for q in ("param", "exp_avg", "exp_avg_sq"):
            for pos in ("value", "increment"):
                cell = [r["results"][key][q][pos] for r in rows]
                summary[f"{key}|{q}|{pos}"] = {
                    "units": len(cell),
                    "default_pass_units": sum(c["default_pass"] for c in cell),
                    "rtol_only_pass_units": sum(c["rtol_only_pass"] for c in cell),
                    "strict_1ulp_pass_units": sum(c["strict_1ulp_pass"] for c in cell),
                    "mean_default_fail_fraction": float(np.mean([c["default_fail_fraction"] for c in cell])),
                    "mean_rtol_only_fail_fraction": float(np.mean([c["rtol_only_fail_fraction"] for c in cell])),
                    "mean_strict_fail_fraction": float(np.mean([c["strict_fail_fraction"] for c in cell])),
                    "mean_relative_rms": float(np.mean([c["relative_rms"] for c in cell]))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"step": a.step, "seeds": [lo, hi], "checks": {
        "default": "torch.testing.assert_close defaults for bfloat16: rtol 1.6e-2, atol 1e-5",
        "rtol_only": "rtol 1.6e-2, atol 0", "strict": "|K - R| <= 1 bf16 ulp of R"},
        "summary": summary, "units": rows, "seconds": round(time.time() - t0, 1)}, indent=1) + "\n")


if __name__ == "__main__":
    main()
