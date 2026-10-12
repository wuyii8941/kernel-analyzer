"""RQ4: the bias mechanisms (protocol section 5), each on the device in a minimal composition, with the prediction
written down from first principles and the catalog before the device result is read.

A deviation K - G decomposes over the rounding sites s (switch the sites from exact to device one at a time; the
increments sum to K - G exactly).  To first order a site contributes c_s rho_s (c_s: the derivative of the output
with respect to the site's value, rho_s: its local residual), so

    E[K - G] = sum_s E[c_s] E[rho_s]  +  sum_s Cov(c_s, rho_s)  +  second-order terms  +  jumps of discrete decisions
                  (B1: site mean)        (B2: covariance)        (B3: curvature x variance)  (B4: flips)

and the declared program differs from the task specification by B5 (constants, storage types, algorithm).  Each
experiment isolates one term:

  E1  B1  tf32 operand truncation in a dot: E[tau] per truncated operand predicted from the significand
          distribution of the data (the catalog: tf32 conversion is RZ);
  E2  B1 conditional on the state: bf16 accumulation of same-sign increments in three composition forms
          (C8 loop, C3 reduce, C4 scan) -- one primitive (bf16 add), three magnitudes of bias;
  E3  B2  the package's D1 / D2 construction on the device: residuals of mean zero, coefficient aligned with
          (or against) their sign;
  E4  B3  mean square of quantized values (bf16, fp8 e5m2): E[rho^2] = ulp^2 / 12 under a uniform residual;
  E5  B4  a threshold after rounding and the maximum of rounded values (selection on the residual).

Units are independent blocks; the mean and its interval come from the tool's statistics (analysis.unit_axes).
Decision (protocol): the prediction lies in the 95% interval, or sign agrees and measured / predicted is in
[0.8, 1.25]; otherwise 不成立; an unresolved interval is 无法判断.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "2_tool/src"))
import exact as X  # noqa: E402
from kernel_analyzer.reference_eval import analysis as A  # noqa: E402

ALPHA = 0.05


def decide(pred, units):
    s = A.unit_axes(np.asarray(units, np.float64), ALPHA)["nonzero"]
    out = {"prediction": pred, "units": len(units)}
    if "t_interval" not in s:
        out.update(measured=None, verdict="无法判断", statistics=s)
        return out
    lo, hi = s["t_interval"]
    m = s["mean_projection"]
    ratio = m / pred if pred else None
    ok = (lo <= pred <= hi) or (ratio is not None and np.sign(m) == np.sign(pred) and 0.8 <= ratio <= 1.25)
    if pred != 0 and lo <= 0 <= hi:
        verdict = "无法判断"          # the effect itself is not resolved: agreement with the prediction says nothing
    else:
        verdict = "成立" if ok else "不成立"
    out.update(measured=m, ci95=[lo, hi], ratio=ratio, verdict=verdict,
               p_nonzero=s["p_value_two_sided_conservative"])
    return out


def maximum_decision(units, data="x ~ 1 + 0.05 N(0,1)"):
    """max_i RN(x_i) >= RN(x_argmax) and E[RN(x) - x] ~ 0, so the prediction is a positive mean (the maximum selects
    the element whose residual is largest)."""
    s = A.unit_axes(np.asarray(units, np.float64), ALPHA)["nonzero"]
    out = {"observable": f"mean of max RN(x_i) - max x_i per group of 8 ({data}, fp8 e5m2)",
           "prediction": "positive", "units": len(units)}
    if "t_interval" not in s:
        return {**out, "verdict": "无法判断"}
    lo, hi = s["t_interval"]
    return {**out, "measured": s["mean_projection"], "ci95": [lo, hi],
            "verdict": "成立" if lo > 0 else ("不成立" if hi < 0 else "无法判断")}


def kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def mm(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        rk = tl.arange(0, BK)
        acc = tl.zeros((M, N), tl.float32)
        for k0 in range(0, K, BK):
            x = tl.load(a + rm[:, None] * K + (k0 + rk)[None, :])
            y = tl.load(b + (k0 + rk)[:, None] * N + rn[None, :])
            acc = tl.dot(x, y, acc, input_precision="tf32")
        tl.store(c + rm[:, None] * N + rn[None, :], acc)

    @triton.jit
    def acc_loop(x, o, N: tl.constexpr, T: tl.constexpr):         # C8: one bf16 accumulator per lane, T steps
        i = tl.arange(0, N)
        acc = tl.zeros([N], tl.bfloat16)
        for t in range(T):
            acc = acc + tl.load(x + t * N + i)
        tl.store(o + i, acc)

    @triton.jit
    def acc_reduce(x, o, N: tl.constexpr):                          # C3: tl.sum of a bf16 block (bf16 combine)
        i = tl.arange(0, N)
        tl.store(o + tl.program_id(0), tl.sum(tl.load(x + tl.program_id(0) * N + i), axis=0))

    @triton.jit
    def acc_scan(x, o, N: tl.constexpr):                            # C4: the last element of a bf16 cumsum
        i = tl.arange(0, N)
        c = tl.cumsum(tl.load(x + tl.program_id(0) * N + i), axis=0)
        tl.store(o + tl.program_id(0) * N + i, c)

    @triton.jit
    def signed_products(a, b, s, o, N: tl.constexpr):              # C1 + C3: sum_i s_i * RN(a_i b_i), f64 sum
        i = tl.arange(0, N)
        p = tl.load(a + tl.program_id(0) * N + i) * tl.load(b + tl.program_id(0) * N + i)
        v = p.to(tl.float64) * tl.load(s + tl.program_id(0) * N + i)
        tl.store(o + tl.program_id(0), tl.sum(v, axis=0))

    @triton.jit
    def mean_square(x, o, N: tl.constexpr, FMT: tl.constexpr):     # quantize, square, sum in f64
        i = tl.arange(0, N)
        v = tl.load(x + tl.program_id(0) * N + i)
        if FMT == 0:
            q = v.to(tl.bfloat16).to(tl.float64)
        else:
            q = v.to(tl.float8e5).to(tl.float32).to(tl.float64)   # e5m2 -> f64 directly is not lowered
        tl.store(o + tl.program_id(0), tl.sum(q * q, axis=0))

    @triton.jit
    def threshold(x, o, m, N: tl.constexpr, G: tl.constexpr):     # count(RN_bf16(x) > 1); max of e5m2 values
        i = tl.arange(0, N)
        v = tl.load(x + tl.program_id(0) * N + i)
        q = v.to(tl.bfloat16).to(tl.float32)
        tl.store(o + tl.program_id(0), tl.sum((q > 1.0).to(tl.int32), axis=0))
        r = tl.reshape(v.to(tl.float8e5).to(tl.float32), (N // G, G))
        tl.store(m + tl.program_id(0) * (N // G) + tl.arange(0, N // G), tl.max(r, axis=1))
    return mm, acc_loop, acc_reduce, acc_scan, signed_products, mean_square, threshold


def tau_rz_tf32(x):
    """Relative truncation error of RZ to 11 significant bits (data side of the E1 prediction)."""
    m, e = np.frexp(x)
    t = np.ldexp(np.trunc(np.ldexp(m, 11)), e - 11)
    return (t - x) / x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/mechanisms.json"))
    a = ap.parse_args()
    import torch
    mm, acc_loop, acc_reduce, acc_scan, signed_products, mean_square, threshold = kernels()
    rng = np.random.default_rng(20261013)
    res = {}

    # ---------------------------------------------------------------- E1: B1, tf32 operand truncation
    M = N = 64
    K = 256
    units, preds = [], []
    for u in range(32):
        Am = rng.standard_normal((M, K)).astype(np.float32).astype(np.float64)
        Bm = rng.standard_normal((K, N)).astype(np.float32).astype(np.float64)
        pred_u = float(np.mean(tau_rz_tf32(Am[Am != 0])) + np.mean(tau_rz_tf32(Bm[Bm != 0])))
        preds.append(pred_u)
        ta, tb = torch.from_numpy(Am).float().cuda(), torch.from_numpy(Bm).float().cuda()
        tc = torch.empty(M, N, device="cuda")
        mm[(1,)](ta, tb, tc, M=M, N=N, K=K, BK=32)
        torch.cuda.synchronize()
        r = Am @ Bm
        e = tc.double().cpu().numpy() - r
        units.append(float((e * r).sum() / (r * r).sum()))
    res["E1_B1_tf32_operand_truncation"] = {
        "form": "C15 dot (tf32, K = 256, BK = 32)", "observable": "alignment <e, r> / <r, r> per matmul",
        "prediction_basis": "E[tau_A] + E[tau_B], tau = relative RZ-to-tf32 error of the operands (catalog: the "
                            "conversion is RZ); the accumulation's own RZ is left out of the prediction",
        **decide(float(np.mean(preds)), units)}
    print("E1", res["E1_B1_tf32_operand_truncation"], flush=True)

    # ---------------------------------------------------------------- E2: B1 conditional, accumulation forms
    # increments x in [0.5, 1) in bf16.  Prediction for a sequential accumulator (C8): below 128 the residuals are
    # unbiased; on [128, 256) the spacing is 1, so every increment rounds to 1 (ties at 0.5 to even); on [256, 512)
    # the spacing is 2 and every increment (< 1 = half the spacing) is lost: each lane stops at exactly 256.
    out2 = {}
    lanes, steps = 128, 4096
    xs = rng.random((steps, lanes)) * 0.5 + 0.5
    xb = torch.from_numpy(xs).to(torch.bfloat16)
    o = torch.empty(lanes, device="cuda", dtype=torch.bfloat16)
    acc_loop[(1,)](xb.cuda().reshape(-1), o, N=lanes, T=steps)
    torch.cuda.synchronize()
    dev = o.double().cpu().numpy()
    lane_exact = xb.double().numpy().sum(axis=0)
    out2["C8_loop_4096_steps"] = {"prediction": "every lane ends at exactly 256",
                                  "lanes_at_256": int((dev == 256).sum()), "lanes": lanes,
                                  "relative_error_mean": float(np.mean(dev / lane_exact - 1)),
                                  "verdict": "成立" if (dev == 256).all() else "不成立"}
    n = 4096
    red_units, scan_units = [], []
    for u in range(16):
        x = torch.from_numpy(rng.random(n) * 0.5 + 0.5).to(torch.bfloat16)
        ex = float(sum(F(float(v)) for v in x.double().numpy()))
        xt = x.cuda()
        orr = torch.empty(1, device="cuda", dtype=torch.bfloat16)
        acc_reduce[(1,)](xt, orr, N=n)
        osc = torch.empty_like(xt)
        acc_scan[(1,)](xt, osc, N=n)
        torch.cuda.synchronize()
        red_units.append((float(orr.double().item()) - ex) / ex)
        scan_units.append((float(osc[-1].double().item()) - ex) / ex)
    for name, units in (("C3_reduce_4096", red_units), ("C4_scan_last_of_4096", scan_units)):
        out2[name] = {"prediction": "partial sums of comparable size: |relative bias| below 0.05, far from the "
                                    "loop's stagnation", "measured": float(np.mean(units)),
                      "min": float(np.min(units)), "max": float(np.max(units)),
                      "verdict": "成立" if abs(float(np.mean(units))) < 0.05 else "不成立"}
    res["E2_B1_conditional_accumulation_forms"] = out2
    print("E2", out2, flush=True)

    # ---------------------------------------------------------------- E3: B2, D1 / D2 construction
    out3 = {}
    for pattern, sign in (("D1 aligned", 1.0), ("D2 anti-aligned", -1.0)):
        units, preds = [], []
        for u in range(24):
            nn = 1024
            av = rng.standard_normal(nn).astype(np.float32).astype(np.float64)
            bv = rng.standard_normal(nn).astype(np.float32).astype(np.float64)
            exact_p = [F(float(x)) * F(float(y)) for x, y in zip(av, bv)]
            rho = [X.round_frac(p, "f32") - p for p in exact_p]
            s = np.array([sign * (1.0 if r > 0 else (-1.0 if r < 0 else 0.0)) for r in rho])
            # prediction: E[c rho] = sign * E|rho|, with |rho| uniform on [0, ulp / 2] -> ulp / 4 on average
            pred_u = sign * float(np.mean([float(X.ulp_at(p, "f32")) / 4 for p in exact_p]))
            preds.append(pred_u)
            ta = torch.from_numpy(av).float().cuda()
            tb = torch.from_numpy(bv).float().cuda()
            ts = torch.from_numpy(s).cuda()
            to = torch.empty(1, device="cuda", dtype=torch.float64)
            signed_products[(1,)](ta, tb, ts, to, N=nn)
            exact_sum = float(sum(F(float(si)) * p for si, p in zip(s, exact_p)))
            units.append((float(to.item()) - exact_sum) / nn)
        out3[pattern] = {"observable": "mean of s_i (RN(a_i b_i) - a_i b_i) per block of 1024",
                         **decide(float(np.mean(preds)), units)}
    res["E3_B2_residual_coefficient_covariance"] = out3
    print("E3", out3, flush=True)

    # ---------------------------------------------------------------- E4: B3 (and the B2 term next to it)
    # f(q) = q^2 with q = RN(x) = x + rho:  E[q^2 - x^2] = 2 E[x rho] (B2: residual against the coefficient 2x)
    # + E[rho^2] (B3).  Prediction 1 keeps only B3 with a uniform residual (ulp^2 / 12, residual independent of x);
    # prediction 2 takes both terms from the exact RN residuals of the data (the answer machine, not the device).
    out4 = {}
    for fmt, code, fx in (("bf16", 0, "bf16"), ("fp8 e5m2", 1, "f8e5m2")):
        units, pred_b3, pred_full, cross, square = [], [], [], [], []
        for u in range(24):
            nn = 4096
            x = rng.standard_normal(nn).astype(np.float32).astype(np.float64)
            fx_ = [F(float(v)) for v in x]
            ulps = np.array([float(X.ulp_at(v, fx)) for v in fx_])
            rho = np.array([float(X.round_frac(v, fx, "rne") - v) for v in fx_])
            sxx = float((x * x).sum())
            pred_b3.append(float((ulps ** 2 / 12).sum() / sxx))
            cross.append(float(2 * (x * rho).sum() / sxx))
            square.append(float((rho * rho).sum() / sxx))
            pred_full.append(cross[-1] + square[-1])
            tx = torch.from_numpy(x).float().cuda()
            to = torch.empty(1, device="cuda", dtype=torch.float64)
            mean_square[(1,)](tx, to, N=nn, FMT=code)
            units.append(float(to.item()) / sxx - 1)
        out4[fmt] = {"observable": "sum RN(x)^2 / sum x^2 - 1 per block of 4096",
                     "prediction_1_B3_only_uniform_residual": decide(float(np.mean(pred_b3)), units),
                     "prediction_2_B2_plus_B3_exact_residuals": decide(float(np.mean(pred_full)), units),
                     "B2_cross_term_2E[x rho]": float(np.mean(cross)), "B3_square_term_E[rho^2]": float(np.mean(square))}
    res["E4_B3_and_B2_mean_square_of_quantized_values"] = out4
    print("E4", out4, flush=True)

    # ---------------------------------------------------------------- E5: B4, threshold and maximum after rounding
    units_t, preds_t, units_m = [], [], []
    units_wide = []
    G = 8
    for u in range(24):
        nn = 4096
        x = (1 + rng.standard_normal(nn) * 0.05).astype(np.float32).astype(np.float64)
        # prediction: values in (1, 1 + 2^-8] round to 1 in bf16 (ulp above 1 is 2^-7; the tie goes to the even 1)
        # and stop counting as > 1; values just below 1 round to 1 or stay below: no change
        from math import erf
        Phi = lambda z: 0.5 * (1 + erf(z / math.sqrt(2)))  # noqa: E731
        preds_t.append(-(Phi(2.0 ** -8 / 0.05) - 0.5))
        tx = torch.from_numpy(x).float().cuda()
        to = torch.empty(1, device="cuda", dtype=torch.int32)
        tm = torch.empty(nn // G, device="cuda", dtype=torch.float32)
        threshold[(1,)](tx, to, tm, N=nn, G=G)
        units_t.append((int(to.item()) - int((x > 1).sum())) / nn)
        units_m.append(float(np.mean(tm.double().cpu().numpy() - x.reshape(nn // G, G).max(axis=1))))
        # the other regime: spread much larger than the spacing (x ~ N(0, 4^2), e5m2 spacing at |x| ~ 4 is 1)
        xw = (rng.standard_normal(nn) * 64).astype(np.float32).astype(np.float64)
        tw = torch.from_numpy(xw).float().cuda()
        threshold[(1,)](tw, to, tm, N=nn, G=G)
        units_wide.append(float(np.mean(tm.double().cpu().numpy() - xw.reshape(nn // G, G).max(axis=1))
                                / np.mean(np.abs(xw))))
        torch.cuda.synchronize()
    res["E5_B4_discrete_decisions"] = {
        "threshold_count_after_bf16_rounding": {"observable": "share of RN_bf16(x) > 1 minus share of x > 1",
                                                **decide(float(np.mean(preds_t)), units_t)},
        "maximum_of_e5m2_rounded_groups_of_8_spacing_above_spread": maximum_decision(units_m),
        "maximum_of_e5m2_rounded_groups_of_8_spread_above_spacing": maximum_decision(
            units_wide, "x ~ 64 N(0,1), divided by mean |x|")}
    print("E5", res["E5_B4_discrete_decisions"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
