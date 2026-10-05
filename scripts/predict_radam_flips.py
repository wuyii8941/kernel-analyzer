#!/usr/bin/env python3
"""Predict, from RAdam's formula alone, where the compiled step's rectification decision rho_t > 5 differs from the
specification (B012), before running anything.

Compiled step (TTIR of torch.compile(RAdam.step), torch 2.10, read from the capture of opt_radam_b9995_step5):
    t = step + 1 (fp32), b = fp32(beta2), rho_inf = fp32(2 / (1 - beta2) - 1) (folded in double, stored as fp32)
    p = powf(b, t); rho_t = rho_inf - 2 t p / (1 - p); rectified iff rho_t > 5
    multiplier S = rect(rho_t) * sqrt(1 - p) / (1 - powf(fp32(beta1), t))   (times m / (sqrt(v) + eps))
Three evaluations:
    f    exact real arithmetic with the double beta2 (the float64 eager specification)
    K_R  exact real arithmetic with the fp32 constants (the tool's reference: e_sem = K_R - f isolates them)
    K    float32 arithmetic (numpy; powf may differ from __nv_powf by an ulp -> flagged when rho_t is that close to 5)

    python scripts/predict_radam_flips.py 0.9993 0.9997 0.99985 0.99995 --steps 12
"""
import argparse
from fractions import Fraction as Fr

import numpy as np


def rect(rho, rho_inf):
    return float(((rho - 4) * (rho - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho)) ** 0.5) if rho > 5 else None


def evaluate(beta2, t, beta1=0.9, constants="f"):
    if constants == "f":
        b, b1 = Fr(beta2), Fr(beta1)
        rho_inf = 2 / (1 - b) - 1
    else:  # fp32 constants, exact arithmetic
        b, b1 = Fr(float(np.float32(beta2))), Fr(float(np.float32(beta1)))
        rho_inf = Fr(float(np.float32(2 / (1 - beta2) - 1)))
    p = b ** t
    rho = rho_inf - 2 * t * p / (1 - p)
    bc1 = 1 - b1 ** t
    r = rect(rho, rho_inf)
    s = None if r is None else r * float(1 - p) ** 0.5 / float(bc1)
    return float(rho), rho > 5, s, 1 / float(bc1)


def evaluate_fp32(beta2, t, beta1=0.9):
    f32 = np.float32
    b, tt = f32(beta2), f32(t)
    rho_inf = f32(2 / (1 - beta2) - 1)
    p = f32(np.power(b, tt))
    rho = f32(rho_inf - f32(f32(f32(2) * tt) * p) / f32(1 - p))
    bc1 = f32(1 - f32(np.power(f32(beta1), tt)))
    r = rect(Fr(float(rho)), Fr(float(rho_inf))) if rho > f32(5) else None
    s = None if r is None else r * float(np.sqrt(f32(1 - p))) / float(bc1)
    return float(rho), bool(rho > f32(5)), float(np.spacing(np.float32(5))), s, 1 / float(bc1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("beta2", type=float, nargs="+")
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()
    print("| beta2 | step | rho_t (f) | rho_t (K_R) | rho_t (K, fp32) | rectified f / K_R / K | e_sem | e_num |")
    print("|---|---:|---:|---:|---:|---|---|---|")
    for beta2 in a.beta2:
        for t in range(1, a.steps + 1):
            rf, df, sf, uf = evaluate(beta2, t)
            rr, dr, sr, ur = evaluate(beta2, t, constants="fp32")
            rk, dk, ulp5, sk, uk = evaluate_fp32(beta2, t)
            if dk == dr:  # e_num factor from the decision quantities alone (same branch in K and K_R)
                num = abs(sk / sr - 1) if (dk and sk and sr) else abs(uk / ur - 1)
                num_txt = f"e_num ~ {num:.1e} x update"
            else:
                num_txt = "e_num: whole update (K and K_R take different branches)"
            if df != dr:
                pred = "e_sem: whole update (decision flips in K_R)"
            elif df and sf and sr:
                pred = f"e_sem ~ {abs(sr / sf - 1):.1e} x update (same decision)"
            else:
                pred = f"e_sem ~ {abs(ur / uf - 1):.1e} x update (both unrectified)"
            if dk != df:
                pred += "; K flips too" if df != dr else "; K flips (e_num carries it)"
            if abs(rk - 5) < 4 * ulp5:
                pred += " [K within 4 ulp of 5: powf ulp decides]"
            print(f"| {beta2} | {t} | {rf:.6f} | {rr:.6f} | {rk:.6f} | {'yes' if df else 'no'} / {'yes' if dr else 'no'} / "
                  f"{'yes' if dk else 'no'} | {pred} | {num_txt} |")


if __name__ == "__main__":
    main()
