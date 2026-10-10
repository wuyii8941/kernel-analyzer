#!/usr/bin/env python3
"""Independent level prediction for the precision hold-out items (protocol.md, H-R1 / H-R2), written and run before
the acceptance run.  It does not call the tool's interval code: the published error bounds the levels use are
evaluated here in 60-digit arithmetic (mpmath) on the same unit inputs, against 1/8 ulp of the float32 binade of the
exact result.

* H-R1 (row sums, SumK, Ogita-Rump-Oishi 2005 Prop. 4.10): |res - s| <= (u + 3 g_{n-1}^2)|s| + g_{2n-2}^K S, with
  K = 3 / 5 / 8 at levels 1 / 2 / 3, g_m = m u / (1 - m u), S = sum |x_i|; width = 2 x bound.
* H-R2 (reverse prefix sums): level 1 the gamma bound g_n * prefix-sum |x|; levels 2 and 3 the Sum2 bound per prefix
  (Prop. 4.5) u |s_j| + g_j^2 S_j (j: 0-based position in the scanned order), intersected with the gamma bound.

A prediction is "uncertain" when the deciding ratio lies within a factor 4 of the threshold (the tool adds directed
rounding of the float64 endpoints that this evaluation leaves out).

    python predict_levels.py --out predictions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
mp.mp.dps = 60
U = mp.mpf(2) ** -53
TARGET = mp.mpf(1) / 8


def gamma(m):
    m = max(int(m), 1)
    return m * U / (1 - m * U)


def ulp32(s):
    s = abs(mp.mpf(s))
    if s == 0:
        return mp.mpf(2) ** -149
    e = int(mp.floor(mp.log(s, 2)))
    return mp.mpf(2) ** (max(e, -126) - 23)


def exact_sum(row):
    return sum((Fraction(float(v)) for v in row), Fraction(0))


def r1_ratios(rows):
    """Worst width / ulp over the rows at each level."""
    out = {1: mp.mpf(0), 2: mp.mpf(0), 3: mp.mpf(0)}
    for row in rows:
        n = len(row)
        s = exact_sum(row)
        S = sum(abs(mp.mpf(float(v))) for v in row)
        sm = mp.mpf(s.numerator) / s.denominator
        for lev, K in ((1, 3), (2, 5), (3, 8)):
            bound = (U + 3 * gamma(n - 1) ** 2) * abs(sm) + gamma(2 * n - 2) ** K * S
            out[lev] = max(out[lev], 2 * bound / ulp32(sm))
    return out


def r2_ratios(rows):
    out = {1: mp.mpf(0), 2: mp.mpf(0), 3: mp.mpf(0)}
    for row in rows:
        y = [float(v) for v in row][::-1]             # reverse scan: prefix sums of the flipped row
        n = len(y)
        g = gamma(n) * (1 + 4 * U)
        s, S = Fraction(0), mp.mpf(0)
        for j, v in enumerate(y):
            s += Fraction(v)
            S += abs(mp.mpf(v))
            sm = mp.mpf(s.numerator) / s.denominator
            b1 = g * S
            b2 = min(b1, U * abs(sm) + gamma(max(j, 1)) ** 2 * S)
            u = ulp32(sm)
            out[1] = max(out[1], 2 * b1 / u)
            out[2] = max(out[2], 2 * b2 / u)
            out[3] = max(out[3], 2 * b2 / u)
    return out


def predict(ratios):
    for lev in (1, 2, 3):
        if ratios[lev] <= TARGET:
            r = ratios[lev]
            uncertain = r > TARGET / 4 or (lev > 1 and ratios[lev - 1] < TARGET * 4)
            return {"category": "met", "level": lev, "uncertain": bool(uncertain)}
    return {"category": "backend limit", "level": 3, "uncertain": bool(ratios[3] < TARGET * 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import holdout_calls as H
    items = json.loads((HERE / "items.json").read_text())["items"]
    out = {}
    for it in items:
        if it["id"] not in ("H-R1", "H-R2"):
            continue
        u = it["declaration"]["units"]
        n_total = min(u["development"] + u["confirmation"], it["declaration"]["budget"]["max_units"])
        seeds = [u["seed_offset"] + i for i in range(n_total)]
        src = it["declaration"]["inputs"]["x"]
        gen = {"H-R1": H.pairs_rows, "H-R2": None}[it["id"]]
        worst = {1: mp.mpf(0), 2: mp.mpf(0), 3: mp.mpf(0)}
        for seed in seeds:
            if it["id"] == "H-R1":
                rows = gen(seed, tuple(src["shape"]), 70, 90, -60, 11).astype("float32")
                r = r1_ratios(rows)
            else:
                rows = H.cancel_rows_rev(seed, tuple(src["shape"]), "float32").cpu().numpy()
                r = r2_ratios(rows)
            for k in worst:
                worst[k] = max(worst[k], r[k])
        out[it["id"]] = {"ratios_width_over_ulp": {str(k): mp.nstr(v, 6) for k, v in worst.items()},
                         "target": float(TARGET), **predict(worst)}
        print(it["id"], out[it["id"]])
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
