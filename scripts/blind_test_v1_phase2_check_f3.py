#!/usr/bin/env python3
"""Post-hoc check for blind_test_v1 phase 2, run after the verdicts were fixed (it changes none of them).

prog_05 / prog_08 (F3) show e_sem = K_R - f of about 1e-12 in the rstd column, beyond the interval widths and
also with float32 runtime scalars.  To rule out an evaluator error, prog_05's source was read: it merges eight
chunks of the row, and the merge coefficients CH / (k + CH) and k * CH / (k + CH) are compile-time Python floats
that become float32 constants in the TTIR.  This script evaluates that merge with the coefficients rounded to
float32 and everything else in float64 (errors ~1e-16, far below the effect), subtracts the cached f midpoint
and prints the mean of the predicted e_sem per column, to compare with phase2_report.json.

    python scripts/blind_test_v1_phase2_check_f3.py --package .cache/blind/blind_test_v1
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FCACHE = ROOT / ".cache" / "blind_v1_phase2_f"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package / "programs"))
    inputs = importlib.import_module("inputs")
    f32 = lambda v: float(np.float32(v))  # noqa: E731
    D, CH = 4096, 512
    cols = {"mean": [], "rstd": []}
    for seed in range(96):
        inp = inputs.make_inputs("F3", seed)
        x = inp["x"].cpu().double().numpy()
        mean = x[:, :CH].sum(1) / CH
        m2 = ((x[:, :CH] - mean[:, None]) ** 2).sum(1)
        for k in range(CH, D, CH):
            xb = x[:, k:k + CH]
            mb = xb.sum(1) / CH
            m2b = ((xb - mb[:, None]) ** 2).sum(1)
            delta = mb - mean
            mean = mean + delta * f32(CH / (k + CH))
            m2 = m2 + m2b + delta * delta * f32(k * CH / (k + CH))
        rstd = 1.0 / np.sqrt(m2 / D + f32(inp["eps"]))  # the kernel receives eps as float32
        f = np.load(FCACHE / f"F3_{seed:03d}.npz")
        fm = (0.5 * (f["given_lo"] + f["given_hi"])).reshape(-1, 2)
        cols["mean"].append(mean - fm[:, 0])
        cols["rstd"].append(rstd - fm[:, 1])
    out = {}
    for name, v in cols.items():
        v = np.concatenate(v)
        out[name] = {"predicted_mean": float(v.mean()), "predicted_negative_frac": float((v < 0).mean())}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
