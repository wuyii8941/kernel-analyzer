#!/usr/bin/env python3
"""Control for the e_num mean effects found on the Unsloth subset (docs/unfamiliar_subset_results_20261006.md):
is an effect inherent to rounding this function's output to its storage format, or specific to how the kernel
computes it?

For each case and output, on the confirmation seeds 32-95: the shrinkage statistics of the kernel output K and of
the correctly rounded output RN(f) (one round-to-nearest-even of the float64 specification to the output dtype),
both against f:
    R2-like  s2 = sum((x - f) * -sign(f)) / sum(|f|)        (magnitude pulled toward zero > 0)
    R3-like  s3 = sum((x - f) * -f) / sum(f^2)
An effect that RN(f) shares is a property of the output format on this input distribution; one that RN(f) lacks
comes from the kernel's own sequence of roundings.  (e_sem is constant-level or zero for every output, so f stands
for K_R here.)

    PYTHONPATH=.cache/pylibs/unsloth_shim:src:. python scripts/unsloth_rounding_control.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import unsloth_subset as us  # noqa: E402

OUT = ROOT / "results/external/unsloth_rounding_control.json"
SEEDS = range(32, 96)


def rn(f, dtype):
    f = np.asarray(f, np.float64)
    if dtype == torch.float32:
        return np.float32(f).astype(np.float64)
    if dtype == torch.bfloat16:  # one rounding from float64 (torch's double -> bf16 goes through float)
        m, e = np.frexp(f)
        return np.ldexp(np.round(m * 256.0), e - 8)
    raise ValueError(dtype)


def stats(x, f):
    ok = np.isfinite(f) & np.isfinite(x)
    x, f = x[ok], f[ok]
    return (float(np.sum((x - f) * -np.sign(f)) / max(np.sum(np.abs(f)), 1e-300)),
            float(np.sum((x - f) * -f) / max(np.sum(f * f), 1e-300)))


def main():
    rows = []
    for case in us.CASES:
        acc = {}
        for seed in SEEDS:
            inp = case.inputs(seed)
            outs = case.launch(inp)
            specs = case.spec(inp)
            for name, t in outs.items():
                f = 0.5 * (specs[name][0] + specs[name][1])
                k = t.detach().float().cpu().numpy().astype(np.float64).reshape(f.shape)
                a = acc.setdefault(name, {"k2": [], "k3": [], "r2": [], "r3": [], "dtype": str(t.dtype)})
                k2, k3 = stats(k, f)
                r2, r3 = stats(rn(f, t.dtype), f)
                a["k2"].append(k2), a["k3"].append(k3), a["r2"].append(r2), a["r3"].append(r3)
        for name, a in acc.items():
            row = {"case": case.name, "output": name, "dtype": a["dtype"], "n": len(a["k3"])}
            for key in ("k2", "k3", "r2", "r3"):
                v = np.array(a[key])
                row[key] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(v.size))}
            rows.append(row)
            print(f"{case.name:34s} {name:8s} K: s2 {row['k2']['mean']: .2e}±{row['k2']['se']:.0e} s3 {row['k3']['mean']: .2e}"
                  f"±{row['k3']['se']:.0e} | RN(f): s2 {row['r2']['mean']: .2e}±{row['r2']['se']:.0e} s3 {row['r3']['mean']: .2e}"
                  f"±{row['r3']['se']:.0e}", flush=True)
    OUT.write_text(json.dumps({"seeds": [SEEDS.start, SEEDS.stop - 1], "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
