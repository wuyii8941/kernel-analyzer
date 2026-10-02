#!/usr/bin/env python3
"""Population average of the approximate-division bias over the count N (round-2 item 4).

Liger divides every dlogit by n_non_ignore = N with ``arith.divf`` (lowered to
div.full.f32).  For each N the relative bias of the approximate quotient
against the correctly rounded one is measured on the GPU over many numerators,
and compared with the model delta(N) = RN32(1/N) * N - 1.  The population
average E_N[delta(N)] is then taken over declared N distributions.  The
hardware reciprocal rcp.approx.ftz.f32 is measured as well: the model
delta_rcp(N) = rcp(N) * N - 1 is what div.full.f32 actually multiplies by.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def model_delta(n: int) -> float:
    r = Fraction(float(np.float32(1.0) / np.float32(n)))  # RN32(1/N): float32 division is correctly rounded
    return float(r * n - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-n", type=int, default=1 << 17)
    args = parser.parse_args()
    import torch
    import triton

    from scripts.reference_eval_kernels import divide_by_count, hardware_reciprocal

    ns = np.arange(1, args.max_n + 1)
    g = torch.Generator(device="cuda").manual_seed(20261004)
    x = (torch.rand(1024, device="cuda", generator=g) * 2 - 1) * 4.0  # numerators of either sign
    measured = np.zeros(ns.size)
    sign_consistent = np.zeros(ns.size, dtype=bool)
    chunk = 4096
    for start in range(0, ns.size, chunk):
        nn = torch.as_tensor(ns[start:start + chunk], device="cuda", dtype=torch.int32)
        qf = torch.empty(nn.numel(), x.numel(), device="cuda")
        qr = torch.empty_like(qf)
        divide_by_count[(nn.numel(),)](x, nn, qf, qr, x.numel(), BLOCK=1024)
        rel = ((qf.double() - qr.double()) / qr.double().abs().clamp_min(1e-30) * torch.sign(qr.double()))
        measured[start:start + chunk] = rel.mean(dim=1).cpu().numpy()
        nz = rel != 0
        pos = (rel > 0).sum(dim=1)
        neg = (rel < 0).sum(dim=1)
        sign_consistent[start:start + chunk] = ((pos == 0) | (neg == 0)).cpu().numpy()
    model = np.array([model_delta(int(n)) for n in ns])
    nn = torch.as_tensor(ns, device="cuda", dtype=torch.int32)
    rcp = torch.empty(ns.size, device="cuda")
    hardware_reciprocal[(triton.cdiv(ns.size, 1024),)](nn, rcp, ns.size, BLOCK=1024)
    rcp = rcp.cpu().numpy()
    model_rcp = np.array([float(Fraction(float(rcp[i])) * int(ns[i]) - 1) for i in range(ns.size)])
    power_of_two = (ns & (ns - 1)) == 0

    def avg(lo, hi, weights=None):
        sel = (ns >= lo) & (ns <= hi)
        w = None if weights is None else weights[sel]
        return {"model_mean": float(np.average(model[sel], weights=w)),
                "model_rcp_mean": float(np.average(model_rcp[sel], weights=w)),
                "measured_mean": float(np.average(measured[sel], weights=w)),
                "model_positive_fraction": float(np.average(model[sel] > 0, weights=w)),
                "model_mean_abs": float(np.average(np.abs(model[sel]), weights=w))}

    report = {
        "schema": "kernel-analyzer-division-bias-over-n-v1",
        "delta_definition": "relative bias of x / N (div.full.f32) against div_rn, averaged over 1024 numerators",
        "model": "delta(N) = RN32(1/N) * N - 1",
        "model_rcp": "delta_rcp(N) = rcp.approx.ftz.f32(N) * N - 1",
        "N_63": {"model": model[62], "model_rcp": model_rcp[62], "measured": measured[62]},
        "measured_vs_model_correlation": float(np.corrcoef(measured, model)[0, 1]),
        "measured_vs_model_rcp_correlation": float(np.corrcoef(measured, model_rcp)[0, 1]),
        "measured_minus_model_rcp_max_abs": float(np.max(np.abs(measured - model_rcp))),
        "rcp_differs_from_RN32_fraction": float(np.mean(model_rcp != model)),
        "rcp_above_RN32_fraction": float(np.mean(model_rcp > model)),
        "rcp_below_RN32_fraction": float(np.mean(model_rcp < model)),
        "measured_sign_matches_model_fraction": float(np.mean(np.sign(measured) == np.sign(model))),
        "sign_consistent_within_N_fraction": float(sign_consistent.mean()),
        "powers_of_two_exact": bool(np.all(model[power_of_two] == 0) and np.all(measured[power_of_two] == 0)),
        "uniform_averages": {f"1..{m}": avg(1, m) for m in (64, 512, 4096, 65536, args.max_n)},
        "near_batch_sizes": {f"{c - w}..{c + w}": avg(c - w, c + w) for c, w in ((2048, 64), (8192, 256), (32768, 1024))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("uniform_averages", "near_batch_sizes")}, indent=1))
    print(json.dumps(report["uniform_averages"], indent=1))
    print(json.dumps(report["near_batch_sizes"], indent=1))


if __name__ == "__main__":
    main()
