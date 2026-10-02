#!/usr/bin/env python3
"""Magnitude-only judgement vs average-effect judgement on the same locations (round-2 item 3).

Same kernels, same input draws, same measurement point (kernel output):

* magnitude side, per draw: torch.allclose(mutant, original) with default
  tolerances, and a magnitude baseline in the style of TTrace: relative
  Frobenius error of mutant vs original, against a threshold obtained by
  perturbing the original's float inputs by about one FP32 ulp and rerunning
  the original.  Only this part of TTrace is implemented, so it is called a
  magnitude baseline, not a TTrace reproduction.  A location "exceeds" when
  the error exceeds the threshold on at least half of the draws;
* average-effect side: 96 draws from the declared input distribution, the
  fixed-direction rule (32 calibration, 64 confirmation) on u = mutant -
  original; and, with the automatic reference, on u = K - K_R for the mutant
  and for the original, the reference interval entering through its endpoints.

    python scripts/run_method_comparison.py --out results/reference_eval/method_comparison.json
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

from kernel_analyzer.reference_eval.analysis import _holm, _summarize  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

N_CAL, N_CONF = 32, 64


def specs():
    import torch

    from scripts import mutation_kernels as mk

    n = 2048

    def gen(seed):
        return torch.Generator(device="cuda").manual_seed(seed)

    def scale(seed):
        g = gen(seed)
        return {"X": torch.randn(n, device="cuda", generator=g), "alpha": 0.1}, \
            lambda t, mut: mk.m_scale[(n // 256,)](t["X"], t["Y"], t["alpha"], n, BLOCK=256, MUT=mut), (n,)

    def sum4(seed):
        g = gen(seed)
        t = {k: torch.randn(n, device="cuda", generator=g) * s for k, s in zip("ABCD", (1.0, 10.0, 100.0, 1.0))}
        return t, lambda t, mut: mk.m_sum4[(n // 256,)](t["A"], t["B"], t["C"], t["D"], t["Y"], n, BLOCK=256,
                                                         MUT=mut), (n,)

    def row_sum(seed):
        g = gen(seed)
        return {"X": torch.randn(32, 300, device="cuda", generator=g)}, \
            lambda t, mut: mk.m_row_sum[(32,)](t["X"], t["Y"], 300, BLOCK=512, MUT=mut), (32,)

    def softmax(seed):
        g = gen(seed)
        return {"X": torch.randn(16, 200, device="cuda", generator=g) * 3.0}, \
            lambda t, mut: mk.m_softmax[(16,)](t["X"], t["Y"], 200, 200, BLOCK=256, MUT=mut), (16, 200)

    def layernorm(seed):
        g = gen(seed)
        t = {"X": torch.randn(16, 200, device="cuda", generator=g) + 0.5,
             "W": torch.randn(200, device="cuda", generator=g), "B": torch.randn(200, device="cuda", generator=g)}
        return t, lambda t, mut: mk.m_layernorm[(16,)](t["X"], t["W"], t["B"], t["Y"], 200, 1e-5, BLOCK=256,
                                                        MUT=mut), (16, 200)

    def matmul(seed):
        g = gen(seed)
        t = {"A": torch.randn(64, 64, device="cuda", generator=g), "B": torch.randn(64, 32, device="cuda", generator=g)}
        return t, lambda t, mut: mk.m_matmul[(2, 2)](t["A"], t["B"], t["Y"], 64, 32, 64, BM=32, BN=16, BK=16,
                                                      MUT=mut), (64, 32)

    def accumulate(seed):
        g = gen(seed)
        t = {"ACC0": torch.randn(n, device="cuda", generator=g) * 100.0, "C": torch.randn(n, device="cuda", generator=g)}

        def launch(t, mut):
            t["Y"].copy_(t["ACC0"])
            mk.m_accumulate[(n // 256,)](t["Y"], t["C"], n, BLOCK=256, MUT=mut)
        return t, launch, (n,)

    return {
        "scale": (scale, {1: "rounding_down", 2: "rounding_up", 3: "precision"}),
        "sum4": (sum4, {1: "order", 2: "rounding_down_local", 3: "precision"}),
        "row_sum": (row_sum, {1: "order", 2: "precision"}),
        "softmax": (softmax, {1: "approximation", 2: "precision", 3: "reformulation"}),
        "layernorm": (layernorm, {1: "approximation", 2: "reformulation", 3: "precision"}),
        "matmul": (matmul, {1: "approximation_tf32", 2: "precision", 3: "order"}),
        "accumulate": (accumulate, {1: "rounding_down", 2: "rounding_up"}),
    }


def run(make, seed, mut, perturb=False, with_reference=False):
    import torch

    tensors, launch, shape = make(seed)
    tensors = dict(tensors)
    if perturb:  # about one FP32 ulp, random sign, on every float input tensor
        g = torch.Generator(device="cuda").manual_seed(seed + 10_000)
        for k, v in list(tensors.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                sign = torch.randint(0, 2, v.shape, device="cuda", generator=g).float() * 2 - 1
                tensors[k] = v * (1 + sign * 2.0 ** -24)
    tensors["Y"] = torch.empty(shape, device="cuda")
    if not with_reference:
        launch(tensors, mut)
        torch.cuda.synchronize()
        return tensors["Y"].double().cpu().numpy().reshape(-1), None
    recorder = TritonLaunchRecorder(select=lambda name, i: name.startswith("m_"))
    with recorder:
        launch(tensors, mut)
        torch.cuda.synchronize()
    launch_rec = recorder.launches[-1]
    result = KernelReferenceEvaluator(parse_ttir(launch_rec.asm["ttir"])).evaluate(launch_rec)
    out_ptr = tensors["Y"].untyped_storage().data_ptr()
    buf = result.buffers[out_ptr]  # the output buffer, matched by storage address
    m = buf.written
    if (buf.st[m] != 0).any():
        raise RuntimeError("reference not established on some outputs")
    return tensors["Y"].double().cpu().numpy().reshape(-1), (buf.lo[m], buf.hi[m])


def fixed_direction(lows, highs):
    direction = (0.5 * (lows[:N_CAL] + highs[:N_CAL])).mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return {"verdict": "UNRESOLVED_MEASUREMENT", "reason": "calibration direction is zero"}
    w = direction / norm
    l = np.array([np.minimum(lows[i] * w, highs[i] * w).sum() for i in range(N_CAL, N_CAL + N_CONF)])
    h = np.array([np.maximum(lows[i] * w, highs[i] * w).sum() for i in range(N_CAL, N_CAL + N_CONF)])
    return _summarize("x", "fixed_direction", l, h, 0.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    rows = []
    for name, (make, table) in specs().items():
        draws = range(1000, 1000 + N_CAL + N_CONF)
        base = {s: run(make, s, 0, with_reference=True) for s in draws}
        base_perturbed = {s: run(make, s, 0, perturb=True)[0] for s in draws}
        for mut, kind in [(0, "rerun")] + list(table.items()):
            mutant = {s: run(make, s, mut, with_reference=(mut != 0)) for s in draws}
            if mut == 0:
                mutant = {s: (run(make, s, 0)[0], base[s][1]) for s in draws}
            allclose_fail, exceed, rel, tol = 0, 0, [], []
            for s in draws:
                k0, k1 = base[s][0], mutant[s][0]
                allclose_fail += int(not torch.allclose(torch.from_numpy(k1), torch.from_numpy(k0)))
                denom = max(np.linalg.norm(k0), 1e-300)
                r = np.linalg.norm(k1 - k0) / denom
                t = np.linalg.norm(base_perturbed[s] - k0) / denom
                rel.append(r)
                tol.append(t)
                exceed += int(r > t)
            n_d = len(draws)
            diff = np.stack([mutant[s][0] - base[s][0] for s in draws])
            mean_mut_vs_orig = fixed_direction(diff, diff)
            mut_lo = np.stack([mutant[s][0] - mutant[s][1][1] for s in draws])
            mut_hi = np.stack([mutant[s][0] - mutant[s][1][0] for s in draws])
            org_lo = np.stack([base[s][0] - base[s][1][1] for s in draws])
            org_hi = np.stack([base[s][0] - base[s][1][0] for s in draws])
            mean_mut_vs_ref = fixed_direction(mut_lo, mut_hi)
            mean_org_vs_ref = fixed_direction(org_lo, org_hi)
            row = {"kernel": name, "mutation": kind, "draws": n_d,
                   "magnitude": {"allclose_fail_fraction": allclose_fail / n_d,
                                 "baseline_exceed_fraction": exceed / n_d,
                                 "relative_error_median": float(np.median(rel)),
                                 "threshold_median": float(np.median(tol)),
                                 "exceeds_tolerance": exceed / n_d >= 0.5},
                   "average_effect_mutant_vs_original": mean_mut_vs_orig,
                   "average_effect_mutant_vs_reference": mean_mut_vs_ref,
                   "average_effect_original_vs_reference": mean_org_vs_ref}
            rows.append(row)
            print(name, kind, "allclose_fail", allclose_fail / n_d, "exceed", exceed / n_d,
                  "mean(mut-orig)", mean_mut_vs_orig.get("verdict"), "mean(mut-KR)", mean_mut_vs_ref.get("verdict"),
                  flush=True)
    tested = [r for r in rows if "p_value_two_sided_conservative" in r["average_effect_mutant_vs_original"]]
    rejections = _holm([r["average_effect_mutant_vs_original"]["p_value_two_sided_conservative"] for r in tested], 0.05)
    for r, rej in zip(tested, rejections):
        r["average_effect_mutant_vs_original"]["holm_reject"] = bool(rej)
    table = {"exceeds_and_detected": [], "exceeds_not_confirmed": [], "within_and_detected": [],
             "within_not_confirmed": []}
    for r in rows:
        if r["mutation"] == "rerun":
            continue
        avg = r["average_effect_mutant_vs_original"]
        detected = avg.get("holm_reject", False) and avg.get("verdict", "").startswith("DETECTED")
        key = ("exceeds" if r["magnitude"]["exceeds_tolerance"] else "within") + \
              ("_and_detected" if detected else "_not_confirmed")
        table[key].append(f"{r['kernel']}:{r['mutation']}")
    payload = {"schema": "kernel-analyzer-method-comparison-v1", "measurement_point": "kernel output",
               "draws": N_CAL + N_CONF, "split": [N_CAL, N_CONF], "rows": rows, "two_by_two": table,
               "notes": ["magnitude baseline: TTrace-style relative Frobenius error with a threshold from "
                         "perturbing the original's inputs by about one FP32 ulp; not a TTrace reproduction",
                         "average effect detected = Holm-corrected endpoint-conservative t test on the "
                         "fixed-direction projection"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
