#!/usr/bin/env python3
"""Captures for the reviewer's independent check (blind_test_v1 protocol v1.1, section 5).

For every program with a detection in phase 1 and for seeds 0 and 32: the input tensors and runtime
scalars, the output K (raw bytes and float64), the tool's K_R interval, the residual interval and the
endpoint-conservative projections [l, h] under R1-R5 for that seed (R5 with the direction learned on
seeds 0-31, saved as well).  The reviewer recomputes K_R with an independent implementation and compares.

    python scripts/blind_test_v1_export.py --package .cache/blind/blind_test_v1 \
        --report results/reference_eval/blind_test_v1/phase1_report.json --out .cache/blind_v1_verification
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402
from scripts.run_blind_test_v1 import direction  # noqa: E402


def evaluate(mod, inputs, seed):
    import torch

    inp = inputs.make_inputs(mod.FAMILY, seed)
    rec = TritonLaunchRecorder()
    with rec:
        y = mod.launch(inp)
        torch.cuda.synchronize()
    launch = rec.launches[0]
    res = KernelReferenceEvaluator(parse_ttir(launch.asm["ttir"])).evaluate(launch)
    b = res.buffers[y.untyped_storage().data_ptr()]
    m = b.written
    k = b.actual_after[m]
    return inp, y, k, b.lo[m], b.hi[m], b.global_indices()[m]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    sys.path.insert(0, str(args.package / "programs"))
    inputs = importlib.import_module("inputs")
    report = json.loads(args.report.read_text())
    detected = sorted(p for p, r in report.items() for o in r.get("outputs", []) if isinstance(o.get("rules"), dict)
                      and any(str(v.get("final_verdict", "")).startswith("DETECTED") for v in o["rules"].values()
                              if isinstance(v, dict)))
    for pid in detected:
        mod = importlib.import_module(pid)
        family = mod.FAMILY
        outputs = [o["output"] for o in report[pid]["outputs"]]
        # R5 directions, learned on seeds 0-31 exactly as in phase 1
        dev = [evaluate(mod, inputs, s) for s in range(32)]
        index = dev[0][5]
        sels = {"output": np.ones(index.size, dtype=bool)} if family != "F3" else \
            {"mean": (index % 2) == 0, "rstd": (index % 2) == 1}
        r5 = {}
        for name in outputs:
            sel = sels[name]
            mid = np.stack([d[2][sel] - 0.5 * (d[3][sel] + d[4][sel]) for d in dev])
            w = mid.mean(axis=0)
            r5[name] = w / np.linalg.norm(w)
        target = args.out / pid
        target.mkdir(parents=True, exist_ok=True)
        for name, w in r5.items():
            np.save(target / f"R5_direction_{name}.npy", w)
        for seed in (0, 32):
            inp, y, k, lo, hi, idx = evaluate(mod, inputs, seed)
            sd = target / f"seed{seed:03d}"
            sd.mkdir(exist_ok=True)
            tensors = {key: v.detach().cpu().numpy() for key, v in inp.items() if torch.is_tensor(v)}
            np.savez_compressed(sd / "inputs.npz", **tensors)
            (sd / "scalars.json").write_text(json.dumps({key: v for key, v in inp.items() if not torch.is_tensor(v)}))
            np.save(sd / "K.npy", y.detach().cpu().numpy())
            (sd / "K_raw.bin").write_bytes(y.detach().cpu().contiguous().numpy().tobytes())
            np.save(sd / "KR_lo.npy", lo)
            np.save(sd / "KR_hi.npy", hi)
            np.save(sd / "output_flat_index.npy", idx)
            e_lo = iv.add_bounds(k, -hi)[0]
            e_hi = iv.add_bounds(k, -lo)[1]
            proj = {}
            for name in outputs:
                sel = sels[name]
                kr = 0.5 * (lo + hi)[sel]
                rules = ["R1", "R2", "R3"] + (["R4"] if family == "F4" else [])
                proj[name] = {}
                for rule in rules + ["R5"]:
                    wv = r5[name] if rule == "R5" else direction(rule, kr, family, tuple(y.shape))
                    a, b = e_lo[sel] * wv, e_hi[sel] * wv
                    proj[name][rule] = [float(np.minimum(a, b).sum()), float(np.maximum(a, b).sum())]
            (sd / "tool_values.json").write_text(json.dumps({
                "program": pid, "family": family, "seed": seed,
                "residual": {"definition": "e = K - K_R as the interval [K - hi, K - lo]",
                             "mean_mid": float((0.5 * (e_lo + e_hi)).mean()),
                             "max_width": float((e_hi - e_lo).max())},
                "projections": proj,
                "directions": "R1 -1/sqrt(n); R2 -sign(K_R)/sqrt(n); R3 -K_R/|K_R|; R4 F4 pattern; R5 saved .npy",
                "kr_and_k_files": "KR_lo.npy / KR_hi.npy / K.npy over output_flat_index.npy (row-major)"}, indent=2))
        print(pid, "exported", flush=True)


if __name__ == "__main__":
    main()
