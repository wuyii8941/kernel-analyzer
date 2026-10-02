#!/usr/bin/env python3
"""Second Triton version: references stay correct and differences are explained.

--capture (run in each environment) captures the exercise workloads of
validate_ttir_reference.py with fixed seeds and saves the packages under
<out>/<triton version>/.  --analyze (ka_main) evaluates both versions with the
same mapping table and reports per kernel: coverage, rounding-check bitwise
reproduction, whether the two versions' references are identical (same
declared computation), whether the device values are identical, and the
version identity

    E[K2 - G2] - E[K1 - G1] = E[K2 - K1] - E[G2 - G1].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def capture(out: Path):
    import torch
    import triton

    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch
    from scripts.validate_ttir_reference import workloads

    target = out / triton.__version__
    target.mkdir(parents=True, exist_ok=True)
    done = {}
    for name, fn in workloads():
        try:
            recorder = TritonLaunchRecorder()
            with recorder:
                fn()
                torch.cuda.synchronize()
            save_launch(recorder.launches[-1], target / name)
            done[name] = "captured"
        except Exception as exc:
            done[name] = f"failed: {type(exc).__name__}: {str(exc)[:160]}"
        print(name, done[name], flush=True)
    (target / "manifest.json").write_text(json.dumps({"triton": triton.__version__, "torch": torch.__version__,
                                                      "kernels": done}, indent=2) + "\n")


def analyze(out: Path, report_path: Path):
    from kernel_analyzer.reference_eval.capture import load_launch
    from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, NumericMode
    from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    versions = sorted(p.name for p in out.iterdir() if p.is_dir())
    if len(versions) != 2:
        raise ValueError(f"expected two version directories, found {versions}")
    v1, v2 = versions
    rows = []
    for kdir in sorted((out / v1).iterdir()):
        if not kdir.is_dir() or not (out / v2 / kdir.name).exists():
            continue
        entry = {"kernel": kdir.name}
        evals = {}
        for v in (v1, v2):
            launch = load_launch(out / v / kdir.name)
            module = parse_ttir(launch.asm["ttir"])
            cov = kernel_coverage(module)
            nd = KernelReferenceEvaluator(module).evaluate(launch)
            rc = KernelReferenceEvaluator(module, mode=NumericMode.ROUNDING_CHECK).evaluate(launch)
            evals[v] = (launch, nd, rc)
            entry[v] = {"coverage_complete": cov["complete"], "nd_aborted": len(nd.aborted),
                        "rc_aborted": len(rc.aborted), "libtriton_sha256": launch.libtriton_sha256[:12],
                        "ttir_ops": cov["operations"]}
        per_buffer = {}
        for (ident1, b1) in evals[v1][1].buffers.items():
            if not b1.written.any():
                continue
            b2 = next((b for b in evals[v2][1].buffers.values() if b.name == b1.name), None)
            if b2 is None or b2.written.sum() != b1.written.sum():
                continue
            m1, m2 = b1.written, b2.written
            ok = (b1.st[m1] == 0) & (b2.st[m2] == 0)
            k1, k2 = b1.actual_after[m1], b2.actual_after[m2]
            info = {"elements": int(m1.sum()), "values_identical": bool(np.array_equal(k1, k2)),
                    "values_differ": int((k1 != k2).sum())}
            if b1.kind == "f":
                info["references_identical"] = bool(np.array_equal(b1.lo[m1], b2.lo[m2]) and np.array_equal(b1.hi[m1], b2.hi[m2]))
                g1 = 0.5 * (b1.lo[m1] + b1.hi[m1])
                g2 = 0.5 * (b2.lo[m2] + b2.hi[m2])
                # E[K2 - G2] - E[K1 - G1] = E[K2 - K1] - E[G2 - G1]  (finite elements)
                info["version_identity"] = {
                    "lhs": float(np.mean((k2 - g2)[ok]) - np.mean((k1 - g1)[ok])) if ok.any() else None,
                    "E_K2_minus_K1": float(np.mean((k2 - k1)[ok])) if ok.any() else None,
                    "E_G2_minus_G1": float(np.mean((g2 - g1)[ok])) if ok.any() else None,
                }
                for v, (launch, nd, rc) in evals.items():
                    rb = next((b for b in rc.buffers.values() if b.name == b1.name), None)
                    if rb is not None and rb.written.any():
                        mm = rb.written & (rb.st == 0)
                        info[f"rc_bitwise_{v}"] = bool(np.array_equal(rb.lo[mm], rb.actual_after[mm])
                                                       and np.array_equal(rb.hi[mm], rb.actual_after[mm]))
            per_buffer[b1.name] = info
        entry["buffers"] = per_buffer
        rows.append(entry)
        print(kdir.name, json.dumps({b: {k: v for k, v in i.items() if k != "version_identity"} for b, i in per_buffer.items()})[:300])
    report = {"schema": "kernel-analyzer-version-migration-v1", "versions": [v1, v2], "rows": rows}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.capture:
        capture(args.out)
    if args.analyze:
        analyze(args.out, args.report)


if __name__ == "__main__":
    main()
