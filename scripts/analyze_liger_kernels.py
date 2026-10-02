#!/usr/bin/env python3
"""Automatic K_R for captured Liger kernels vs hand-written FP64 formulas.

* liger_cross_entropy_kernel (one launch per chunk): the kernel overwrites the
  logits row with d loss / d logits and writes the row loss.  Manual FP64:
  grad = (softmax(x) - onehot(y)) / N, ignored rows give zeros.
* RMSNorm forward/backward: evaluated and compared with the device output.

Agreement: the manual value lies in the reference interval widened by a
generous bound on the manual FP64 error.  Runs in ka_main.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.capture import load_launch  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, decode_storage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402


def buffer(result, name):
    for buf in result.buffers.values():
        if buf.name == name:
            return buf
    raise KeyError(name)


def residual_signs(actual, lo, hi):
    r_lo = iv.add_bounds(actual, -hi)[0]
    r_hi = iv.add_bounds(actual, -lo)[1]
    return {"positive": int((r_lo > 0).sum()), "negative": int((r_hi < 0).sum()),
            "contains_zero": int(((r_lo <= 0) & (r_hi >= 0)).sum()),
            "max_abs": float(np.max(np.maximum(np.abs(r_lo), np.abs(r_hi)))) if r_lo.size else 0.0}


def ce_launch(launch, modules):
    module = modules.setdefault(launch.asm["ttir"], parse_ttir(launch.asm["ttir"]))
    t0 = time.time()
    result = KernelReferenceEvaluator(module).evaluate(launch)
    seconds = time.time() - t0
    args = {a.name: a for a in launch.args}
    x_arg, y_arg = args["X_ptr"], args["Y_ptr"]
    xbuf = buffer(result, "X_ptr")
    n_cols = int(args["n_cols"].value)
    stride_arg = args.get("X_stride")
    stride = int(stride_arg.value) if stride_arg is not None and stride_arg.value is not None else n_cols
    n_non_ignore = args["n_non_ignore"]
    n_valid = float(n_non_ignore.value) if n_non_ignore.kind in ("int", "float") else None
    x_before, _ = decode_storage(x_arg.before.numpy(), x_arg.dtype)
    y_vals, _ = decode_storage(y_arg.before.numpy(), y_arg.dtype)
    base = (x_arg.data_ptr - x_arg.storage_ptr) // 4
    rows = launch.grid[0]
    m = xbuf.written
    report = {"kernel": launch.kernel_name, "rows": rows, "seconds": round(seconds, 3),
              "aborted": list(result.aborted.values()),
              "classes": {"complete_composed": int(((xbuf.st == 0) & ~xbuf.cond & m).sum()),
                          "not_established": int(((xbuf.st >= 4) & m).sum())},
              "dlogits_residual": residual_signs(xbuf.actual_after[m], xbuf.lo[m], xbuf.hi[m])}
    agree = disagree = 0
    for r in range(rows):
        start = base + r * stride
        x = x_before[start:start + n_cols]
        target = int(y_vals[(y_arg.data_ptr - y_arg.storage_ptr) // y_arg.element_size + r])
        if target == -100:
            manual = np.zeros(n_cols)
        else:
            e = np.exp(x - x.max())
            manual = e / e.sum()
            manual[target] -= 1.0
            manual = manual / (n_valid if n_valid else 1.0)
        lo, hi = xbuf.lo[start:start + n_cols], xbuf.hi[start:start + n_cols]
        bound = 1e-13 * (np.abs(manual).max() + 1e-30)
        ok = (manual >= lo - bound) & (manual <= hi + bound)
        agree += int(ok.sum())
        disagree += int((~ok).sum())
    report["manual_fp64_agrees"] = agree
    report["manual_fp64_disagrees"] = disagree
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    modules, ce_reports, other = {}, [], {}
    for row in manifest["launches"]:
        launch = load_launch(args.capture / f"launch{row['launch']:03d}")
        coverage = kernel_coverage(parse_ttir(launch.asm["ttir"]))
        if launch.kernel_name == "liger_cross_entropy_kernel":
            rep = ce_launch(launch, modules)
            rep["coverage_complete"] = coverage["complete"]
            ce_reports.append(rep)
        else:
            t0 = time.time()
            result = KernelReferenceEvaluator(parse_ttir(launch.asm["ttir"])).evaluate(launch)
            other[launch.kernel_name] = {"coverage_complete": coverage["complete"],
                                         "operations": coverage["operations"],
                                         "seconds": round(time.time() - t0, 3),
                                         "aborted": list(result.aborted.values())[:2],
                                         "outputs": result.compare()}
    summary = {
        "launches": len(ce_reports),
        "coverage_complete": all(r["coverage_complete"] for r in ce_reports),
        "aborted": sum(len(r["aborted"]) for r in ce_reports),
        "manual_fp64_agrees": sum(r["manual_fp64_agrees"] for r in ce_reports),
        "manual_fp64_disagrees": sum(r["manual_fp64_disagrees"] for r in ce_reports),
        "dlogits_residual_positive": sum(r["dlogits_residual"]["positive"] for r in ce_reports),
        "dlogits_residual_negative": sum(r["dlogits_residual"]["negative"] for r in ce_reports),
        "dlogits_residual_contains_zero": sum(r["dlogits_residual"]["contains_zero"] for r in ce_reports),
        "not_established": sum(r["classes"]["not_established"] for r in ce_reports),
        "seconds": round(sum(r["seconds"] for r in ce_reports), 2),
    }
    payload = {"schema": "kernel-analyzer-liger-kernels-auto-reference-v1", "manifest": manifest,
               "cross_entropy_summary": summary, "cross_entropy_launches": ce_reports, "other_kernels": other}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=1))
    for name, rep in other.items():
        print(name, rep["coverage_complete"], rep["aborted"],
              {k: {kk: v[kk] for kk in ("classes", "residual_positive", "residual_negative",
                                        "residual_contains_zero") if kk in v} for k, v in rep["outputs"].items()})


if __name__ == "__main__":
    main()
