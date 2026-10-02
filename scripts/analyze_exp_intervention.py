#!/usr/bin/env python3
"""Exp intervention on the Liger cross-entropy kernel (analysis, ka_main).

Fixed reference: K_R of the original kernel (declared semantics: real exp).
For every variant the residual K_variant - K_R(original) is classified by
sign over all dlogits.  The residual against each variant's own K_R is also
reported: for the exp2 rewrites the rounded constant is part of the declared
expression, so it moves from K - K_R into K_R - f.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.capture import load_launch  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, decode_storage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

PREDICTION = {
    "original": "positive skew as originally observed",
    "exp2_rn": "same as original if math.exp lowers to ex2.approx(x * RN32(log2 e))",
    "exp2_up": "skew reverses (constant above log2 e makes exp of negative arguments too small)",
    "libdevice": "skew largely removed (compensated reduction)",
    "div_rn": "isolates the approximate division (exp unchanged)",
    "libdevice_div_rn": "both exp and division replaced",
}


def x_buffer(result):
    return next(b for b in result.buffers.values() if b.name == "X_ptr")


def signs(actual, lo, hi):
    r_lo = iv.add_bounds(actual, -hi)[0]
    r_hi = iv.add_bounds(actual, -lo)[1]
    pos, neg = int((r_lo > 0).sum()), int((r_hi < 0).sum())
    zero = int(actual.size - pos - neg)
    mid = 0.5 * (r_lo + r_hi)
    return {"positive": pos, "negative": neg, "contains_zero": zero,
            "positive_fraction_of_signed": pos / max(pos + neg, 1),
            "mean_residual": float(mid.mean()), "mean_abs_residual": float(np.abs(mid).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="capture of the unmodified kernel (bitwise control)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    variants = list(manifest["variants"])
    n_launch = manifest["variants"]["original"]["launches"]
    modules = {}

    def evaluate(launch):
        module = modules.setdefault(launch.asm["ttir"], parse_ttir(launch.asm["ttir"]))
        return KernelReferenceEvaluator(module).evaluate(launch)

    acc = {v: {"fixed": [], "own": []} for v in variants}
    bitwise_control = []
    for j in range(n_launch):
        ref = evaluate(load_launch(args.capture / "original" / f"launch{j:03d}"))
        rb = x_buffer(ref)
        m = rb.written
        for v in variants:
            launch = load_launch(args.capture / v / f"launch{j:03d}")
            x_arg = next(a for a in launch.args if a.name == "X_ptr")
            actual, _ = decode_storage(x_arg.after.numpy(), x_arg.dtype)
            acc[v]["fixed"].append((actual[m], rb.lo[m], rb.hi[m]))
            own = x_buffer(evaluate(launch)) if v != "original" else rb
            acc[v]["own"].append((actual[m], own.lo[m], own.hi[m]))
            if v == "original" and args.baseline is not None:
                base = load_launch(args.baseline / f"launch{j:03d}")
                b_arg = next(a for a in base.args if a.name == "X_ptr")
                bitwise_control.append(bool(np.array_equal(b_arg.after.numpy(), x_arg.after.numpy())))
    report = {"schema": "kernel-analyzer-exp-intervention-v1", "kernel": "liger_cross_entropy_kernel",
              "fixed_reference": "K_R of the original kernel (real exp)", "log2e_rn": manifest["log2e_rn"],
              "log2e_up": manifest["log2e_up"], "launches": n_launch,
              "same_inputs": {v: manifest["variants"][v]["same_logits_as_first_variant"] for v in variants},
              "original_bitwise_equal_to_unmodified_kernel": all(bitwise_control) if bitwise_control else None,
              "variants": {}}
    for v in variants:
        cat = lambda key: [np.concatenate([t[i] for t in acc[v][key]]) for i in range(3)]  # noqa: E731
        report["variants"][v] = {"prediction": PREDICTION[v],
                                 "against_original_reference": signs(*cat("fixed")),
                                 "against_own_declared_reference": signs(*cat("own"))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for v, r in report["variants"].items():
        f, o = r["against_original_reference"], r["against_own_declared_reference"]
        print(f"{v:10s} fixed: +{f['positive']:>8d} -{f['negative']:>8d} 0:{f['contains_zero']:>6d} "
              f"frac+={f['positive_fraction_of_signed']:.3f} mean={f['mean_residual']:.3e} | own: "
              f"frac+={o['positive_fraction_of_signed']:.3f} mean={o['mean_residual']:.3e}")
    print("original bitwise equal to unmodified kernel:", report["original_bitwise_equal_to_unmodified_kernel"])


if __name__ == "__main__":
    main()
