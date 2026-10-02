#!/usr/bin/env python3
"""Automatic K_R for captured torchao AdamW8bit step kernels (TorchInductor Triton).

Every captured launch is evaluated over all program instances in both
reference modes.  The 8-bit codes are compared exactly; codes whose
real-number comparison is undecided by the reference interval are reported as
not established.  Runs in ka_main.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.capture import load_launch  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, NumericMode  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    rows = []
    for row in manifest["launches"]:
        launch = load_launch(args.capture / f"launch{row['launch']:03d}")
        module = parse_ttir(launch.asm["ttir"])
        coverage = kernel_coverage(module)
        entry = {**row, "operations": coverage["operations"], "categories": coverage["categories"],
                 "coverage_complete": coverage["complete"]}
        for mode in (NumericMode.NUMERICAL_DIFFERENCE, NumericMode.ROUNDING_CHECK):
            t0 = time.time()
            result = KernelReferenceEvaluator(module, mode=mode).evaluate(launch)
            entry[mode] = {"seconds": round(time.time() - t0, 2), "aborted": len(result.aborted),
                           "outputs": result.compare(),
                           "reasons": dict(sorted(result.reasons.items())[:10])}
        rows.append(entry)
        nd = entry[NumericMode.NUMERICAL_DIFFERENCE]["outputs"]
        print(json.dumps({"launch": row["launch"], "step": row["step"],
                          "outputs": {k: {kk: v[kk] for kk in ("classes", "integer_mismatches", "residual_positive",
                                                               "residual_negative") if kk in v}
                                      for k, v in nd.items()}}))
    payload = {"schema": "kernel-analyzer-torchao-adamw8bit-auto-reference-v1", "manifest": manifest,
               "launches": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
