#!/usr/bin/env python3
"""Step 4: automatic K_R for the Liger FP32 dW accumulation vs the manual reference.

For every captured unit and chunk order, the 64 accumulate launches are
chained through one reference memory (composed reference).  The final
accumulator reference on the declared rows D is compared with the
hand-written FP64 accumulation of the same contributions:

    |manual - x| <= gamma_63 * sum_c |G_c|   for the exact sum x, and
    x in [lo, hi]                            (reference interval),

so the two agree iff the manual value lies in [lo, hi] widened by the manual
error bound.  The compact per-unit arrays (reference interval, actual
gradient, weights) are kept for step 5; the capture packages are deleted
after a successful analysis unless --keep is given.

Runs in the mainline environment (ka_main).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.capture import load_launch  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402


def analyze_variant(variant_dir: Path, rows: np.ndarray, block: int) -> dict:
    launch_dirs = sorted(p for p in variant_dir.iterdir() if p.name.startswith("launch"))
    launches = [load_launch(p) for p in launch_dirs]
    coverage = kernel_coverage(parse_ttir(launches[0].asm["ttir"]))
    t0 = time.time()
    sequence = evaluate_sequence(launches, programs_for=lambda launch: [(int(r), 0, 0) for r in rows])
    seconds = time.time() - t0
    acc_arg = next(a for a in launches[0].args if a.name == "ACC")
    buf = sequence.memory[acc_arg.storage_ptr]
    window = buf.index
    arrays = np.load(variant_dir / "arrays.npz")
    manual, abs_sum, grad = arrays["manual"], arrays["abs_sum"], arrays["grad"]
    order = np.argsort((rows[:, None] * block + np.arange(block)[None, :]).reshape(-1), kind="stable")
    # Window indices are storage-relative and sorted; the npz arrays follow the row-major window order.
    expected_window = (rows[:, None] * block + np.arange(block)[None, :]).reshape(-1)
    base = acc_arg.data_ptr - acc_arg.storage_ptr
    if not np.array_equal(window, np.sort(expected_window + base // 4)):
        raise RuntimeError("captured window does not match the declared rows")
    manual, abs_sum, grad = manual[order], abs_sum[order], grad[order]
    lo, hi, st, cond = buf.lo, buf.hi, buf.st, buf.cond
    actual = buf.actual_after
    if not np.array_equal(actual, grad):
        raise RuntimeError("final captured accumulator differs from the saved gradient")
    n_c = len(launches)
    bound = iv.up(abs_sum * iv.gamma(n_c) * (1 + 8 * iv.U))
    agree = (manual >= iv.down(lo - bound)) & (manual <= iv.up(hi + bound))
    r_lo = iv.add_bounds(actual, -hi)[0]
    r_hi = iv.add_bounds(actual, -lo)[1]
    manual_residual = actual - manual
    return {
        "variant": variant_dir.name,
        "launches": n_c,
        "coverage_complete": coverage["complete"],
        "external_inputs": len(sequence.external_writes),
        "accumulator_rewritten_outside": any(e["buffer"] == "ACC" for e in sequence.external_writes),
        "aborted_programs": sum(len(r.aborted) for r in sequence.launches),
        "classes": {
            "complete_composed": int(((st == 0) & ~cond).sum()),
            "conditional_local": int(((st == 0) & cond).sum()),
            "not_established": int((st >= 4).sum()),
        },
        "coordinates": int(lo.size),
        "reference_width_max": float(np.max(hi - lo)),
        "reference_width_zero_fraction": float(np.mean(hi == lo)),
        "manual_agrees": int(agree.sum()),
        "manual_disagrees": int((~agree).sum()),
        "manual_bound_max": float(bound.max()),
        "residual_positive": int((r_lo > 0).sum()),
        "residual_negative": int((r_hi < 0).sum()),
        "residual_contains_zero": int(((r_lo <= 0) & (r_hi >= 0)).sum()),
        "residual_sign_matches_manual": int(((r_lo > 0) & (manual_residual > 0)).sum()
                                            + ((r_hi < 0) & (manual_residual < 0)).sum()),
        "mean_residual_bounds": [float(np.mean(r_lo)), float(np.mean(r_hi))],
        "max_abs_residual": float(np.max(np.maximum(np.abs(r_lo), np.abs(r_hi)))),
        "evaluation_seconds": round(seconds, 2),
        "_arrays": {"ref_lo": lo, "ref_hi": hi, "grad": actual, "weight": arrays["weight"][order],
                    "manual": manual, "abs_sum": abs_sum, "window": window},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for compact per-unit results")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    design = json.loads((args.capture / "design.json").read_text())
    rows = np.asarray(design["rows"], dtype=np.int64)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "design.json").write_text(json.dumps(design) + "\n")
    for unit_dir in sorted(p for p in args.capture.iterdir() if p.name.startswith("unit")):
        summary = {"unit": unit_dir.name}
        arrays = {}
        for variant_dir in sorted(p for p in unit_dir.iterdir() if p.is_dir()):
            meta = json.loads((variant_dir / "unit.json").read_text())
            report = analyze_variant(variant_dir, rows, design["block"])
            arr = report.pop("_arrays")
            arrays.update({f"{variant_dir.name}_{k}": v for k, v in arr.items()})
            report.update({k: meta[k] for k in ("bank_index", "state_id", "bitwise_equal_to_torch_path")})
            summary[variant_dir.name] = report
            print(json.dumps({"unit": unit_dir.name, **{k: report[k] for k in (
                "variant", "classes", "manual_disagrees", "reference_width_max", "residual_positive",
                "residual_negative", "residual_contains_zero", "evaluation_seconds")}}), flush=True)
        np.savez_compressed(args.out / f"{unit_dir.name}.npz", **arrays)
        (args.out / f"{unit_dir.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
        if not args.keep:
            shutil.rmtree(unit_dir)


if __name__ == "__main__":
    main()
