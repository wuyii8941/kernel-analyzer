#!/usr/bin/env python3
"""Blind bias detection for an operator the tool was not adapted to (kernel output level).

The user supplies only a binding module -- how to draw one unit's inputs and how to call the operator:

    # binding.py
    KERNELS = None                      # optional: names of the Triton kernels to measure (default: all)
    def make_inputs(seed: int) -> dict  # one independent draw from the input / state source
    def run(inputs: dict) -> None       # calls the operator (it may launch several Triton kernels)

No reference, mechanism, direction or suspect node is given.  For every unit the operator is run under the
launch recorder; for every measured launch the automatic reference K_R is evaluated (version-locked
TTIR semantics) on a fixed set of program instances, and e = K - RN(K_R) is formed on the written output
elements (and e = K - RN(K_R), beyond the final rounding, as a secondary definition).  The coordinate set
is fixed on the development units (complete references only); the default
detector (detect.py) then tests the vector mean and the alignment with the output.  Holm is applied over
all tests of all measured outputs within each family.  Kernel-internal emulation is not used: detection
does not depend on it.

    python scripts/run_detection.py --binding path/to/binding.py --units 128 --out report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.detect import ALPHA, detect  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import ST_OK, KernelReferenceEvaluator, TORCH_TO_ELEM  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

MAX_PROGRAMS = 16
# real: e = K - K_R (bias against the declared real semantics; the primary target)
# rounded: e = K - RN(K_R) (bias beyond the unavoidable rounding of the result to its storage format)
DEFINITIONS = ("real", "rounded")


def load_binding(path: Path):
    spec = importlib.util.spec_from_file_location("binding", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def program_subset(grid, count=MAX_PROGRAMS, seed=0):
    gx, gy, gz = grid
    total = gx * gy * gz
    if total <= count:
        return None
    rng = np.random.default_rng(seed)
    flat = sorted({0, total - 1} | set(rng.choice(total, size=count - 2, replace=False).tolist()))
    return [(i % gx, (i // gx) % gy, i // (gx * gy)) for i in flat]


def measure_launch(launch, programs):
    """Per written float output buffer: (global indices, e_lo, e_hi, actual, status ok, shape)."""

    module = parse_ttir(launch.asm["ttir"])
    result = KernelReferenceEvaluator(module).evaluate(launch, programs=programs)
    out = {}
    for ident, b in result.buffers.items():
        if b.kind != "f" or not b.written.any() or b.actual_after is None:
            continue
        arg = next(a for a in launch.args if a.kind == "tensor" and a.storage_ptr == ident)
        fmt = TORCH_TO_ELEM.get(arg.dtype)
        if fmt not in iv.FLOAT_FORMATS:
            continue
        m = b.written
        lo, _ = iv.round_nearest_even(b.lo[m], fmt)
        hi, _ = iv.round_nearest_even(b.hi[m], fmt)
        k = b.actual_after[m]
        ok = (b.st[m] == ST_OK) & ~b.cond[m] & (b.actual_after_st[m] == ST_OK if b.actual_after_st is not None else True)
        r_lo = iv.add_bounds(k, -b.hi[m])[0]  # K - K_R against the real reference
        r_hi = iv.add_bounds(k, -b.lo[m])[1]
        out[b.name] = {"index": b.global_indices()[m], "real": (r_lo, r_hi), "rounded": (k - hi, k - lo), "k": k,
                       "ok": ok, "shape": tuple(arg.shape) if arg.shape else None, "aborted": len(result.aborted)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--units", type=int, default=128)
    parser.add_argument("--development", type=int, default=None, help="default: half of the units")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch

    binding = load_binding(args.binding)
    wanted = getattr(binding, "KERNELS", None)
    n_dev = args.development or args.units // 2
    t_start = time.time()
    per_target = {}  # (launch position, kernel, buffer) -> list over units
    statuses, programs_of = {}, {}
    seconds = {"operator_and_capture": 0.0, "reference": 0.0}
    for unit in range(args.units):
        inputs = binding.make_inputs(args.seed + unit)
        t0 = time.time()
        rec = TritonLaunchRecorder(select=lambda name, i: wanted is None or name in wanted)
        with rec:
            binding.run(inputs)
            torch.cuda.synchronize()
        seconds["operator_and_capture"] += time.time() - t0
        for pos, launch in enumerate(rec.launches):
            key = (pos, launch.kernel_name)
            if key not in statuses:
                cov = kernel_coverage(parse_ttir(launch.asm["ttir"]))
                statuses[key] = {"coverage_complete": cov["complete"], "operations": cov["operations"],
                                 "rejected": [r["rejected"] for r in cov["rejected"]][:5], "grid": list(launch.grid)}
                programs_of[key] = program_subset(launch.grid)
            if not statuses[key]["coverage_complete"]:
                continue
            t0 = time.time()
            try:
                bufs = measure_launch(launch, programs_of[key])
            except Exception as exc:  # an unsupported construct during evaluation
                statuses[key].setdefault("evaluation_errors", []).append(f"{type(exc).__name__}: {str(exc)[:160]}")
                continue
            seconds["reference"] += time.time() - t0
            for name, data in bufs.items():
                per_target.setdefault((pos, launch.kernel_name, name), []).append(data)
        del rec
        print(f"unit {unit}: {len(per_target)} targets", flush=True)

    targets = []
    for (pos, kernel, buf), units in per_target.items():
        entry = {"launch_position": pos, "kernel": kernel, "buffer": buf, **statuses[(pos, kernel)],
                 "programs_evaluated": programs_of[(pos, kernel)] and len(programs_of[(pos, kernel)])}
        if len(units) != args.units:
            entry.update(verdict="CANNOT_JUDGE", reason=f"measured in {len(units)} of {args.units} units")
            targets.append(entry)
            continue
        same = all(np.array_equal(u["index"], units[0]["index"]) for u in units)
        if not same:
            entry.update(verdict="CANNOT_JUDGE", reason="written elements differ between units (no common coordinates)")
            targets.append(entry)
            continue
        valid = np.logical_and.reduce([u["ok"] for u in units[:n_dev]])
        conf_bad = [i for i in range(n_dev, args.units) if (valid & ~units[i]["ok"]).any()]
        entry["coordinates"] = {"written": int(valid.size), "complete_reference_on_development": int(valid.sum())}
        if not valid.any() or conf_bad:
            entry.update(verdict="CANNOT_JUDGE",
                         reason="no coordinate with a complete reference" if not valid.any() else
                         f"{len(conf_bad)} confirmation units with references not established on the coordinate set")
            targets.append(entry)
            continue
        k = np.stack([u["k"][valid] for u in units])
        shape = None
        if valid.all() and units[0]["shape"] and len(units[0]["shape"]) >= 2 and \
                int(np.prod(units[0]["shape"])) == valid.size:
            shp = units[0]["shape"]
            shape = (int(np.prod(shp[:-1])), int(shp[-1]))
        for definition in DEFINITIONS:
            e_lo = np.stack([u[definition][0][valid] for u in units])
            e_hi = np.stack([u[definition][1][valid] for u in units])
            mid = 0.5 * (e_lo + e_hi)
            spread = float(np.std(mid))
            entry.setdefault("reference_width_relative_to_residual_spread", {})[definition] = \
                float(np.max(e_hi - e_lo) / spread) if spread > 0 else None
            if spread == 0.0:
                entry.setdefault("residual_identically_zero", []).append(definition)
            entry.setdefault("detection", {})[definition] = detect(mid, k, n_dev, shape=shape, seed=args.seed)
        targets.append(entry)

    # Holm over all tests of all measured outputs, within each family and residual definition
    for definition, fam in [(d, f) for d in DEFINITIONS for f in ("vector_mean", "alignment")]:
        tests = [(t, row) for t in targets if "detection" in t for row in t["detection"][definition][fam]["tests"]]
        order = sorted(range(len(tests)), key=lambda i: tests[i][1]["p"])
        stop = False
        for rank, i in enumerate(order):
            rej = (not stop) and tests[i][1]["p"] <= ALPHA / (len(tests) - rank)
            stop = stop or not rej
            row = tests[i][1]
            supported = row.get("diagnostics", {}).get("tail_assumption_supported", True)
            row["holm_reject_over_targets"] = bool(rej)
            row["verdict"] = ("DETECTED" if supported else "EXPLORATORY_ONLY") if rej else "NOT_CONFIRMED"
        for t in targets:
            if "detection" in t:
                vs = [r["verdict"] for r in t["detection"][definition][fam]["tests"]]
                t["detection"][definition][fam]["verdict"] = (
                    "DETECTED" if "DETECTED" in vs else "EXPLORATORY_ONLY" if "EXPLORATORY_ONLY" in vs else "NOT_CONFIRMED")
    for t in targets:
        if "detection" in t:
            t["verdict"] = {d: {f: t["detection"][d][f]["verdict"] for f in ("vector_mean", "alignment")}
                            for d in DEFINITIONS}
    report = {"schema": "kernel-analyzer-blind-detection-v1", "binding": str(args.binding), "units": args.units,
              "development_units": n_dev, "seconds": {**{k: round(v, 1) for k, v in seconds.items()},
                                                      "total": round(time.time() - t_start, 1)},
              "launch_kinds": {f"{p}:{k}": v for (p, k), v in statuses.items()}, "targets": targets,
              "how_to_read": ["real: e = K - K_R (primary); rounded: e = K - RN(K_R), beyond the final rounding",
                              "vector_mean: E[e] != 0; alignment: <e, K>/|K| or e/K has a nonzero mean",
                              "NOT_CONFIRMED is not evidence of zero bias; CANNOT_JUDGE gives the reason",
                              "rare values absent from the sample cannot be diagnosed"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    for t in targets:
        print(t["launch_position"], t["kernel"][:40], t["buffer"], t.get("verdict"), t.get("reason", ""))


if __name__ == "__main__":
    main()
