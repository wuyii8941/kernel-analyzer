#!/usr/bin/env python3
"""Run the tool on operators captured from the end-to-end training run (ka_main).

For every capture package written by run_e2e_training_capture.py:

1. coverage of its TTIR by the version-locked mapping table;
2. the automatic reference K_R (numerical-difference mode) on a sample of program instances: classes of
   the written elements and the residual K - K_R;
3. bitwise emulation of the device execution on the same programs, rules and choices frozen (nothing is
   fitted to these outputs; if an output does not reproduce, a fitted search is run and reported apart);
4. for the cross-entropy kernel: node contributions of the division by n_non_ignore, the division by the
   row sum, the second-pass exp and the bf16 store (model counterfactual, float64 substitution), and the
   deployable change "division by n_non_ignore correctly rounded" (substitution rn);
5. the hardware-reciprocal factor rcp(N) * N - 1 over the N of every training step.

    python scripts/analyze_e2e_captures.py --captures .cache/e2e --log results/reference_eval/e2e/training_log.json \
        --out results/reference_eval/e2e/operator_tests.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.reference_eval.capture import load_launch  # noqa: E402
from kernel_analyzer.reference_eval.emulate import HardwareOracle, localize, verify  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

PROGRAMS = {"liger_cross_entropy_kernel": 8}
DEFAULT_PROGRAMS = 16


def sample_programs(grid, count, seed=0):
    gx, gy, gz = grid
    total = gx * gy * gz
    if total <= count:
        return None
    rng = np.random.default_rng(seed)
    flat = sorted({0, total - 1} | set(rng.choice(total, size=count - 2, replace=False).tolist()))
    return [(i % gx, (i // gx) % gy, i // (gx * gy)) for i in flat]


def ce_nodes(module) -> dict:
    defs = {r: o for o in module.entry().walk() for r in o.results}

    def source(v):
        o = defs.get(v)
        while o is not None and o.name in ("tt.splat", "tt.broadcast"):
            o = defs.get(o.operands[0])
        return o

    nodes = {}
    for o in module.entry().walk():
        if o.name == "arith.divf" and o.result_types[0].is_tensor:
            den, num = source(o.operands[1]), source(o.operands[0])
            if den is not None and den.name == "arith.sitofp" and den.operands[0] == "%n_non_ignore":
                nodes["div_by_N"] = o.node_id
            elif num is not None and num.name == "math.exp":
                nodes["div_by_row_sum"] = o.node_id
                nodes["exp_second_pass"] = num.node_id
        if o.name == "arith.truncf" and o.result_types[0].is_tensor:
            nodes["store_rounding_bf16"] = o.node_id
    return nodes


def reference_summary(module, launch, programs):
    t0 = time.time()
    result = KernelReferenceEvaluator(module).evaluate(launch, programs=programs)
    out = {"seconds": round(time.time() - t0, 2), "aborted_programs": len(result.aborted),
           "abort_reasons": sorted(set(result.aborted.values()))[:2], "buffers": {}}
    for name, v in result.compare().items():
        out["buffers"][name] = {k: v[k] for k in ("classes", "residual_positive", "residual_negative",
                                                  "residual_contains_zero", "max_reference_width", "max_abs_residual")
                                if k in v}
    return out


def hardware_factors(ns):
    import torch

    from scripts.reference_eval_kernels import hardware_reciprocal

    n = torch.as_tensor(np.asarray(ns), device="cuda", dtype=torch.int32)
    r = torch.empty(n.numel(), device="cuda")
    hardware_reciprocal[((n.numel() + 1023) // 1024,)](n, r, n.numel(), BLOCK=1024)
    rcp = r.cpu().numpy()
    d_rcp = np.array([float(Fraction(float(x)) * int(k) - 1) for x, k in zip(rcp, ns)])
    d_rn = np.array([float(Fraction(float(np.float32(1) / np.float32(k))) * int(k) - 1) for k in ns])
    return d_rcp, d_rn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    oracle = HardwareOracle(options={"num_warps": 4, "enable_fp_fusion": True})
    rows = []
    for step_dir in sorted(p for p in args.captures.iterdir() if p.name.startswith("step")):
        batch = json.loads((step_dir / "batch.json").read_text())
        for pkg in sorted(p for p in step_dir.iterdir() if (p / "launch.json").exists()):
            launch = load_launch(pkg)
            module = parse_ttir(launch.asm["ttir"])
            cov = kernel_coverage(module)
            programs = sample_programs(launch.grid, PROGRAMS.get(launch.kernel_name, DEFAULT_PROGRAMS))
            row = {"step": batch["step"], "package": pkg.name, "kernel": launch.kernel_name,
                   "grid": list(launch.grid), "programs_evaluated": len(programs) if programs else
                   launch.grid[0] * launch.grid[1] * launch.grid[2],
                   "coverage_complete": cov["complete"], "operations": cov["operations"],
                   "rejected": [r["rejected"] for r in cov["rejected"]][:3]}
            if cov["complete"]:
                row["reference"] = reference_summary(module, launch, programs)
                t0 = time.time()
                v = verify(launch, oracle=oracle, ungrouped=[], swapped=[], programs=programs)
                row["emulation"] = {k: v[k] for k in ("status", "buffers", "nodes", "not_emulable_nodes",
                                                     "aborted_programs", "ttir_sha256")}
                row["emulation"]["seconds"] = round(time.time() - t0, 2)
                if v["status"] == "mismatch":
                    fitted = verify(launch, oracle=oracle, programs=programs)
                    row["emulation_fitted_to_output"] = {"status": fitted["status"],
                                                         "lowering_choices": fitted["lowering_choices"],
                                                         "buffers": fitted["buffers"]}
                if launch.kernel_name == "liger_cross_entropy_kernel" and v["status"] == "bit_identical":
                    nodes = ce_nodes(module)
                    n_value = next(a.value for a in launch.args if a.name == "n_non_ignore")
                    loc = localize(launch, nodes=list(nodes.values()), oracle=oracle, programs=programs)
                    rn = localize(launch, nodes=[nodes["div_by_N"]], oracle=oracle, programs=programs,
                                  substitution="rn")
                    names = {v_: k for k, v_ in nodes.items()}
                    row["n_non_ignore"] = n_value
                    row["localization"] = {names[r["node"]]: r["buffers"].get("X_ptr") for r in loc["nodes"]}
                    row["div_by_N_correctly_rounded"] = rn["nodes"][0]["buffers"].get("X_ptr")
            rows.append(row)
            print(batch["step"], pkg.name[:50], row["coverage_complete"],
                  row.get("emulation", {}).get("status"), (row.get("reference") or {}).get("seconds"), flush=True)
    log = json.loads(args.log.read_text())
    ns = [s["n_non_ignore"] for s in log["steps"]]
    d_rcp, d_rn = hardware_factors(ns)
    per_kernel = collections.defaultdict(collections.Counter)
    for r in rows:
        k = per_kernel[r["kernel"]]
        k["packages"] += 1
        k["coverage_complete"] += int(r["coverage_complete"])
        if "emulation" in r:
            k["status:" + r["emulation"]["status"]] += 1
            for b in r["emulation"]["buffers"].values():
                k["emulated"] += b["emulated"]
                k["bit_identical"] += b["bit_identical"]
                k["mismatch"] += b["mismatch"]
        if "reference" in r:
            for b in r["reference"]["buffers"].values():
                for c, n in b.get("classes", {}).items():
                    k["reference:" + c] += n
    report = {"schema": "kernel-analyzer-e2e-operator-tests-v1", "training": {k: v for k, v in log.items()
                                                                             if k != "steps"},
              "training_steps": len(log["steps"]),
              "loss_first_last_10": [float(np.mean([s["loss"] for s in log["steps"][:10]])),
                                     float(np.mean([s["loss"] for s in log["steps"][-10:]]))],
              "n_non_ignore": {"min": int(min(ns)), "max": int(max(ns)), "mean": float(np.mean(ns))},
              "hardware_reciprocal_factor": {"mean": float(d_rcp.mean()), "positive_fraction": float((d_rcp > 0).mean()),
                                             "zero_fraction": float((d_rcp == 0).mean()),
                                             "mean_abs": float(np.abs(d_rcp).mean())},
              "rn32_reciprocal_factor": {"mean": float(d_rn.mean()), "positive_fraction": float((d_rn > 0).mean())},
              "per_kernel": {k: dict(v) for k, v in per_kernel.items()}, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("n_non_ignore", "hardware_reciprocal_factor", "rn32_reciprocal_factor")},
                     indent=1))
    print(json.dumps(report["per_kernel"], indent=1))


if __name__ == "__main__":
    main()
