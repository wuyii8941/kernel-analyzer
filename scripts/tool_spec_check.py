#!/usr/bin/env python3
"""Kernel against its specification with the tool: e_num = K - K_R and e_sem = K_R - f.

For each case the kernel call runs under ``TritonLaunchRecorder``; the automatic reference K_R of the measured
buffer comes from the captured TTIR (``evaluate_sequence``, composed over all launches of the call), and the
specification f is the documented math of the function the kernel implements or replaces, evaluated on the same
inputs as a rigorous float64 enclosure (``reference_eval.intervals``).  Both residuals enter as directed intervals
and go through the unified decision layer (``analysis.assess_units``: rules R1, R2, R3, R5, endpoint-conservative
inference on the confirmation seeds, default detector 2.1).  Seeds 0-31 are development, 32-95 confirmation.

Reading: e_num detected / e_sem not -> numerical (class 1); e_sem detected with |e_sem| far above the K_R width
and the rounding scale -> the kernel's own semantics differ from f (class 4).  The per-coordinate profile of
|e_sem| (``sem_profile``) shows where in the output the deviation sits.

    python scripts/tool_spec_check.py --group flex --case all --out results/tool_spec/flex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# the engine (cases, runner, report) lives in the package; this script is its CLI for the case groups
from kernel_analyzer.check import (CONF, DEV, RULES, Case, exact_mul, f64, f64_point_spec,  # noqa: E402,F401
                                   run, to_storage_order, torch_intermediates, verdicts)
from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402,F401  (used by case modules)


# ---------------------------------------------------------------------------------------------------------------
# Cases: Case subclasses live in tool_spec_cases_<group>.py; launch() returns {output name: tensor} and spec()
# returns {output name: (lo, hi)} in the tensor's logical shape.
# ---------------------------------------------------------------------------------------------------------------


GROUPS = {"liger": "tool_spec_cases_liger", "flex": "tool_spec_cases_flex", "inductor": "tool_spec_cases_inductor",
          "tridao": "tool_spec_cases_tridao", "fla": "tool_spec_cases_fla", "inductor2": "tool_spec_cases_inductor2",
          "inductor3": "tool_spec_cases_inductor3", "tutorials": "tool_spec_cases_triton_tutorials",
          "inductor4": "tool_spec_cases_inductor4", "vllm": "tool_spec_cases_vllm", "opinfo": "tool_spec_cases_opinfo", "optim2": "tool_spec_cases_optim2", "scatter1": "tool_spec_cases_scatter1", "vllm2": "tool_spec_cases_vllm2"}


def load_cases(group):
    import importlib

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return {c.name: c for c in importlib.import_module(GROUPS[group]).CASES}


# ---------------------------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    parser.add_argument("--case", required=True, help="case name, or 'all'")
    parser.add_argument("--out", type=Path, required=True, help="directory; one JSON per case")
    parser.add_argument("--seeds", type=int, default=96, help="development = first third")
    parser.add_argument("--zero-fill", default="auto", choices=("auto", "off"),
                        help="auto: take masked lanes without `other` as 0 when the PTX of every launch zero-fills them")
    args = parser.parse_args()
    import torch._inductor.config as inductor_config

    inductor_config.use_static_cuda_launcher = False  # Inductor's static launcher bypasses the launch hook
    cases = load_cases(args.group)
    names = sorted(cases) if args.case == "all" else args.case.split(",")
    n_dev = args.seeds // 3
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            report = run(cases[name], dev=range(0, n_dev), conf=range(n_dev, args.seeds), zero_fill_mode=args.zero_fill)
        except Exception as exc:  # noqa: BLE001
            import traceback

            report = {"case": name, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-1500:]}
            print(name, "ERROR", report["error"][:300], flush=True)
        (args.out / f"{name}.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
        for oname, entry in report.get("outputs", {}).items():
            sem, num = entry["semantic"], entry["numerical"]
            print(f"{name:34s} {oname:6s} complete={entry['reference_classes']['complete_fraction']:.2f} "
                  f"num={verdicts(num)} sem={verdicts(sem)} "
                  f"sem_rel_rms={sem.get('scale', {}).get('relative_rms', float('nan')):.1e} "
                  f"num_rel_rms={num.get('scale', {}).get('relative_rms', float('nan')):.1e}", flush=True)


if __name__ == "__main__":
    main()
