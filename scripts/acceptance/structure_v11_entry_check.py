#!/usr/bin/env python3
"""Check that the adapter reproduces the frozen entry: run the frozen harness (scripts/general/acceptance_run.py
run_case -> measure.run + its baseline) on a few v1.1 programs, written in the harness's case layout, and save its
report next to the run.  structure_v11_report.py compares it with the adapter's main-round mode-B jobs.

    python scripts/acceptance/structure_v11_entry_check.py --run-id ID --program prog_01 prog_03

The declaration's single input is a state source that returns the package's input dict for the seed (its shape and
dtype fields are placeholders the declaration format requires; the state function ignores them).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import structure_v11 as S  # noqa: E402
import acceptance_run  # noqa: E402  (frozen harness; path added by structure_v11)

LOADER = '''# Generated for the entry check (adapter), not a detector or a reference.
import importlib.util
_s = importlib.util.spec_from_file_location("entry_check_{pid}", "{binding}")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
'''


def write_case(case_dir: Path, pid: str, out_names):
    case_dir.mkdir(parents=True, exist_ok=True)
    head = LOADER.format(pid=pid, binding=S.BINDINGS / f"{pid}_modeB.py")
    (case_dir / "decl_binding.py").write_text(head + '''
def state(seed, shape, dtype):
    return _m.make_inputs(seed)


def call(inputs):
    return _m.run(inputs["inp"])


def spec(inputs):
    return _m.spec(inputs["inp"])
''')
    (case_dir / "ordinary_reference.py").write_text(head + f'''
def reference(inputs):
    return _m.bs.ordinary_reference("{pid}", inputs["inp"])
''')
    decl = {"call": "decl_binding.py:call",
            "inputs": {"inp": {"state": "decl_binding.py:state", "shape": [1], "dtype": "float32"}},
            "compare": {"mode": "B", "measure": list(out_names), "spec": "decl_binding.py:spec"},
            "budget": {"cpu_seconds": 3600, "gpu_seconds": 3600, "case_timeout": 3600, "max_units": 96}}
    (case_dir / "declaration.json").write_text(json.dumps(decl, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--program", nargs="+", required=True)
    a = ap.parse_args()
    man = S.manifest()
    out = S.RESULTS / a.run_id / "entry_equivalence"
    out.mkdir(parents=True, exist_ok=True)
    fz = acceptance_run.frozen_check()
    for pid in a.program:
        case_dir = S.RUNS_CACHE / a.run_id / "entry_check_cases" / pid
        write_case(case_dir, pid, man[pid]["output_names"])
        res = acceptance_run.run_case(case_dir) if fz["ok"] else {"status": "refused", "frozen": fz}
        res["frozen"] = fz
        res["case_files"] = {p.name: p.read_text() for p in sorted(case_dir.glob("*")) if p.is_file()}
        (out / f"{pid}.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
        print(pid, res.get("status"), res.get("seconds"), flush=True)


if __name__ == "__main__":
    main()
