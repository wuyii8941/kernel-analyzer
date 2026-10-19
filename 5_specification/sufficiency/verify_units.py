#!/usr/bin/env python3
"""K−Gπ check of new captures for the sufficiency pipeline (docs/SUFFICIENCY_PROTOCOL.md section 3, step 2).

Runs only the tool's `emulate.verify` on the programs that three_way's `analyze_units.programs_for` selects, and writes
the units file that `corpus/bind_inductor.py --units` reads ({unit, kernel, verify}). The bias analyses of
analyze_units are not run. Each unit runs in its own process with a wall-clock budget; a timeout or an error is
recorded in the verify field.

    python 5_specification/sufficiency/verify_units.py OUT.jsonl CAPTURE_DIR [CAPTURE_DIR ...] [--workers 12] [--timeout 600]
    python 5_specification/sufficiency/verify_units.py --one UNIT_DIR      (internal)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def one(unit):
    sys.path.insert(0, str(REPO / "1_experiments/three_way"))
    sys.path.insert(0, str(REPO / "2_tool/src"))
    from analyze_units import programs_for
    from kernel_analyzer.reference_eval import emulate as E
    from kernel_analyzer.reference_eval.capture import load_launch
    L = load_launch(Path(unit))
    progs = programs_for(L)
    v = E.verify(L, ungrouped=[], swapped=[], programs=progs)
    return {"unit": unit, "kernel": L.kernel_name, "programs": len(progs), "verify": v["status"]}


def run(unit, timeout, gpu):
    try:
        p = subprocess.run([sys.executable, __file__, "--one", unit], capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
    except subprocess.TimeoutExpired:
        return {"unit": unit, "verify": "budget_exceeded"}
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode == 0 and lines:
        return json.loads(lines[-1])
    return {"unit": unit, "verify": f"error rc={p.returncode}: {p.stderr.strip().splitlines()[-1][:160] if p.stderr.strip() else ''}"}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print(json.dumps(one(sys.argv[2])))
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("capture_dirs", type=Path, nargs="+", help="directories containing <unit>/launch.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--gpus", default="0,1,2,3", help="emulate.verify replays launches on a GPU; round-robin per unit")
    a = ap.parse_args()
    units = sorted(str(p.parent) for d in a.capture_dirs for p in d.glob("*/launch.json"))
    if not units:
        raise SystemExit("no captured units")
    with open(a.out, "x") as fh, ThreadPoolExecutor(a.workers) as ex:
        gpus = itertools.cycle(a.gpus.split(","))
        for row in ex.map(lambda ug: run(ug[0], a.timeout, ug[1]), [(u, next(gpus)) for u in units]):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
