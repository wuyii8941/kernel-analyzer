#!/usr/bin/env python3
"""Regression of the structure acceptance v1.1 programs under the dsl-v2 branch (tool 4.0, unreleased).

The 33 programs are a development / regression set from now on (rc3 03 7.5): these are not blind results.  Each
program runs once through ``check.run`` (main-round seeds: development 0-31, confirmation 32-95, mode B, repeated
launches per rc3 02 8.7) and is compared with its general-v3.1 main-round mode-B job of run r20261008T2030.

    python scripts/dsl_v2/regression_v11.py --run-id ID --all --workers 16
    python scripts/dsl_v2/regression_v11.py --run-id ID --program prog_17          # one program (worker)

Expectations were registered before the run in docs/dsl_v2/increment_01.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "acceptance"))
from kernel_analyzer import check, measure  # noqa: E402

import structure_v11 as S  # noqa: E402  (manifest and binding loader only)

# the frozen v1.1 run, extracted from 1_experiments/structure_acceptance_v1_1/structure_v1_1_rc1_r20261008T2030.tar.gz
V31 = ROOT.parent / ".cache/acceptance/structure_v1_1_rc1/r20261008T2030/jobs"
OUT = ROOT.parent / "1_experiments/dsl_v2/regression_v11"
DEV, CONF = list(range(0, 32)), list(range(32, 96))
PY = "/data1/tzh/envs/ka_main/bin/python"


def _canon(x):
    return json.dumps(x, sort_keys=True, default=str)


def compare_v31(pid, outputs):
    p = V31 / f"main_B_{pid}.json"
    if not p.exists():
        return {"status": "no v3.1 job"}
    old = json.loads(p.read_text())
    rows = {}
    for name, o in outputs.items():
        oo = old.get("outputs", {}).get(name, {})
        r = {"v31_status": oo.get("status"), "v40_status": o["status"]}
        if oo.get("status") == "evaluated" and o["status"] == "evaluated":
            r["reference_equal"] = _canon(oo["reference"]) == _canon(o["reference"])
            r["k_first_launch_equal"] = oo.get("k_sha256_per_unit") == o.get("k_sha256_per_unit")
            new_rec = {k: v for k, v in (o.get("numerical") or {}).items() if k != "scale"}
            old_rec = {k: v for k, v in (oo["comparisons"]["FR_e_num"]["record"] or {}).items() if k != "scale"}
            r["e_num_record_equal"] = _canon(new_rec) == _canon(old_rec)
            r["e_num_summary_equal"] = all(
                oo["comparisons"]["FR_e_num"]["class_statistics_protocol"][c]["summary"] == o["e_num_classes"][c]["summary"]
                for c in S.RULE_CLASSES)
        rows[name] = r
    return rows


def run_one(pid, run_id):
    e = S.manifest()[pid]
    mod, bpath = S.load_binding(pid, "B")
    keep = {}
    job = {"schema": "dsl-v2-regression-v11-v1", "run_id": run_id, "program": pid, "family": e["family"],
           "tool_version": check.TOOL_VERSION, "binding_sha256": S.sha256_file(bpath),
           "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "seeds": {"development": [0, 31], "confirmation": [32, 95]}, "versions": measure._versions()}
    t0 = time.time()
    try:
        rep = check.run(check.BindingCase(mod), dev=DEV, conf=CONF, keep=keep)
    except Exception as exc:  # noqa: BLE001
        job.update(status="error", reason=f"{type(exc).__name__}: {exc}"[:400], traceback=traceback.format_exc()[-3000:],
                   seconds=round(time.time() - t0, 1))
        return job
    outputs = {}
    for name in e["output_names"]:
        o = rep["outputs"].get(name)
        if o is None:
            why = "; ".join((rep.get("outputs_whose_writing_programs_aborted") or {}).get(name, [])) or "not evaluated"
            outputs[name] = {"status": "not established", "reason": why, "failure_class": measure.classify_failure(why)}
            continue
        rows = keep.get(name, [])
        q = measure.reference_quality(rows, "float32", 0.125)
        outputs[name] = {
            "status": "evaluated", "reference": q, "execution": o["execution"],
            "reference_classes": o["reference_classes"],
            "not_established_reasons_seed0": o["not_established_reasons_seed0"],
            "k_sha256_per_unit": [hashlib.sha256(np.asarray(r["k"], np.float32).tobytes()).hexdigest() for r in rows],
            "numerical": o.get("numerical"), "semantic": o.get("semantic"),
            "e_num_classes": measure.class_statistics(o.get("numerical"), S.RULE_CLASSES, S.ALPHA),
            "e_sem_classes": measure.class_statistics(o.get("semantic"), S.RULE_CLASSES, S.ALPHA) if o.get("semantic") else None}
    job.update(status="ok", launches_per_input=rep.get("launches_per_input"), outputs=outputs,
               notes={k: rep.get(k) for k in ("outputs_whose_writing_programs_aborted", "ttir_coverage_complete",
                                              "launches")},
               timing_seconds=rep.get("timing_seconds"), seconds=round(time.time() - t0, 1),
               v31_comparison=compare_v31(pid, outputs))
    return job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--program")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = OUT / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    if a.program:
        job = run_one(a.program, a.run_id)
        (out / f"{a.program}.json").write_text(json.dumps(job, indent=1, default=str) + "\n")
        print(a.program, job["status"], job.get("seconds"), flush=True)
        return
    man = S.manifest()
    todo = [p for p, e in sorted(man.items()) if e["execution_lane"] == "measurement" and not (out / f"{p}.json").exists()]
    todo.sort(key=lambda p: -int(man[p].get("execution_repeats", 1)))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    logs = ROOT.parent / ".cache" / "dsl_v2" / a.run_id
    logs.mkdir(parents=True, exist_ok=True)
    running, slot = {}, 0
    while todo or running:
        while todo and len(running) < a.workers:
            pid = todo.pop(0)
            e2 = dict(env, CUDA_VISIBLE_DEVICES=str(slot % 4))
            slot += 1
            log = open(logs / f"{pid}.log", "w")
            p = subprocess.Popen([PY, __file__, "--run-id", a.run_id, "--program", pid], cwd=ROOT, env=e2, stdout=log,
                                 stderr=subprocess.STDOUT)
            running[p.pid] = (p, pid, log, time.time())
        time.sleep(5)
        for k, (p, pid, log, t) in list(running.items()):
            if p.poll() is not None:
                log.close()
                print(f"{time.strftime('%H:%M:%S')} {pid} exit={p.returncode} {time.time() - t:.0f}s", flush=True)
                del running[k]


if __name__ == "__main__":
    main()
