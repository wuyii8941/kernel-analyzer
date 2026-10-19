#!/usr/bin/env python3
"""Sufficiency evaluation E4, synthetic source S2 (docs/SUFFICIENCY_PROTOCOL.md section 6): mechanism discrimination
and normal controls on reviewed real-call bindings.

For each case of sufficiency/e4_cases.py: a scratch copy of the frozen specification tree receives the edited call
(a copy of the base call with the case's textual edits) and a copy of the base binding plan with a new id (plus the
case's plan edits). Both are retained under validated/ IN THE SCRATCH COPY ONLY, as an evaluation device: the case is
the candidate implementation under test, not a reviewed binding. measure.py then runs in contract mode with
development 1 + confirmation 2 units. The repository is never modified.

Verdict per case, from every measured output's program-semantics comparison (semantic: G_pi - f):
  detected        some output has coordinates whose G_pi - f enclosure excludes zero
  consistent      every output evaluated completely and no coordinate excludes zero
  not_established measurement refused, incomplete, over budget, or an execution error (reason kept)
Rates are then: errors detected / errors; normal controls judged "detected" (false alarm) / controls; not_established
separately. Cases whose expected behavior is "refuse" count as correct when the verdict is not_established with an
admission or task-choice refusal.

    python 5_specification/sufficiency/run_e4.py --spec-src FROZEN_5_SPEC --root NEW_DIR [--gpu 0] [--only ID ...]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def set_pointer(doc, pointer, value):
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.strip("/").split("/")]
    cur = doc
    for p in parts[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def prepare(spec, case):
    base = case["base"]
    decl = json.loads((spec / "corpus/measure_cases" / f"{base}.json").read_text())
    call_rel = decl["call"].split(":")[0]
    base_call = (spec / "corpus/measure_cases" / call_rel).resolve()
    src = base_call.read_text()
    for old, new in case.get("edits", []):
        if src.count(old) != 1:
            raise ValueError(f"edit target occurs {src.count(old)} times: {old[:60]!r}")
        src = src.replace(old, new)
    name = f"e4_{case['id'].replace('-', '_')}"
    plan_path = (spec / "corpus/measure_cases" / decl["compare"]["binding_plan"]).resolve()
    plan = json.loads(plan_path.read_text())
    plan = copy.deepcopy(plan)
    plan["id"] = f"e4.{case['id']}.{plan['id']}"
    plan["subject"]["measure_call"] = f"../calls/{name}.py:call"
    for pointer, value in case.get("plan_edits", []):
        set_pointer(plan, pointer, value)
    (spec / "corpus/calls" / f"{name}.py").write_text(src)
    plan_text = json.dumps(plan, indent=2) + "\n"
    (spec / "corpus/call_bindings" / f"{name}.json").write_text(plan_text)
    (spec / "validated/bindings" / f"{plan['id']}.json").write_text(plan_text)
    (spec / "validated/calls" / f"{plan['id']}.py").write_text(src)
    decl["call"] = f"../calls/{name}.py:call"
    decl["compare"]["binding_plan"] = f"../call_bindings/{name}.json"
    decl["units"] = {"development": 1, "confirmation": 2, "seed_offset": 0}
    decl.setdefault("budget", {})["max_units"] = 3
    decl["inputs"].update(case.get("input_edits", {}))
    dpath = spec / "corpus/measure_cases" / f"{name}.json"
    dpath.write_text(json.dumps(decl, indent=1))
    return dpath


def verdict(report):
    if report is None:
        return "not_established", "no report"
    outs = {}
    for level in report.get("levels", []):
        for name, o in (level.get("outputs") or {}).items():
            outs[name] = o
    if not outs:
        return "not_established", "no evaluated outputs: " + json.dumps(report.get("failure_counts"))[:200]
    excl, complete, reasons = 0, True, []
    for name, o in outs.items():
        sem = o.get("semantic") or {}
        excl += int(sem.get("reference_difference_excludes_zero") or 0)
        if o.get("status") != "evaluated" or (o.get("complete_rate") or 0) < 1.0 or not sem:
            complete = False
            reasons.append(f"{name}: {o.get('status')} complete_rate={o.get('complete_rate')}")
    if excl:
        return "detected", f"{excl} coordinates with G_pi - f excluding zero"
    if complete:
        return "consistent", "all outputs complete, no coordinate excludes zero"
    return "not_established", "; ".join(reasons)[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-src", type=Path, required=True, help="frozen 5_specification tree (worktree at F)")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    from e4_cases import CASES
    spec = a.root / "spec"
    if not spec.exists():
        shutil.copytree(a.spec_src, spec)
    # the M6 input generators are referenced as ../../sufficiency/e4_cases.py:<fn> from corpus/measure_cases
    (spec / "sufficiency").mkdir(exist_ok=True)
    shutil.copy2(HERE / "e4_cases.py", spec / "sufficiency" / "e4_cases.py")
    out = a.root / "results.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    tool = a.spec_src.parent / "2_tool"
    env = dict(os.environ, HOME=str(REPO / ".cache"), XDG_CACHE_HOME=str(REPO / ".cache"),
               TRITON_CACHE_DIR=str(REPO / ".cache/triton"), TMPDIR=str(REPO / ".cache/tmp"),
               PYTHONPATH=str(tool / "src"), CUDA_VISIBLE_DEVICES=a.gpu)
    for case in CASES:
        if case["id"] in done or (a.only and case["id"] not in a.only):
            continue
        row = {k: case.get(k) for k in ("id", "mechanism", "kind", "source", "base", "expected")}
        t = time.time()
        report = None
        try:
            dpath = prepare(spec, case)
            rpath = a.root / "reports" / f"{case['id']}.json"
            rpath.parent.mkdir(exist_ok=True)
            p = subprocess.run([sys.executable, str(tool / "scripts/measure.py"), "--declaration", str(dpath),
                                "--out", str(rpath)], env=env, capture_output=True, text=True, timeout=a.timeout)
            row["rc"] = p.returncode
            if rpath.exists():
                report = json.loads(rpath.read_text())
            else:
                row["stderr"] = p.stderr[-400:]
        except subprocess.TimeoutExpired:
            row["rc"] = "budget_exceeded"
        except Exception as err:  # noqa: BLE001 -- recorded, never hidden
            row["rc"] = f"error {type(err).__name__}: {str(err)[:200]}"
        row["verdict"], row["reason"] = verdict(report)
        if report is None and "stderr" in row:
            row["reason"] += ": " + row["stderr"][-200:]
        row["seconds"] = round(time.time() - t, 1)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
