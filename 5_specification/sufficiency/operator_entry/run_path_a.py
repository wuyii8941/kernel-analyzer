#!/usr/bin/env python3
"""Path A runner (docs/OPERATOR_ENTRY_PROTOCOL.md): record the registered workloads, bind and measure every candidate
call key, analyze.

    run_path_a.py --root NEW_OR_EXISTING_ROOT --stage record|bind|analyze|all [--workers 8] [--only NAME ...]

Layout: ROOT/rec/<workload>/ (record.py output), ROOT/spec/ (scratch copy of 5_specification),
ROOT/results/<workload>__<key>.json (+ measurement reports), ROOT/log.jsonl, ROOT/analysis.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC_SRC = HERE.parents[1]
REPO = HERE.parents[2]
PY = "/data1/tzh/envs/hf_ga_5190/bin/python"
PYDEPS = REPO / ".cache/pydeps"
GPUS = ("0", "1", "2", "3")

MODELS = {"bert_tiny": "/data1/tzh/models/prajjwal1/bert-tiny",
          "gpt2": "/data1/tzh/kernel-analyzer/.cache/models/openai-community__gpt2",
          "qwen25_05b": "/data1/tzh/kernel-analyzer/.cache/models/Qwen__Qwen2.5-0.5B-Instruct"}
WORKLOADS = [(f"{m}_{c}", ["--model", d, "--config", c]) for m, d in MODELS.items() for c in ("fp32", "bf16")] + \
            [(f"trigger_{t}", ["--trigger", t]) for t in ("D3", "D5", "D18", "D20", "D8")]

ENV = dict(os.environ, HOME=str(REPO / ".cache"), XDG_CACHE_HOME=str(REPO / ".cache"),
           TRITON_CACHE_DIR=str(REPO / ".cache/triton"), TMPDIR=str(REPO / ".cache/tmp"), PYTHONDONTWRITEBYTECODE="1",
           LD_LIBRARY_PATH="/data1/tzh/envs/ka_main/lib",
           PYTHONPATH=":".join([str(SPEC_SRC), str(PYDEPS / "flag_gems_5_0_2"), str(PYDEPS / "sqlalchemy_2_0_48"),
                                str(REPO / "2_tool/src")]))


def log(root, rec):
    with open(root / "log.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


RUN_FAILURES = HERE / "run_failures_20261020.json"   # deviation D-A3


def extra_exclude(name):
    if not RUN_FAILURES.exists():
        return []
    return sorted({f["flaggems_function"] for f in json.loads(RUN_FAILURES.read_text()) if f["workload"] == name})


def stage_record(root, only, protocol="A2"):
    (root / "rec").mkdir(parents=True, exist_ok=True)
    for i, (name, args) in enumerate(WORKLOADS):
        ex = extra_exclude(name)
        if ex:
            args = [*args, "--extra-exclude", *ex]
        out = root / "rec" / name
        if (only and name not in only) or out.exists():
            continue
        t = time.time()
        try:
            p = subprocess.run([PY, str(HERE / "record_workload.py"), *args, "--protocol", protocol, "--out", str(out)],
                               cwd=REPO,
                               env=dict(ENV, CUDA_VISIBLE_DEVICES=GPUS[i % len(GPUS)]), capture_output=True, text=True,
                               timeout=3600)
            rc, tail = p.returncode, (p.stdout.strip().splitlines() or [""])[-1] if p.returncode == 0 else p.stderr[-600:]
        except subprocess.TimeoutExpired:
            rc, tail = "timeout", ""
        log(root, {"stage": "record", "workload": name, "rc": rc, "tail": tail, "seconds": round(time.time() - t)})


# G12 (A2): the recorded ATen operation through which each excluded FlagGems function is replayed
REPLAY_THROUGH = {"_flash_attention_forward": "aten._scaled_dot_product_flash_attention.default"}


def replay_ops(name):
    """recorded op -> excluded FlagGems function to replay on it (run failures of deviation D-A3)."""
    if not RUN_FAILURES.exists():
        return {}
    out = {}
    for f in json.loads(RUN_FAILURES.read_text()):
        if f["workload"] == name:
            fn = f["flaggems_function"]
            registered = {"flash_attention_forward": "_flash_attention_forward",
                          "layer_norm_backward": "native_layer_norm_backward", "layer_norm": "native_layer_norm"}
            reg = registered.get(fn, fn)
            out[REPLAY_THROUGH.get(reg, f["aten"])] = fn
    return out


def stage_bind(root, workers, only, protocol="v1", work_range=(0, float("inf"))):
    spec = root / "spec"
    if not spec.exists():
        shutil.copytree(SPEC_SRC, spec, ignore=shutil.ignore_patterns("__pycache__", "sufficiency"))
    (root / "results").mkdir(exist_ok=True)
    jobs = []
    for wdir in sorted((root / "rec").iterdir()):
        if only and wdir.name not in only or not (wdir / "keys.jsonl").exists():
            continue
        replay = replay_ops(wdir.name) if protocol == "A2" else {}
        for line in open(wdir / "keys.jsonl"):
            k = json.loads(line)
            if not work_range[0] <= _work(k) < work_range[1]:
                continue
            if k["launches"]:
                jobs.append((wdir, k["key"], None))
            elif k["op"] in replay:
                jobs.append((wdir, k["key"], replay[k["op"]]))

    def one(i_job):
        i, (wdir, key, enable) = i_job
        out = root / "results" / (f"{wdir.name}__{key}.json" if enable is None else
                                  f"{wdir.name}__{key}__enable_{enable}.json")
        if out.exists():
            return
        t = time.time()
        try:
            p = subprocess.run([PY, str(HERE / "bind_key.py"), "--workload", str(wdir), "--key", str(key),
                                "--spec", str(spec), "--out", str(out), "--protocol", protocol,
                                "--timeout", "2000" if protocol == "A2" else "700",
                                *(["--enable", enable] if enable else [])], cwd=REPO,
                               env=dict(ENV, CUDA_VISIBLE_DEVICES=GPUS[i % len(GPUS)]), capture_output=True, text=True,
                               timeout=40000 if protocol == "A2" else 4000)
            rc, err = p.returncode, p.stderr[-600:]
        except subprocess.TimeoutExpired:
            rc, err = "timeout", ""
        if not out.exists():
            out.write_text(json.dumps({"workload": wdir.name, "key": key, "class": "run or evidence insufficient",
                                       "reason": f"harness error rc={rc}: {err[-300:]}"}))
        log(root, {"stage": "bind", "workload": wdir.name, "key": key, "rc": rc, "seconds": round(time.time() - t)})

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, enumerate(jobs)))


def _work(k):
    sys.path.insert(0, str(HERE))
    from bind_key import key_work
    return key_work(k)


def wilson(k, n):
    if n == 0:
        return None
    z, p = 1.959963984540054, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 4), round(max(0, c - h), 4), round(min(1, c + h), 4)]


def stage_analyze(root):
    out = {"workloads": {}, "classes": {}, "classes_call_weighted": {}, "reasons": {}, "verdicts": {}, "by_op": {},
           "by_workload": {}, "triggers": {}}
    rows = [json.loads(f.read_text()) for f in sorted((root / "results").glob("*.json"))
            if "__" in f.stem and not f.name.endswith(".report.json")]
    for wdir in sorted((root / "rec").iterdir()):
        if (wdir / "run.json").exists():
            r = json.loads((wdir / "run.json").read_text())
            keys = [json.loads(l) for l in open(wdir / "keys.jsonl")]
            out["workloads"][wdir.name] = {"t1": r["t1_launch_sequences_agree"], "calls": r["calls"], "keys": r["keys"],
                                           "candidate_keys": sum(1 for k in keys if k["launches"]),
                                           "candidate_calls": sum(k["calls"] for k in keys if k["launches"]),
                                           "launches_outside_recorded_calls": r["launches_outside_recorded_calls"],
                                           "launches": r["launches_with_recorder"],
                                           "keys_over_cap": len(r["keys_over_capture_cap"])}
    for name, _ in WORKLOADS:
        if name not in out["workloads"]:
            out["workloads"][name] = {"recorded": False}
    model_rows = [r for r in rows if not r["workload"].startswith("trigger_")]
    n, ncalls = len(model_rows), sum(r.get("calls", 1) for r in model_rows)
    cls = collections.Counter(r["class"] for r in model_rows)
    clsw = collections.Counter()
    for r in model_rows:
        clsw[r["class"]] += r.get("calls", 1)
    out["candidate_keys"], out["candidate_calls"] = n, ncalls
    out["classes"] = {k: {"k": v, "wilson": wilson(v, n)} for k, v in cls.most_common()}
    out["classes_call_weighted"] = {k: {"k": v, "share": round(v / ncalls, 4)} for k, v in clsw.most_common()}
    out["reasons"] = dict(collections.Counter(f"{r['class']} | {r['reason'][:110]}" for r in model_rows).most_common())
    out["verdicts"] = dict(collections.Counter(r.get("verdict") for r in model_rows if r["class"] == "usable as is"))
    byop = collections.defaultdict(collections.Counter)
    byw = collections.defaultdict(collections.Counter)
    for r in model_rows:
        byop[r.get("op", "?")][r["class"]] += 1
        byw[r["workload"]][r["class"]] += 1
    out["by_op"] = {k: dict(v) for k, v in sorted(byop.items())}
    out["by_workload"] = {k: dict(v) for k, v in sorted(byw.items())}
    for r in rows:
        if r["workload"].startswith("trigger_"):
            out["triggers"].setdefault(r["workload"], []).append(
                {"op": r.get("op"), "class": r["class"], "verdict": r.get("verdict"), "reason": r["reason"][:200]})
    (root / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("candidate_keys", "candidate_calls", "classes", "verdicts", "triggers")},
                     indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--stage", choices=("record", "bind", "analyze", "all"), default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--protocol", choices=("v1", "A2"), default="A2")
    ap.add_argument("--work-range", nargs=2, type=float, default=[0, float("inf")],
                    help="bind only keys whose scalar-operation estimate lies in [lo, hi) (large keys get fewer workers)")
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    if a.stage in ("record", "all"):
        stage_record(a.root, a.only, a.protocol)
    if a.stage in ("bind", "all"):
        stage_bind(a.root, a.workers, a.only, a.protocol, tuple(a.work_range))
    if a.stage in ("analyze", "all"):
        stage_analyze(a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
