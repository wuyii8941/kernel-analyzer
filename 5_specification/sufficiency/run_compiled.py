#!/usr/bin/env python3
"""Compiled-workload pipeline for sufficiency evaluations E1 and E3 (docs/SUFFICIENCY_PROTOCOL.md sections 3 and 5).

Only schedules the frozen scripts; it does not change their logic. Stages, each resumable and never overwriting:

  capture  prov_model.py MODEL ROOT/prov/<model>/<config>.jsonl --config C --capture-dir ROOT/capture/<model>/<config>
           with a fresh TORCHINDUCTOR_CACHE_DIR=ROOT/inductor/<model>/<config> (4 GPUs, 7200 s per model-config)
  verify   sufficiency/verify_units.py ROOT/units/<model>_<config>.jsonl ROOT/capture/<model>/<config>
  bind     corpus/bind_inductor.py --units ... --capture-root ROOT/capture --code-dir ROOT/inductor/<model>/<config>
  merge    ROOT/candidates.json.gz (all model-configs)
  dynamic  corpus/dynamic_three_way.py --bindings ROOT/candidates.json.gz --only UNIT --full-output-limit 4194304
           --capture-root ROOT/capture --provenance-root ROOT/prov, one process per unit, 600 s budget, CPU only;
           units: all, or a seeded simple random sample (--sample N, seed 20261019)

Every attempted model-config and unit stays in the records, with its failure if any.

    python 5_specification/sufficiency/run_compiled.py --root NEW_OR_EXISTING_ROOT --models MODELS.json --stage all
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "5_specification/corpus"
HERE = Path(__file__).resolve().parent
PY = "/data1/tzh/envs/liger/bin/python"
PY_TF5 = "/data1/tzh/envs/latest_train/bin/python"
TF5 = ("gemma4", "mistral3", "ministral3")
SEED = 20261019
FULL_LIMIT = 4194304
BASE_ENV = dict(os.environ, HOME=str(REPO / ".cache"), XDG_CACHE_HOME=str(REPO / ".cache"),
                TMPDIR=str(REPO / ".cache/tmp"), PYTHONPATH=str(REPO / "2_tool/src"), TOKENIZERS_PARALLELISM="false",
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def key_of(model_dir: Path) -> str:
    """org__name, as three_way names its capture directories."""
    return model_dir.name if "__" in model_dir.name else f"{model_dir.parent.name}__{model_dir.name}"


def log(root, row):
    with open(root / "pipeline_log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def capture(root, jobs, gpus):
    q = Queue()
    for g in gpus:
        q.put(g)

    def one(job):
        model_dir, config = job
        key = key_of(model_dir)
        prov = root / "prov" / key / f"{config}.jsonl"
        cap = root / "capture" / key / config
        if prov.exists() or (root / "prov" / key / f"{config}.log").exists():
            return
        prov.parent.mkdir(parents=True, exist_ok=True)
        mt = json.loads((model_dir / "config.json").read_text()).get("model_type", "")
        env = dict(BASE_ENV, TORCHINDUCTOR_CACHE_DIR=str(root / "inductor" / key / config),
                   TRITON_CACHE_DIR=str(root / "triton"))
        g = q.get()
        t = time.time()
        try:
            with open(prov.with_suffix(".log"), "w") as fh:
                p = subprocess.run([PY_TF5 if mt in TF5 else PY, str(CORPUS / "prov_model.py"), str(model_dir), str(prov),
                                    "--config", config, "--capture-dir", str(cap)],
                                   env=dict(env, CUDA_VISIBLE_DEVICES=g), stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=7200, cwd=REPO)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        finally:
            q.put(g)
        log(root, {"stage": "capture", "model": key, "config": config, "rc": rc, "seconds": round(time.time() - t)})

    with ThreadPoolExecutor(len(gpus)) as ex:
        list(ex.map(one, jobs))


def verify(root, jobs, workers, gpus):
    for model_dir, config in jobs:
        key = key_of(model_dir)
        cap = root / "capture" / key / config
        out = root / "units" / f"{key}_{config}.jsonl"
        if out.exists() or not cap.is_dir() or not any(cap.glob("*/launch.json")):
            continue
        out.parent.mkdir(exist_ok=True)
        p = subprocess.run([PY, str(HERE / "verify_units.py"), str(out), str(cap), "--workers", str(workers),
                            "--gpus", gpus], env=BASE_ENV, cwd=REPO)
        log(root, {"stage": "verify", "model": key, "config": config, "rc": p.returncode})


def bind(root, jobs):
    for model_dir, config in jobs:
        key = key_of(model_dir)
        units = root / "units" / f"{key}_{config}.jsonl"
        out = root / "bind" / f"{key}_{config}.json.gz"
        if out.exists() or not units.exists():
            continue
        out.parent.mkdir(exist_ok=True)
        # bind_inductor takes unit paths ending in <model>/<config>/<name>
        fixed = root / "bind" / f"{key}_{config}.units.jsonl"
        with open(fixed, "w") as fh:
            for l in open(units):
                r = json.loads(l)
                r["unit"] = f"{key}/{config}/{Path(r['unit']).name}"
                fh.write(json.dumps(r) + "\n")
        p = subprocess.run([PY, str(CORPUS / "bind_inductor.py"), "--out", str(out), "--units", str(fixed),
                            "--capture-root", str(root / "capture"), "--code-dir", str(root / "inductor" / key / config)],
                           env=dict(BASE_ENV, CUDA_VISIBLE_DEVICES=""), cwd=REPO, capture_output=True, text=True)
        log(root, {"stage": "bind", "model": key, "config": config, "rc": p.returncode,
                   "tail": p.stdout.strip().splitlines()[-1][:300] if p.stdout.strip() else p.stderr[-300:]})


def merge(root):
    out = root / "candidates.json.gz"
    if out.exists():
        return
    rows = []
    for f in sorted((root / "bind").glob("*.json.gz")):
        rows += json.loads(gzip.open(f).read())
    with gzip.open(out, "wt") as fh:
        json.dump(rows, fh)
    log(root, {"stage": "merge", "units": len(rows)})


def dynamic(root, sample, workers, timeout):
    recs = json.loads(gzip.open(root / "candidates.json.gz").read())
    units = sorted(r["unit"] for r in recs)
    chosen = sorted(random.Random(SEED).sample(units, sample)) if sample and sample < len(units) else units
    (root / "dynamic_units.json").write_text(json.dumps({"seed": SEED, "of": len(units), "units": chosen}))
    out_dir = root / "dynamic"
    out_dir.mkdir(exist_ok=True)

    def one(i_unit):
        i, unit = i_unit
        out = out_dir / f"{i:05d}.jsonl"
        if out.exists() or out.with_suffix(".err").exists():
            return
        t = time.time()
        try:
            p = subprocess.run([PY, str(CORPUS / "dynamic_three_way.py"), "--bindings", str(root / "candidates.json.gz"),
                                "--only", unit, "--full-output-limit", str(FULL_LIMIT), "--capture-root",
                                str(root / "capture"), "--provenance-root", str(root / "prov"), "--out", str(out)],
                               env=dict(BASE_ENV, CUDA_VISIBLE_DEVICES=""), cwd=REPO, capture_output=True, text=True,
                               timeout=timeout)
            rc, err = p.returncode, p.stderr[-600:]
        except subprocess.TimeoutExpired:
            rc, err = "budget_exceeded", ""
        if not out.exists() or rc != 0:
            out.with_suffix(".err").write_text(json.dumps({"unit": unit, "rc": rc, "stderr": err,
                                                           "seconds": round(time.time() - t)}))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, enumerate(chosen)))
    log(root, {"stage": "dynamic", "units": len(chosen)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True, help="JSON list of model directories")
    ap.add_argument("--configs", nargs="+", default=["bf16", "fp32tf32"])
    ap.add_argument("--stage", choices=("capture", "verify", "bind", "merge", "dynamic", "all"), default="all")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    models = [Path(m) for m in json.loads(a.models.read_text())]
    jobs = [(m, c) for m in models for c in a.configs]
    if a.stage in ("capture", "all"):
        capture(a.root, jobs, a.gpus.split(","))
    if a.stage in ("verify", "all"):
        verify(a.root, jobs, a.workers, a.gpus)
    if a.stage in ("bind", "all"):
        bind(a.root, jobs)
    if a.stage in ("merge", "all"):
        merge(a.root)
    if a.stage in ("dynamic", "all"):
        dynamic(a.root, a.sample, a.workers, a.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
