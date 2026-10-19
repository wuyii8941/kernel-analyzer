#!/usr/bin/env python3
"""Run prov_model.py for every model-config that three_way captured successfully (4 GPUs, same Python as the capture).

    python 5_specification/corpus/run_provenance.py [--gpus 0,1,2,3] [--out .cache/three_way/provenance_v2]
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TF5 = ("gemma4", "mistral3", "ministral3")
PY_TF5 = "/data1/tzh/envs/latest_train/bin/python"
PY = "/data1/tzh/envs/liger/bin/python"

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--out", default=".cache/three_way/provenance", help="output directory under the repository")
a = ap.parse_args()
OUT = ROOT / a.out
rows = [json.loads(l) for l in open(ROOT / ".cache/three_way/capture/status.jsonl") if l.strip()]
jobs = sorted({(r["model"], r["config"]) for r in rows if r["returncode"] == 0})
gpus = Queue()
for g in a.gpus.split(","):
    gpus.put(g)
env = dict(os.environ, HOME=str(ROOT / ".cache"), XDG_CACHE_HOME=str(ROOT / ".cache"), TRITON_CACHE_DIR=str(ROOT / ".cache/triton"),
           TMPDIR=str(ROOT / ".cache/tmp"), TORCHINDUCTOR_CACHE_DIR=str(ROOT / ".cache/inductor"),
           PYTHONPATH=str(ROOT / "2_tool/src"), TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def run(job):
    model, config = job
    m = Path(model)
    key = m.name if m.parent == ROOT / ".cache/models" else f"{m.parent.name}__{m.name}"
    out = OUT / key / f"{config}.jsonl"
    if out.exists():
        return
    mt = json.loads((m / "config.json").read_text()).get("model_type", "")
    py = PY_TF5 if mt in TF5 else PY
    g = gpus.get()
    t = time.time()
    try:
        log = OUT / key / f"{config}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            p = subprocess.run([py, str(HERE / "prov_model.py"), model, str(out), "--config", config],
                               env=dict(env, CUDA_VISIBLE_DEVICES=g), stdout=f, stderr=subprocess.STDOUT, timeout=7200)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    finally:
        gpus.put(g)
    print(json.dumps({"model": key, "config": config, "rc": rc, "seconds": round(time.time() - t)}), flush=True)


with ThreadPoolExecutor(len(a.gpus.split(","))) as ex:
    list(ex.map(run, jobs))
