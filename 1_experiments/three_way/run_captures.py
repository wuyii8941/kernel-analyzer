"""Capture every model x config (protocol.md section 3): one capture_model.py process per GPU at a time.

    python run_captures.py [--configs bf16,fp32tf32] [--gpus 0,1,2,3] [--timeout 5400]
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = Path("/data1/tzh/models")

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="bf16,fp32tf32")
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--timeout", type=float, default=5400)
ap.add_argument("--only", default="")
a = ap.parse_args()

LOCAL = ROOT / ".cache/models"          # weights downloaded for the model directories that have none


def has_weights(p):
    return any(p.glob("*.safetensors")) or any(p.glob("*.bin"))


models = []
for p in sorted(p for org in MODELS.iterdir() if org.is_dir() for p in org.iterdir() if (p / "config.json").exists()):
    alt = LOCAL / f"{p.parent.name}__{p.name}"
    models.append(p if has_weights(p) or not (alt / "config.json").exists() else alt)
TF5 = ("gemma4", "mistral3", "ministral3")     # architectures transformers 4.57 does not know: latest_train (Triton 3.6.0)
PY_TF5 = "/data1/tzh/envs/latest_train/bin/python"


def python_for(m):
    model_type = json.loads((m / "config.json").read_text()).get("model_type", "")
    return PY_TF5 if model_type in TF5 else sys.executable
if a.only:
    models = [m for m in models if any(s in str(m) for s in a.only.split(","))]
jobs = [(m, c) for c in a.configs.split(",") for m in models]
gpus = Queue()
for g in a.gpus.split(","):
    gpus.put(g)
env = dict(os.environ, HOME=str(ROOT / ".cache"), XDG_CACHE_HOME=str(ROOT / ".cache"),
           TRITON_CACHE_DIR=str(ROOT / ".cache/triton"), TMPDIR=str(ROOT / ".cache/tmp"),
           TORCHINDUCTOR_CACHE_DIR=str(ROOT / ".cache/inductor"), PYTHONPATH=str(ROOT / "2_tool/src"),
           TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
status_file = HERE / "capture" / "status.jsonl"
status_file.parent.mkdir(parents=True, exist_ok=True)
done = set()
if status_file.exists():
    done = {(r["model"], r["config"]) for r in map(json.loads, filter(str.strip, open(status_file)))
            if r["returncode"] == 0}


def run(job):
    m, c = job
    key = m.name if m.parent == LOCAL else f"{m.parent.name}__{m.name}"
    if (str(m), c) in done:
        return
    out = HERE / "capture" / key / c
    log = HERE / "capture" / key / f"{c}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    g = gpus.get()
    t = time.time()
    try:
        with open(log, "w") as f:
            p = subprocess.run([python_for(m), str(HERE / "capture_model.py"), str(m), str(out), "--config", c],
                               env=dict(env, CUDA_VISIBLE_DEVICES=g), stdout=f, stderr=subprocess.STDOUT,
                               timeout=a.timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    finally:
        gpus.put(g)
    tail = log.read_text().strip().splitlines()[-1:] if log.exists() else []
    rec = {"model": str(m), "config": c, "python": python_for(m), "returncode": rc, "seconds": round(time.time() - t), "last_line": tail[0][:300] if tail else ""}
    with open(status_file, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


with ThreadPoolExecutor(len(a.gpus.split(","))) as ex:
    list(ex.map(run, jobs))
