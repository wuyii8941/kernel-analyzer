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

models = sorted(p for org in MODELS.iterdir() if org.is_dir() for p in org.iterdir() if (p / "config.json").exists())
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
    done = {(json.loads(l)["model"], json.loads(l)["config"]) for l in open(status_file) if l.strip()}


def run(job):
    m, c = job
    key = f"{m.parent.name}__{m.name}"
    if (str(m), c) in done:
        return
    out = HERE / "capture" / key / c
    log = HERE / "capture" / key / f"{c}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    g = gpus.get()
    t = time.time()
    try:
        with open(log, "w") as f:
            p = subprocess.run([sys.executable, str(HERE / "capture_model.py"), str(m), str(out), "--config", c],
                               env=dict(env, CUDA_VISIBLE_DEVICES=g), stdout=f, stderr=subprocess.STDOUT,
                               timeout=a.timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    finally:
        gpus.put(g)
    tail = log.read_text().strip().splitlines()[-1:] if log.exists() else []
    rec = {"model": str(m), "config": c, "returncode": rc, "seconds": round(time.time() - t), "last_line": tail[0][:300] if tail else ""}
    with open(status_file, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


with ThreadPoolExecutor(len(a.gpus.split(","))) as ex:
    list(ex.map(run, jobs))
