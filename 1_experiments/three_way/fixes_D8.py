"""Deviation D8: the fix verifications capped by D6, in a random order (seed 20261016), then the unfinished mamba outputs.
Same code and criteria as fixes.py; each row records its rank in that order.

    python fixes_D8.py units.jsonl fixes.jsonl fixes_D8.jsonl [--workers N]
    python fixes_D8.py units.jsonl fixes.jsonl fixes_recheck.jsonl --recheck     # negative ranks
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SEED = 20261016


def order(units: str, fixes: str) -> list:
    from summarize import records
    recs, _ = records(units)
    todo = sorted({(r["unit"], r["output"]) for r in recs if r["elem"] == "f32" and r["truth"] not in (None, 0)})
    done = {(json.loads(l)["unit"], json.loads(l)["output"]) for l in open(fixes) if l.strip()}
    rest = [t for t in todo if t not in done]
    tf32 = [t for t in rest if Path(t[0]).name.split("_", 1)[1].startswith("triton_tem")]
    other = [t for t in rest if t not in tf32]
    random.Random(SEED).shuffle(tf32)
    return tf32 + other


def _worker(args):
    """One output; the patched launch goes to its own directory (fixes.py's per-unit directory is shared by the
    outputs of one unit, which can overwrite each other when they run in parallel)."""
    import os
    import traceback

    from fixes import one
    rank, unit, output, gpu = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    u = Path(unit)
    out = HERE / "fixes_D8" / u.parent.parent.name / u.parent.name / f"{u.name}__{output}"
    try:
        return {"D8_rank": rank, **one(unit, output, out_dir=out)}
    except Exception as e:
        return {"D8_rank": rank, "unit": unit, "output": output, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "trace": traceback.format_exc()[-800:]}


def recheck(fixes: str) -> list:
    """Fixed rows of the main run whose unit had several fixed outputs patched at different lines."""
    import collections
    rows = [json.loads(l) for l in open(fixes) if l.strip()]
    lines = collections.defaultdict(set)
    for r in rows:
        if r.get("status") == "fixed":
            lines[r["unit"]].add(r["fix"]["line"])
    return [(r["unit"], r["output"]) for r in rows if r.get("status") == "fixed" and len(lines[r["unit"]]) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("fixes")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--recheck", action="store_true", help="rerun the main-run rows that may have shared a directory")
    a = ap.parse_args()
    if a.recheck:
        todo = [(-1 - i, t) for i, t in enumerate(recheck(a.fixes))]
    else:
        todo = list(enumerate(order(a.units, a.fixes)))
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["D8_rank"] for l in open(a.out) if l.strip()}
    gpus = a.gpus.split(",")
    jobs = [(i, u, o, gpus[i % len(gpus)]) for i, (u, o) in todo if i not in done]
    ctx = mp.get_context("spawn")
    with open(a.out, "a") as f, ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_worker, jobs):
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
