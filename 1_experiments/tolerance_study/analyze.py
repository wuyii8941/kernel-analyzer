#!/usr/bin/env python3
"""Runs the frozen analyses of the tolerance study (protocol 5A and appendices B, C) on the confirmation data and writes
compact summaries to results/ (full unified-entry reports gzipped to .cache/round2/reports).

    python analyze.py jobs A|B|C            # print the frozen job list
    python analyze.py run A|B|C --gpus 0,1,2,3 --parallel 8
    python analyze.py one '<job json>'      # one job (used by run)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
REPORTS = ROOT / ".cache" / "round2" / "reports"
PY = "/data1/tzh/envs/latest_train/bin/python"
CAPTURES = {"A": ROOT / ".cache/round2/captures", "B": ROOT / ".cache/round2/captures_b",
            "C": ROOT / ".cache/round2/captures_c"}
UNITS = {"development": 10, "confirmation": 22}
N_AB = 6 * 65536 + 6 * 896


def jobs(group):
    out = []
    if group == "A":
        for t in (10, 100, 300):
            for impl, mode in (("compiled", "A"), ("compiled", "B"), ("fused", "black-box")):
                out.append({"group": "A", "kind": "measure", "stratum": f"s{t}", "step": t, "impl": impl, "mode": mode})
            out.append({"group": "A", "kind": "tolerance", "stratum": f"s{t}", "step": t})
    elif group == "B":
        for t in (6, 7, 10, 20, 100):
            for impl, mode in (("compiled", "A"), ("compiled", "B"), ("eager", "black-box")):
                out.append({"group": "B", "kind": "measure", "stratum": f"s{t}", "step": t, "impl": impl, "mode": mode})
            out.append({"group": "B", "kind": "tolerance", "stratum": f"s{t}", "step": t})
    elif group == "C":
        for layer in (2, 12, 22):
            for mat in ("q_proj", "down_proj"):
                for impl, mode in (("triton_default", "A"), ("triton_default", "B"), ("triton_ieee", "A"),
                                   ("inductor_high", "A"), ("cublas_high", "black-box"),
                                   ("cublas_highest", "black-box")):
                    out.append({"group": "C", "kind": "measure", "stratum": f"l{layer}_{mat}", "layer": layer,
                                "mat": mat, "impl": impl, "mode": mode})
                out.append({"group": "C", "kind": "tolerance", "stratum": f"l{layer}_{mat}", "layer": layer,
                            "mat": mat})
    return out


def _summary_output(o):
    """The compact view of one measured output."""
    s = {"status": o.get("status"), "reason": o.get("reason"), "dtype": o.get("dtype")}
    ref = o.get("reference") or {}
    s["reference"] = {k: ref.get(k) for k in ("complete_rate", "reference_scope", "resolved_fraction",
                                               "width_over_ulp", "resolution_met")} if ref else None
    ob = o.get("observation") or {}
    if ob:
        s["observation"] = {"summary": ob.get("summary"),
                            "decomposition": {k: (ob.get("decomposition") or {}).get(k) for k in ("total", "semantic")}
                            if ob.get("decomposition") else None}
    st = o.get("statistics") or {}
    s["statistics"] = {cls: {"summary": v.get("summary"),
                             "rules": {r: {k: x.get(k) for k in ("judgment", "holm_adjusted_p", "n",
                                                                  "mean_projection", "mde_approximate")}
                                       for r, x in v.get("rules", {}).items()},
                             "bounded": ((v.get("axes") or {}).get("nonzero") or {}).get("bounded", {}).get(
                                 "available")}
                       for cls, v in st.items()}
    s["scale"] = o.get("scale")
    s["semantic"] = o.get("semantic")
    s["failure_classes"] = o.get("failure_classes")
    return s


def run_one(job):
    sys.path.insert(0, str(HERE))
    g = job["group"]
    out_dir = RESULTS / g / job["stratum"]
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if job["kind"] == "tolerance":
        if g == "A":
            cmd = [PY, str(HERE / "tolerance_checks.py"), "--captures", str(CAPTURES["A"]), "--step", str(job["step"]),
                   "--seeds", "1000-1031", "--out", str(out_dir / "tolerance.json")]
        elif g == "B":
            cmd = [PY, str(HERE / "tolerance_bc.py"), "B", "--captures", str(CAPTURES["B"]), "--step",
                   str(job["step"]), "--seeds", "4000-4031", "--out", str(out_dir / "tolerance.json")]
        else:
            cmd = [PY, str(HERE / "tolerance_bc.py"), "C", "--captures", str(CAPTURES["C"]), "--layer",
                   str(job["layer"]), "--mat", job["mat"], "--units", "0-31", "--out", str(out_dir / "tolerance.json")]
        p = subprocess.run(cmd, capture_output=True, text=True)
        (out_dir / "tolerance.log").write_text(p.stdout[-5000:] + "\n--- stderr ---\n" + p.stderr[-5000:])
        return {"job": job, "returncode": p.returncode, "seconds": round(time.time() - t0, 1)}
    from kernel_analyzer import measure
    os.environ["KA_TOL_CAPTURES" if g == "A" else f"KA_TOL_CAPTURES_{g}"] = str(CAPTURES[g])
    if g == "A":
        import calls_adamw as C
        d = C.declaration(job["step"], job["impl"], job["mode"], N_AB, UNITS, 1000)
    elif g == "B":
        import calls_radam as C
        d = C.declaration(job["step"], job["impl"], job["mode"], N_AB, UNITS, 4000)
    else:
        import calls_tf32 as C
        k = 896 if job["mat"] == "q_proj" else 4864
        d = C.declaration(job["layer"], job["mat"], job["impl"], job["mode"], UNITS, 0, k)
    d["_base_dir"] = str(HERE)
    name = f"{job['impl']}_{job['mode']}"
    REPORTS.mkdir(parents=True, exist_ok=True)
    rep = measure.run(d)
    with gzip.open(REPORTS / f"{g}_{job['stratum']}_{name}.json.gz", "wt") as fh:
        json.dump(rep, fh, default=str)
    lv = rep["levels"][0]
    summary = {"job": job, "declaration_sha256": rep["expanded_declaration"]["declaration_sha256"],
               "status": lv.get("status"), "reason": lv.get("reason"), "notes": lv.get("notes"),
               "precision_environment": lv.get("precision_environment"),
               "producer_trace_per_seed": lv.get("producer_trace_per_seed"),
               "refinement": (lv.get("refinement") or {}).get("outcome"),
               "units": lv.get("units"), "seconds": lv.get("seconds"), "timing": lv.get("timing_seconds"),
               "outputs": {n: _summary_output(o) for n, o in (lv.get("outputs") or {}).items()}}
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return {"job": job, "status": lv.get("status"), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("jobs", "run", "one"))
    ap.add_argument("arg")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "jobs":
        print(json.dumps(jobs(a.arg), indent=1))
        return
    if a.cmd == "one":
        print(json.dumps(run_one(json.loads(a.arg))))
        return
    js = jobs(a.arg)
    gpus = a.gpus.split(",")
    env = dict(os.environ)
    c = str(ROOT / ".cache")
    env.update(HOME=c, XDG_CACHE_HOME=c, TRITON_CACHE_DIR=c + "/triton", TMPDIR=c + "/tmp", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(ROOT / "2_tool" / "src"), HF_HOME=c + "/hf", HF_HUB_OFFLINE="1")
    log = ROOT / ".cache" / "round2" / f"analyze_{a.arg}.log"

    def go(i_job):
        i, job = i_job
        e = dict(env, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)])
        p = subprocess.run([PY, str(HERE / "analyze.py"), "one", json.dumps(job)], env=e, capture_output=True,
                           text=True)
        line = f"{time.strftime('%H:%M:%S')} {json.dumps(job)} exit={p.returncode} {p.stdout.strip()[-300:]}"
        if p.returncode:
            line += " STDERR " + p.stderr.strip()[-1500:].replace("\n", " | ")
        with open(log, "a") as fh:
            fh.write(line + "\n")
        return p.returncode
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        codes = list(pool.map(go, enumerate(js)))
    with open(log, "a") as fh:
        fh.write(f"DONE {a.arg}: {sum(c == 0 for c in codes)}/{len(codes)} ok\n")


if __name__ == "__main__":
    main()
