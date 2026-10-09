#!/usr/bin/env python3
"""prog_26, the isolated boundary item of structure acceptance v1.1-rc1, under the issuer's approval of 2026-10-08:
a fresh child process with a timeout, every input executed 8 times, execution randomness recorded only.  Not scored:
no reference, no statistics.

    python scripts/acceptance/structure_v11_boundary.py --out DIR [--first-seed 0 --last-seed 95] [--timeout 900]

Each program of prog_26 stores its row mean to buf[row] and loads buf[(row + 1) % R], written by another program
without synchronization (the index is always in range).  Per execution: the digest of K, and per row which value the
load returned, estimated as mean(x - y) over the row in float64: the initial 0, the neighbouring row's mean, or
something else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT.parent / ".cache" / "acceptance" / "structure_acceptance_v1_1_rc1"
PID = "prog_26"
REPEATS = 8
TOL = 1e-4  # row means are about 0.5 (inputs N(0.5, 1), D = 2048); the three classes are far apart


def child(first, last):
    sys.path.insert(0, str(PKG))
    import numpy as np
    import torch
    import triton
    import binding_support as bs
    out_rows, order = [], 0
    for s in range(first, last + 1):
        per = []
        for r in range(REPEATS):
            inp = bs.make_inputs(PID, s)
            t = time.time_ns()
            y = bs.run(PID, inp, allow_boundary=True)["output"]
            torch.cuda.synchronize()
            k = y.detach().cpu().numpy()
            x = inp["x"].detach().cpu().double().numpy()
            read = (x - k.astype(np.float64)).mean(axis=1)
            nb = np.roll(x.mean(axis=1), -1)  # the mean of row (row + 1) % R
            cls = np.where(np.abs(read) < TOL, "0", np.where(np.abs(read - nb) < TOL, "n", "?"))
            per.append({"execution": r, "order": order, "launch_start_ns": t,
                        "k_sha256": hashlib.sha256(k.tobytes()).hexdigest(),
                        "rows_read_initial_zero": int((cls == "0").sum()),
                        "rows_read_neighbour_mean": int((cls == "n").sum()), "rows_other": int((cls == "?").sum()),
                        "row_classes": "".join(cls.tolist())})
            order += 1
        out_rows.append({"seed": s, "distinct_outputs": len({p["k_sha256"] for p in per}), "executions": per})
    env = {"torch": torch.__version__, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(0)}
    print("BOUNDARY_JSON:" + json.dumps({"environment": env, "inputs": out_rows}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--last-seed", type=int, default=95)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a.first_seed, a.last_seed)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = {"purpose": "prog_26 isolated boundary run (issuer approval 2026-10-08): execution randomness only, not scored",
           "program": PID, "seeds": [a.first_seed, a.last_seed], "executions_per_input": REPEATS,
           "isolation": f"fresh child process (python -I), timeout {a.timeout} s",
           "tool_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip(),
           "read_classes": {"0": "initial value 0 (torch.zeros)", "n": "the neighbouring row's mean",
                            "?": "neither (tolerance %g on mean(x - y))" % TOL}}
    try:
        p = subprocess.run([sys.executable, "-I", __file__, "--child", "--first-seed", str(a.first_seed),
                            "--last-seed", str(a.last_seed)], capture_output=True, text=True, timeout=a.timeout,
                           env=dict(os.environ))
        marker = [x for x in p.stdout.splitlines() if x.startswith("BOUNDARY_JSON:")]
        if p.returncode != 0 or not marker:
            rec.update(status="FAILED", returncode=p.returncode, stderr=p.stderr[-6000:])
        else:
            data = json.loads(marker[-1].split(":", 1)[1])
            ins = data["inputs"]
            ex = [e for i in ins for e in i["executions"]]
            rows = len(ex[0]["row_classes"]) if ex else 0
            rec.update(status="RAN", environment=data["environment"], inputs=ins, summary={
                "inputs": len(ins), "executions": len(ex),
                "inputs_with_more_than_one_distinct_output": sum(i["distinct_outputs"] > 1 for i in ins),
                "distinct_outputs_per_input": {str(n): sum(i["distinct_outputs"] == n for i in ins)
                                               for n in sorted({i["distinct_outputs"] for i in ins})},
                "row_reads": {"initial_zero": sum(e["rows_read_initial_zero"] for e in ex),
                              "neighbour_mean": sum(e["rows_read_neighbour_mean"] for e in ex),
                              "other": sum(e["rows_other"] for e in ex), "total": rows * len(ex)}})
    except subprocess.TimeoutExpired:
        rec.update(status="TIMEOUT")
    rec["seconds"] = round(time.time() - t0, 1)
    (a.out / "prog_26_isolated.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(rec["status"], json.dumps(rec.get("summary")), rec["seconds"])


if __name__ == "__main__":
    main()
