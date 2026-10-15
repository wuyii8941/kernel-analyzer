"""Per captured launch (unit): truth K - G, the textbook premises (A), the tool (C) and their classes on a common set
of elements (protocol.md sections 2-5).  One JSON line per unit.

    python analyze_units.py OUT.jsonl UNIT_DIR [UNIT_DIR ...] [--workers N] [--timeout S]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

ELEMENT_BUDGET = 32768
MAX_PROGRAMS = 256
INPUT_BUDGET = 1 << 23          # deviation D4: input elements read by the selected programs


def select_programs(grid, per_program: int, input_per_program: int = 0) -> list:
    """Evenly spaced programs incl. the first and the last, until ELEMENT_BUDGET elements or MAX_PROGRAMS, and (D4)
    at most INPUT_BUDGET input elements read; at least 2 programs."""
    gx, gy, gz = (int(g) for g in grid)
    allp = [(x, y, z) for z in range(gz) for y in range(gy) for x in range(gx)]
    want = min(len(allp), MAX_PROGRAMS, max(2, -(-ELEMENT_BUDGET // max(per_program, 1))))
    if input_per_program > 0:
        want = min(want, max(2, INPUT_BUDGET // input_per_program))
    if want >= len(allp):
        return allp
    idx = sorted({round(i * (len(allp) - 1) / (want - 1)) for i in range(want)})
    return [allp[i] for i in idx]


def _per_program_elements(L) -> int:
    """Elements one program writes, estimated as the largest output tensor's size over the number of programs
    (Inductor names its outputs out_ptr* / in_out_ptr*; block sizes are constants in the source, not arguments)."""
    import numpy as np
    total = int(np.prod([int(g) for g in L.grid]))
    outs = [a for a in L.args if a.kind == "tensor" and "out" in (a.name or "")] or \
        [a for a in L.args if a.kind == "tensor"]
    numel = max((int(np.prod(a.shape)) if a.shape else 1) for a in outs) if outs else total
    return max(1, numel // max(total, 1))


def _input_per_program(L) -> int:
    """Input elements one program reads, estimated as all non-output tensor elements over the number of programs."""
    import numpy as np
    total = int(np.prod([int(g) for g in L.grid]))
    ins = [a for a in L.args if a.kind == "tensor" and "out" not in (a.name or "")]
    return sum(int(np.prod(a.shape)) if a.shape else 1 for a in ins) // max(total, 1)


def programs_for(L) -> list:
    return select_programs(L.grid, _per_program_elements(L), _input_per_program(L))


def analyze(unit: str) -> dict:
    import numpy as np

    from kernel_analyzer.reference_eval import bias_predict as BP
    from kernel_analyzer.reference_eval import emulate as E
    from kernel_analyzer.reference_eval import intervals as iv
    from kernel_analyzer.reference_eval.audit import _alignment, _se
    from kernel_analyzer.reference_eval.capture import load_launch
    from kernel_analyzer.reference_eval.premises import PREMISES, textbook_emulate
    from kernel_analyzer.reference_eval.ttir_eval import ST_OK, KernelReferenceEvaluator, NumericMode
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    t0 = time.time()
    L = load_launch(Path(unit))
    progs = programs_for(L)
    row = {"unit": unit, "kernel": L.kernel_name, "grid": [int(g) for g in L.grid], "programs": len(progs),
           "programs_total": int(np.prod(L.grid))}
    v = E.verify(L, ungrouped=[], swapped=[], programs=progs)
    row["verify"] = v["status"]
    module = parse_ttir(L.asm["ttir"])
    module.source_text = L.asm["ttir"]
    ev = KernelReferenceEvaluator(module, mode=NumericMode.NUMERICAL_DIFFERENCE, ttgir=L.asm["ttgir"])
    original = iv.elementary_bounds
    iv.elementary_bounds = BP._float64_elementary(original)
    try:
        gres = ev.evaluate(L, programs=progs)
    finally:
        iv.elementary_bounds = original
    G = {b.name: b for b in gres.buffers.values() if b.kind == "f" and b.written.any()}
    if not G:
        row["outputs"] = {}
        row["note"] = "no float output"
        return row
    runs = {"C-exec": (), "A-exec": PREMISES, **{f"A-{p}": (p,) for p in PREMISES}}
    emu_vals, trace = {}, {}
    for label, prem in runs.items():
        try:
            res, emu = textbook_emulate(L, premises=prem, programs=progs)
            emu_vals[label] = {b.name: (b.lo, b.st) for b in res.buffers.values() if b.name in G}
            if label == "C-exec":
                kinds = {}
                for (node, how) in emu.trace:
                    kinds.setdefault(how.split(":")[0], 0)
                    kinds[how.split(":")[0]] += 1
                trace = kinds
        except Exception as e:  # an emulation outside the model: recorded, that run not compared
            emu_vals[label] = None
            row.setdefault("emulation_errors", {})[label] = f"{type(e).__name__}: {str(e)[:120]}"
    row["lowering_kinds"] = trace
    pred = None
    try:
        pred = BP.summarize(BP.predict(L, programs=progs))
        row["prediction_status"] = pred["status"]
    except Exception as e:
        row["prediction_status"] = f"error {type(e).__name__}: {str(e)[:120]}"
    outs = {}
    for name, b in G.items():
        ok = b.written & (b.st == ST_OK) & (b.actual_after_st == 0) & np.isfinite(b.actual_after)
        for label, vals in emu_vals.items():
            if vals is not None:
                ok &= vals[name][1] == ST_OK
        n = int(ok.sum())
        o = {"elem": b.elem, "n": n, "written": int(b.written.sum()),
             "resolution": 2.0 ** -iv.FLOAT_FORMATS[b.elem][0] if b.elem in iv.FLOAT_FORMATS else None}
        if n:
            g = 0.5 * (b.lo[ok] + b.hi[ok])
            e = b.actual_after[ok] - g
            o["truth"] = [_alignment(e, g), _se(e, g)]
            for label, vals in emu_vals.items():
                if vals is not None:
                    ee = vals[name][0][ok] - g
                    o[label] = [_alignment(ee, g), _se(ee, g)]
            if pred is not None and name in pred.get("outputs", {}):
                p = pred["outputs"][name]
                o["C-model"] = [p.get("alignment"), p.get("alignment_noise_se")]
        outs[name] = o
    row["outputs"] = outs
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    """Each unit in its own subprocess (killed at the timeout); results written as they complete."""
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("units", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--single", action="store_true", help="analyse units[0] and print one JSON line (internal)")
    a = ap.parse_args()
    if a.single:
        try:
            print(json.dumps(analyze(a.units[0]), default=str))
        except Exception as e:
            print(json.dumps({"unit": a.units[0], "error": f"{type(e).__name__}: {str(e)[:200]}",
                              "trace": traceback.format_exc()[-800:]}))
        return
    gpus = a.gpus.split(",")
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["unit"] for l in open(a.out) if l.strip()}
    todo = [u for u in a.units if u not in done]

    def run(i, u):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)])
        try:
            p = subprocess.run([sys.executable, __file__, "-", u, "--single"], env=env, capture_output=True, text=True,
                               timeout=a.timeout)
            lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
            return json.loads(lines[-1]) if lines else {"unit": u, "error": f"no output (rc {p.returncode})",
                                                        "trace": p.stderr[-800:]}
        except subprocess.TimeoutExpired:
            return {"unit": u, "error": f"timeout after {a.timeout} s"}

    with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(run, i, u) for i, u in enumerate(todo)]):
            f.write(json.dumps(fut.result(), default=str) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
