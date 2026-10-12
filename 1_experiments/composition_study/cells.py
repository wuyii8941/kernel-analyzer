"""RQ2: every (primitive, composition form) cell on the device, checked against the composition rules.

A cell is one primitive inside one form (forms.py) under one set of compile options.  The cell's kernel is compiled
for sm_86, launched on inputs drawn per program from six distributions, captured with the tool's launch recorder and
emulated with the tool's lowering rules (kernel_analyzer.reference_eval.emulate.verify with every lowering choice
frozen: no fitting to the outputs).  The emulator is the system under test; the device output is the judge.

Per cell: build status, compile status (the compiler's rejection is a result), the bit comparison of every written
element (bit identical / mismatch / not emulated, with the emulator's not-emulable reasons), and the run-to-run
reproducibility of the device output (two launches on the same inputs).

Run:  cells.py --cells-file F --out-jsonl O       (child: one JSON line per cell, flushed as it goes)
      cells.py --plan default --out results/...   (driver: all cells, child processes on the GPUs)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "2_tool/src"))
import forms as FM  # noqa: E402
import primitives as P  # noqa: E402

BLOCK_DISTS = ["symmetric", "same-sign", "heavy-tailed", "small-integer", "wide", "near-format-boundary"]
DEFAULT_OPTIONS = {"num_warps": 4, "enable_fp_fusion": True, "enable_reflect_ftz": True}


def cell_id(pid, form, param, options):
    return json.dumps([pid, form, param, sorted(options.items())])


def make_inputs(cell, prim, rng):
    import torch
    import catalog as C
    tensors = {}
    n_elems = cell.grid * cell.block
    for name, e, role in cell.inputs:
        sizes = getattr(cell, "input_sizes", {})
        if name in sizes:
            n = sizes[name]
            if e in C.FLOAT_T:
                v = C.fmt_round_array(C.float_dist(rng, "symmetric", e, n), e)
            else:
                v = C.int_dist(rng, "small-integer", e, n)
            tensors[name] = C.to_device(v, e)
            continue
        if role == "trip count":
            tensors[name] = torch.tensor([3], dtype=torch.int32, device="cuda")
            continue
        if role == "permutation within the program":
            perm = np.concatenate([rng.permutation(cell.block) for _ in range(cell.grid)]).astype(np.int32)
            tensors[name] = torch.from_numpy(perm).cuda()
            continue
        reps = FM.TRIP if "per iteration" in role else 1
        cols = []
        for r in range(reps):
            for g in range(cell.grid):
                d = BLOCK_DISTS[g % len(BLOCK_DISTS)]
                if e in C.FLOAT_T:
                    mpos = re.search(r"\bop(\d)", role)
                    cols.append(C.fmt_round_array(C.fold(prim.sem, int(mpos.group(1)) if mpos else 0,
                                                         C.float_dist(rng, d, e, cell.block)), e))
                else:
                    mpos = re.search(r"\bop(\d)", role)                 # "op1", "stage2.op1", "op1 per iteration"
                    pos = int(mpos.group(1)) if mpos else -1
                    if prim.family == "int" and prim.sem in ("shli", "shrsi", "shrui") and pos == 1:
                        cols.append(np.array(C.shift_amounts(rng, cell.block, C.INT_W[e]), dtype=object))
                    elif prim.sem in ("jn", "yn") and pos == 0:             # Bessel order: small, as in the catalog
                        cols.append(np.array([int(v) for v in rng.integers(0, 6, cell.block)], dtype=object))
                    elif prim.sem in ("powi", "pow", "ldexp", "scalbn") and pos == 1 and e not in C.FLOAT_T:
                        cols.append(np.array([int(v) for v in rng.integers(-12, 13, cell.block)], dtype=object))
                    elif prim.sem == "ipowi" and pos == 1:
                        cols.append(np.array([int(v) for v in rng.integers(0, 9, cell.block)], dtype=object))
                    else:
                        cols.append(np.array(C.int_dist(rng, d, e, cell.block), dtype=object))
        v = np.concatenate(cols)
        tensors[name] = C.to_device(v.tolist() if v.dtype == object else v, e)
    for name, e, n in cell.outputs:
        size = n if n else n_elems
        tensors[name] = C.to_device(np.zeros(size) if e in C.FLOAT_T else [0] * size, e)
    return tensors


def run_cell(prim, form, param, options, seed_text):
    import torch
    import catalog as C
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    from kernel_analyzer.reference_eval.emulate import verify
    rec = {"pid": prim.pid, "form": form, "param": param, "options": options}
    options = dict(options)
    block = options.pop("block", 128)
    try:
        cell = FM.build(prim, form, param, block=block)
    except Exception as e:
        rec.update(status="build error", error=f"{type(e).__name__}: {e}"[:300])
        return rec
    if not cell.applicable:
        rec.update(status="not applicable", reason=cell.reason)
        return rec
    rec["ttir_sha256"] = hashlib.sha256(cell.text.encode()).hexdigest()
    try:
        k1 = C.compile_text(cell.text, options)
        k2 = C.compile_text(cell.second_text, options) if cell.launches == 2 else None
    except Exception as e:
        rec.update(status="not compiled", error=str(e).splitlines()[0][:300])
        return rec
    import compile_sweep as cs
    ops = cs.ptx_opcodes(k1.asm["ptx"])
    rec["ptx_numeric"] = {o: n for o, n in sorted(ops.items()) if any(s in o for s in (
        "fma", "add.f", "add.rn", "sub.f", "mul.f", "mul.rn", "div", "rcp", "sqrt", "ex2", "lg2", "sin.", "cos.",
        "tanh", "cvt.r", "mma", "atom", "shfl", "max.f", "min.f", "selp.f", "call"))}
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little"))
    inputs = make_inputs(cell, prim, rng)
    args1 = [inputs[n] for n, _, _ in cell.inputs] + [inputs[n] for n, _, _ in cell.outputs]
    grid = (cell.grid, 1, 1)
    snapshot = {n: t.clone() for n, t in inputs.items()}
    try:
        with TritonLaunchRecorder() as R:
            if cell.form == "C13":
                k1[grid](*args1)
                state = inputs[cell.outputs[0][0]]
                extra = {n: make_extra(e, cell, rng) for n, e, _ in cell.second_inputs}
                out2 = torch.zeros_like(state)
                order = [n for n, _, _ in cell.second_inputs]
                ys = [extra[n] for n in order if n.startswith("y")]
                alts = [extra[n] for n in order if n.startswith("alt")]
                k2[grid](state, *ys, *alts, out2)
            elif cell.form == "C15":
                k1[(1, 1, 1)](*args1)
            else:
                k1[grid](*args1)
            torch.cuda.synchronize()
    except Exception as e:
        rec.update(status="launch error", error=f"{type(e).__name__}: {str(e)[:300]}")
        return rec
    first = {n: inputs[n].clone() for n, _, _ in cell.outputs}
    # reproducibility: the same launch again on the same inputs
    for n, t in snapshot.items():
        inputs[n].copy_(t)
    if cell.form not in ("C13",):
        k1[grid if cell.form != "C15" else (1, 1, 1)](*args1)
        torch.cuda.synchronize()
        rec["reproducible"] = all(bool(torch.equal(first[n].view(torch.uint8), inputs[n].view(torch.uint8)))
                                  for n, _, _ in cell.outputs)
    reports = [compare_launch(L) for L in R.launches]
    rec["launches"] = reports
    sts = [r["status"] for r in reports]
    if not reports:
        rec["status"] = "not captured"
    elif all(s == "bit_identical" for s in sts):
        rec["status"] = "bit identical"
    elif any(s == "mismatch" for s in sts):
        rec["status"] = "mismatch"
    elif any(s == "emulator error" for s in sts):
        rec["status"] = "emulator error"
    else:
        rec["status"] = "not established"
    return rec


def compare_launch(L) -> dict:
    """Emulate one captured launch with every lowering choice frozen and compare every written element with the
    device: float buffers as storage bits (the tool's comparison), integer buffers as values (added here: the tool's
    verify compares float buffers only)."""
    from kernel_analyzer.reference_eval import emulate as E
    t0 = time.time()
    try:
        result, emu = E.emulate(L, ungrouped=[], swapped=[])
    except Exception as e:
        return {"status": "emulator error", "error": f"{type(e).__name__}: {str(e)[:300]}",
                "where": traceback.format_exc().splitlines()[-3:]}
    tot = Counter()
    for name, o in E._outputs(result, L).items():
        m = E._compare(*o)
        written = o[0]
        tot["written"] += int(written.sum())
        tot["emulated"] += int(m["emulated"].sum())
        tot["bit_identical"] += int(m["bit_equal"].sum())
        tot["nan_both"] += int(m["nan_both"].sum())
    for ptr, b in result.buffers.items():
        if b.kind not in ("i", "b") or not b.written.any():   # integer and boolean (i1 stored as bytes) buffers
            continue
        w = b.written
        point = w & (b.st == 0) & ((b.lo == b.hi) if b.hi is not None else True)   # hi None: a point value
        tot["written"] += int(w.sum())
        tot["emulated"] += int(point.sum())
        tot["bit_identical"] += int((point & (b.lo == b.actual_after)).sum())
        tot["int_elements"] += int(w.sum())
    tot["mismatch"] = tot["emulated"] - tot["bit_identical"] - tot["nan_both"]
    tot["not_emulated"] = tot["written"] - tot["emulated"]
    nodes, not_em = Counter(), []
    for (node, how), _ in emu.trace.items():
        nodes[how.split(":")[0]] += 1
        if how.startswith("not_emulable"):
            not_em.append(f"{node} ({how.split(':', 1)[1]})")
    status = ("not_established" if tot["emulated"] == 0 else ("bit_identical" if tot["mismatch"] == 0 else "mismatch"))
    return {"status": status, "elements": dict(tot), "nodes": dict(nodes), "not_emulable": not_em[:4],
            "aborted_programs": len(result.aborted), "abort_reasons": sorted(set(result.aborted.values()))[:3],
            "contractions_from_sass": emu.sass_choices, "product_choices_from_ptx": emu.ptx_choices,
            "seconds": round(time.time() - t0, 2)}


def make_extra(e, cell, rng):
    import catalog as C
    n = cell.grid * cell.block
    if e in C.FLOAT_T:
        return C.to_device(C.fmt_round_array(C.float_dist(rng, "symmetric", e, n), e), e)
    return C.to_device(C.int_dist(rng, "small-integer", e, n), e)


def child(cells_file, out_jsonl):
    prims = {p.pid: p for p in P.all_prims()}
    todo = [json.loads(l) for l in Path(cells_file).read_text().splitlines() if l.strip()]
    with open(out_jsonl, "a") as f:
        for pid, form, param, opts in todo:
            options = dict(opts)
            t0 = time.time()
            try:
                rec = run_cell(prims[pid], form, param, options, cell_id(pid, form, param, options))
            except Exception as e:
                rec = {"pid": pid, "form": form, "param": param, "options": options, "status": "harness error",
                       "error": f"{type(e).__name__}: {str(e)[:300]}", "where": traceback.format_exc().splitlines()[-4:]}
            rec["seconds"] = round(time.time() - t0, 2)
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()


def plan_default(catalog_json=None):
    """Every primitive that compiled in the catalog x every form, default options."""
    prims = P.all_prims()
    ok = None
    if catalog_json and Path(catalog_json).exists():
        cat = json.loads(Path(catalog_json).read_text())
        ok = {r["pid"] for r in cat["records"] if r.get("status") in ("measured", "compiled (not executed)")}
    cells = []
    for p in prims:
        if p.family in ("literal", "scalar", "mx"):
            continue
        if ok is not None and p.pid not in ok:
            continue
        for form, param in FM.FORMS:
            cells.append([p.pid, form, param, sorted(DEFAULT_OPTIONS.items())])
    return cells


OPTION_VALUES = {"enable_fp_fusion": (True, False), "num_warps": (1, 4, 8), "enable_reflect_ftz": (True, False),
                 "block": (32, 128, 512)}


def representatives(catalog_summary):
    """One primitive per (catalog class, family, result type): the lowering classes of the option sweep."""
    import hashlib as H
    rows = json.loads(Path(catalog_summary).read_text())["rows"]
    prims = {p.pid: p for p in P.all_prims()}
    groups = {}
    for r in rows:
        p = prims.get(r["pid"])
        if p is None or p.family in ("literal", "scalar", "mx") or r["class"] in ("not compiled", "no template"):
            continue
        key = (r["class"], p.family, p.outs[0] if p.outs else "-")
        groups.setdefault(key, []).append(p.pid)
    reps = []
    for key, pids in sorted(groups.items()):
        pids.sort(key=lambda x: H.sha256(x.encode()).hexdigest())
        reps.extend(pids[:2])
    return reps


def plan_options(catalog_summary):
    """Greedy pairwise covering over (form, representative primitive, fusion, num_warps, reflect_ftz, block): every
    pair of values of two factors that occurs in an applicable cell is covered at least once."""
    import itertools as I
    reps = representatives(catalog_summary)
    prims = {p.pid: p for p in P.all_prims()}
    forms = list(FM.FORMS)
    app = {(pid, f): FM.build(prims[pid], f[0], f[1]).applicable for pid in reps for f in forms}
    names = ["form", "prim"] + list(OPTION_VALUES)
    vals = [forms, reps] + [list(v) for v in OPTION_VALUES.values()]

    def pairs(row):
        return {(i, j, row[i], row[j]) for i, j in I.combinations(range(len(row)), 2)}
    required = set()
    for f in forms:
        for pid in reps:
            if not app[(pid, f)]:
                continue
            for combo in I.product(*vals[2:]):
                required |= pairs((f, pid) + combo)
    covered, rows = set(), []
    cand_opts = list(I.product(*vals[2:]))
    while required - covered:
        best, gain = None, -1
        need = required - covered
        # seed with an uncovered pair, then pick the best completion
        i, j, vi, vj = next(iter(sorted(need, key=str)))
        for f in forms:
            if i == 0 and f != vi:
                continue
            for pid in reps:
                if (i == 1 and pid != vi) or (j == 1 and pid != vj) or not app[(pid, f)]:
                    continue
                for combo in cand_opts:
                    row = (f, pid) + combo
                    if row[i] != vi or row[j] != vj:
                        continue
                    g = len(pairs(row) & need)
                    if g > gain:
                        best, gain = row, g
        if best is None:
            covered.add((i, j, vi, vj))
            continue
        rows.append(best)
        covered |= pairs(best)
    cells = []
    for f, pid, fus, nw, ftz, blk in rows:
        cells.append([pid, f[0], f[1], sorted({"enable_fp_fusion": fus, "num_warps": nw, "enable_reflect_ftz": ftz,
                                                "block": blk}.items())])
    return cells


def drive(cells, out_jsonl, workers, gpus, batch=25):
    out_jsonl = Path(out_jsonl)
    done = set()
    if out_jsonl.exists():
        for l in out_jsonl.read_text().splitlines():
            r = json.loads(l)
            done.add(cell_id(r["pid"], r["form"], r["param"], r["options"]))
    todo = [c for c in cells if cell_id(c[0], c[1], c[2], dict(c[3])) not in done]
    batches = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    work = HERE.parents[1] / ".cache/composition_study" / f"cell_batches_{out_jsonl.stem}"   # one per run
    work.mkdir(parents=True, exist_ok=True)

    def run_batch(ib):
        i, b = ib
        part = work / f"part_{i}.jsonl"
        remaining = list(b)
        attempts = 0
        while remaining and attempts < len(b) + 2:
            attempts += 1
            cf = work / f"batch_{i}_{attempts}.txt"
            cf.write_text("\n".join(json.dumps(c) for c in remaining))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)])
            timed_out = False
            try:
                subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--cells-file", str(cf),
                                "--out-jsonl", str(part)], env=env, capture_output=True, text=True,
                               timeout=900 * len(remaining))
            except subprocess.TimeoutExpired:
                timed_out = True
            got = set()
            if part.exists():
                for l in part.read_text().splitlines():
                    r = json.loads(l)
                    got.add(cell_id(r["pid"], r["form"], r["param"], r["options"]))
            nxt = [c for c in remaining if cell_id(c[0], c[1], c[2], dict(c[3])) not in got]
            if nxt and len(nxt) == len(remaining):
                # the first remaining cell crashed the child: record it and move on
                c = nxt[0]
                with open(part, "a") as f:
                    f.write(json.dumps({"pid": c[0], "form": c[1], "param": c[2], "options": dict(c[3]),
                                        "status": "harness timeout" if timed_out
                                        else "child crashed (compiler abort)"}) + "\n")
                nxt = nxt[1:]
            remaining = nxt
        return i

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run_batch, list(enumerate(batches))))
    with open(out_jsonl, "a") as f:
        for part in sorted(work.glob("part_*.jsonl")):
            f.write(part.read_text())
            part.unlink()
    for cf in work.glob("batch_*.txt"):
        cf.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-file")
    ap.add_argument("--out-jsonl")
    ap.add_argument("--plan")
    ap.add_argument("--catalog", default=str(HERE / "results/catalog.json"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--filter", default="")
    a = ap.parse_args()
    if a.cells_file:
        child(a.cells_file, a.out_jsonl)
        return
    if a.plan == "default":
        cells = plan_default(a.catalog)
    elif a.plan == "options":
        cells = plan_options(a.catalog)
    else:
        cells = [json.loads(l) for l in Path(a.plan).read_text().splitlines() if l.strip()]
    cells = [c for c in cells if a.filter in json.dumps(c)]
    print(len(cells), "cells")
    drive(cells, a.out_jsonl, a.workers, a.gpus.split(","))


if __name__ == "__main__":
    main()
