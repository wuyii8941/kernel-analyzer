"""Fix verification (protocol.md section 7): for a unit with a significant bias on an fp32 output, the tool's
attribution names the top site; if the site's source line matches a registered fix, C and A predict the bias after the
fix (site replaced by a correctly rounded implementation), then the fix is applied to the source and replayed.

    python fixes.py units.jsonl fixes.jsonl [--workers N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _align(vals, g, ok):
    from kernel_analyzer.reference_eval.audit import _alignment, _se
    e = vals[ok] - g
    return [_alignment(e, g), _se(e, g)]


def one(unit: str, output: str) -> dict:
    import numpy as np

    from analyze_units import _per_program_elements, analyze, select_programs
    from kernel_analyzer.reference_eval import bias_predict as BP
    from kernel_analyzer.reference_eval import emulate as E
    from kernel_analyzer.reference_eval import intervals as iv
    from kernel_analyzer.reference_eval.capture import load_launch
    from kernel_analyzer.reference_eval.premises import PREMISES, textbook_emulate
    from kernel_analyzer.reference_eval.ttir_eval import ST_OK, KernelReferenceEvaluator, NumericMode
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    from patch_replay import fix_edit, run_patched, source_line

    L = load_launch(Path(unit))
    progs = select_programs(L.grid, _per_program_elements(L))
    module = parse_ttir(L.asm["ttir"])
    module.source_text = L.asm["ttir"]
    ev = KernelReferenceEvaluator(module, mode=NumericMode.NUMERICAL_DIFFERENCE, ttgir=L.asm["ttgir"])
    original = iv.elementary_bounds
    iv.elementary_bounds = BP._float64_elementary(original)
    try:
        b = next(x for x in ev.evaluate(L, programs=progs).buffers.values() if x.name == output)
    finally:
        iv.elementary_bounds = original
    base, emu = E.emulate(L, programs=progs)
    bb = next(x for x in base.buffers.values() if x.name == output)
    ok = b.written & (b.st == ST_OK) & (b.actual_after_st == 0) & np.isfinite(b.actual_after) & (bb.st == ST_OK)
    g = 0.5 * (b.lo[ok] + b.hi[ok])
    row = {"unit": unit, "output": output, "kernel": L.kernel_name, "truth": _align(b.actual_after, g, ok)}
    contrib = {}
    for node in E.rounding_nodes(emu):
        try:
            r, _ = E.emulate(L, exact_nodes=[node], programs=progs)
            x = next(y for y in r.buffers.values() if y.name == output)
            m = ok & (x.st == ST_OK)
            contrib[node] = float(((bb.lo[m] - x.lo[m]) * (0.5 * (b.lo[m] + b.hi[m]))).sum()
                                  / ((0.5 * (b.lo[m] + b.hi[m])) ** 2).sum())
        except Exception as e:
            contrib[node] = None
    ranked = sorted((n for n in contrib if contrib[n] is not None), key=lambda n: -abs(contrib[n]))
    row["contributions"] = {n: contrib[n] for n in ranked[:8]}
    if not ranked:
        row["status"] = "no attributable site"
        return row
    top = ranked[0]
    src = source_line(L, top)
    row["top_site"] = {"node": top, "source": src[:2] if src else None, "code": src[2].strip() if src else None,
                       "contribution": contrib[top]}
    fix = fix_edit(src[2]) if src else None
    if fix is None:
        row["status"] = "no registered fix for the top site"
        return row
    old, new, mech = fix
    row["fix"] = {"mechanism": mech, "line": src[1], "old": old, "new": new}
    # predictions made before the fix is applied
    rc, _ = E.emulate(L, exact_nodes=[top], substitution="rn", programs=progs)
    xc = next(y for y in rc.buffers.values() if y.name == output)
    ra, _ = textbook_emulate(L, premises=PREMISES, exact_nodes=[top], substitution="rn", programs=progs)
    xa = next(y for y in ra.buffers.values() if y.name == output)
    ra0, _ = textbook_emulate(L, premises=PREMISES, programs=progs)
    xa0 = next(y for y in ra0.buffers.values() if y.name == output)
    m = ok & (xc.st == ST_OK) & (xa.st == ST_OK) & (xa0.st == ST_OK)
    gm = 0.5 * (b.lo[m] + b.hi[m])
    row["predicted_after"] = {"C": _align(xc.lo, gm, m), "A": _align(xa.lo, gm, m)}
    row["A_before"] = _align(xa0.lo, gm, m)
    # the real fix on the device
    out = HERE / "fixes" / Path(unit).parent.parent.name / Path(unit).parent.name / Path(unit).name
    run_patched(unit, [[src[1], old, new]], out)
    after = analyze(str(out))
    row["after"] = {"verify": after.get("verify"), "truth": after["outputs"].get(output, {}).get("truth"),
                    "elements": after["outputs"].get(output, {}).get("n")}
    row["status"] = "fixed"
    return row


def _worker(args):
    unit, output, gpu = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        return one(unit, output)
    except Exception as e:
        return {"unit": unit, "output": output, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "trace": traceback.format_exc()[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", default="0,1,2,3")
    a = ap.parse_args()
    from summarize import records
    recs, _ = records(a.units)
    todo = sorted({(r["unit"], r["output"]) for r in recs if r["elem"] == "f32" and r["truth"] not in (None, 0)})
    done = set()
    if Path(a.out).exists():
        done = {(json.loads(l)["unit"], json.loads(l)["output"]) for l in open(a.out) if l.strip()}
    todo = [t for t in todo if t not in done]
    gpus = a.gpus.split(",")
    ctx = mp.get_context("spawn")
    with open(a.out, "a") as f, ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_worker, [(u, o, gpus[i % len(gpus)]) for i, (u, o) in enumerate(todo)]):
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
