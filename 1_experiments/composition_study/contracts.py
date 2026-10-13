"""Contract table of the bias predictor: local deviation relations of the approximate primitives.

For every measured catalog primitive that may be approximate (an approximate math op of the F1 space or a libdevice
real function), the catalog decides the class: correctly rounded on every distribution -> the RN contract (key in
"correctly_rounded"); otherwise its error K - f(x) in output ulps is measured again, on the primitive alone on the
device against the exact value (MPFR), per cell:

* unary: the cell of the input -- sign, binade (-40..40; f16: -14..15) and one of 8 significand buckets of x, 1024
  inputs per (sign, binade), and the zero input; for three significand classes of the input: uniform ("l", the main table), on an 11-bit grid ("s11"),
  on an 8-bit grid ("s8") -- f16- and bf16-valued inputs have their own errors;
* otherwise: the cell of the exact output -- sign, binade and one of 4 significand buckets (zero results in cell
  "z") -- per class pair of the operands' significands (p: power of two, s: <= 11 bits, l: longer; "ll" is the main
  table): operands with independent signs and binades in half the unary range (49152 per pair) and the catalog's
  input distributions (4096 per distribution, classed per element).

Stored per cell: [n, mean, max |e|, rms] in output ulps; and the same over all inputs ("global").  Subnormal operands
are left out (the FTZ domain; the predictor flags subnormal values).  A primitive with an error beyond 2^20 output ulps
has no bounded local relation -- another function (arith.remf, 4_bugs_cases B024) or an error unbounded near a zero of
the function / for huge arguments: no contract, listed as skipped (the predictor abstains on it).  The predictor
(kernel_analyzer.reference_eval.bias_predict) reads the table from 2_tool/src/kernel_analyzer/reference_eval/data/
bias_contracts.json.gz (compact JSON, 4 significant digits).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "2_tool/src"))
sys.path.insert(0, str(HERE))
OUT = ROOT / "2_tool/src/kernel_analyzer/reference_eval/data/bias_contracts.json.gz"
APPROX_OPS = {"arith.divf", "arith.remf", "math.exp", "math.exp2", "math.log", "math.log2", "math.log10",
              "math.log1p", "math.expm1", "math.sqrt", "math.rsqrt", "math.sin", "math.cos", "math.tan",
              "math.tanh", "math.erf", "math.powf", "math.atan", "math.sinh", "math.cosh"}
BINADES = {"f16": range(-14, 16)}
PER_CELL = 1024          # per (sign, binade) of a unary input: 128 per significand bucket
PER_DIST = 4096
WRONG_FUNCTION = 2.0 ** 20
SUBNORMAL_BELOW = {"f32": 2.0 ** -126, "f16": 2.0 ** -14, "bf16": 2.0 ** -126, "f64": 2.0 ** -1022}


def key_of(p) -> str:
    if p.source == "libdevice":
        return f"libdevice|{p.attr}|{p.outs[0]}"
    return f"{p.op}|{p.outs[0]}"


SUB = 8               # significand buckets per binade for a unary primitive's input cells
SUB_BINARY = 4        # ... and for a binary primitive's output cells


def cell_of(v: float, sub: int = 1) -> str:
    """'+e' / '-e' (sign, binade), with a significand bucket '+e/b' when sub > 1; 'z' zero, 'n' non-finite."""
    if v == 0:
        return "z"
    if not math.isfinite(v):
        return "n"
    m, e = math.frexp(abs(v))
    name = ("-" if v < 0 else "+") + str(e - 1)
    return name if sub == 1 else f"{name}/{min(int((2 * m - 1) * sub), sub - 1)}"


def errors(job):
    """[(cell, error in output ulps)] for one chunk: (sem, ins, out_t, cond, rows) with rows (args, K)."""
    sem, ins, out_t, cond, rows = job
    out = []
    for args, K in rows:
        try:
            r = _one(sem, ins, out_t, cond, args, K)
        except ArithmeticError:                          # results at the edge of float64: not a cell
            r = None
        if r is not None:
            out.append(r)
    return out


def _one(sem, ins, out_t, cond, args, K):
    import catalog as C
    import exact as X
    import refs
    if not math.isfinite(K) or any((not math.isfinite(a)) for a in args):
        return None
    if any(a != 0 and abs(a) < SUBNORMAL_BELOW.get(t, 0.0) for a, t in zip(args, ins)):
        return None                                      # subnormal operands: the FTZ domain, flagged by the predictor
    try:
        G = refs.real_value(sem, [F(a) for a in args])
    except Exception:
        return None
    if G is None or (isinstance(G, float) and not math.isfinite(G)) or C.magnitude_class(G, out_t) is not None:
        return None
    Gf = refs.as_frac(G)
    if isinstance(Gf, float):
        return None
    if Gf == 0:                                          # an exact zero result: the cell of a zero input or output
        c = cell_of(args[0], SUB) if cond == "input" else "z"
        return c, (0.0 if K == 0 else math.inf)
    p, emin, emax, _ = X.FORMATS[out_t]
    if abs(Gf) < F(2) ** emin:                           # subnormal results: FTZ / gradual underflow, not a cell
        return None
    u = X.ulp_at(Gf, out_t)
    c = cell_of(args[0], SUB) if cond == "input" else cell_of(float(Gf), SUB_BINARY)
    e = (F(K) - Gf) / u
    return c, (float(e) if abs(e) < 2 ** 60 else math.copysign(math.inf, e))


def short(v, bits):
    """float64 values rounded to ``bits`` significant bits (inputs on a shorter grid: bf16 / f16 values in f32)."""
    m, e = np.frexp(v)
    return np.ldexp(np.rint(np.ldexp(m, bits)), e - bits)


UNARY_CLASSES = {"l": None, "s11": 11, "s8": 8}       # significand class of the input


def inputs_unary(t, rng, bits=None):
    rows = [0.0] * PER_CELL                              # the zero cell
    for e in BINADES.get(t, range(-40, 41)):
        for sign in (1.0, -1.0):
            m = 1.0 + rng.random(PER_CELL)
            v = sign * np.ldexp(m, e)
            if bits is not None:
                v = short(v, bits)
            rows.extend(float(x) for x in v)
    import catalog as C
    return [list(C.fmt_round_array(np.array(rows), t))]


def operand_class(v) -> str:
    """'p' power of two (one significant bit), 's' short (<= 11 bits), 'l' long -- per operand of a binary op."""
    m, _ = np.frexp(np.abs(v))
    i = np.ldexp(m, 53).astype(np.int64)
    low = i & -i
    sb = np.where(i > 0, 53 - np.log2(np.where(low > 0, low, 1)).astype(np.int64), 0)
    return np.where(sb <= 1, "p", np.where(sb <= 11, "s", "l"))


def inputs_dense(p, rng, n=PER_DIST * 12, classes=None):
    """Operands with independent signs and binades (uniform over half the unary range); each operand's significand
    uniform ('l'), on an 8- or 11-bit grid ('s'), or a power of two ('p'), so that the output cells of a binary
    operation are covered densely for every class pair."""
    import catalog as C
    cols = []
    for k, t in enumerate(p.ins):
        lo, hi = (min(BINADES.get(t, range(-40, 41))), max(BINADES.get(t, range(-40, 41))))
        e = rng.integers(lo // 2, hi // 2 + 1, n)
        m = 1.0 + rng.random(n)
        cls = classes[k] if classes else "l"
        if cls == "p":
            m = np.ones(n)
        elif cls == "s":
            m = np.where(rng.random(n) < 0.5, short(m, 8), short(m, 11))
        v = np.where(rng.random(n) < 0.5, -1.0, 1.0) * np.ldexp(m, e)
        cols.append(list(C.fmt_round_array(v, t)))
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--binary-only", action="store_true",
                    help="recompute only the binary primitives and merge them into the existing table at --out")
    a = ap.parse_args()
    import catalog as C
    import primitives as P
    cat = json.loads((HERE / "results/catalog.json").read_text())
    prims = {p.pid: p for p in P.all_prims()}
    table = {"schema": 1, "target": cat.get("target"), "triton": cat.get("triton"),
             "source": "1_experiments/composition_study/contracts.py", "unit": "output ulps", "min_n": 64,
             "approx": {}, "correctly_rounded": [], "skipped": {}}
    todo = []
    for r in cat["records"]:
        p = prims.get(r["pid"])
        if r.get("status") != "measured" or p is None or p.family != "real":
            continue
        if not (p.op in APPROX_OPS or p.source == "libdevice") or p.outs[0] not in C.FLOAT_T:
            continue
        if a.only and a.only not in p.pid:
            continue
        if a.binary_only and len(p.ins) == 1:
            continue
        shares = [v.get("cr_share") for v in (r.get("per_distribution") or {}).values()
                  if isinstance(v, dict) and v.get("n")]
        if shares and all(s == 1 for s in shares if s is not None):
            table["correctly_rounded"].append(key_of(p))
        else:
            todo.append(p)
    print(len(table["correctly_rounded"]), "correctly rounded;", len(todo), "to characterize", flush=True)
    pool = ProcessPoolExecutor(a.workers)
    for p in todo:
        key = key_of(p)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
        try:
            text, _ = C.module_text(p)
            k = C.compile_text(text, {"num_warps": 4})
        except Exception as e:
            table["skipped"][key] = f"{type(e).__name__}: {str(e)[:120]}"
            continue
        unary = len(p.ins) == 1
        designs = {}
        if unary:
            for cls, bits in UNARY_CLASSES.items():
                designs[cls] = inputs_unary(p.ins[0], rng, bits)
        else:
            import itertools
            for pair in itertools.product("psl", repeat=len(p.ins)):
                designs["".join(pair)] = inputs_dense(p, rng, classes=pair)
            for d in C.DISTS:                            # the catalog's distributions, classed per element below
                for c, v in zip(designs["l" * len(p.ins)], C.draw_operands(p, rng, d, PER_DIST)):
                    c.extend(v)
        by_class = {}
        for cls, ops in designs.items():
            dev = C.run_elementwise(k, p, ops)[0]
            rows = list(zip(zip(*ops), dev))
            if not unary:                                # reclass every row by its operands' significands
                labels = ["".join(z) for z in zip(*[operand_class(np.array(col)) for col in ops])]
                groups = defaultdict(list)
                for lab, row in zip(labels, rows):
                    groups[lab].append(row)
            else:
                groups = {cls: rows}
            for lab, rws in groups.items():
                chunks = [(p.sem, p.ins, p.outs[0], "input" if unary else "output", rws[i:i + 2000])
                          for i in range(0, len(rws), 2000)]
                cells = by_class.setdefault(lab, defaultdict(list))
                for part in pool.map(errors, chunks):
                    for c, e in part:
                        cells[c].append(e)
        main = "l" if unary else "l" * len(p.ins)
        cells = by_class.get(main, {})
        if not cells:
            table["skipped"][key] = "no finite normal results"
            continue
        allv = np.concatenate([np.array(v) for cl in by_class.values() for v in cl.values()])
        if not np.all(np.abs(allv) <= WRONG_FUNCTION):
            # no bounded local relation: another function (arith.remf, 4_bugs_cases B024) or an error unbounded in
            # output ulps (near a zero of the function, huge arguments of a fast intrinsic); the predictor abstains
            table["skipped"][key] = (f"errors beyond {WRONG_FUNCTION:.0f} output ulps (largest "
                                     f"{np.abs(allv).max():.3g}): no bounded local relation")
            print(key, table["skipped"][key], flush=True)
            continue

        def stat(v):
            v = np.asarray(v)
            return [int(v.size), float(v.mean()), float(np.abs(v).max()), float(np.sqrt(np.mean(v * v)))]
        table["approx"][key] = {"cond": "input" if unary else "output", "pid": p.pid,
                                "sub": SUB if unary else SUB_BINARY,
                                "cells": {c: stat(v) for c, v in sorted(cells.items())}, "global": stat(allv),
                                "classes": {lab: {c: stat(v) for c, v in sorted(cl.items())}
                                            for lab, cl in sorted(by_class.items()) if lab != main}}
        print(key, table["approx"][key]["global"], len(cells), "cells", flush=True)
    table["correctly_rounded"] = sorted(set(table["correctly_rounded"]))
    if a.binary_only and Path(a.out).exists():
        old = json.loads(gzip.decompress(Path(a.out).read_bytes()))
        for k in [k for k, v in old["approx"].items() if v.get("cond") == "output"]:
            del old["approx"][k]
        for k in [k for k in old["skipped"] if k in table["skipped"] or k in table["approx"]]:
            del old["skipped"][k]
        old["approx"].update(table["approx"])
        old["skipped"].update(table["skipped"])
        table = old
    write_table(table, Path(a.out))


def _round4(x):
    """Four significant digits (the table holds errors in output ulps)."""
    if isinstance(x, float):
        return float(f"{x:.4g}") if math.isfinite(x) else x
    if isinstance(x, list):
        return [_round4(v) for v in x]
    if isinstance(x, dict):
        return {k: _round4(v) for k, v in x.items()}
    return x


def write_table(table: dict, path: Path):
    """Compact JSON, values to 4 significant digits, gzip with a fixed timestamp (byte-reproducible)."""
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(json.dumps(_round4(table), separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode())


if __name__ == "__main__":
    main()
