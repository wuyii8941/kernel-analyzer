#!/usr/bin/env python3
"""FPCore cross-check of the automatic reference (evaluation plan, work item B / RQ1).

Representative pure fragments of the external corpus kernels (docs/external_eval_protocol_20261006.md) are exported
from their captured TTIR to FPCore: one closed FPCore per output element, the real-number program of that element
with the captured inputs as exact rational literals (``let*`` keeps shared subexpressions once).  A third-party
interpreter (titanfp, MPFR with a 241-bit significand, round to nearest) evaluates each FPCore in its own
environment; a violation is a value outside [K_R lo, K_R hi] beyond a slack of 2^-200 |v|.

The exporter shares only the TTIR parser and its literal parsing with the tool: integers, masks and addresses are
executed concretely, floats symbolically, and the interval evaluator is not used.  Fragments the exporter cannot
express (unsupported op, data-dependent ``scf.if``, an undefined masked lane reaching an output) stay in the
denominator.

    python scripts/fpcore_crosscheck.py export                                # ka_main -> results/fpcore/fragments.jsonl
    /data1/tzh/envs/fpcore/bin/python -I scripts/fpcore_crosscheck.py evaluate   # titanfp -> results/fpcore/evaluated.jsonl
    python scripts/fpcore_crosscheck.py summary                               # -> results/fpcore/summary.json
"""
import argparse
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/fpcore"
PRECISION = "(float 15 256)"
SLACK_LOG2 = -200

# representative fragments: (entry, condition); seeds 0 and 1, every program of the launch
FRAGMENTS = [(e, "B1_S1_H256") for e in ("elu_triton", "gelu_triton", "gelu_triton_buggy", "leaky_relu_triton",
                                         "leaky_relu_triton_buggy", "relu_triton", "sigmoid_triton", "silu_triton",
                                         "silu_triton_buggy", "tanh_triton")]
FRAGMENTS += [(e, c) for e in ("softmax_triton", "rmsnorm_triton", "rmsnorm_triton_buggy", "l2norm_triton",
                                "l2norm_triton_buggy") for c in ("B1_S7_H3", "B1_S1_H256")]
FRAGMENTS += [("softmax_triton_buggy", "B1_S7_H3"), ("softmax_triton_buggy", "B1_S1_H256"),
              ("attention_triton", "M2_N16_D16"), ("attention_triton_buggy", "M2_N16_D16"),
              ("flash_attention_triton", "M2_N64_D16"), ("flash_attention_triton_buggy", "M2_N64_D16"),
              ("matmul_triton", "M7_K16_N15"), ("matmul_triton_buggy", "M7_K16_N15")]
SEEDS = (0, 1)


# ----------------------------------------------------------------------------------------------------------------
# symbolic execution of one TTIR program (export side; runs in ka_main)
# ----------------------------------------------------------------------------------------------------------------

class Unsupported(Exception):
    pass


NAME = re.compile(r"^t\d+$")
FOPS = {"arith.addf": "+", "arith.subf": "-", "arith.mulf": "*", "arith.divf": "/", "arith.maxnumf": "fmax",
        "arith.maximumf": "fmax", "arith.minnumf": "fmin", "arith.minimumf": "fmin", "math.exp": "exp",
        "math.exp2": "exp2", "math.log": "log", "math.log2": "log2", "math.sqrt": "sqrt", "math.tanh": "tanh",
        "math.erf": "erf", "math.absf": "fabs", "math.sin": "sin", "math.cos": "cos", "math.fma": "fma"}
CMPF = {"olt": "<", "ult": "<", "ole": "<=", "ule": "<=", "ogt": ">", "ugt": ">", "oge": ">=", "uge": ">=",
        "oeq": "==", "ueq": "==", "one": "!=", "une": "!="}
CMPI = {"slt": np.less, "ult": np.less, "sle": np.less_equal, "ule": np.less_equal, "sgt": np.greater,
        "ugt": np.greater, "sge": np.greater_equal, "uge": np.greater_equal, "eq": np.equal, "ne": np.not_equal}


def literal(v):
    if math.isnan(v):
        return "NAN"
    if math.isinf(v):
        return "INFINITY" if v > 0 else "(- INFINITY)"
    p, q = Fraction(v).as_integer_ratio()
    return str(p) if q == 1 else f"{p}/{q}"


class Graph:
    """Float subexpressions as let-bound nodes (deduplicated by their text)."""

    def __init__(self):
        self.nodes, self.index = [], {}

    def node(self, op, *args):
        expr = f"({op} {' '.join(args)})"
        name = self.index.get(expr)
        if name is None:
            name = f"t{len(self.nodes)}"
            self.nodes.append((expr, [a for a in args if NAME.match(a)]))
            self.index[expr] = name
        return name

    def fpcore(self, root):
        if not NAME.match(root):
            return f"(FPCore () :precision {PRECISION} {root})"
        need, stack = set(), [root]
        while stack:
            n = stack.pop()
            if n in need:
                continue
            need.add(n)
            stack.extend(self.nodes[int(n[1:])][1])
        binds = " ".join(f"[{n} {self.nodes[int(n[1:])][0]}]" for n in sorted(need, key=lambda s: int(s[1:])))
        return f"(FPCore () :precision {PRECISION} (let* ({binds}) {root}))", len(need)


class V:
    """kind: 'i' concrete ints / bools, 'p' pointers (buffer, element offsets), 'f' symbolic floats, 'b' symbolic bools."""

    def __init__(self, kind, data, buf=None):
        self.kind, self.data, self.buf = kind, data, buf


def _obj(shape, fill):
    a = np.empty(shape, dtype=object)
    a[...] = fill
    return a


def _sym(v, shape=None):
    """A float value as an object array of node names / literals."""
    if v.kind == "f":
        return v.data
    raise Unsupported(f"float expected, got {v.kind}")


def _map(fn, *arrays):
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=object) for a in arrays])
    out = np.empty(arrays[0].shape, dtype=object)
    for idx in np.ndindex(out.shape):
        out[idx] = fn(*(a[idx] for a in arrays))
    return out


class Program:
    def __init__(self, func, bindings, memory, pid, grid, graph):
        self.func, self.memory, self.pid, self.grid, self.g = func, memory, pid, grid, graph
        self.env = dict(bindings)

    def run(self):
        self._block(self.func.body.entry, [])

    def _block(self, block, args):
        for (name, _t), v in zip(block.args, args):
            self.env[name] = v
        for op in block.ops:
            if op.name in ("scf.yield", "tt.reduce.return"):
                return [self.env[o] for o in op.operands]
            if op.name == "tt.return":
                return []
            res = self._op(op)
            if res is None:
                continue
            res = res if isinstance(res, list) else [res]
            for r, v in zip(op.results, res):
                self.env[r] = v
        return []

    def _op(self, op):
        n, a = op.name, [self.env[o] for o in op.operands]
        rt = op.result_types[0] if op.result_types else None
        if n == "arith.constant":
            from kernel_analyzer.reference_eval.ttir_eval import _parse_scalar_literal

            text = op.attrs["value"].strip()
            if text.startswith("dense<"):
                text = text[len("dense<"):-1]
                if text.startswith("["):
                    raise Unsupported("non-splat dense constant")
            val = _parse_scalar_literal(text, rt.elem)
            if str(rt.elem).startswith(("f", "bf")):
                return V("f", _obj(rt.shape, literal(float(val))))
            return V("i", np.full(rt.shape, int(val), dtype=np.int64))
        if n == "tt.get_program_id":
            return V("i", np.array(self.pid[{"x": 0, "y": 1, "z": 2}[op.attrs["axis"]]], dtype=np.int64))
        if n == "tt.get_num_programs":
            return V("i", np.array(self.grid[{"x": 0, "y": 1, "z": 2}[op.attrs["axis"]]], dtype=np.int64))
        if n == "tt.make_range":
            return V("i", np.arange(int(op.attrs["start"].split(":")[0]), int(op.attrs["end"].split(":")[0])))
        if n in ("tt.splat", "tt.broadcast"):
            x = a[0]
            return V(x.kind, np.broadcast_to(x.data, rt.shape).copy(), x.buf)
        if n == "tt.expand_dims":
            x = a[0]
            return V(x.kind, np.expand_dims(x.data, int(op.attrs["axis"].split(":")[0])), x.buf)
        if n == "tt.reshape":
            if "allow_reorder" in op.attrs:
                raise Unsupported("reshape with allow_reorder")
            return V(a[0].kind, np.reshape(a[0].data, rt.shape), a[0].buf)
        if n == "tt.addptr":
            p, off = a
            return V("p", p.data + off.data, p.buf)
        if n in ("arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.remsi", "arith.shli", "arith.shrsi",
                 "arith.maxsi", "arith.minsi"):
            x, y = a[0].data, a[1].data
            f = {"arith.addi": np.add, "arith.subi": np.subtract, "arith.muli": np.multiply,
                 "arith.divsi": lambda p, q: np.trunc(p / q).astype(np.int64), "arith.remsi": np.fmod,
                 "arith.shli": np.left_shift, "arith.shrsi": np.right_shift, "arith.maxsi": np.maximum,
                 "arith.minsi": np.minimum}[n]
            return V("i", np.asarray(f(x, y), dtype=np.int64))
        if n in ("arith.andi", "arith.ori", "arith.xori"):
            if a[0].kind == "i" and a[1].kind == "i":
                f = {"arith.andi": np.bitwise_and, "arith.ori": np.bitwise_or, "arith.xori": np.bitwise_xor}[n]
                return V("i", f(a[0].data, a[1].data))
            if n == "arith.xori":
                raise Unsupported("symbolic xor")
            word = "and" if n == "arith.andi" else "or"
            return V("b", _map(lambda p, q: self._bool(word, p, q), self._as_b(a[0]), self._as_b(a[1])))
        if n == "arith.cmpi":
            return V("i", CMPI[op.attrs["predicate"]](a[0].data, a[1].data))
        if n in ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast"):
            return V("i", a[0].data)
        if n in ("arith.sitofp", "arith.uitofp"):
            return V("f", _map(lambda q: literal(float(q)), a[0].data))
        if n in ("arith.extf", "arith.truncf"):
            return V("f", _sym(a[0]))  # rounding to the destination format is numerical, not semantic
        if n == "arith.negf":
            return V("f", _map(lambda x: self.g.node("-", x), _sym(a[0])))
        if n == "math.rsqrt":
            return V("f", _map(lambda x: self.g.node("/", "1", self.g.node("sqrt", x)), _sym(a[0])))
        if n in FOPS:
            return V("f", _map(lambda *xs: self.g.node(FOPS[n], *xs), *[_sym(x) for x in a]))
        if n == "arith.cmpf":
            sym = CMPF[op.attrs["predicate"]]
            return V("b", _map(lambda x, y: self.g.node(sym, x, y), _sym(a[0]), _sym(a[1])))
        if n == "arith.select":
            c, x, y = a
            if c.kind == "i":
                if x.kind == "i":
                    return V("i", np.where(c.data, x.data, y.data))
                return V("f", np.where(np.broadcast_to(c.data, np.shape(x.data)).astype(bool), x.data, y.data))
            return V("f", _map(lambda q, p, r: self.g.node("if", q, p, r), c.data, _sym(x), _sym(y)))
        if n == "tt.load":
            return self._load(op, a)
        if n == "tt.store":
            self._store(a)
            return None
        if n == "tt.reduce":
            return self._reduce(op, a)
        if n == "scf.for":
            lb, ub, step = (int(np.asarray(x.data)) for x in a[:3])
            carried = a[3:]
            i = lb
            while (step > 0 and i < ub) or (step < 0 and i > ub):
                carried = self._block(op.regions[0].entry, [V("i", np.array(i, dtype=np.int64))] + carried)
                i += step
            return carried
        if n == "scf.if":
            c = a[0]
            if c.kind != "i":
                raise Unsupported("data-dependent scf.if")
            region = op.regions[0] if bool(np.asarray(c.data)) else (op.regions[1] if len(op.regions) > 1 else None)
            return self._block(region.entry, []) if region is not None else []
        raise Unsupported(n)

    def _as_b(self, v):
        if v.kind == "b":
            return v.data
        return _map(lambda q: "TRUE" if q else "FALSE", v.data)

    def _bool(self, word, p, q):
        if "FALSE" in (p, q) and word == "and":
            return "FALSE"
        if "TRUE" in (p, q) and word == "or":
            return "TRUE"
        if p in ("TRUE", "FALSE"):
            return q
        if q in ("TRUE", "FALSE"):
            return p
        return self.g.node(word, p, q)

    def _load(self, op, a):
        ptr = a[0]
        mask = a[1].data if len(a) > 1 else np.ones(np.shape(ptr.data), dtype=bool)
        if len(a) > 1 and a[1].kind != "i":
            raise Unsupported("data-dependent load mask")
        other = _sym(a[2]) if len(a) > 2 else None
        buf = self.memory[ptr.buf]
        offs = np.asarray(ptr.data)
        out = np.empty(offs.shape, dtype=object)
        for idx in np.ndindex(offs.shape):
            if np.broadcast_to(mask, offs.shape)[idx]:
                o = int(offs[idx])
                out[idx] = buf["stored"].get(o, literal(float(buf["values"][o])))
            else:
                out[idx] = np.broadcast_to(other, offs.shape)[idx] if other is not None else "UNDEF"
        return V("f", out)

    def _store(self, a):
        ptr, val = a[0], a[1]
        mask = a[2].data if len(a) > 2 else np.ones(np.shape(ptr.data), dtype=bool)
        offs = np.asarray(ptr.data)
        vals = np.broadcast_to(_sym(val), offs.shape)
        for idx in np.ndindex(offs.shape):
            if np.broadcast_to(mask, offs.shape)[idx]:
                self.memory[ptr.buf]["stored"][int(offs[idx])] = vals[idx]

    def _reduce(self, op, a):
        if len(a) != 1:
            raise Unsupported("multi-operand reduce")
        axis = int(op.attrs["axis"].split(":")[0])
        x = np.moveaxis(np.asarray(a[0].data, dtype=object), axis, 0)
        acc = V(a[0].kind, x[0])
        for k in range(1, x.shape[0]):
            (acc,) = self._block(op.regions[0].entry, [acc, V(a[0].kind, x[k])])
        return V(acc.kind, np.asarray(acc.data, dtype=object).reshape(op.result_types[0].shape) if op.result_types[0].shape
                 else np.asarray(acc.data, dtype=object).reshape(()))


def _decode(arg):
    raw = arg.before.numpy() if hasattr(arg.before, "numpy") else np.asarray(arg.before)
    dt = {"float32": np.float32, "float64": np.float64, "int32": np.int32,
          "int64": np.int64}.get(str(arg.dtype).replace("torch.", ""))
    if dt is None:
        raise Unsupported(f"dtype {arg.dtype}")
    return np.asarray(raw).view(dt).astype(np.float64)


def export_launch(launch):
    """{storage element offset: (fpcore or None, reason)} for every float element the launch stores."""
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    func = parse_ttir(launch.asm["ttir"]).entry()
    params, captured = func.params, launch.ttir_params()
    if len(params) != len(captured):
        raise Unsupported("parameter / argument mismatch")
    memory, bindings = {}, {}
    for (name, ttype, _a), arg in zip(params, captured):
        if ttype.is_ptr:
            memory.setdefault(arg.storage_ptr, {"values": _decode(arg), "stored": {}})
            bindings[name] = V("p", np.array((arg.data_ptr - arg.storage_ptr) // arg.element_size), arg.storage_ptr)
        elif str(ttype.elem).startswith("f"):
            v = float(arg.value)
            v = float(np.float32(v)) if ttype.elem == "f32" else v  # a Python float reaches an f32 parameter rounded
            bindings[name] = V("f", np.array(literal(v), dtype=object))
        else:
            bindings[name] = V("i", np.array(int(arg.value), dtype=np.int64))
    graph = Graph()
    grid = tuple(launch.grid) + (1,) * (3 - len(launch.grid))
    for pid in np.ndindex(grid[::-1]):
        Program(func, bindings, memory, pid[::-1], grid, graph).run()
    out = {}
    for ptr, buf in memory.items():
        for off, expr in buf["stored"].items():
            if "UNDEF" in str(expr):
                out[(ptr, off)] = (None, "undefined masked lane reaches the output")
                continue
            fp = graph.fpcore(expr)
            out[(ptr, off)] = (fp[0] if isinstance(fp, tuple) else fp, None)
    return out, graph


def export():
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    import external_corpus as ec
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    from kernel_analyzer.reference_eval.ttir_eval import ST_OK, evaluate_sequence

    TritonLaunchRecorder.install_hook()
    entries = {e["name"]: e for e in ec.entries()}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fragments.jsonl"
    path.write_text("")
    for entry_name, cname in FRAGMENTS:
        e = entries[entry_name]
        cond = next(c for c in ec.conditions(e["meta"]) if c["name"] == cname)
        for seed in SEEDS:
            case = ec.CorpusCase(e, cond)
            inp = case.inputs(seed)
            rec = TritonLaunchRecorder()
            with rec:
                outs = case.launch(inp)
                torch.cuda.synchronize()
            launch = rec.launches[0]
            out_ptr = outs["out"].untyped_storage().data_ptr()
            row = {"entry": entry_name, "condition": cname, "seed": seed, "launches": len(rec.launches)}
            try:
                exported, graph = export_launch(launch)
                row["graph_nodes"] = len(graph.nodes)
            except Unsupported as exc:
                row.update(exported=False, reason=str(exc))
                with open(path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(entry_name, cname, seed, "unsupported:", exc, flush=True)
                continue
            seq = evaluate_sequence(rec.launches)
            buf = seq.memory[out_ptr]
            elems = []
            for (ptr, off), (fp, why) in sorted(exported.items()):
                if ptr != out_ptr:
                    continue
                ok = bool(buf.st[off] == ST_OK and not buf.cond[off])
                elems.append({"offset": off, "fpcore": fp, "reason": why, "kr_ok": ok,
                              "kr_lo": float(buf.lo[off]), "kr_hi": float(buf.hi[off]) if buf.hi is not None else float(buf.lo[off]),
                              "K": float(buf.actual_after[off])})
            row.update(exported=True, elements=elems)
            with open(path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(entry_name, cname, seed, len(elems), "elements,", row["graph_nodes"], "nodes", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# evaluation (titanfp; runs in /data1/tzh/envs/fpcore)
# ----------------------------------------------------------------------------------------------------------------

def evaluate():
    from titanfp.arithmetic import ieee754
    from titanfp.fpbench import fpcparser

    interp = ieee754.Interpreter()
    src = OUT / "fragments.jsonl"
    dst = OUT / "evaluated.jsonl"
    dst.write_text("")
    for line in src.read_text().splitlines():
        row = json.loads(line)
        res = {k: row[k] for k in ("entry", "condition", "seed")}
        res.update(exported=row["exported"], reason=row.get("reason"), checked=0, violations=0, not_exported=0,
                   kr_not_complete=0, max_excess_rel=0.0, examples=[])
        for el in row.get("elements", []):
            if el["fpcore"] is None:
                res["not_exported"] += 1
                continue
            if not el["kr_ok"]:
                res["kr_not_complete"] += 1
                continue
            r = interp.interpret(fpcparser.compile1(el["fpcore"]), [])
            if r._isnan or r._isinf:
                res["violations"] += 1
                res["examples"].append({"offset": el["offset"], "value": str(r)})
                continue
            v = Fraction(int(r._c)) * (Fraction(2) ** int(r._exp))
            v = -v if r._negative else v
            slack = abs(v) * Fraction(2) ** SLACK_LOG2
            lo, hi = Fraction(el["kr_lo"]), Fraction(el["kr_hi"])
            res["checked"] += 1
            if v < lo - slack or v > hi + slack:
                res["violations"] += 1
                excess = max(lo - v, v - hi) / max(abs(v), Fraction(1, 10 ** 300))
                res["max_excess_rel"] = max(res["max_excess_rel"], float(excess))
                if len(res["examples"]) < 3:
                    res["examples"].append({"offset": el["offset"], "value": float(v), "kr": [el["kr_lo"], el["kr_hi"]]})
        with open(dst, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(res["entry"], res["condition"], res["seed"], "checked", res["checked"], "violations", res["violations"],
              flush=True)


def summary():
    rows = [json.loads(line) for line in (OUT / "evaluated.jsonl").read_text().splitlines()]
    frag = [json.loads(line) for line in (OUT / "fragments.jsonl").read_text().splitlines()]
    s = {"evaluator": "titanfp 0.1.2 (ieee754 interpreter, MPFR), precision " + PRECISION + ", round to nearest even",
         "slack": f"2^{SLACK_LOG2} |v|", "fragments": len(rows),
         "fragments_exported": sum(r["exported"] for r in rows),
         "fragments_not_exported": [{k: r[k] for k in ("entry", "condition", "seed", "reason")} for r in rows if not r["exported"]],
         "elements_checked": sum(r["checked"] for r in rows), "violations": sum(r["violations"] for r in rows),
         "elements_not_exported": sum(r["not_exported"] for r in rows),
         "elements_with_incomplete_reference": sum(r["kr_not_complete"] for r in rows),
         "max_graph_nodes": max((f.get("graph_nodes", 0) for f in frag), default=0),
         "per_fragment": rows}
    (OUT / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps({k: v for k, v in s.items() if k != "per_fragment"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["export", "evaluate", "summary"])
    a = ap.parse_args()
    {"export": export, "evaluate": evaluate, "summary": summary}[a.command]()


if __name__ == "__main__":
    main()
