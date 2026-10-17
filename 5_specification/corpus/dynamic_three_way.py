#!/usr/bin/env python3
"""PLAN_v3 V3, dynamic part for three_way: on a random sample of 100 bit-identical units (seed 20261016), evaluate f
(the kernel's ATen graph fragment, GRAPH contract, lazily) on the captured inputs at sampled output elements and compare
with the device output K.

Argument binding, without matching values:
  kernel parameters -> call-site buffers: the first `.run(` of the kernel in its output code, zipped with the kernel's
  parameter names; placeholders named like a buffer bind directly; a placeholder named after a graph node binds to the
  remaining input of the same shape and dtype if there is exactly one; a kernel output binds to the node that a later
  kernel of the same file reads from that buffer, else to the only sink node of matching shape and dtype. Anything else
  is reported as ambiguous.

    CUDA_VISIBLE_DEVICES= python 5_specification/corpus/dynamic_three_way.py [--n 100]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent
REPO = SPEC.parent
sys.path.insert(0, str(SPEC))
sys.path.insert(0, str(REPO / "2_tool/src"))

from kernel_spec import aten as A  # noqa: E402
from kernel_spec import fx_text, lazy  # noqa: E402
from kernel_spec.num import NotEstablished, Undefined, ends  # noqa: E402
from kernel_spec.real import R  # noqa: E402

SEED = 20261016
TORCH_DT = {"float32": "f32", "bfloat16": "bf16", "float16": "f16", "float64": "f64", "int64": "i64", "int32": "i32",
            "bool": "b8", "int8": "i8", "uint8": "u8", "int16": "i16"}
META_DT = {"f32": "f32", "bf16": "bf16", "f16": "f16", "f64": "f64", "i64": "i64", "i32": "i32", "b8": "b8", "i8": "i8",
           "u8": "u8", "i16": "i16"}
EPS = {"f32": 2.0 ** -23, "bf16": 2.0 ** -7, "f16": 2.0 ** -10, "f64": 2.0 ** -52}


def meta_shape(meta):
    if meta is None:
        return None, None
    m = re.match(r'Tensor "(\w+)\[([^\]]*)\]', meta)
    if not m:
        return None, None
    dims = [d.strip() for d in m.group(2).split(",") if d.strip()]
    try:
        return META_DT.get(m.group(1), m.group(1)), tuple(int(d) for d in dims)
    except ValueError:
        return META_DT.get(m.group(1), m.group(1)), None


def _dims(text):
    try:
        return tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError:                             # symbolic sizes (dynamic shapes)
        return None


def is_buffer(arg):
    return re.fullmatch(r"[A-Za-z_]\w*", arg) is not None


class OutputCode:
    """Kernel blocks, fragments, signatures, call sites and buffer shapes of one Inductor output-code file."""

    def __init__(self, path):
        t = open(path, errors="ignore").read()
        self.frag, self.params, self.calls, self.bufshape = {}, {}, [], {}
        for m in re.finditer(r"^(\w+) = async_compile\.triton\('(\w+)', '''", t, re.M):
            head = t.rfind("# Topologically Sorted Source Nodes:", 0, m.start())
            end = t.find("''', device_str=", m.end())
            comment = t[head:m.start()]
            lines = comment.split("# Graph fragment:", 1)[1].splitlines() if "# Graph fragment:" in comment else []
            self.frag[m.group(2)] = fx_text.parse_fragment([l for l in lines if l.startswith("#   %")])
            d = re.search(r"def %s\(([^)]*)\)" % m.group(2), t[m.end():end])
            self.params[m.group(2)] = [p.strip().split(":")[0].strip() for p in d.group(1).split(",")] if d else []
        for m in re.finditer(r"^\s+(\w+)\.run\((.*?)(?:, stream=\w+)?\)\s*$", t, re.M):
            self.calls.append((m.group(1), [a.strip() for a in m.group(2).split(",")]))
        for m in re.finditer(r"^\s+(buf\d+) = empty_strided_\w+\(\(([^)]*)\), \(([^)]*)\), torch\.(\w+)\)", t, re.M):
            dims = _dims(m.group(2))
            if dims is not None:
                self.bufshape[m.group(1)] = (TORCH_DT.get(m.group(4)), dims)
        for m in re.finditer(r"assert_size_stride\((\w+), \(([^)]*)\), \(([^)]*)\)", t):
            dims = _dims(m.group(2))
            if dims is not None:
                self.bufshape.setdefault(m.group(1), (None, dims))
        self.node_of_buffer = self._consumers()

    def bind_inputs(self, kernel, args, shapes=None):
        """placeholder name -> buffer name for one call; shapes overrides the buffer shapes known from the code."""
        frag = self.frag.get(kernel, [])
        ptrs = [(p, a) for p, a in zip(self.params[kernel], args) if is_buffer(a)]
        bufshape = dict(self.bufshape, **(shapes or {}))
        names = {a for _, a in ptrs}
        out, unresolved = {}, []
        for n in frag:
            if n["kind"] != "placeholder":
                continue
            if n["name"] in names:
                out[n["name"]] = n["name"]
            else:
                unresolved.append(n)
        used = set(out.values())
        for n in unresolved:
            dt, shape = meta_shape(n["meta"])
            cand = [a for p, a in ptrs if a not in used and not p.startswith("out_ptr")
                    and bufshape.get(a, (None, None))[1] == shape
                    and (bufshape.get(a, (None, None))[0] in (None, dt))]
            if len(cand) == 1:
                out[n["name"]] = cand[0]
                used.add(cand[0])
            else:
                out[n["name"]] = None
        return out

    def _consumers(self):
        m = {}
        for kernel, args in self.calls:
            if kernel not in self.frag:
                continue
            for ph, buf in self.bind_inputs(kernel, args).items():
                if buf and ph != buf:
                    m.setdefault(buf, ph)
        return m


def first_call(oc, kernel):
    return next((args for k, args in oc.calls if k == kernel), None)


def logical(arg, which):
    import torch
    flat = getattr(arg, which)
    if flat is None:
        return None
    off = (arg.data_ptr - arg.storage_ptr) // arg.element_size
    dt = getattr(torch, arg.dtype)
    base = torch.empty(0, dtype=dt).set_(flat.untyped_storage(), 0, (flat.untyped_storage().nbytes() // arg.element_size,))
    v = base.as_strided(arg.shape, arg.stride, off)
    return (v.double() if v.is_floating_point() else v).numpy().copy()


def relerr(f, k, dt):
    """|K - f| / max(|f|, tiny) at the enclosure midpoint (exact agreement gives 0)."""
    if isinstance(f, A.AllowedSet):
        return min(relerr(m, k, dt) for m in f.members)
    if not isinstance(f, R):
        return 0.0 if f == k else float("inf")
    if not f.finite:
        return 0.0 if (f.v == "+inf" and k == float("inf")) or (f.v == "-inf" and k == float("-inf")) else float("inf")
    a, b = f.bounds()
    mid = float((a + b) / 2)
    if not np.isfinite(k):
        return float("inf")
    return abs(k - mid) / max(abs(mid), 2.0 ** -126)


def run_unit(rec, rng):
    from kernel_analyzer.reference_eval.capture import load_launch
    unit_dir = REPO / ".cache/three_way/capture" / rec["unit"]
    oc = OutputCode(REPO / rec["output_code"])
    kernel = rec["kernel"]
    frag_kernel = kernel if kernel in oc.frag else next((k for k in oc.frag if k == kernel), None)
    args = first_call(oc, kernel)
    if args is None or frag_kernel is None:
        return {"status": "no call site in the output code"}
    L = load_launch(unit_dir)
    byname = {a.name: a for a in L.args}
    param_buf = {p: a for p, a in zip(oc.params[kernel], args) if is_buffer(a) and p in byname and byname[p].kind == "tensor"}
    buf_param = {b: p for p, b in param_buf.items()}
    shapes = {b: (TORCH_DT.get(byname[p].dtype), tuple(byname[p].shape)) for p, b in param_buf.items()}
    try:
        ph = oc.bind_inputs(kernel, args, shapes)
    except TypeError:
        return {"status": "ambiguous input binding (symbolic shapes)"}
    if any(v is None for v in ph.values()):
        return {"status": "ambiguous input binding", "unbound": [k for k, v in ph.items() if v is None]}
    inputs = {}
    for name, buf in ph.items():
        arg = byname[buf_param[buf]]
        inputs[name] = lazy.from_array(logical(arg, "before"), TORCH_DT[arg.dtype])
    frag = oc.frag[frag_kernel]
    graph = fx_text.to_graph(frag)
    defined = set(inputs) | {n["name"] for n in graph["nodes"]}
    missing = sorted({r for n in graph["nodes"] for r in fx_text_refs(n)} - defined)
    free = [b for p, b in param_buf.items() if not p.startswith("out_ptr") and b not in ph.values()]
    for name in list(missing):                     # references Inductor listed without a placeholder line
        if name in buf_param and name in free:
            pick = name
        elif len(missing) == 1 and len(free) == 1:
            pick = free[0]
        else:
            continue
        arg = byname[buf_param[pick]]
        inputs[name] = lazy.from_array(logical(arg, "before"), TORCH_DT[arg.dtype])
        free.remove(pick)
        missing.remove(name)
    if missing:
        return {"status": "fragment references values without a placeholder", "missing": missing[:5]}
    computed = {n["name"]: n for n in frag if n["kind"] == "call"}
    used_inside = {r for n in graph["nodes"] for r in fx_text_refs(n)}
    outs = {}
    for p, buf in param_buf.items():
        if not (p.startswith("out_ptr") or p.startswith("in_out_ptr")):
            continue
        arg = byname[p]
        node = oc.node_of_buffer.get(buf)
        if node not in computed:
            dt = TORCH_DT[arg.dtype]
            cand = [nm for nm, n in computed.items() if meta_shape(n["meta"]) == (dt, tuple(arg.shape)) and nm not in used_inside]
            node = cand[0] if len(cand) == 1 else None
        if node is None:
            return {"status": "ambiguous output binding", "param": p}
        outs[p] = node
    graph["outputs"] = list(outs.values())
    result = {"status": "evaluated", "outputs": {}}
    for reading, elide in (("eager", False), ("inductor_default", True)):
        env = lazy.evaluate_lazy(graph, inputs, elide_intermediate_casts=elide)
        for p, node in outs.items():
            arg = byname[p]
            K = logical(arg, "after")
            lt = env[node]
            entry = result["outputs"].setdefault(p, {"node": node})
            if isinstance(lt, Exception):
                entry[reading] = {"outcome": type(lt).__name__, "note": str(lt)[:120]}
                continue
            idxs = {tuple(0 for _ in K.shape), tuple(s - 1 for s in K.shape)} if K.size else set()
            while len(idxs) < min(34, K.size):
                idxs.add(tuple(rng.randrange(s) for s in K.shape))
            errs, undefined, not_established = [], 0, 0
            for i in sorted(idxs):
                try:
                    errs.append(relerr(lt.get(i), float(K[i]) if K.dtype != bool else bool(K[i]), TORCH_DT[arg.dtype]))
                except Undefined:
                    undefined += 1
                except NotEstablished:
                    not_established += 1
            entry[reading] = {"outcome": "value", "elements": len(idxs), "undefined": undefined,
                              "not_established": not_established,
                              "max_rel": max(errs) if errs else None, "median_rel": float(np.median(errs)) if errs else None}
    return result


def fx_text_refs(node):
    out = []

    def walk(x):
        if isinstance(x, dict) and "ref" in x:
            out.append(x["ref"])
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(node.get("args", []))
    walk(list((node.get("kwargs") or {}).values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    recs = json.loads(gzip.open(HERE / "bindings/three_way.json.gz").read())
    frame = sorted((r for r in recs if r["verify"] == "bit_identical" and r["output_code"]), key=lambda r: r["unit"])  # same frame as before
    sample = random.Random(SEED).sample(frame, a.n) if not a.only else [r for r in frame if r["unit"] in a.only]
    rng = random.Random(SEED)
    out_path = HERE / "results/three_way_dynamic.jsonl"
    stat = collections.Counter()
    with open(out_path, "w") as fh:
        for r in sample:
            t = time.time()
            try:
                res = run_unit(r, rng)
            except Exception as e:  # noqa: BLE001 -- recorded, never hidden
                res = {"status": f"error {type(e).__name__}: {str(e)[:160]}"}
            res.update(unit=r["unit"], kernel=r["kernel"], seconds=round(time.time() - t, 1), contract_status=r["status"])
            stat[res["status"].split(":")[0]] += 1
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(r["unit"], res["status"], {p: {k: v.get("max_rel") if isinstance(v, dict) and "max_rel" in v else v
                                              for k, v in o.items() if k != "node"} for p, o in res.get("outputs", {}).items()},
                  res["seconds"], flush=True)
    print(dict(stat))


if __name__ == "__main__":
    sys.exit(main())
