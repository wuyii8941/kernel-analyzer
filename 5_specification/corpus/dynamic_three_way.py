#!/usr/bin/env python3
"""PLAN_v3 V3, dynamic part for three_way: on a random sample of 100 bit-identical units (seed 20261016), evaluate f
(the kernel's ATen graph fragment, GRAPH contract, lazily) on the captured inputs at sampled output elements and compare
with the device output K.

Argument binding uses recorded compiler origins for the captured source, the call site of the captured (first) launch,
checked against the provenance records of every call site, and IR write analysis.
Names and matching shapes alone do not establish origins. Every possible write needs an output node. The result is
structural binding evidence, not an independent review of the graph's task semantics. Numerical comparisons sample
output elements and do not certify full outputs.

    CUDA_VISIBLE_DEVICES= python 5_specification/corpus/dynamic_three_way.py --bindings CANDIDATES.json.gz --out NEW.jsonl [--n 100]
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import gzip
import json
import math
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
from bind_inductor import norm_template  # noqa: E402
from kernel_spec.num import Invalid, NotEstablished, Undefined, Unsupported, ends  # noqa: E402
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


def compiler_view_metadata(source):
    """Literal layouts and uniquely assigned reinterpret_tensor views in generated code."""
    views, layouts, assignments = {}, {}, collections.Counter()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return views, layouts

    def layout(shape, stride):
        try:
            pair = tuple(ast.literal_eval(x) for x in (shape, stride))
            if (all(isinstance(x, (tuple, list)) for x in pair) and len(pair[0]) == len(pair[1])
                    and all(type(n) is int and n >= 0 for x in pair for n in x)):
                return tuple(map(tuple, pair))
        except (ValueError, TypeError):
            pass
        return None

    def record(name, value):
        if name not in layouts:
            layouts[name] = value
        elif layouts[name] != value:
            layouts[name] = None  # conflicting declarations need control-flow evidence

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] += 1
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name, value = node.targets[0].id, node.value
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
                continue
            if value.func.id.startswith("empty_strided_") and len(value.args) >= 2:
                record(name, layout(*value.args[:2]))
            if (value.func.id == "reinterpret_tensor" and len(value.args) == 4
                    and not value.keywords and isinstance(value.args[0], ast.Name)):
                pair = layout(*value.args[1:3])
                try:
                    offset = ast.literal_eval(value.args[3])
                except (ValueError, TypeError):
                    offset = None
                if pair is not None and type(offset) is int:
                    views[name] = dict(source=value.args[0].id, shape=pair[0], stride=pair[1], offset=offset)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "assert_size_stride" and len(node.args) >= 3
                and isinstance(node.args[0], ast.Name)):
            record(node.args[0].id, layout(*node.args[1:3]))
    return {name: view for name, view in views.items()
            if assignments[name] == 1 and assignments[view["source"]] <= 1
            and view["source"] not in views}, layouts


def is_buffer(arg):
    return re.fullmatch(r"[A-Za-z_]\w*", arg) is not None


class OutputCode:
    """Kernel blocks, fragments, signatures, call sites and buffer shapes of one Inductor output-code file."""

    def __init__(self, path):
        t = open(path, errors="ignore").read()
        self.views, self.layout_assertions = compiler_view_metadata(t)
        self.frag, self.params, self.calls, self.bufshape = {}, {}, [], {}
        self.launcher_meta = {}
        for m in re.finditer(r"^(\w+) = async_compile\.triton\('(\w+)', '''", t, re.M):
            head = t.rfind("# Topologically Sorted Source Nodes:", 0, m.start())
            end = t.find("''', device_str=", m.end())
            comment = t[head:m.start()]
            lines = comment.split("# Graph fragment:", 1)[1].splitlines() if "# Graph fragment:" in comment else []
            self.frag[m.group(2)] = fx_text.parse_fragment([l for l in lines if l.startswith("#   %")])
            d = re.search(r"def %s\(([^)]*)\)" % m.group(2), t[m.end():end])
            self.params[m.group(2)] = [p.strip().split(":")[0].strip() for p in d.group(1).split(",")] if d else []
            self.launcher_meta[m.group(2)] = launcher_metadata(t[m.end():end])
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
        for n in unresolved:
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


_PROV = {}
PROV_DIR = REPO / ".cache/three_way/provenance_v3"      # prov_model.py with origin_node and in-place alias origins


def provenance(rec, capture_root=None, provenance_root=None):
    """The provenance records of this unit's kernel (corpus/prov_model.py), in definition order: those whose kernel file
    path is the one the captured TTIR names; for a template kernel that the final code does not contain (an autotuning
    candidate), those of the same name whose kernel file equals the captured one after removing the autotuning
    configuration lines (corpus/bind_inductor.py:norm_template). Inductor reuses one kernel for identical source, so a
    kernel may have one record per call site. Returns (records, method), or (None, reason)."""
    model, config, name = rec["unit"].split("/")
    f = (provenance_root or PROV_DIR) / model / f"{config}.jsonl"
    if not f.exists():
        return None, "no provenance file for the model-config"
    if f not in _PROV:
        recs = []
        for r in map(json.loads, open(f)):
            # define_kernel can run twice for the same graph (identical records); they are one definition
            if "kernel_path" in r and r not in recs[-1:]:
                recs.append(r)
        _PROV[f] = recs
    records = _PROV[f]
    if any("output_code" in r for r in records):
        selected = rec.get("output_code")
        records = [r for r in records if selected and r.get("output_code")
                   and Path(r["output_code"]).resolve() == (REPO / selected).resolve()]
        if not records:
            return None, "no provenance for the selected output code"
    src = next(((capture_root or REPO / ".cache/three_way/capture") / rec["unit"] / "compiled").glob("*.source")).read_text(errors="ignore")
    m = re.search(r'#loc = loc\("([^"]+\.py)"', src)
    if not m:
        return None, "captured source names no kernel file"
    same = [r for r in records if r["kernel_path"] == m.group(1)]
    if same:
        return same, "kernel file path"
    kfile = Path(m.group(1))
    # A reused compiled binary can retain its original #loc after the source
    # is emitted into a new cache directory. Compare the complete source text;
    # matching names or matching basenames do not establish correspondence.
    if kfile.is_file():
        body = kfile.read_text()
        same = [r for r in records if Path(r["kernel_path"]).is_file()
                and Path(r["kernel_path"]).read_text() == body]
        if same:
            return same, "identical kernel source at a different cache path"
    if "_tem_" in rec["kernel"] and kfile.is_file():
        want = norm_template(kfile.read_text())
        same = [r for r in records if r["kernel"] == rec["kernel"] and Path(r["kernel_path"]).is_file()
                and norm_template(Path(r["kernel_path"]).read_text()) == want]
        if same and len({r["kernel_path"] for r in same}) == 1:
            return same, "template source without autotuning configuration"
        if same:
            return None, "template matches several final kernels"
    return None, "no provenance for the captured kernel source"


def select_call(oc, kernel, records):
    """The call site of the captured launch. The capture records the first launch of a kernel, and Inductor's call()
    is straight-line code, so it is the first call site, provided every call site of the kernel in this output code
    has exactly one provenance record with the same buffers, in the same order (this also excludes launches of the
    same kernel file from another graph). Returns (args, record) or (None, reason)."""
    params = oc.params.get(kernel)
    calls = [args for k, args in oc.calls if k == kernel]
    if not calls or params is None:
        return None, "no call site in the output code"

    def bufs(args):
        return {p: a for p, a in zip(params, args) if is_buffer(a)}

    def agrees(r, args):
        b = bufs(args)
        return all(b.get(p) == info["buffer"] for p, info in r["params"].items())

    if len(records) != len(calls) or not all(agrees(r, a) for r, a in zip(records, calls)):
        return None, f"{len(calls)} call sites and {len(records)} provenance records do not correspond in order"
    return calls[0], records[0]


def launcher_metadata(source):
    """Read only literal launcher metadata from the source; do not import it."""
    try:
        tree = ast.parse(source)
        entries = [ast.literal_eval(k.value) for node in ast.walk(tree) if isinstance(node, ast.Call)
                   for k in node.keywords if k.arg == "inductor_meta"]
        return entries[0] if len(entries) == 1 and isinstance(entries[0], dict) else {}
    except (SyntaxError, ValueError, TypeError):
        return {}


def bind_call_scalars(params, args, launch, launcher_meta=None):
    """Check explicit call operands against the capture; never evaluate call-site code.

    Autotuning constexpr parameters may be omitted from the call. Their recorded
    values are not a claim that the mathematical graph uses the same profile.
    Extra launcher operands require a separate mapping, not silent zip truncation.
    """
    captured = {a.name: a for a in launch.args}
    if len(captured) != len(launch.args) or len(set(params)) != len(params) or set(params) != set(captured):
        return None, "source signature and captured parameters do not correspond exactly"
    values = {}
    if len(args) > len(params):
        meta = launcher_meta or {}
        extra = meta.get("extra_launcher_args", [])
        if (meta.get("grid_type") != "FixedGrid" or len(extra) != 3 or
                not all(isinstance(n, str) for n in extra) or len(set(extra)) != 3 or
                set(extra) & set(params) or meta.get("fixed_grid") != extra or
                len(args) != len(params) + 3):
            return None, "extra launcher operands have no parameter mapping"
        try:
            grid = [ast.literal_eval(a) for a in args[len(params):]]
        except (ValueError, SyntaxError):
            return None, "launcher grid is not resolved"
        if any(type(v) is not int for v in grid) or tuple(grid) != tuple(launch.grid):
            return None, "launcher grid differs from the captured grid"
        values.update(zip(extra, grid))
        args = args[:len(params)]
    for i, name in enumerate(params):
        arg = captured[name]
        if i >= len(args):
            if arg.constexpr and arg.kind != "tensor":
                continue
            return None, f"call omits captured parameter {name}"
        if arg.kind == "tensor":
            if not is_buffer(args[i]):
                return None, f"tensor argument {name} is not a recorded buffer name"
            continue
        try:
            value = ast.literal_eval(args[i])
        except (ValueError, SyntaxError):
            return None, f"scalar {name} is not a resolved literal"
        expected_type = {"int": int, "float": float, "bool": bool, "none": type(None)}.get(arg.kind)
        if expected_type is None or type(value) is not expected_type or type(arg.value) is not expected_type:
            return None, f"scalar {name} has an unsupported or different type"
        same = value.hex() == arg.value.hex() if expected_type is float else value == arg.value
        if not same:
            return None, f"scalar {name} differs from the captured value"
        values[name] = value
    return values, None


def input_buffers_of(name, params, param_buf, among=None):
    """Call-site buffers of the input parameters that hold the value of FX node `name`: the buffer is named after it,
    Inductor recorded it as the buffer's origin node, or as an alias name of an in-place buffer; only when none of
    these exact records exists, a buffer with `name` among its fused origins."""
    def hits(pred):
        return {param_buf.get(p) for p, info in params.items() if info["role"] in ("input", "inplace") and pred(info)
                and (among is None or param_buf.get(p) in among)} - {None}
    exact = hits(lambda i: name in (i["buffer"], i.get("origin_node")) or name in i.get("other_names", ())
                 or name in (i.get("other_origin_nodes") or {}).values())
    return exact or hits(lambda i: name in i["origins"])


def _output_nodes(params, frag, written_params):
    """output param -> fragment node: the buffer's recorded origin node, else the unique sink among its origins."""
    computed = {n["name"]: n for n in frag if n["kind"] == "call"}
    used_by = {}
    for n in computed.values():
        for r in fx_text_refs({"args": n["args"], "kwargs": n["kwargs"]}):
            used_by.setdefault(r, set()).add(n["name"])
    outs = {}
    for p, info in params.items():
        if p not in written_params:
            continue
        if info["role"] not in ("output", "inplace"):
            return None, f"write to parameter {p} recorded as {info['role']}"
        if info.get("origin_node") in computed:
            outs[p] = info["origin_node"]
            continue
        origin_nodes = [o for o in info["origins"] if o in computed]
        sinks = [o for o in origin_nodes if not (used_by.get(o, set()) & set(origin_nodes))]
        if len(sinks) != 1:
            return None, f"output {p} has {len(sinks)} candidate nodes among its origins"
        outs[p] = sinks[0]
    return outs, None


def cut_fragment(frag, params, outs):
    """The part of the fragment this kernel computes. Inductor's fragment can include nodes that an earlier kernel
    computed: when an input parameter's buffer is recorded as holding the value of a computed node (origin_node), the
    kernel receives that value, so the node becomes a placeholder. A placeholder line for a node the fragment itself
    computes is dropped (the value is internal). Nodes are then removed only if every path from them to a sink of the
    fragment or to an output node passes through such a cut; all other nodes, unused placeholders included, stay and
    must be bound."""
    computed = {n["name"] for n in frag if n["kind"] == "call"}
    out_nodes = set(outs.values())
    cut = {info["origin_node"] for info in params.values() if info["role"] == "input"
           and info.get("origin_node") in computed and info["origin_node"] not in out_nodes}
    nodes = []
    for n in frag:
        if n["kind"] == "placeholder" and n["name"] in computed:
            continue
        if n["kind"] == "call" and n["name"] in cut:
            n = {"name": n["name"], "kind": "placeholder", "meta": n.get("meta")}
        nodes.append(n)
    def refs_of(ns):
        return {n["name"]: (fx_text_refs({"args": n["args"], "kwargs": n["kwargs"]}) if n["kind"] == "call" else set())
                for n in ns}
    refs = refs_of(nodes)
    used_before_cut = {r for rs in refs_of(frag).values() for r in rs}
    stack = [n["name"] for n in nodes if n["name"] not in used_before_cut] + sorted(out_nodes)
    keep = set()
    while stack:
        x = stack.pop()
        if x in keep or x not in refs:
            continue
        keep.add(x)
        stack.extend(refs[x])
    return [n for n in nodes if n["name"] in keep]


def bind_from_provenance(prov, frag, param_buf, written_params=None):
    """placeholder -> buffer and output param -> node from the recorded parameter -> buffer -> origin nodes, on the
    fragment cut to what the kernel computes (cut_fragment)."""
    params = prov["params"]
    if written_params is None:
        return None, "write effects not established"
    if not written_params:
        return None, "no observable writes established"
    missing = (set(params) - set(param_buf)) | (set(written_params) - set(params))
    if missing:
        return None, f"parameters missing from capture or provenance: {sorted(missing)}"
    outs, why = _output_nodes(params, frag, written_params)
    if outs is None:
        return None, why
    ph = {}
    for n in cut_fragment(frag, params, outs):
        if n["kind"] != "placeholder":
            continue
        bufs = input_buffers_of(n["name"], params, param_buf)
        if len(bufs) != 1:
            return None, f"placeholder {n['name']} matches {len(bufs)} parameters"
        ph[n["name"]] = bufs.pop()
    return ph, outs


def tensor_corresponds(node, arg):
    """A tuple-valued node or a reshape requires an explicit output mapping."""
    dtype, shape = meta_shape(node.get("meta"))
    return (dtype is not None and shape is not None
            and dtype == TORCH_DT.get(arg.dtype) and shape == tuple(arg.shape))


def complete_fragment(frag, task_graph, params, param_buf):
    """Restore omitted FX definitions; captured input values remain boundaries.

    This uses the compiler graph as task-binding evidence, not kernel arithmetic.
    No tuple component is selected by its shape or by its observed value.
    """
    full = {}
    for record in task_graph["nodes"]:
        name = record["name"]
        if name in full:
            raise ValueError(f"duplicate task graph node {name}")
        if record["op"] not in ("placeholder", "call_function"):
            continue
        value = record.get("value", {})
        meta = ""
        if value.get("kind") == "tensor":
            dtype = TORCH_DT.get(value["dtype"].removeprefix("torch."))
            shape, stride = value["shape"], value["stride"]
            if dtype and all(re.fullmatch(r"\d+", x) for x in shape + stride):
                meta = f'Tensor "{dtype}[{", ".join(shape)}][{", ".join(stride)}]"'
        if record["op"] == "placeholder":
            node = {"name": name, "kind": "placeholder", "meta": meta}
        else:
            expression = record["expression"]
            if not expression.startswith(f"%{name} : ") or " = " not in expression:
                raise ValueError(f"invalid task graph expression for {name}")
            node, = fx_text.parse_fragment([f'#   %{name} : {meta} = {expression.split(" = ", 1)[1]}'])
            if set(fx_text_refs({"args": node["args"], "kwargs": node["kwargs"]})) != set(record["inputs"]):
                raise ValueError(f"task graph dependencies disagree for {name}")
        full[name] = node
    local = {}
    for node in frag:
        original = full.get(node["name"])
        if (node["kind"] == "call" and original is not None
                and (original["kind"] != "call" or any(node.get(k) != original.get(k)
                                                        for k in ("target", "args", "kwargs")))):
            raise ValueError(f"fragment and complete task graph disagree for {node['name']}")
        if node["name"] not in local or node["kind"] == "call":
            local[node["name"]] = node
    restored, visiting, done, result = [], set(), set(), []

    def visit(name):
        if name in done:
            return
        if name in visiting:
            raise ValueError(f"cyclic task graph dependency {name}")
        visiting.add(name)
        node = local.get(name)
        supplied = input_buffers_of(name, params, param_buf)
        if (node is None or node["kind"] == "placeholder") and name in full:
            candidate = full[name]
            if supplied:
                if node is None:
                    node = dict(name=name, kind="placeholder", meta=candidate.get("meta"))
            elif candidate["kind"] == "call":
                node = candidate
                restored.append(name)
            elif node is None:
                node = candidate
        if node is not None:
            if node["kind"] == "call":
                for ref in sorted(fx_text_refs({"args": node["args"], "kwargs": node["kwargs"]})):
                    visit(ref)
            result.append(node)
        visiting.remove(name)
        done.add(name)

    for name in local:
        visit(name)
    return result, restored


def input_view(node, arg, parameter, buffer, prov, output_code):
    """Use a recorded pre-state alias layout, never reshape merely by element count.

    Currently accepts a single zero-offset reinterpretation of a dense storage.
    Nonzero offsets, chained/reassigned aliases and layouts with gaps remain unbound.
    """
    match = re.match(r'Tensor "\w+\[[^\]]*\]\[([^\]]*)\]', node.get("meta") or "")
    stride = _dims(match.group(1)) if match else None
    if tensor_corresponds(node, arg) and (stride is None or stride == tuple(arg.stride)):
        return arg
    info = prov["params"].get(parameter, {})
    view = output_code.views.get(buffer)
    if info.get("role") != "inplace" or view is None or view["offset"] != 0:
        return None
    if (info.get("other_origin_nodes") or {}).get(view["source"]) != node["name"]:
        return None
    if view["shape"] != tuple(arg.shape) or view["stride"] != tuple(arg.stride):
        return None
    dtype, shape = meta_shape(node.get("meta"))
    if (dtype != TORCH_DT.get(arg.dtype) or shape is None or stride is None
            or output_code.layout_assertions.get(view["source"]) != (shape, stride)):
        return None

    def dense(shape, stride):
        if len(shape) != len(stride) or any(type(n) is not int or n < 0 for n in (*shape, *stride)):
            return False
        expected = 1
        for step, count in sorted((s, n) for s, n in zip(stride, shape) if n > 1):
            if step != expected:
                return False
            expected *= count
        return True

    if (math.prod(shape) != math.prod(arg.shape) or not dense(shape, stride)
            or not dense(tuple(arg.shape), tuple(arg.stride))):
        return None
    mapped = copy.copy(arg)
    mapped.shape, mapped.stride = shape, stride
    return mapped


def write_parameters(launch):
    """Conservative write set from the captured IR, including unchanged-value writes."""
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    from kernel_analyzer.reference_eval.ttir_eval import _may_write_params, _ptr_like
    func = parse_ttir(launch.asm["ttir"]).entry()
    args = launch.ttir_params()
    if len(func.params) != len(args):
        return None
    names, _ = _may_write_params(func)
    if names is None:
        return None
    # the analysis taints every value that flows into an address, integers included; a non-pointer parameter has no
    # memory (int_to_ptr makes names None above), so only pointer parameters can be written
    return {a.name for param, a in zip(func.params, args) if param[0] in names and _ptr_like(param[1])}


def write_only_parameters(launch):
    """Prove a restricted pointer flow contains stores but no reads or escapes.

    Unknown operations, calls, atomics, pointer casts and region-carried pointers
    are refused. This is independent of the compiler's input/output role names.
    """
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    from kernel_analyzer.reference_eval.ttir_eval import _ptr_like
    func = parse_ttir(launch.asm["ttir"]).entry()
    args = launch.ttir_params()
    if len(func.params) != len(args):
        return set()
    ops = list(func.walk())
    pure = {"tt.addptr", "tt.splat", "tt.broadcast", "tt.expand_dims"}
    proven = set()
    for param, arg in zip(func.params, args):
        name, typ = param[:2]
        if not _ptr_like(typ):
            continue
        derived = {name}
        changed = True
        while changed:
            changed = False
            for op in ops:
                if op.name in pure and derived.intersection(op.operands):
                    new = set(op.results) - derived
                    derived.update(new)
                    changed |= bool(new)
        stored, safe = False, True
        for op in ops:
            if any(derived.intersection(values) for _, values in op.successors):
                safe = False
            if not derived.intersection(op.operands):
                continue
            if op.name == "tt.store" and op.operands[0] in derived and not derived.intersection(op.operands[1:]):
                stored = True
            elif op.name not in pure:
                safe = False
        if safe and stored:
            proven.add(arg.name)
    return proven


def remove_unused_scratch(frag, prov, write_only):
    """Remove only unreferenced synthetic aliases of proven write-only outputs."""
    used = {r for n in frag if n["kind"] == "call" for r in fx_text_refs(n)}
    aliases = set()
    updated = copy.deepcopy(prov)
    for parameter in write_only:
        info = updated["params"].get(parameter, {})
        if info.get("role") != "inplace":
            continue
        info["role"] = "output"
        aliases.update(name for name, origin in info.get("other_origin_nodes", {}).items()
                       if origin is None and name in info.get("other_names", ()) and name not in used)
    removed = [n["name"] for n in frag if n["kind"] == "placeholder" and n["name"] in aliases]
    return [n for n in frag if not (n["kind"] == "placeholder" and n["name"] in aliases)], updated, removed


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


def check_storage_changes(arg, writable):
    """Check observed changes in the entire captured storage, including view gaps.

    This does not prove the absence of unchanged-value writes or accesses outside
    captured storage. Those still require IR memory-access evidence. Supported
    layouts have nonnegative, nonoverlapping strides, including transpose/slices.
    """
    if arg.before is None or arg.after is None or arg.window is not None:
        return "complete storage before and after the launch is required"
    before, after = np.asarray(arg.before), np.asarray(arg.after)
    if (before.dtype != np.uint8 or after.dtype != np.uint8 or before.ndim != 1 or after.ndim != 1 or
            before.size != arg.storage_nbytes or after.size != arg.storage_nbytes):
        return "storage snapshots do not match the declared byte length"
    size = arg.element_size
    if type(size) is not int or size <= 0:
        return "invalid element size"
    offset = arg.data_ptr - arg.storage_ptr
    if offset < 0 or offset % size or len(arg.shape) != len(arg.stride):
        return "invalid storage offset or layout"
    if any(type(n) is not int or n < 0 for n in (*arg.shape, *arg.stride)):
        return "unsupported shape or stride"
    empty = any(n == 0 for n in arg.shape)
    dims = sorted((s, n) for s, n in zip(arg.stride, arg.shape) if n > 1)
    span = 1
    for stride, count in dims:
        if stride < span:
            return "overlapping layout needs a separate storage mapping"
        span += (count - 1) * stride
    if not empty and offset + span * size > arg.storage_nbytes:
        return "logical tensor extends outside captured storage"
    for start in range(0, before.size, 1 << 20):
        changed = np.flatnonzero(before[start:start + (1 << 20)] != after[start:start + (1 << 20)]) + start
        if not changed.size:
            continue
        if not writable:
            return "storage of a parameter without a declared output changed"
        if empty:
            return "storage changed for an empty output"
        remaining = (changed - offset) // size
        valid = remaining >= 0
        for stride, count in reversed(dims):
            coordinate = remaining // stride
            valid &= coordinate < count
            remaining %= stride
        if not np.all(valid & (remaining == 0)):
            return "storage changed outside the declared logical output"
    return None


def reference_write_scope(launch, outputs, max_programs=64):
    """Full-grid reference writes, including writes that leave bytes unchanged.

    This is evidence about the IR reference execution, not device access tracing.
    Budget exhaustion is explicit; no prefix of the grid is treated as complete.
    """
    from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, NumericMode
    tensors = [a for a in launch.args if a.kind == "tensor"]
    if (len({a.storage_ptr for a in tensors}) != len(tensors) or
            not set(outputs).issubset({a.name for a in tensors}) or launch.implicit):
        return {"outcome": "not_established", "note": "aliases or indirect memory need an explicit task mapping"}
    if len(launch.grid) != 3 or any(type(n) is not int or n < 0 for n in launch.grid):
        return {"outcome": "not_established", "note": "invalid grid"}
    if math.prod(launch.grid) > max_programs:
        return {"outcome": "not_established", "note": "full grid exceeds reference write budget"}
    evaluator = KernelReferenceEvaluator.from_text(launch.asm["ttir"], mode=NumericMode.ROUNDING_CHECK,
                                                   ttgir=launch.asm.get("ttgir"))
    ref = evaluator.evaluate(launch)
    return inspect_reference_writes(launch, outputs, ref)


def inspect_reference_writes(launch, outputs, ref):
    """Inspect a complete execution result produced for this launch.

    The caller retains the evaluator's identity and its validation evidence.
    This does not equate hardware emulation with rounding-check evaluation.
    """
    tensors = [a for a in launch.args if a.kind == "tensor"]
    if (len({a.storage_ptr for a in tensors}) != len(tensors) or launch.implicit
            or not set(outputs).issubset({a.name for a in tensors})):
        return {"outcome": "not_established", "note": "aliases or indirect memory need an explicit task mapping"}
    if len(launch.grid) != 3 or any(type(n) is not int or n <= 0 for n in launch.grid):
        return {"outcome": "not_established", "note": "invalid or empty grid"}
    if ref.aborted or ref.reasons or ref.notes:
        return {"outcome": "not_established", "note": "reference has incomplete or qualified execution",
                "aborted": len(ref.aborted), "abort_reasons": dict(collections.Counter(ref.aborted.values())),
                "reasons": list(ref.reasons), "notes": ref.notes}
    if (len(ref.programs) != math.prod(launch.grid) or len(set(ref.programs)) != len(ref.programs)
            or any(len(p) != 3 or any(type(i) is not int or not 0 <= i < n for i, n in zip(p, launch.grid))
                   for p in ref.programs)):
        return {"outcome": "not_established", "note": "reference did not evaluate the complete grid"}
    args = {a.storage_ptr: a for a in launch.args if a.kind == "tensor"}
    if set(ref.buffers) != set(args):
        return {"outcome": "not_established", "note": "reference does not account for every captured storage"}
    written_elements = 0
    for ident, buf in ref.buffers.items():
        arg = args.get(ident)
        if arg is None or buf.index is not None or buf.dtype != arg.dtype:
            return {"outcome": "not_established", "note": "reference storage lacks a complete argument mapping"}
        if np.any(buf.cond) or (buf.premise is not None and np.any(buf.premise)):
            return {"outcome": "not_established", "note": "reference memory depends on a condition or premise"}
        mask = np.asarray(buf.written, dtype=bool)
        written_elements += int(mask.sum())
        # Reuse the byte/layout membership check with write events rather than
        # changed values. An unchanged-value store therefore remains observable.
        events = copy.copy(arg)
        events.after = np.repeat(mask.astype(np.uint8), arg.element_size)
        events.before = np.zeros_like(events.after)
        why = check_storage_changes(events, writable=arg.name in outputs)
        if why:
            return {"outcome": "not_established", "note": f"{arg.name}: {why}"}
    return {"outcome": "checked", "programs": len(ref.programs), "written_elements": written_elements,
            "scope": "IR reference execution; device access correspondence remains required"}


def relerr(f, k, dt):
    """|K - f| / max(|f|, tiny) at the enclosure midpoint (exact agreement gives 0)."""
    if isinstance(f, A.AllowedSet):
        return min(relerr(m, k, dt) for m in f.members)
    if not isinstance(f, R):
        return 0.0 if f == k else float("inf")
    if not f.finite:
        return 0.0 if (f.v == "+inf" and k == float("inf")) or (f.v == "-inf" and k == float("-inf")) else float("inf")
    a, b = f.bounds()
    mid = (a + b) / 2
    if not np.isfinite(k):
        return float("inf")
    return float(abs(Fraction(k) - mid) / max(abs(mid), Fraction(1, 2**126)))


def sample_reference(lt, K, dt, rng):
    """Diagnostic samples only; preserve discrete values and typed failures."""
    failures = {Undefined: "domain_undefined", NotEstablished: "not_established",
                Unsupported: "unsupported", Invalid: "invalid_input"}
    if isinstance(lt, Exception):
        for cls, outcome in failures.items():
            if isinstance(lt, cls):
                return {"outcome": outcome, "note": str(lt)[:120]}
        raise lt
    if tuple(lt.shape) != tuple(K.shape):
        return {"outcome": "not_established", "note": "reference and captured output shapes differ"}
    idxs = {tuple(0 for _ in K.shape), tuple(s - 1 for s in K.shape)} if K.size else set()
    while len(idxs) < min(34, K.size):
        idxs.add(tuple(rng.randrange(s) for s in K.shape))
    errs, counts = [], collections.Counter()
    for i in sorted(idxs):
        try:
            errs.append(relerr(lt.get(i), K[i].item(), dt))
        except tuple(failures) as e:
            counts[next(outcome for cls, outcome in failures.items() if isinstance(e, cls))] += 1
    outcome = next((s for s in ("invalid_input", "unsupported", "not_established", "domain_undefined")
                    if counts[s]), "value")
    return {"outcome": outcome, "elements": len(idxs), "total_elements": int(K.size),
            "indices": [list(i) for i in sorted(idxs)],
            "undefined": counts["domain_undefined"], "not_established": counts["not_established"],
            "failures": dict(counts), "max_rel": max(errs) if errs else None,
            "median_rel": float(np.median(errs)) if errs else None}


def full_reference(lt, K, limit):
    """Compare every coordinate with exact values or enclosures, without tolerances.

    This is K versus the selected reference, not a semantic-error verdict.
    A value inside a non-point enclosure is only compatible with that enclosure.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("full-output limit must be a positive integer")
    failures = {Undefined: "domain_undefined", NotEstablished: "not_established",
                Unsupported: "unsupported", Invalid: "invalid_input"}
    total, done, relations = int(K.size), 0, collections.Counter()
    base = {"total_elements": total, "elements": 0, "complete": False,
            "unestablished_elements": 0, "unestablished_examples": []}
    if total > limit:
        return dict(base, outcome="not_established", note="complete output exceeds reference budget")
    try:
        if isinstance(lt, Exception):
            raise lt
        if tuple(lt.shape) != tuple(K.shape):
            raise NotEstablished("reference and captured output shapes differ")
        for index in np.ndindex(K.shape):
            try:
                value, observed = lt.get(index), K[index].item()
                if isinstance(value, A.AllowedSet):
                    raise NotEstablished("allowed outputs need a joint complete-output membership check")
                if isinstance(value, R):
                    if not value.finite:
                        equal = ((value.v == "+inf" and observed == float("inf"))
                                 or (value.v == "-inf" and observed == -float("inf")))
                        relation = "exact_equal" if equal else "outside_reference"
                    else:
                        lo, hi = value.bounds()
                        if isinstance(observed, float) and not math.isfinite(observed):
                            relation = "outside_reference"
                        else:
                            point = Fraction(observed)
                            relation = ("exact_equal" if lo == hi == point else "outside_reference"
                                        if point < lo or point > hi else "within_enclosure")
                elif isinstance(value, (int, bool, Fraction, np.integer, np.bool_)):
                    if isinstance(observed, float) and not math.isfinite(observed):
                        relation = "outside_reference"
                    else:
                        exact = value.item() if isinstance(value, np.generic) else value
                        relation = "exact_equal" if Fraction(exact) == Fraction(observed) else "outside_reference"
                else:
                    raise NotEstablished(f"no strict output comparison for {type(value).__name__}")
                relations[relation] += 1
                done += 1
            except NotEstablished as error:
                if lazy.evaluation_budget_exhausted():
                    raise
                base["unestablished_elements"] += 1
                if len(base["unestablished_examples"]) < 3:
                    base["unestablished_examples"].append({"index": list(index), "note": str(error)[:160]})
    except tuple(failures) as error:
        outcome = next(name for cls, name in failures.items() if isinstance(error, cls))
        return dict(base, outcome=outcome, elements=done, note=str(error)[:160], relations=dict(relations))
    if base["unestablished_elements"]:
        return dict(base, outcome="not_established", elements=done, relations=dict(relations),
                    all_exact_equal=False, note="some output coordinates remain unestablished; known comparisons retained")
    return dict(base, outcome="value", elements=done, complete=True, relations=dict(relations),
                all_exact_equal=relations["exact_equal"] == total)


def reference_status(outputs):
    outcomes = {v["outcome"] for entry in outputs.values() for k, v in entry.items() if k != "node"}
    for state in ("invalid_input", "unsupported", "not_established", "domain_undefined"):
        if state in outcomes:
            return state
    if outcomes == {"value"}:
        return "complete" if all(v.get("complete") is True for entry in outputs.values()
                                 for k, v in entry.items() if k != "node") else "sampled"
    return "not_evaluated"


def capture_task_state(launch):
    if launch.executed is not True:
        return "captured execution is absent or not established"
    if not launch.implicit_recorded:
        return "capture does not establish completeness of indirect memory"
    if launch.implicit:
        return "indirect memory requires an explicit task mapping"
    return None


def candidate_origin_problem(rec):
    status = rec.get("candidate_status")
    if status == "ambiguous":
        return "ambiguous candidate origins"
    if status == "not_matched":
        return "no candidate origin"
    if status != "matched":
        return "candidate origin has not been revalidated"
    candidates = rec.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        return "candidate origin is not unique"
    if not rec.get("output_code") or candidates[0].get("file") != rec["output_code"]:
        return "candidate origin and selected output code disagree"
    return None


def run_unit(rec, rng, binding_only=False, capture_root=None, provenance_root=None, full_output_limit=None):
    problem = candidate_origin_problem(rec)
    if problem:
        return {"status": problem, "binding_status": "not_established"}
    from kernel_analyzer.reference_eval.capture import load_launch
    unit_dir = (capture_root or REPO / ".cache/three_way/capture") / rec["unit"]
    oc = OutputCode(REPO / rec["output_code"])
    kernel = rec["kernel"]
    frag_kernel = kernel if kernel in oc.frag else next((k for k in oc.frag if k == kernel), None)
    if frag_kernel is None:
        return {"status": "no graph fragment for the kernel"}
    records, why = provenance(rec, capture_root, provenance_root)
    if records is None:
        return {"status": why}
    args, prov = select_call(oc, kernel, records)
    if args is None:
        return {"status": prov}
    L = load_launch(unit_dir)
    why = capture_task_state(L)
    if why is not None:
        return {"status": why}
    scalars, why = bind_call_scalars(oc.params[kernel], args, L, oc.launcher_meta.get(kernel))
    if why is not None:
        return {"status": f"call parameter mapping not established: {why}"}
    byname = {a.name: a for a in L.args}
    param_buf = {p: a for p, a in zip(oc.params[kernel], args) if is_buffer(a) and p in byname and byname[p].kind == "tensor"}
    if len(set(param_buf.values())) != len(param_buf):
        return {"status": "aliased call arguments need an explicit storage mapping"}
    buf_param = {b: p for p, b in param_buf.items()}
    storage = [byname[p].storage_ptr for p in param_buf]
    if len(set(storage)) != len(storage):
        return {"status": "shared storage needs an explicit alias mapping"}
    restored = []
    if prov.get("task_graph"):
        with Path(prov["task_graph"]).open() as source:
            task_graph = json.load(source)
        if (task_graph.get("error") or not task_graph.get("output_code")
                or Path(task_graph["output_code"]).resolve() != (REPO / rec["output_code"]).resolve()):
            return {"status": "complete task graph and selected output code disagree"}
        oc.frag[frag_kernel], restored = complete_fragment(oc.frag[frag_kernel], task_graph, prov["params"], param_buf)
    write_only = write_only_parameters(L)
    oc.frag[frag_kernel], prov, scratch = remove_unused_scratch(oc.frag[frag_kernel], prov, write_only)
    ph, outs = bind_from_provenance(prov, oc.frag[frag_kernel], param_buf, write_parameters(L))
    if ph is None:
        return {"status": f"provenance binding incomplete: {outs}"}
    nodes = {n["name"]: n for n in oc.frag[frag_kernel]}
    input_args = {}
    for name, buf in ph.items():
        parameter = buf_param[buf]
        mapped = input_view(nodes[name], byname[parameter], parameter, buf, prov, oc)
        if mapped is None:
            return {"status": f"input type or shape mapping not established: {name}"}
        input_args[name] = mapped
    for p, name in outs.items():
        if not tensor_corresponds(nodes[name], byname[p]):
            return {"status": f"output type or shape mapping not established: {p} -> {name}"}
    inputs = {}
    for name, buf in ph.items():
        arg = input_args[name]
        if arg.before is None or arg.window is not None:
            return {"status": f"complete pre-state not captured: {arg.name}"}
        inputs[name] = lazy.from_array(logical(arg, "before"), TORCH_DT[arg.dtype], physical_input=True)
    frag = cut_fragment(oc.frag[frag_kernel], prov["params"], outs)
    graph = fx_text.to_graph(frag)
    defined = set(inputs) | {n["name"] for n in graph["nodes"]}
    missing = sorted({r for n in graph["nodes"] for r in fx_text_refs(n)} - defined)
    free = [b for p, b in param_buf.items()
            if prov["params"].get(p, {}).get("role") in ("input", "inplace") and b not in ph.values()]
    for name in list(missing):                     # references Inductor listed without a placeholder line
        cands = input_buffers_of(name, prov["params"], param_buf, among=set(free))
        if len(cands) != 1:
            continue
        pick = cands.pop()
        arg = byname[buf_param[pick]]
        if arg.before is None or arg.window is not None:
            return {"status": f"complete pre-state not captured: {arg.name}"}
        inputs[name] = lazy.from_array(logical(arg, "before"), TORCH_DT[arg.dtype], physical_input=True)
        ph[name] = pick
        input_args[name] = arg
        free.remove(pick)
        missing.remove(name)
    if missing:
        return {"status": "fragment references values without a placeholder", "missing": missing[:5]}
    if free:
        return {"status": "recorded input or pre-state has no graph mapping", "unmapped": free}
    for p in outs:
        if byname[p].after is None or byname[p].window is not None:
            return {"status": f"complete post-state not captured: {p}"}
    for p in param_buf:
        why = check_storage_changes(byname[p], writable=p in outs)
        if why is not None:
            return {"status": f"storage correspondence not established: {p}: {why}"}
    writes = reference_write_scope(L, outs)
    graph["outputs"] = list(outs.values())
    result = {"status": "structural binding checked", "outputs": {},
              "binding": {"inputs": ph, "outputs": outs, "writes": sorted(outs), "call_scalars": scalars},
              "candidate": "fragment matched", "semantic_verdict": "not_established",
              "storage_changes": "all captured bytes checked against declared outputs",
              "reference_write_scope": writes,
              "reference_status": "not_evaluated",
              "reference_scope": "declared output dependency slice",
              "graph_admission": {
                  "status": "not_admitted",
                  "scope": "output_slice",
                  "implementation_status": "pending",
                  "implementation_version": "graph-common-node-gate-2026-10-24",
                  "reason": "development lazy reference uses an explicit output slice; whole-graph admission was not run",
              },
              "readings": {"eager": "selected GRAPH task reading",
                           "inductor_default": "counterfactual diagnostic only; not actual compiler semantics"}}
    result["binding"]["input_layouts"] = {name: {"shape": list(arg.shape), "stride": list(arg.stride),
                                                "storage_offset_bytes": arg.data_ptr - arg.storage_ptr}
                                         for name, arg in input_args.items()}
    result["binding"]["restored_task_nodes"] = restored
    result["binding"]["task_graph"] = prov.get("task_graph")
    result["binding"]["write_only_parameters"] = sorted(write_only)
    result["binding"]["unused_scratch_placeholders"] = scratch
    if binding_only:
        return result
    result.update(status="full reference comparison" if full_output_limit is not None else "sampled reference comparison",
                  reference_status="not_evaluated")
    for reading, elide in (("eager", False), ("inductor_default", True)):
        try:
            env = lazy.evaluate_lazy(graph, inputs, elide_intermediate_casts=elide, scope="output_slice")
        except (Undefined, NotEstablished, Unsupported, Invalid) as error:
            env = {node: error for node in outs.values()}
        for p, node in outs.items():
            arg = byname[p]
            K = logical(arg, "after")
            lt = env[node]
            entry = result["outputs"].setdefault(p, {"node": node})
            # The same coordinates for both readings, independent of earlier refused units.
            sample_rng = random.Random(f"{SEED}:{rec['unit']}:{p}")
            entry[reading] = (full_reference(lt, K, full_output_limit) if full_output_limit is not None
                              else sample_reference(lt, K, TORCH_DT[arg.dtype], sample_rng))
    result["reference_status"] = reference_status(result["outputs"])
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
    ap.add_argument("--bindings", type=Path, required=True, help="explicit static candidate record file")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--out", type=Path, required=True, help="new result file; existing records are never overwritten")
    ap.add_argument("--binding-only", action="store_true")
    ap.add_argument("--full-output-limit", type=int, help="compare every output coordinate; refuse outputs above this limit")
    ap.add_argument("--capture-root", type=Path, help="explicit root of new captures")
    ap.add_argument("--provenance-root", type=Path, help="compiler-origin records for those captures")
    ap.add_argument("--include-unverified", action="store_true",
                    help="binding-only inspection of all captures, including ones without K-G verification")
    a = ap.parse_args()
    if a.n <= 0:
        ap.error("--n must be positive")
    if a.include_unverified and not a.binding_only:
        ap.error("--include-unverified requires --binding-only")
    if a.full_output_limit is not None and (a.full_output_limit <= 0 or a.binding_only):
        ap.error("--full-output-limit must be positive and cannot be combined with --binding-only")
    recs = json.loads(gzip.open(a.bindings).read())
    # Selection includes unresolved candidates; dropping them would inflate binding coverage.
    frame = sorted((r for r in recs if a.include_unverified or r["verify"] == "bit_identical"), key=lambda r: r["unit"])
    if a.only is not None:
        missing = sorted(set(a.only) - {r["unit"] for r in frame})
        if missing:
            ap.error("requested units absent from the eligible capture records: " + ", ".join(missing))
        sample = [r for r in frame if r["unit"] in a.only]
    else:
        if a.n > len(frame):
            ap.error(f"--n exceeds the {len(frame)} eligible capture records")
        sample = random.Random(SEED).sample(frame, a.n)
    rng = random.Random(SEED)
    out_path = a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stat = collections.Counter()
    errors = 0
    with open(out_path, "x") as fh:
        for r in sample:
            t = time.time()
            try:
                res = run_unit(r, rng, binding_only=a.binding_only,
                               capture_root=a.capture_root, provenance_root=a.provenance_root,
                               full_output_limit=a.full_output_limit)
            except Exception as e:  # noqa: BLE001 -- recorded, never hidden
                errors += 1
                res = {"status": f"error {type(e).__name__}: {str(e)[:160]}"}
            res.setdefault("binding_status", "not_established")
            res.setdefault("semantic_verdict", "not_established")
            res.setdefault("reference_status", "not_evaluated")
            res.update(unit=r["unit"], kernel=r["kernel"], seconds=round(time.time() - t, 1), contract_status=r["status"],
                       candidate_status=r.get("candidate_status", "not_revalidated"), candidate_records=str(a.bindings))
            res.update(verify=r.get("verify"), include_unverified=a.include_unverified,
                       capture_root=str(a.capture_root or REPO / ".cache/three_way/capture"),
                       provenance_root=str(a.provenance_root or PROV_DIR))
            stat[res["status"].split(":")[0]] += 1
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(r["unit"], res["status"], {p: {k: v.get("max_rel") if isinstance(v, dict) and "max_rel" in v else v
                                              for k, v in o.items() if k != "node"} for p, o in res.get("outputs", {}).items()},
                  res["seconds"], flush=True)
    print(dict(stat))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
