"""Parse the post-grad graph fragment that Inductor writes as comments above each kernel in its output code
("# Graph fragment:" followed by one "#   %name : ... = call_function[target=...](args = (...), kwargs = {...})"
line per node) into the GRAPH contract's graph format, and resolve ATen overloads to the operations of kernel_spec.aten.
"""
from __future__ import annotations

import math
import re

NODE = re.compile(r"^#\s+%(\w+) : (.*?) = (PlaceHolder\[target=(\w+)\]|call_function\[target=([\w.<>]+)\]\(args = \((.*)\), "
                  r"kwargs = \{(.*)\}\))\s*$")
META = re.compile(r'^Tensor "(\w+)\[([\d, s*]*)\]')


class ParseError(ValueError):
    pass


def _split_top(s, sep=","):
    out, depth, cur, q = [], 0, [], None
    for ch in s:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
            continue
        if ch in "\"'":
            q = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def parse_value(tok: str):
    tok = tok.strip()
    if tok.startswith("%"):
        return {"ref": tok[1:]}
    if tok.startswith("[") and tok.endswith("]"):
        return [parse_value(t) for t in _split_top(tok[1:-1])]
    if tok.startswith("(") and tok.endswith(")"):
        return [parse_value(t) for t in _split_top(tok[1:-1])]
    if tok in ("None",):
        return None
    if tok in ("True", "False"):
        return tok == "True"
    if tok in ("inf", "-inf"):
        return tok
    if tok == "nan":
        return "nan"
    if tok.startswith("torch.") and tok.split(".")[1] in {"float64", "float32", "float16", "bfloat16", "int64", "int32",
                                                          "int16", "int8", "uint8", "bool", "float8_e4m3fn", "float8_e5m2"}:
        return {"dtype": tok}
    if re.fullmatch(r"-?\d+", tok):
        return int(tok)
    if re.fullmatch(r"-?(\d+\.\d*|\.\d+|\d+)(e[-+]?\d+)?", tok, re.I):
        v = float(tok)
        return v if math.isfinite(v) else ("inf" if v > 0 else "-inf")
    return tok                                   # identifiers such as rand, cuda:0, torch.strided


def parse_fragment(lines):
    """[(name, kind, meta, op, args, kwargs)] from the comment lines of one graph fragment."""
    nodes = []
    for line in lines:
        m = NODE.match(line.rstrip())
        if not m:
            if line.startswith("#   %"):
                raise ParseError(f"unparsed node line: {line[:120]}")
            continue
        name, meta = m.group(1), m.group(2)
        if m.group(4):
            nodes.append({"name": name, "kind": "placeholder", "meta": meta})
            continue
        target = m.group(5)
        args = [parse_value(t) for t in _split_top(m.group(6))] if m.group(6).strip() else []
        kwargs = {}
        for kv in _split_top(m.group(7)):
            if not kv:
                continue
            k, _, v = kv.partition(":")
            kwargs[k.strip()] = parse_value(v)
        nodes.append({"name": name, "kind": "call", "meta": meta, "target": target, "args": args, "kwargs": kwargs})
    return nodes


def dtype_of(meta: str):
    m = META.match(meta)
    return m.group(1) if m else None


# ------------------------------------------------------------------ overloads -> (op name, argument rewrite)
IGNORED_KWARGS = {"layout", "device", "pin_memory", "memory_format", "requires_grad", "non_blocking"}


def resolve(target: str, args, kwargs):
    """Map 'torch.ops.aten.scatter.value' etc. to the GRAPH operation name and arguments, or raise ParseError."""
    if target == "operator.getitem":
        return "operator.getitem", list(args), {}
    if target == "torch._inductor.fx_passes.post_grad.mm_plus_mm":
        return "inductor.mm_plus_mm", [], dict(kwargs)
    parts = target.split(".")
    if parts[:2] != ["torch", "ops"] or len(parts) < 4:
        raise ParseError(f"unexpected target {target}")
    ns, op, overload = parts[2], parts[3], (parts[4] if len(parts) > 4 else "default")
    name = f"{ns}.{op}"
    kwargs = {k: v for k, v in kwargs.items() if k not in IGNORED_KWARGS}
    if name == "aten.scatter" and overload in ("value", "value_reduce"):
        a = list(args)
        kwargs = dict(kwargs, value=a[3])
        args = a[:3]
        if len(a) > 4:
            kwargs["reduce"] = a[4]
    elif name == "aten.scatter" and overload in ("src", "reduce"):
        a = list(args)
        kwargs = dict(kwargs, src=a[3])
        args = a[:3]
        if len(a) > 4:
            kwargs["reduce"] = a[4]
    elif name in ("aten.max", "aten.min") and overload == "default":
        name = "aten.amax" if op == "max" else "aten.amin"
    elif name in ("aten.var_mean", "aten.var", "aten.std") and overload == "correction":
        pass
    elif name == "aten.sum" and overload == "default" and args and len(args) == 1:
        pass
    elif name == "aten.full" and "dtype" in kwargs and isinstance(kwargs["dtype"], dict):
        kwargs["dtype"] = kwargs["dtype"]["dtype"]
    for k in ("dtype",):
        if isinstance(kwargs.get(k), dict):
            kwargs[k] = kwargs[k]["dtype"]
    return name, list(args), kwargs


def to_graph(nodes, inputs=None, outputs=None):
    """The GRAPH contract's input format from parsed fragment nodes (placeholders become inputs)."""
    g = {"inputs": dict(inputs or {}), "nodes": [], "outputs": list(outputs or [])}
    for n in nodes:
        if n["kind"] == "placeholder":
            continue
        name, args, kwargs = resolve(n["target"], n["args"], n["kwargs"])
        if name == "prims.convert_element_type" and len(args) > 1 and isinstance(args[1], dict):
            args = [args[0], args[1]["dtype"]]
        g["nodes"].append({"name": n["name"], "op": name, "args": args, "kwargs": kwargs})
    return g
