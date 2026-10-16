"""Evaluation of an ATen graph (the GRAPH contract): nodes in topological order, each operation by the semantics of
the base contract that covers it.

A graph is {"inputs": {name: tensor}, "nodes": [{"name", "op", "args", "kwargs"}], "outputs": [ref, ...]} where an
argument {"ref": name} (optionally "index": i for tuple results) refers to an input or an earlier node. The value of a
node that is undefined in the reals poisons only the nodes that use it.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import aten as A
from .aten_adapter import decode, encode
from .num import Invalid, NotEstablished, Undefined, Unsupported

BASE_CONTRACTS = ("ARITH", "MATHFN", "LOGIC", "CAST", "LAYOUT", "INDEX", "REDUCE", "SOFTMAX", "MATMUL", "CONV", "RANDOM")
CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


def covered_ops():
    ops = {"operator.getitem"}
    for cid in BASE_CONTRACTS:
        p = CONTRACTS / f"{cid}.yaml"
        if p.is_file():
            ops |= set(yaml.safe_load(p.read_text()).get("covers", []))
    return ops


class _Poison:
    def __init__(self, why):
        self.why = why


def _resolve(x, env):
    if isinstance(x, dict) and "ref" in x:
        if x["ref"] not in env:
            raise Invalid(f"reference to an unknown value {x['ref']!r} (not in topological order?)")
        v = env[x["ref"]]
        if isinstance(v, _Poison):
            raise Undefined(v.why)
        if "index" in x:
            v = v[x["index"]]
            if v is None:
                raise Invalid(f"{x['ref']}[{x['index']}] is absent")
        return v
    if isinstance(x, list):
        return [_resolve(v, env) for v in x]
    if isinstance(x, dict) and "t" in x:
        return decode(x)
    return decode(x)


def evaluate(graph: dict, elide_intermediate_casts=False):
    ops = covered_ops()
    env = {name: decode(t) for name, t in graph.get("inputs", {}).items()}
    consumers = {}
    for n in graph["nodes"]:
        for a in _walk_refs(n.get("args", [])) + _walk_refs(list((n.get("kwargs") or {}).values())):
            consumers.setdefault(a, []).append(n["name"])
    outputs = {o["ref"] if isinstance(o, dict) else o for o in graph["outputs"]}
    for n in graph["nodes"]:
        op = n["op"]
        if op not in ops:
            raise Unsupported(f"{op} is not covered by a base contract")
        if n["name"] in env:
            raise Invalid(f"duplicate node name {n['name']!r}")
        try:
            args = [_resolve(a, env) for a in n.get("args", [])]
            kwargs = {k: _resolve(v, env) for k, v in (n.get("kwargs") or {}).items()}
            if op == "operator.getitem":
                env[n["name"]] = args[0][args[1]]
                continue
            if (elide_intermediate_casts and op in ("prims.convert_element_type", "aten._to_copy")
                    and n["name"] not in outputs and isinstance(args[0], A.Ten) and args[0].dtype in A.FLOAT):
                target = A.dt(args[1] if len(args) > 1 else kwargs.get("dtype"))
                if target in A.FLOAT:                       # the reading: no rounding at an intermediate float cast
                    env[n["name"]] = A.Ten(args[0].a, target)
                    continue
            env[n["name"]] = A.OPS[op](*args, **kwargs)
        except Undefined as e:
            env[n["name"]] = _Poison(f"{n['name']}: {e}")
    out = []
    for o in graph["outputs"]:
        out.append(_resolve(o if isinstance(o, dict) else {"ref": o}, env))
    return [encode(v) for v in out]


def _walk_refs(xs):
    refs = []
    for x in xs:
        if isinstance(x, dict) and "ref" in x:
            refs.append(x["ref"])
        elif isinstance(x, list):
            refs += _walk_refs(x)
    return refs
