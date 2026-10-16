"""Adapter for contracts/GRAPH.yaml: kernel_spec.graph.evaluate; the reading and the mutants change one aspect of the
composition."""
from __future__ import annotations

from kernel_spec import graph as G
from kernel_spec.num import Result, Undefined, Unsupported


def _check(profile):
    if profile != "eager":
        raise Unsupported(f"profile {profile!r}")


def reference(profile, inp):
    _check(profile)
    return Result("value", {"out": G.evaluate(inp["graph"])})


def _inductor_default(profile, inp):
    _check(profile)
    return Result("value", {"out": G.evaluate(inp["graph"], elide_intermediate_casts=True)})


def _ignore_casts(profile, inp):
    _check(profile)
    g = dict(inp["graph"])
    g["nodes"] = [dict(n, op="aten.clone", args=n["args"][:1]) if n["op"] == "prims.convert_element_type" else n
                  for n in g["nodes"]]
    return Result("value", {"out": G.evaluate(g)})


def _strict_poison(profile, inp):
    _check(profile)
    g = inp["graph"]
    for n in g["nodes"]:                       # evaluate every node on its own; any undefined value is fatal
        G.evaluate(dict(g, outputs=[n["name"]]))
    return Result("value", {"out": G.evaluate(g)})


def _getitem_from_end(profile, inp):
    _check(profile)
    g = dict(inp["graph"])
    g["nodes"] = [dict(n, args=[n["args"][0], -1 - n["args"][1]]) if n["op"] == "operator.getitem" else n for n in g["nodes"]]
    return Result("value", {"out": G.evaluate(g)})


def _swap_operands(profile, inp):
    _check(profile)
    g = dict(inp["graph"])
    g["nodes"] = [dict(n, args=[n["args"][1], n["args"][0]] + n["args"][2:]) if n["op"] in ("aten.sub", "aten.div") else n
                  for n in g["nodes"]]
    return Result("value", {"out": G.evaluate(g)})


READINGS = {"inductor_default": _inductor_default}
MUTANTS = {"GR.M1": _ignore_casts, "GR.M2": _strict_poison, "GR.M3": _getitem_from_end, "GR.M4": _swap_operands}
