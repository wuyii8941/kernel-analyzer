"""Evaluation of an ATen graph (the GRAPH contract): nodes in topological order, each operation by the semantics of
the base contract that covers it.

A graph is {"inputs": {name: tensor}, "nodes": [{"name", "op", "args", "kwargs"}], "outputs": [ref, ...]} where an
argument {"ref": name} (optionally "index": i for tuple results) refers to an input or an earlier node. The value of a
node that is undefined in the reals poisons only the nodes that use it.
"""
from __future__ import annotations

from pathlib import Path
import itertools
import inspect

import numpy as np
import yaml

from . import aten as A
from . import sorting
from .aten_adapter import decode, encode, bind_reference_call
from .num import Invalid, NotEstablished, Undefined, Unsupported

BASE_CONTRACTS = ("ARITH", "MATHFN", "LOGIC", "CAST", "LAYOUT", "INDEX", "REDUCE", "SOFTMAX", "MATMUL", "CONV", "RANDOM", "ACT",
                  "SORT")
CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
MUTATING_OPS = {"aten.index_put_"}
# These operations are admitted only by rules in the drafted GRAPH contract:
# GR.EFFECTS covers a fresh full allocation and its functional index_put
# replacement, while GR.JOINT_CHOICES covers a complete scatter result.  They
# are deliberately not put back into the reviewed INDEX contract.
GRAPH_LOCAL_VALUE_OPS = {"aten.full", "aten.index_put", "aten.scatter"}


def covered_ops():
    ops = {"operator.getitem"}
    for cid in BASE_CONTRACTS:
        p = CONTRACTS / f"{cid}.yaml"
        if p.is_file():
            ops |= set(yaml.safe_load(p.read_text()).get("covers", []))
    return (ops | GRAPH_LOCAL_VALUE_OPS) - MUTATING_OPS


class _Poison:
    def __init__(self, why):
        self.why = why


def _reference_name(record):
    if set(record) - {"ref", "index"}:
        raise Unsupported("graph reference contains unimplemented metadata")
    if "index" in record and not isinstance(record["index"], (int, np.integer)):
        raise Invalid("graph result index must be an integer")
    return record.get("ref")


def select_result(value, index):
    """Projection of a multi-result value, including a declared absent result."""
    if not isinstance(index, (int, np.integer)):
        raise Invalid("graph result index must be an integer")
    if not isinstance(value, (list, tuple)):
        raise Unsupported("graph getitem only selects from multi-result operations")
    if not -len(value) <= index < len(value):
        raise Invalid("graph result index is out of range")
    return value[index]


def _resolve(x, env):
    if isinstance(x, dict) and "ref" in x:
        if x["ref"] not in env:
            raise Invalid(f"reference to an unknown value {x['ref']!r} (not in topological order?)")
        v = env[x["ref"]]
        if isinstance(v, _Poison):
            raise Undefined(v.why)
        if "index" in x:
            v = select_result(v, x["index"])
        return v
    if isinstance(x, list):
        return [_resolve(v, env) for v in x]
    if isinstance(x, dict) and "t" in x:
        return decode(x)
    return decode(x)


class _Branch(Exception):
    def __init__(self, name, alternatives):
        self.name, self.alternatives = name, alternatives


def evaluate(graph: dict, elide_intermediate_casts=False, max_choices=256):
    """Evaluate shared choices consistently; return a set of complete exact outputs when needed."""
    if isinstance(max_choices, bool) or not isinstance(max_choices, int) or max_choices < 1:
        raise Invalid("max_choices must be a positive integer")
    graph, _ = functional_value_graph(graph)
    pending, results = [{}], []
    ops = covered_ops()
    attempts = 0
    while pending:
        choices = pending.pop()
        attempts += 1
        if attempts > max_choices:
            raise NotEstablished("graph joint-choice evaluation exceeded its budget")
        try:
            results.append(_evaluate(graph, elide_intermediate_casts, choices, max_choices, ops))
        except _Branch as branch:
            if len(pending) + len(branch.alternatives) + len(results) > max_choices:
                raise NotEstablished("graph joint-choice evaluation exceeded its budget")
            pending.extend(dict(choices, **{branch.name: value}) for value in branch.alternatives)
        except Undefined:
            if choices:
                raise NotEstablished("a joint-choice branch has undefined output; no complete result established")
            raise
    if len(results) == 1:
        return results[0]
    from .compare import _canon
    from .expr import ExpressionError
    try:
        complete = {_canon(r) for r in results}
    except ExpressionError as error:
        raise NotEstablished(f"joint outputs are not representable as an exact finite set: {error}") from error
    return results[0] if len(complete) == 1 else frozenset(complete)


def require_value_graph(graph, input_names=None, *, _defer_mutation_check=False):
    """The base contracts do not yet compose storage aliases and mutation.

    Raw updates are rejected here. functional_value_graph can first establish
    the restricted no-alias local-allocation case; all other updates still need
    explicit alias and state composition.
    """
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), (list, tuple)):
        raise Invalid("graph requires an ordered node sequence")
    if set(graph) - {"inputs", "nodes", "outputs"}:
        raise Unsupported("graph contains fields outside the functional value contract")
    if not isinstance(graph.get("outputs"), (list, tuple)):
        raise Invalid("graph requires an output sequence")
    if input_names is None:
        inputs = graph.get("inputs", {})
        if not isinstance(inputs, dict):
            raise Invalid("graph inputs must be a name-to-value mapping")
        input_names = inputs
    names = set()
    for name in input_names:
        if not isinstance(name, str) or not name:
            raise Invalid("graph input names must be nonempty strings")
        names.add(name)

    def check_ref(name):
        if not isinstance(name, str) or name not in names:
            raise Invalid(f"reference to an unknown or later graph value {name!r}")

    for node in graph["nodes"]:
        if not isinstance(node, dict) or not isinstance(node.get("name"), str) or not node["name"]:
            raise Invalid("graph node names must be nonempty strings")
        if set(node) - {"name", "op", "args", "kwargs"}:
            raise Unsupported("graph node contains fields outside the functional value contract")
        if node["name"] in names:
            raise Invalid(f"duplicate graph value name {node['name']!r}")
        if not isinstance(node.get("op"), str):
            raise Invalid("graph node operation must be a string")
        args, kwargs = node.get("args", []), node.get("kwargs", {})
        if kwargs is None:
            kwargs = {}
        if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict):
            raise Invalid("graph node requires positional arguments and a keyword mapping")
        if any(not isinstance(key, str) for key in kwargs):
            raise Invalid("graph keyword names must be strings")
        fn = A.OPS.get(node["op"]) or sorting.OPS.get(node["op"])
        if fn is not None:
            bind_reference_call(node["op"], inspect.signature(fn), args, kwargs)
        if node["op"] == "operator.getitem" and (len(args) != 2 or kwargs):
            raise Invalid("operator.getitem requires exactly two positional arguments")
        for name in _walk_refs(args) + _walk_refs(list(kwargs.values())):
            check_ref(name)
        names.add(node["name"])
        if node["op"] in MUTATING_OPS and not _defer_mutation_check:
            raise Unsupported("aten.index_put_ in a graph requires explicit alias and post-state semantics; "
                              "the current GRAPH contract composes functional values only")
    for output in graph["outputs"]:
        check_ref(_reference_name(output) if isinstance(output, dict) else output)


def functional_value_graph(graph, input_names=None):
    """Remove only updates to fresh allocations with exactly one use.

    The allocation cannot be an input, returned, aliased, read elsewhere, or used
    as an update index/value. Its sole consumer is the update destination, so the
    post-value returned by index_put represents every observable effect. Any
    additional use requires real alias/state composition and is refused.
    """
    require_value_graph(graph, input_names, _defer_mutation_check=True)
    nodes = {n["name"]: n for n in graph["nodes"]}
    uses = {}
    for node in graph["nodes"]:
        for name in _walk_refs(node.get("args", [])) + _walk_refs(list((node.get("kwargs") or {}).values())):
            uses.setdefault(name, []).append(node["name"])
    for output in graph["outputs"]:
        name = output["ref"] if isinstance(output, dict) else output
        uses.setdefault(name, []).append("graph output")
    evidence, replacements = [], set()
    for node in graph["nodes"]:
        if node["op"] not in MUTATING_OPS:
            continue
        args, kwargs = node.get("args", []), node.get("kwargs") or {}
        if len(args) > 4 or set(kwargs) - {"a", "indices", "values", "accumulate"}:
            raise Unsupported("local index_put_ has unimplemented parameters")
        try:
            bound = inspect.signature(A.index_put).bind(*args, **kwargs)
        except TypeError as error:
            raise Invalid("local index_put_ arguments do not match its signature") from error
        destination = bound.arguments["a"]
        if not isinstance(destination, dict) or set(destination) != {"ref"}:
            raise Unsupported("local update destination must be an explicit whole allocation")
        name = destination["ref"]
        allocation = nodes.get(name)
        if (allocation is None or allocation["op"] != "aten.full"
                or not 2 <= len(allocation.get("args", [])) <= 3
                or set(allocation.get("kwargs") or {}) - {"dtype"}
                or uses.get(name) != [node["name"]]):
            raise Unsupported("mutation requires a fresh local full tensor used only as this update destination")
        replacements.add(node["name"])
        evidence.append({"update": node["name"], "allocation": name,
                         "allocation_uses": list(uses[name]),
                         "observable_result": node["name"], "external_state_changed": False,
                         "scope": "logical task graph; captured memory effects require separate checks"})
    normalized = dict(graph, nodes=[dict(n, op="aten.index_put") if n["name"] in replacements else n
                                    for n in graph["nodes"]])
    require_value_graph(normalized, input_names)
    return normalized, evidence


def scatter_outputs(result, budget):
    """The scatter contract permits one source value at each duplicated target.

    Concretise complete tensors before graph consumers run; each reuse then
    sees the same choice. This rule is specific to scatter, not arbitrary sets.
    """
    from .compare import _canon
    from .expr import ExpressionError
    positions, choices, total = [], [], 1
    for position in np.ndindex(result.a.shape):
        value = result.a[position]
        if not isinstance(value, A.AllowedSet):
            continue
        members = {}
        try:
            for member in value.members:
                members.setdefault(_canon(encode(member)), member)
        except ExpressionError as error:
            raise NotEstablished("scatter choices are not exact finite values") from error
        if not members:
            raise NotEstablished("scatter has an empty choice set")
        total *= len(members)
        if total > budget:
            raise NotEstablished("scatter joint-choice enumeration exceeded its budget")
        positions.append(position)
        choices.append(list(members.values()))
    if not positions:
        return [result]
    alternatives = []
    for assignment in itertools.product(*choices):
        array = result.a.copy()
        for position, value in zip(positions, assignment):
            array[position] = value
        alternatives.append(A.Ten(array, result.dtype))
    return alternatives


def _evaluate(graph, elide_intermediate_casts, choices, max_choices, ops):
    env = {name: decode(t) for name, t in graph.get("inputs", {}).items()}
    consumers = {}
    for n in graph["nodes"]:
        for a in _walk_refs(n.get("args", [])) + _walk_refs(list((n.get("kwargs") or {}).values())):
            consumers.setdefault(a, []).append(n["name"])
    outputs = {o["ref"] if isinstance(o, dict) else o for o in graph["outputs"]}
    for n in graph["nodes"]:
        op = n["op"]
        if op not in ops:
            raise Unsupported(f"{op} is not covered by a base contract or a GRAPH rule")
        if n["name"] in env:
            raise Invalid(f"duplicate node name {n['name']!r}")
        try:
            args = [_resolve(a, env) for a in n.get("args", [])]
            kwargs = {k: _resolve(v, env) for k, v in (n.get("kwargs") or {}).items()}
            if op == "operator.getitem":
                env[n["name"]] = select_result(args[0], args[1])
                continue
            fn = A.OPS.get(op) or sorting.OPS[op]
            bound = bind_reference_call(op, inspect.signature(fn), args, kwargs)
            args, kwargs = bound.args, bound.kwargs
            if (elide_intermediate_casts and op in ("prims.convert_element_type", "aten._to_copy")
                    and n["name"] not in outputs and isinstance(args[0], A.Ten) and args[0].dtype in A.FLOAT):
                target = A.dt(args[1] if len(args) > 1 else kwargs.get("dtype"))
                if target in A.FLOAT:                       # the reading: no rounding at an intermediate float cast
                    env[n["name"]] = A.Ten(args[0].a, target)
                    continue
            env[n["name"]] = (A.OPS.get(op) or sorting.OPS[op])(*args, **kwargs)
            if op in sorting.OPS or op == "aten.scatter":
                if n["name"] in choices:
                    env[n["name"]] = choices[n["name"]]
                else:
                    if op == "aten.scatter":
                        alternatives = scatter_outputs(env[n["name"]], max_choices)
                    else:
                        dimpos = 2 if op == "aten.topk" else 1
                        dim = args[dimpos] if len(args) > dimpos else kwargs.get("dim", -1)
                        alternatives = sorting.concrete_outputs(env[n["name"]], op, dim, max_choices)
                    if len(alternatives) > 1:
                        raise _Branch(n["name"], alternatives)
                    if not alternatives:
                        raise NotEstablished(f"{op} produced no joint output")
                    env[n["name"]] = alternatives[0]
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
            refs.append(_reference_name(x))
        elif isinstance(x, list):
            refs += _walk_refs(x)
    return refs
