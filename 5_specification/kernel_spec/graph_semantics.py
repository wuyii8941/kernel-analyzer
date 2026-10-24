"""Admitted node semantics for the GRAPH composition contract.

The GRAPH evaluator must select a contract/profile and its executable meaning,
not merely find an operation name in a helper table.  This module is the one
shared gate used by the complete and lazy evaluators.  It deliberately refuses
operations without a reviewed base contract or an explicit GRAPH-local rule.

Physical graph inputs are checked at the boundary.  Node results are mathematical
values (often rigorous intervals), so downstream nodes use the explicitly
declared interval lift below instead of pretending that those values are new
physical tensors in a base-contract profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import inspect
from pathlib import Path

import numpy as np
import yaml

from . import aten as A
from . import ieee
from . import real as rl
from . import sorting
from .aten_adapter import bind_reference_call
from .num import Invalid, NotEstablished, Undefined, Unsupported
from .real import R


BASE_CONTRACTS = ("ARITH", "MATHFN", "LOGIC", "CAST", "LAYOUT", "INDEX", "REDUCE", "SOFTMAX", "MATMUL",
                  "CONV", "RANDOM", "ACT", "SORT")
CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
GRAPH_LOCAL_VALUE_OPS = {"aten.full", "aten.index_put", "aten.scatter"}


@dataclass(frozen=True)
class NodeRule:
    contract: str
    profile: str
    function: object
    guarantee: str
    aliases: dict


def _contract_owners():
    owners = {}
    for contract in BASE_CONTRACTS:
        record = yaml.safe_load((CONTRACTS / f"{contract}.yaml").read_text())
        if (record.get("activation") or {}).get("implementation_status") != "validated":
            raise RuntimeError(f"GRAPH cannot import unvalidated base contract {contract}")
        for operation in record.get("covers") or []:
            if operation in owners:
                raise RuntimeError(f"GRAPH operation {operation} belongs to more than one base contract")
            owners[operation] = contract
    return owners


def _tables():
    # Import the reviewed semantic cores, not their Result/serialization wrappers.
    from adapters import ACT, ARITH, CAST, INDEX, LAYOUT, LOGIC, MATHFN, MATMUL, RANDOM, REDUCE

    common = dict(A.OPS)
    tables = {
        "ARITH": {**common, **ARITH.OPS},
        "MATHFN": {**common, **MATHFN.STRICT_OPS},
        "LOGIC": {**common, **LOGIC.OPS},
        "CAST": {**common, **CAST.OPS},
        "LAYOUT": {**common, **LAYOUT.OPS},
        "INDEX": {**common, **INDEX.OPS},
        "REDUCE": {**common, **REDUCE.REVIEWED_OPS},
        "SOFTMAX": common,
        "MATMUL": common,
        "CONV": common,
        "RANDOM": {**common, **RANDOM.FORWARD_OPS, **RANDOM.BACKWARD_OPS},
        "ACT": {**common, **ACT.OPS},
        # GRAPH extends reviewed unique sorting with the explicitly relational
        # GR.JOINT_CHOICES rule.  Validation below keeps all other SORT limits.
        "SORT": {**common, **sorting.OPS},
    }
    aliases = {**INDEX.ARGUMENT_ALIASES, **MATMUL.ARGUMENT_ALIASES}
    return tables, aliases


_PROFILE = {
    "ARITH": "real", "MATHFN": "real", "LOGIC": "real", "CAST": "finite_rne", "LAYOUT": "real",
    "INDEX": "real", "REDUCE": "real", "SOFTMAX": "real", "MATMUL": "real", "CONV": "real",
    "RANDOM": "deterministic_forward", "ACT": "torch", "SORT": "finite_unique+GR.JOINT_CHOICES",
}


def _profile(contract, operation, args):
    if contract == "LOGIC" and operation == "aten.isnan":
        return "ieee_values"
    if contract == "LAYOUT":
        tensors = list(_tensors(args))
        if tensors and all(t.dtype == "b8" for t in tensors):
            return "boolean"
    if contract == "RANDOM" and operation == "aten.native_dropout_backward":
        return "finite_backward"
    # native_dropout needs its returned mask as an observed input.  A plain
    # graph node does not carry that extra observation, so it has no rule here.
    if contract == "RANDOM" and operation == "aten.native_dropout":
        raise Unsupported("aten.native_dropout needs the observed returned mask; a plain GRAPH node is insufficient")
    return _PROFILE[contract]


def _build_rules():
    owners, (tables, aliases), rules = _contract_owners(), _tables(), {}
    for operation, contract in owners.items():
        function = tables[contract].get(operation)
        if function is None:
            continue
        rules[operation] = (contract, function, aliases.get(operation, {}))
    for operation in GRAPH_LOCAL_VALUE_OPS:
        function = A.OPS.get(operation)
        if function is None:
            raise RuntimeError(f"GRAPH-local operation {operation} has no value semantics")
        rules[operation] = ("GRAPH", function, {})
    return rules


_RULES = _build_rules()


def covered_ops():
    return {"operator.getitem", *_RULES}


def rule_for(operation, args=()):
    if operation == "operator.getitem":
        return NodeRule("GRAPH", "result_projection", None, "exact result identity", {})
    try:
        contract, function, aliases = _RULES[operation]
    except KeyError:
        raise Unsupported(f"{operation} has no admitted base-contract or GRAPH-local node semantics") from None
    profile = "graph_local" if contract == "GRAPH" else _profile(contract, operation, args)
    guarantee = "complete finite relation" if operation in sorting.OPS or operation == "aten.scatter" \
        else "rigorous enclosure"
    return NodeRule(contract, profile, function, guarantee, aliases)


def bind_node_call(operation, args, kwargs):
    rule = rule_for(operation, args)
    normalized = {}
    for name, value in kwargs.items():
        target = rule.aliases.get(name, name)
        if target in normalized:
            raise Invalid(f"{operation}: multiple keyword arguments for {target!r}")
        normalized[target] = value
    bound = bind_reference_call(operation, inspect.signature(rule.function), args, normalized)
    if rule.contract != "GRAPH":
        rule = NodeRule(rule.contract, _profile(rule.contract, operation, (bound.args, bound.kwargs)), rule.function,
                        rule.guarantee, rule.aliases)
    return rule, bound


def _tensors(value):
    if isinstance(value, A.Ten):
        yield value
    elif hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "get"):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _values(values):
    for tensor in _tensors(values):
        yield from tensor.a.flat


def _finite(values, contract):
    for item in _values(values):
        if isinstance(item, R) and not item.finite:
            raise Unsupported(f"{contract} node contains infinity outside its selected GRAPH profile")
        if isinstance(item, (float, np.floating)) and not np.isfinite(float(item)):
            raise Invalid(f"{contract} node contains NaN or infinity outside its selected GRAPH profile")


def _float_tensors(values, contract, *, allow_bool=False):
    allowed = set(A.FLOAT) | ({"b8"} if allow_bool else set())
    for tensor in _tensors(values):
        if tensor.dtype not in allowed:
            raise Unsupported(f"{contract} node dtype {tensor.dtype!r} is outside its selected GRAPH profile")


def validate_graph_inputs(raw_inputs):
    """Check that graph placeholders denote physical values of their declared dtype.

    Operation-specific restrictions are checked when a node consumes the value.
    This boundary check prevents a logical rational such as 1/3 from being called
    an actual f32 input.
    """
    if not isinstance(raw_inputs, dict):
        raise Invalid("graph inputs must be a name-to-value mapping")
    from .aten_adapter import decode
    for name, raw in raw_inputs.items():
        try:
            value = decode(raw)
        except Undefined as error:
            raise Invalid(f"graph input {name!r} is not a physical value: {error}") from error
        if not isinstance(value, A.Ten):
            raise Invalid(f"graph input {name!r} must be a tensor record")
        if value.dtype in A.FLOAT:
            for item in value.a.flat:
                if not isinstance(item, R) or not item.exact:
                    raise Invalid(f"graph input {name!r} must contain exact physical floating values")
                if item.finite:
                    stored = ieee.round_to(item, value.dtype)
                    if not stored.exact or stored.v != item.v:
                        raise Invalid(f"graph input {name!r} contains {item!r}, not representable as {value.dtype}")
        elif value.dtype in A.INT:
            low, high = A.INT_RANGE[value.dtype]
            for item in value.a.flat:
                if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                    raise Invalid(f"graph input {name!r} must contain literal {value.dtype} integers")
                if not low <= int(item) <= high:
                    raise Invalid(f"graph input {name!r} contains an integer outside {value.dtype}")
        elif value.dtype == "b8":
            if any(not isinstance(item, (bool, np.bool_)) for item in value.a.flat):
                raise Invalid(f"graph input {name!r} must contain literal booleans")
        else:
            raise Unsupported(f"graph input {name!r} dtype {value.dtype!r} has no GRAPH boundary rule")


_MATH_DOMAIN = {
    "aten.log": (lambda lo, hi: lo > 0, "log requires x > 0"),
    "aten.log1p": (lambda lo, hi: lo > -1, "log1p requires x > -1"),
    "aten.sqrt": (lambda lo, hi: lo >= 0, "sqrt requires x >= 0"),
    "aten.rsqrt": (lambda lo, hi: lo > 0, "rsqrt requires x > 0"),
}
_MATH_UNBOUNDED = {"aten.sin", "aten.cos", "aten.tanh"}
_MATH_REAL = {
    "aten.exp": rl.exp, "aten.expm1": rl.expm1, "aten.log": rl.log, "aten.log1p": rl.log1p,
    "aten.sqrt": rl.sqrt, "aten.rsqrt": lambda x: R(1) / rl.sqrt(x), "aten.sin": rl.sin,
    "aten.cos": rl.cos, "aten.tanh": rl.tanh, "aten.sigmoid": rl.sigmoid, "aten.erf": rl.erf,
}


def mathfn_scalar(operation, value):
    """The reviewed point oracle plus its explicit rigorous interval lift."""
    from adapters import MATHFN
    value = value if isinstance(value, R) else R(value)
    if not value.finite:
        raise Unsupported(f"{operation} contains infinity outside the finite MATHFN profile")
    lo, hi = value.bounds()
    if operation not in _MATH_UNBOUNDED and (lo < -100 or hi > 100):
        raise Unsupported(f"{operation} contains a value with magnitude above 100")
    if operation in _MATH_DOMAIN:
        predicate, message = _MATH_DOMAIN[operation]
        if not predicate(lo, hi):
            # An interval straddling the boundary is insufficient to establish
            # either a value or whole-node undefinedness.
            if (operation in {"aten.log", "aten.rsqrt"} and hi > 0) or \
                    (operation == "aten.log1p" and hi > -1) or (operation == "aten.sqrt" and hi >= 0):
                raise NotEstablished(f"{message}; an input enclosure crosses the domain boundary")
            raise Undefined(message)
    if value.exact:
        one = A.Ten(np.array([value], dtype=object), "f64")
        return MATHFN.STRICT_OPS.get(operation, A.OPS[operation])(one).a[0]
    return _MATH_REAL[operation](value)


def _mathfn(operation, a):
    if not isinstance(a, A.Ten):
        raise Invalid(f"{operation} requires one tensor")
    if a.dtype not in A.FLOAT:
        raise Unsupported(f"{operation} requires f16, bf16, f32 or f64")
    out = np.empty(a.shape, dtype=object)
    for index in np.ndindex(a.shape):
        out[index] = mathfn_scalar(operation, a.a[index])
    return A.Ten(out, a.dtype)


def _validate_sort(operation, args, kwargs):
    from adapters import SORT
    if not args or not isinstance(args[0], A.Ten):
        raise Invalid(f"{operation} requires a tensor")
    a = args[0]
    if a.dtype not in SORT.FLOAT_DTYPES or len(a.shape) == 0:
        raise Unsupported("GRAPH sorting requires a positive-rank f16/bf16/f32/f64 tensor")
    for value in a.a.flat:
        if not isinstance(value, R) or not value.finite:
            raise Unsupported("GRAPH joint sorting requires finite real values or rigorous enclosures")
    names = ("descending", "stable") if operation != "aten.topk" else ("largest", "sorted")
    signature = inspect.signature(sorting.OPS[operation])
    bound = bind_reference_call(operation, signature, args, kwargs)
    for name in names:
        if name in bound.arguments and not isinstance(bound.arguments[name], (bool, np.bool_)):
            raise Invalid(f"{operation} {name} must be boolean")
    if operation == "aten.topk" and bound.arguments.get("sorted", True) is not True:
        raise Unsupported("GRAPH topk keeps the reviewed sorted=true scope")


_FINITE_CONTRACTS = {"MATHFN", "LOGIC", "CAST", "LAYOUT", "INDEX", "REDUCE", "MATMUL", "CONV", "RANDOM",
                     "ACT", "SORT"}


def _validate_rule(rule, operation, args, kwargs):
    values = (args, kwargs)
    if rule.contract == "MATHFN":
        if len(args) != 1 or kwargs:
            raise Invalid(f"{operation} requires exactly one positional tensor")
        _float_tensors(values, "MATHFN")
        return
    if rule.contract in _FINITE_CONTRACTS:
        _finite(values, rule.contract)
    if rule.contract in {"REDUCE", "SOFTMAX", "MATMUL", "CONV", "RANDOM", "ACT"}:
        _float_tensors(values, rule.contract)
    if rule.contract == "SORT":
        _validate_sort(operation, args, kwargs)


def execute_node(operation, args, kwargs):
    """Validate and execute one admitted node under its selected rule."""
    rule = rule_for(operation, args)
    if operation == "operator.getitem":
        raise Invalid("operator.getitem is handled by GRAPH result projection")
    function = rule.function
    rule, bound = bind_node_call(operation, args, kwargs)
    args, kwargs = bound.args, bound.kwargs
    _validate_rule(rule, operation, args, kwargs)
    if rule.contract == "MATHFN":
        return _mathfn(operation, args[0]), rule
    return function(*args, **kwargs), rule


def validate_lazy_metadata(operation, args, kwargs):
    """Common admission/profile gate for a lazy node.

    Value-dependent checks occur through ``mathfn_scalar`` and the lazy helper;
    complete-graph evaluation forces every node, while an explicit output slice
    is reported as partial and cannot certify the original whole graph.
    """
    rule = rule_for(operation, args)
    if operation == "operator.getitem":
        return rule, tuple(args), dict(kwargs)
    # Bind against the admitted semantic function, not an arbitrary lazy helper.
    rule, bound = bind_node_call(operation, args, kwargs)
    args, kwargs = bound.args, bound.kwargs
    tensors = list(_lazy_tensors((args, kwargs)))
    if rule.contract == "MATHFN":
        if len(args) != 1 or kwargs or not tensors or any(t.dtype not in A.FLOAT for t in tensors):
            raise Unsupported(f"{operation} is outside the MATHFN GRAPH profile")
    if rule.contract in {"REDUCE", "SOFTMAX", "MATMUL", "CONV", "RANDOM", "ACT"} \
            and any(t.dtype not in A.FLOAT for t in tensors):
        raise Unsupported(f"{operation} has a dtype outside the selected {rule.contract} profile")
    return rule, bound.args, bound.kwargs


def validate_lazy_result(rule, value):
    """Check value-dependent profile conditions after complete lazy materialisation."""
    if rule.contract not in _FINITE_CONTRACTS:
        return

    def leaves(item):
        if hasattr(item, "shape") and hasattr(item, "dtype") and hasattr(item, "get") and not isinstance(item, A.Ten):
            for index in np.ndindex(*item.shape):
                yield item.get(index)
        elif isinstance(item, A.Ten):
            yield from item.a.flat
        elif isinstance(item, (list, tuple)):
            for child in item:
                yield from leaves(child)

    for item in leaves(value):
        if isinstance(item, R) and not item.finite:
            raise Unsupported(f"{rule.contract} lazy node produced infinity outside its selected GRAPH profile")
        if isinstance(item, (float, np.floating)) and not np.isfinite(float(item)):
            raise Invalid(f"{rule.contract} lazy node produced NaN or infinity")


def _lazy_tensors(value):
    # Avoid importing lazy at module import time (lazy imports this module).
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "get") and not isinstance(value, A.Ten):
        yield value
    elif isinstance(value, A.Ten):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _lazy_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _lazy_tensors(item)
