"""Reference evaluator: local, composed and derivative references.

Three executions share one set of operation rules:

``local``
    Each node is evaluated on the inputs it actually read (captured values).
    The local residual ``K_n(x_hat) - g_n(x_hat)`` is term (1) of the node
    decomposition.  Nodes whose value does not exist on the device (for
    example a product fused into an fma) are not observable; their reference
    is folded into the nearest observable consumer (region boundary).

``composed``
    Reference values are propagated from the captured external inputs through
    the whole program.  The evaluator keeps its own reference memory, decides
    branches with reference values, and takes the union of both branches when
    the reference condition is undecided.  Every value carries taints that
    record whether the reference had to fall back to an actual value, an
    actual path, or could not be established.

``derivative``
    Forward-mode differentiation on the declared expression with interval
    tangents.  It gives ``J(x) u`` without truncation error, for the
    forward/backward dot-product check.

Reference semantics is a declared mode.  ``NUMERICAL_DIFFERENCE`` (default)
treats every conversion and every rounding-suffixed operation as the real
operation; ``ROUNDING_CHECK`` applies the declared IEEE rounding.  A node may
declare ``declared_quantization=True`` to round in either mode.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from gmpy2 import mpq

from .numbers import (
    ELEMENTARY,
    DomainError,
    Interval,
    Overflow,
    elementary_derivative,
    is_representable,
    round_interval,
)
from .program import Op, Program


# ---------------------------------------------------------------------------
# Element kinds
# ---------------------------------------------------------------------------


class Special(Enum):
    NAN = "nan"
    POS_INF = "+inf"
    NEG_INF = "-inf"


class _Sentinel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


UNDEFINED = _Sentinel("UNDEFINED")
# Undecided boolean: the reference interval does not determine a comparison.
MAYBE = _Sentinel("MAYBE")


class Mode(Enum):
    NUMERICAL_DIFFERENCE = "numerical_difference"
    ROUNDING_CHECK = "rounding_check"


class OutputClass(Enum):
    COMPLETE_COMPOSED = "complete_composed_reference"
    CONDITIONAL_LOCAL = "conditional_local_reference"
    NOT_ESTABLISHED = "reference_not_established"


# Taint prefixes.  Conditional taints downgrade an output to a conditional
# local reference; not-established taints remove it from the reference.
CONDITIONAL_PREFIXES = ("pinned_load:", "actual_path:")
NOT_ESTABLISHED_PREFIXES = ("not_established:", "undefined:")
INFO_PREFIXES = ("path_union:", "nonsmooth_union:")


@dataclass(frozen=True)
class Ref:
    """A reference element with provenance taints and an optional tangent."""

    value: Any
    taints: frozenset = frozenset()
    tangent: Optional[Interval] = None


def classify(ref: Ref) -> tuple[OutputClass, list[str]]:
    reasons = sorted(ref.taints)
    if ref.value is UNDEFINED or any(t.startswith(NOT_ESTABLISHED_PREFIXES) for t in ref.taints):
        return OutputClass.NOT_ESTABLISHED, reasons
    if any(t.startswith(CONDITIONAL_PREFIXES) for t in ref.taints):
        return OutputClass.CONDITIONAL_LOCAL, reasons
    return OutputClass.COMPLETE_COMPOSED, reasons


def _worst(classes: Sequence[OutputClass]) -> OutputClass:
    order = [OutputClass.COMPLETE_COMPOSED, OutputClass.CONDITIONAL_LOCAL, OutputClass.NOT_ESTABLISHED]
    return max(classes, key=order.index) if classes else OutputClass.COMPLETE_COMPOSED


class UnsupportedOperation(ValueError):
    """An op or attribute has no reference rule; nothing is silently skipped."""


class NotObservable(LookupError):
    """The node has no actual value on the device (fused, eliminated)."""


class _NotEstablished(Exception):
    pass


# ---------------------------------------------------------------------------
# Value conversion
# ---------------------------------------------------------------------------


def to_element(value) -> Any:
    """Convert a captured Python value into an exact reference element."""

    if isinstance(value, (Interval, Special)) or value is UNDEFINED or value is MAYBE:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, type(mpq(0))):
        return Interval(value)
    if isinstance(value, float):
        if value != value:
            return Special.NAN
        if value == float("inf"):
            return Special.POS_INF
        if value == float("-inf"):
            return Special.NEG_INF
        return Interval(value)
    if hasattr(value, "item"):
        return to_element(value.item())
    raise TypeError(f"cannot use {type(value).__name__} as a reference element")


def _block(value) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, (Interval,)):
        listed = value.tolist()
        return list(listed) if isinstance(listed, list) else [listed]
    return [value]


def _special_to_float(value) -> float:
    if value is Special.NAN:
        return float("nan")
    if value is Special.POS_INF:
        return float("inf")
    if value is Special.NEG_INF:
        return float("-inf")
    if isinstance(value, Interval) and value.is_point:
        return float(value.lo)
    raise _NotEstablished("special value combined with a non-point interval")


def _float_to_element(value: float):
    if math.isnan(value):
        return Special.NAN
    if math.isinf(value):
        return Special.POS_INF if value > 0 else Special.NEG_INF
    if value == 0.0:
        return Interval(0)
    raise _NotEstablished("finite result from special-value arithmetic")


def _hull_values(a, b):
    if isinstance(a, Interval) and isinstance(b, Interval):
        return a.hull(b)
    if a == b:
        return a
    if isinstance(a, bool) and isinstance(b, bool):
        return MAYBE
    if (a is MAYBE and isinstance(b, bool)) or (b is MAYBE and isinstance(a, bool)):
        return MAYBE
    raise _NotEstablished("union of incompatible branch values")


def _hull_tangents(a: Optional[Interval], b: Optional[Interval]) -> Optional[Interval]:
    if a is None and b is None:
        return None
    return (a or Interval(0)).hull(b or Interval(0))


def _hull_refs(a: Ref, b: Ref, info: str) -> Ref:
    taints = a.taints | b.taints | {info}
    try:
        value = _hull_values(a.value, b.value)
    except _NotEstablished as exc:
        return Ref(UNDEFINED, taints | {f"not_established:{exc}"})
    return Ref(value, taints, _hull_tangents(a.tangent, b.tangent))


# ---------------------------------------------------------------------------
# Elementwise rules
# ---------------------------------------------------------------------------

FLOAT_BINARY = {"add", "sub", "mul", "div"}
ROUNDING_SUFFIX = {f"{name}_rn": name for name in FLOAT_BINARY}
COMMON_ATTRS = {"observable", "loc", "source"}


def _round_if_declared(value: Interval, attrs: Mapping, mode: Mode, tangent):
    rounds = attrs.get("declared_quantization") or (mode is Mode.ROUNDING_CHECK and attrs.get("format"))
    if not rounds:
        return value, tangent
    fmt = attrs.get("format")
    if not fmt:
        raise _NotEstablished("declared rounding without a format")
    try:
        rounded = round_interval(value, fmt)
    except Overflow as exc:
        if value.is_point:
            return (Special.POS_INF if exc.sign > 0 else Special.NEG_INF), None
        raise _NotEstablished("rounding overflow inside the interval") from None
    if tangent is not None:
        surrogate = attrs.get("surrogate_gradient")
        if surrogate == "identity":
            pass
        else:
            raise _NotEstablished("derivative of a rounding node without a declared surrogate gradient")
    return rounded, tangent


def _float_args(args: Sequence[Ref]):
    values = [a.value for a in args]
    for v in values:
        if not isinstance(v, (Interval, Special)):
            raise _NotEstablished(f"float operation on {type(v).__name__}")
    return values


def _tangent(ref: Ref) -> Interval:
    return ref.tangent if ref.tangent is not None else Interval(0)


def _tracks(args: Sequence[Ref]) -> bool:
    return any(a.tangent is not None for a in args)


def _real_float_op(name: str, args: Sequence[Ref], same_operand: bool):
    """Real result and tangent of a float op on interval arguments."""

    values = [a.value for a in args]
    track = _tracks(args)
    if name == "add":
        return values[0] + values[1], (_tangent(args[0]) + _tangent(args[1])) if track else None
    if name == "sub":
        return values[0] - values[1], (_tangent(args[0]) - _tangent(args[1])) if track else None
    if name == "mul":
        a, b = values
        value = a.square() if same_operand else a * b
        if not track:
            return value, None
        if same_operand:
            return value, Interval(2) * a * _tangent(args[0])
        return value, a * _tangent(args[1]) + _tangent(args[0]) * b
    if name == "div":
        a, b = values
        value = a / b
        if not track:
            return value, None
        return value, (_tangent(args[0]) * b - a * _tangent(args[1])) / b.square()
    if name == "neg":
        return -values[0], (-_tangent(args[0])) if track else None
    if name == "abs":
        x = values[0]
        if x.lo >= 0:
            return x, _tangent(args[0]) if track else None
        if x.hi <= 0:
            return -x, (-_tangent(args[0])) if track else None
        value = Interval(0, max(-x.lo, x.hi))
        if not track:
            return value, None
        t = _tangent(args[0])
        return value, t.hull(-t)
    if name == "fma":
        a, b, c = values
        value = a * b + c
        if not track:
            return value, None
        return value, a * _tangent(args[1]) + _tangent(args[0]) * b + _tangent(args[2])
    if name in ELEMENTARY:
        x = values[0]
        value = ELEMENTARY[name](x)
        if not track:
            return value, None
        return value, elementary_derivative(name, x) * _tangent(args[0])
    raise KeyError(name)


_SPECIAL_FLOAT = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if b != 0 else (math.copysign(float("inf"), a) * math.copysign(1.0, b) if a != 0 and a == a else float("nan")),
    "neg": lambda a: -a,
    "abs": abs,
    "fma": lambda a, b, c: a * b + c,
    "exp": lambda a: math.exp(a) if a != float("-inf") else 0.0,
    "log": lambda a: math.log(a) if a > 0 else (float("nan") if a == a and a < 0 else (float("-inf") if a == 0 else a)),
    "sqrt": lambda a: math.sqrt(a) if a >= 0 else float("nan"),
}


def float_rule(op: Op, args: Sequence[Ref], mode: Mode) -> Ref:
    base = ROUNDING_SUFFIX.get(op.name, op.name)
    taints = frozenset().union(*(a.taints for a in args))
    if any(a.value is UNDEFINED for a in args):
        return Ref(UNDEFINED, taints)
    values = _float_args(args)
    if any(isinstance(v, Special) for v in values):
        if base not in _SPECIAL_FLOAT:
            raise _NotEstablished(f"special value in {base}")
        floats = [_special_to_float(v) for v in values]
        with _ieee_floats():
            result = _SPECIAL_FLOAT[base](*floats)
        return Ref(_float_to_element(result), taints)
    same = len(op.inputs) == 2 and op.inputs[0] == op.inputs[1]
    try:
        value, tangent = _real_float_op(base, args, same)
    except DomainError as exc:
        raise _NotEstablished(f"domain: {exc}") from None
    value, tangent = _round_if_declared(value, op.attrs, mode, tangent)
    return Ref(value, taints, tangent)


class _ieee_floats:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type in (OverflowError, ZeroDivisionError, ValueError):
            raise _NotEstablished("special-value arithmetic outside IEEE model") from None
        return False


def cast_rule(op: Op, args: Sequence[Ref], mode: Mode) -> Ref:
    (arg,) = args
    value = arg.value
    if value is UNDEFINED or isinstance(value, Special):
        return Ref(value, arg.taints)
    if isinstance(value, int) and not isinstance(value, bool):
        value = Interval(value)
    if not isinstance(value, Interval):
        raise _NotEstablished("cast of a non-float value")
    attrs = dict(op.attrs)
    attrs.setdefault("format", attrs.get("to_format"))
    value, tangent = _round_if_declared(value, attrs, mode, arg.tangent)
    return Ref(value, arg.taints, tangent)


def fptosi_rule(op: Op, args: Sequence[Ref], mode: Mode) -> Ref:
    (arg,) = args
    value = arg.value
    if not isinstance(value, Interval):
        raise _NotEstablished("float-to-int conversion of a non-finite value")
    lo, hi = int(value.lo), int(value.hi)  # truncation toward zero, exact on mpq
    if lo != hi:
        raise _NotEstablished("float-to-int conversion is discontinuous inside the interval")
    return Ref(lo, arg.taints)


def _minmax(op: Op, args: Sequence[Ref], mode: Mode) -> Ref:
    a, b = args
    taints = a.taints | b.taints
    if a.value is UNDEFINED or b.value is UNDEFINED:
        return Ref(UNDEFINED, taints)
    is_max = op.name in ("maxnum", "maximum")
    propagate_nan = op.name in ("maximum", "minimum")
    va, vb = a.value, b.value
    if va is Special.NAN or vb is Special.NAN:
        if propagate_nan:
            return Ref(Special.NAN, taints)
        other = b if va is Special.NAN else a
        return Ref(other.value, taints, other.tangent)
    if isinstance(va, Special) or isinstance(vb, Special):
        fa, fb = _special_to_float(va), _special_to_float(vb)
        pick_a = (fa >= fb) if is_max else (fa <= fb)
        chosen = a if pick_a else b
        return Ref(chosen.value, taints, chosen.tangent)
    if is_max:
        value = Interval(max(va.lo, vb.lo), max(va.hi, vb.hi))
        a_wins, b_wins = va.lo > vb.hi, vb.lo > va.hi
    else:
        value = Interval(min(va.lo, vb.lo), min(va.hi, vb.hi))
        a_wins, b_wins = va.hi < vb.lo, vb.hi < va.lo
    tangent = None
    if _tracks(args):
        if a_wins:
            tangent = _tangent(a)
        elif b_wins:
            tangent = _tangent(b)
        else:
            tangent = _tangent(a).hull(_tangent(b))
            taints = taints | {f"nonsmooth_union:{op.node_id}"}
    return Ref(value, taints, tangent)


_PREDICATES = {"lt", "le", "gt", "ge", "eq", "ne"}


def cmp_rule(op: Op, args: Sequence[Ref], mode: Mode) -> Ref:
    a, b = args
    taints = a.taints | b.taints
    if a.value is UNDEFINED or b.value is UNDEFINED:
        return Ref(UNDEFINED, taints)
    pred = op.attrs["predicate"]
    va, vb = a.value, b.value
    if isinstance(va, int) and not isinstance(va, bool) and isinstance(vb, int) and not isinstance(vb, bool):
        return Ref(_compare_exact(pred, va, vb), taints)
    if va is Special.NAN or vb is Special.NAN:
        # Ordered predicates are false on NaN; "ne" is the unordered not-equal.
        return Ref(pred == "ne", taints)
    if isinstance(va, Special) or isinstance(vb, Special):
        return Ref(_compare_exact(pred, _special_to_float(va), _special_to_float(vb)), taints)
    va, vb = _as_interval(va), _as_interval(vb)
    # Decide the predicate on the whole box [va] x [vb].
    if pred in ("lt", "le", "gt", "ge"):
        lo_holds = _compare_exact(pred, va.hi, vb.lo) if pred in ("lt", "le") else _compare_exact(pred, va.lo, vb.hi)
        hi_holds = _compare_exact(pred, va.lo, vb.hi) if pred in ("lt", "le") else _compare_exact(pred, va.hi, vb.lo)
        if lo_holds:
            return Ref(True, taints)
        if not hi_holds:
            return Ref(False, taints)
        return Ref(MAYBE, taints)
    equal_possible = va.lo <= vb.hi and vb.lo <= va.hi
    surely_equal = va.is_point and vb.is_point and va.lo == vb.lo
    if pred == "eq":
        return Ref(True if surely_equal else (MAYBE if equal_possible else False), taints)
    return Ref(False if surely_equal else (MAYBE if equal_possible else True), taints)


def _as_interval(value) -> Interval:
    if isinstance(value, Interval):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Interval(value)
    raise _NotEstablished(f"comparison on {type(value).__name__}")


def _compare_exact(pred, a, b) -> bool:
    return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b, "eq": a == b, "ne": a != b}[pred]


def isnan_rule(op: Op, args: Sequence[Ref], mode: Mode) -> Ref:
    (a,) = args
    if a.value is UNDEFINED:
        return Ref(UNDEFINED, a.taints)
    if op.name == "isnan":
        return Ref(a.value is Special.NAN, a.taints)
    return Ref(a.value in (Special.POS_INF, Special.NEG_INF), a.taints)


def select_rule(op: Op, args: Sequence[Ref], mode: Mode) -> Ref:
    cond, a, b = args
    if cond.value is True:
        return Ref(a.value, a.taints | cond.taints, a.tangent)
    if cond.value is False:
        return Ref(b.value, b.taints | cond.taints, b.tangent)
    if cond.value is MAYBE:
        return _hull_refs(Ref(a.value, a.taints | cond.taints, a.tangent), b, f"path_union:{op.node_id}")
    return Ref(UNDEFINED, cond.taints | {f"undefined:select condition@{op.node_id}"})


def logic_rule(op: Op, args: Sequence[Ref], mode: Mode) -> Ref:
    taints = frozenset().union(*(a.taints for a in args))
    values = [a.value for a in args]
    if any(v is UNDEFINED for v in values):
        return Ref(UNDEFINED, taints)
    if op.name == "not":
        v = values[0]
        return Ref(MAYBE if v is MAYBE else (not v), taints)
    a, b = values
    if op.name == "and":
        if a is False or b is False:
            return Ref(False, taints)
        return Ref(True if (a is True and b is True) else MAYBE, taints)
    if a is True or b is True:
        return Ref(True, taints)
    return Ref(False if (a is False and b is False) else MAYBE, taints)


def _wrap(value: int, width: int) -> int:
    value &= (1 << width) - 1
    return value - (1 << width) if value >= 1 << (width - 1) else value


def int_rule(op: Op, args: Sequence[Ref], mode: Mode) -> Ref:
    taints = frozenset().union(*(a.taints for a in args))
    values = [a.value for a in args]
    if any(v is UNDEFINED for v in values):
        return Ref(UNDEFINED, taints)
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        raise _NotEstablished("integer operation on a non-integer value")
    a, b = values
    width = op.attrs.get("width", 32)
    if op.name == "addi":
        r = a + b
    elif op.name == "subi":
        r = a - b
    elif op.name == "muli":
        r = a * b
    elif op.name in ("divsi", "remsi"):
        if b == 0:
            raise _NotEstablished("integer division by zero")
        q = abs(a) // abs(b) * (1 if (a >= 0) == (b >= 0) else -1)
        r = q if op.name == "divsi" else a - q * b
    else:
        raise KeyError(op.name)
    return Ref(_wrap(r, width), taints)


_BITCAST_LAYOUT = {
    ("fp32", "i32"): ("<f", "<i"),
    ("i32", "fp32"): ("<i", "<f"),
    ("fp64", "i64"): ("<d", "<q"),
    ("i64", "fp64"): ("<q", "<d"),
}


def bitcast_rule(op: Op, args: Sequence[Ref], mode: Mode) -> Ref:
    """Bit reinterpretation is exact only on a value with known bits."""

    (arg,) = args
    key = (op.attrs["from_format"], op.attrs["to_format"])
    if key not in _BITCAST_LAYOUT:
        raise _NotEstablished(f"bitcast {key} not supported")
    value = arg.value
    if isinstance(value, Interval):
        if not value.is_point or not is_representable(value.lo, key[0]):
            raise _NotEstablished("bit-level reinterpretation of a value without definite bits")
        raw = float(value.lo)
    elif isinstance(value, Special):
        raw = _special_to_float(value)
    else:
        raw = value
    src, dst = _BITCAST_LAYOUT[key]
    out = struct.unpack(dst, struct.pack(src, raw))[0]
    return Ref(to_element(out), arg.taints)


ELEMENTWISE_RULES = {
    **{name: float_rule for name in FLOAT_BINARY | set(ROUNDING_SUFFIX) | {"neg", "abs", "fma"} | set(ELEMENTARY)},
    "cast": cast_rule,
    "sitofp": cast_rule,
    "fptosi": fptosi_rule,
    "maxnum": _minmax,
    "minnum": _minmax,
    "maximum": _minmax,
    "minimum": _minmax,
    "cmp": cmp_rule,
    "isnan": isnan_rule,
    "isinf": isnan_rule,
    "select": select_rule,
    "and": logic_rule,
    "or": logic_rule,
    "not": logic_rule,
    "addi": int_rule,
    "subi": int_rule,
    "muli": int_rule,
    "divsi": int_rule,
    "remsi": int_rule,
    "bitcast": bitcast_rule,
}

STRUCTURAL_OPS = {"const", "arange", "splat", "reduce_sum", "reduce_max", "reduce_min",
                  "load", "store", "atomic_add", "if", "for"}

ALLOWED_ATTRS = {
    **{name: {"format", "declared_quantization", "surrogate_gradient"} for name in ELEMENTWISE_RULES},
    "cast": {"to_format", "format", "declared_quantization", "surrogate_gradient"},
    "sitofp": {"to_format", "format", "declared_quantization", "surrogate_gradient"},
    "cmp": {"predicate"},
    "addi": {"width"}, "subi": {"width"}, "muli": {"width"}, "divsi": {"width"}, "remsi": {"width"},
    "bitcast": {"from_format", "to_format"},
    "const": {"value", "n"},
    "arange": {"start", "end"},
    "splat": {"n"},
    "reduce_sum": {"format", "declared_quantization", "surrogate_gradient", "order"},
    "reduce_max": set(), "reduce_min": set(),
    "load": {"buffer"},
    "store": {"buffer"},
    "atomic_add": {"buffer", "order"},
    "if": set(),
    "for": set(),
}


def coverage_report(program: Program) -> dict:
    """Instruction-processing coverage: every op is accepted or rejected."""

    accepted, rejected = {}, []
    for op in program.walk():
        if op.name not in ELEMENTWISE_RULES and op.name not in STRUCTURAL_OPS:
            rejected.append({"node": op.node_id, "op": op.name, "reason": "no reference rule"})
            continue
        unknown = set(op.attrs) - ALLOWED_ATTRS.get(op.name, set()) - COMMON_ATTRS
        if unknown:
            rejected.append({"node": op.node_id, "op": op.name, "reason": f"unknown attributes {sorted(unknown)}"})
            continue
        accepted[op.name] = accepted.get(op.name, 0) + 1
    return {"accepted": accepted, "rejected": rejected, "complete": not rejected}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ComposedResult:
    values: dict[str, list[Ref]]
    memory: dict[str, list[Ref]]
    written: dict[str, set[int]]
    path_events: list[dict] = field(default_factory=list)

    def element_class(self, ref: Ref):
        return classify(ref)

    def output_classes(self, outputs: Sequence[str] = ()) -> dict[str, dict]:
        report = {}
        for name in outputs:
            classes = [classify(r) for r in self.values[name]]
            report[name] = {
                "class": _worst([c for c, _ in classes]),
                "reasons": sorted({reason for _, rs in classes for reason in rs}),
            }
        for buffer, indices in self.written.items():
            classes = [classify(self.memory[buffer][i]) for i in sorted(indices)]
            report[f"buffer:{buffer}"] = {
                "class": _worst([c for c, _ in classes]),
                "reasons": sorted({reason for _, rs in classes for reason in rs}),
            }
        return report

    def residual(self, ref: Ref, actual) -> Interval:
        """``actual - reference`` as an interval (requires a finite reference)."""

        if not isinstance(ref.value, Interval):
            raise _NotEstablished("residual of a non-finite or undefined reference")
        return to_element(actual) - ref.value


@dataclass
class LocalNode:
    node: str
    output: str
    status: str  # "MEASURED" | "NOT_OBSERVABLE" | "NOT_ESTABLISHED"
    reference: Optional[Any] = None
    residual: Optional[Interval] = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ReferenceEvaluator:
    def __init__(self, program: Program, mode: Mode = Mode.NUMERICAL_DIFFERENCE):
        report = coverage_report(program)
        if not report["complete"]:
            raise UnsupportedOperation(report["rejected"])
        self.program = program
        self.mode = mode
        self._used = self._used_names(program)

    @staticmethod
    def _used_names(program: Program) -> set[str]:
        used = set(program.outputs)
        for op in program.walk():
            used.update(op.inputs)
            for region in op.regions:
                used.update(region.outputs)
        return used

    # -- public executions -------------------------------------------------

    def composed(
        self,
        inputs: Mapping[str, Any],
        memory: Optional[Mapping[str, Sequence[Any]]] = None,
        actual: Optional[Mapping[str, Any]] = None,
        pin_loads: Sequence[str] = (),
        follow_actual_path: bool = False,
    ) -> ComposedResult:
        """Composed reference from captured external inputs and memory.

        ``actual`` maps SSA names to captured device values; it is only read
        for branch conditions (to record path flips), for ``pin_loads`` (a
        load fixed to its captured value) and when ``follow_actual_path``.
        Either fallback downgrades the affected outputs to conditional local
        references.
        """

        state = _State(
            env={name: [Ref(to_element(v)) for v in _block(value)] for name, value in inputs.items()},
            memory={name: [Ref(to_element(v)) for v in _block(values)] for name, values in (memory or {}).items()},
            written={},
            actual={k: _block(v) for k, v in (actual or {}).items()},
            pin_loads=set(pin_loads),
            follow_actual_path=follow_actual_path,
            events=[],
        )
        self._run(self.program, state, frozenset())
        return ComposedResult(state.env, state.memory, state.written, state.events)

    def derivative(
        self,
        inputs: Mapping[str, Any],
        tangents: Mapping[str, Any],
        memory: Optional[Mapping[str, Sequence[Any]]] = None,
        memory_tangents: Optional[Mapping[str, Sequence[Any]]] = None,
    ) -> ComposedResult:
        """Forward-mode derivative: tangent of every value along ``tangents``."""

        env = {}
        for name, value in inputs.items():
            vals = _block(value)
            tans = _block(tangents.get(name, [0.0] * len(vals)))
            if len(tans) == 1 and len(vals) > 1:
                tans = tans * len(vals)
            env[name] = [Ref(to_element(v), frozenset(), to_element(t)) for v, t in zip(vals, tans)]
        mem = {}
        for name, values in (memory or {}).items():
            vals = _block(values)
            tans = _block((memory_tangents or {}).get(name, [0.0] * len(vals)))
            mem[name] = [Ref(to_element(v), frozenset(), to_element(t)) for v, t in zip(vals, tans)]
        state = _State(env=env, memory=mem, written={}, actual={}, pin_loads=set(),
                       follow_actual_path=False, events=[])
        self._run(self.program, state, frozenset())
        return ComposedResult(state.env, state.memory, state.written, state.events)

    def local(self, actual: Mapping[str, Any]) -> list[LocalNode]:
        """Local residual of every computational node on its actual inputs.

        ``actual`` must hold the captured value of every external input and of
        every observable node output.  A node marked ``observable=False`` has
        no device value; its reference is composed into each consumer from
        the nearest observable producers.
        """

        captured = {k: _block(v) for k, v in actual.items()}
        producers = {}
        for op in self.program.ops:
            if op.regions:
                raise UnsupportedOperation("local residuals inside regions need per-iteration capture")
            for out in op.outputs:
                producers[out] = op

        cache: dict[str, list[Ref]] = {}

        def reference_of(name: str) -> list[Ref]:
            # Value seen by a consumer: actual if observable, else recomposed.
            op = producers.get(name)
            if op is None or op.attrs.get("observable", True):
                if name not in captured:
                    raise KeyError(f"no captured value for observable {name}")
                return [Ref(to_element(v)) for v in captured[name]]
            if name not in cache:
                cache[name] = self._eval_local_op(op, [reference_of(i) for i in op.inputs])[0]
            return cache[name]

        nodes = []
        for op in self.program.ops:
            if op.name in ("load", "store", "atomic_add", "const", "arange", "splat"):
                continue
            for index, out in enumerate(op.outputs):
                if not op.attrs.get("observable", True):
                    nodes.append(LocalNode(op.node_id, out, "NOT_OBSERVABLE",
                                           reason="value does not exist on the device"))
                    continue
                try:
                    refs = self._eval_local_op(op, [reference_of(i) for i in op.inputs])[index]
                except _NotEstablished as exc:
                    nodes.append(LocalNode(op.node_id, out, "NOT_ESTABLISHED", reason=str(exc)))
                    continue
                residuals = []
                for ref, act in zip(refs, _broadcast_to(captured[out], len(refs))):
                    if isinstance(ref.value, Interval) and isinstance(to_element(act), Interval):
                        residuals.append(to_element(act) - ref.value)
                    else:
                        residuals.append(None)
                nodes.append(LocalNode(op.node_id, out, "MEASURED", reference=refs,
                                       residual=residuals[0] if len(residuals) == 1 else residuals))
        return nodes

    def local_residual(self, actual: Mapping[str, Any], output: str):
        for node in self.local(actual):
            if node.output == output:
                if node.status == "NOT_OBSERVABLE":
                    raise NotObservable(f"{output} ({node.node}) has no device value")
                return node.residual
        raise KeyError(output)

    # -- interpreter -------------------------------------------------------

    def _eval_local_op(self, op: Op, args: list[list[Ref]]) -> list[list[Ref]]:
        state = _State(env={}, memory={}, written={}, actual={}, pin_loads=set(),
                       follow_actual_path=False, events=[])
        for name, refs in zip(op.inputs, args):
            state.env[name] = refs
        self._run_op(op, state, frozenset())
        return [state.env[o] for o in op.outputs]

    def _run(self, program: Program, state: "_State", context: frozenset):
        for op in program.ops:
            self._run_op(op, state, context)

    def _run_op(self, op: Op, state: "_State", context: frozenset):
        args = [state.env[name] for name in op.inputs]
        try:
            handler = getattr(self, f"_op_{op.name}", None)
            if handler is not None:
                handler(op, args, state, context)
                return
            rule = ELEMENTWISE_RULES[op.name]
            n = max((len(a) for a in args), default=1)
            out = []
            for lanes in zip(*(_broadcast_to(a, n) for a in args)):
                try:
                    ref = rule(op, lanes, self.mode)
                except _NotEstablished as exc:
                    taints = frozenset().union(*(r.taints for r in lanes))
                    ref = Ref(UNDEFINED, taints | {f"not_established:{exc}@{op.node_id}"})
                out.append(_with_context(ref, context))
            state.env[op.outputs[0]] = out
        except _NotEstablished as exc:
            bad = Ref(UNDEFINED, context | {f"not_established:{exc}@{op.node_id}"})
            for name in op.outputs:
                state.env[name] = [bad]

    # Structural ops ---------------------------------------------------------

    def _op_const(self, op, args, state, context):
        n = op.attrs.get("n", 1)
        state.env[op.outputs[0]] = [Ref(to_element(op.attrs["value"]), context)] * n

    def _op_arange(self, op, args, state, context):
        state.env[op.outputs[0]] = [Ref(i, context) for i in range(op.attrs["start"], op.attrs["end"])]

    def _op_splat(self, op, args, state, context):
        (value,) = args
        if len(value) != 1:
            raise _NotEstablished("splat of a non-scalar")
        state.env[op.outputs[0]] = [_with_context(value[0], context)] * op.attrs["n"]

    def _op_reduce_sum(self, op, args, state, context):
        (values,) = args
        taints = frozenset().union(context, *(r.taints for r in values))
        if any(r.value is UNDEFINED for r in values):
            state.env[op.outputs[0]] = [Ref(UNDEFINED, taints)]
            return
        if any(isinstance(r.value, Special) for r in values):
            with _ieee_floats():
                total = math.fsum(_special_to_float(r.value) for r in values)
            state.env[op.outputs[0]] = [Ref(_float_to_element(total), taints)]
            return
        rounds = op.attrs.get("declared_quantization") or (self.mode is Mode.ROUNDING_CHECK and op.attrs.get("format"))
        if rounds and op.attrs.get("order") is None:
            # A rounded reduction needs its combination tree; reals do not.
            raise _NotEstablished("rounded reduction without a declared order")
        if rounds:
            raise _NotEstablished("ordered rounded reductions are not implemented")
        total = Interval(0)
        for r in values:
            total = total + r.value
        tangent = None
        if any(r.tangent is not None for r in values):
            tangent = Interval(0)
            for r in values:
                tangent = tangent + _tangent(r)
        state.env[op.outputs[0]] = [Ref(total, taints, tangent)]

    def _op_reduce_max(self, op, args, state, context):
        self._fold_minmax(op, args, state, context, "maxnum")

    def _op_reduce_min(self, op, args, state, context):
        self._fold_minmax(op, args, state, context, "minnum")

    def _fold_minmax(self, op, args, state, context, name):
        (values,) = args
        fold_op = Op(name, ("a", "b"), ("c",), {}, (), op.node_id)
        acc = values[0]
        for r in values[1:]:
            acc = _minmax(fold_op, [acc, r], self.mode)
        state.env[op.outputs[0]] = [_with_context(acc, context)]

    # Memory ---------------------------------------------------------------------

    def _addresses(self, op, offsets: list[Ref], buffer: list) -> list[int]:
        addresses = []
        for r in offsets:
            if not isinstance(r.value, int) or isinstance(r.value, bool):
                raise _NotEstablished("address is not a definite integer")
            if not 0 <= r.value < len(buffer):
                raise _NotEstablished(f"address {r.value} outside buffer")
            addresses.append(r.value)
        return addresses

    def _mask(self, args, index, n):
        if len(args) > index:
            return [r.value for r in _broadcast_to(args[index], n)]
        return [True] * n

    def _op_load(self, op, args, state, context):
        buffer = state.memory[op.attrs["buffer"]]
        offsets = args[0]
        n = len(offsets)
        masks = self._mask(args, 1, n)
        other = _broadcast_to(args[2], n) if len(args) > 2 else None
        pinned = op.node_id in state.pin_loads or op.outputs[0] in state.pin_loads
        actual = state.actual.get(op.outputs[0]) if pinned else None
        if pinned and actual is None:
            raise KeyError(f"pinned load {op.node_id} has no captured value")
        out = []
        offset_taints = frozenset().union(*(r.taints for r in offsets))
        for lane, (off, mask) in enumerate(zip(offsets, masks)):
            if mask is False:
                if other is None:
                    out.append(Ref(UNDEFINED, context | offset_taints | {f"undefined:masked load without other@{op.node_id}"}))
                else:
                    out.append(_with_context(other[lane], context))
                continue
            if not isinstance(off.value, int) or not 0 <= off.value < len(buffer):
                out.append(Ref(UNDEFINED, context | off.taints | {f"not_established:invalid address@{op.node_id}"}))
                continue
            if pinned:
                loaded = Ref(to_element(_broadcast_to(actual, n)[lane]), off.taints | {f"pinned_load:{op.node_id}"})
            else:
                loaded = Ref(buffer[off.value].value, buffer[off.value].taints | off.taints, buffer[off.value].tangent)
            if mask is MAYBE:
                fallback = other[lane] if other is not None else Ref(UNDEFINED, frozenset({f"undefined:masked load without other@{op.node_id}"}))
                loaded = _hull_refs(loaded, fallback, f"path_union:{op.node_id}")
            out.append(_with_context(loaded, context))
        state.env[op.outputs[0]] = out

    def _op_store(self, op, args, state, context):
        name = op.attrs["buffer"]
        buffer = state.memory[name]
        offsets, values = args[0], args[1]
        n = max(len(offsets), len(values))
        offsets, values = _broadcast_to(offsets, n), _broadcast_to(values, n)
        masks = self._mask(args, 2, n)
        for off, value, mask in zip(offsets, values, masks):
            if mask is False:
                continue
            address = self._addresses(op, [off], buffer)[0]
            new = _with_context(Ref(value.value, value.taints | off.taints, value.tangent), context)
            if mask is MAYBE:
                new = _hull_refs(buffer[address], new, f"path_union:{op.node_id}")
            buffer[address] = new
            state.written.setdefault(name, set()).add(address)

    def _op_atomic_add(self, op, args, state, context):
        """``(v_old, M') = AtomicAdd(M, p, a)`` with an explicit order.

        If the returned old values are unused, the contributions fold into one
        exact sum (order-free in real arithmetic).  If they are used, each old
        value depends on the order of the participating updates; without a
        declared order the returned values are not established.
        """

        name = op.attrs["buffer"]
        buffer = state.memory[name]
        offsets, values = args[0], args[1]
        n = max(len(offsets), len(values))
        offsets, values = _broadcast_to(offsets, n), _broadcast_to(values, n)
        masks = self._mask(args, 2, n)
        result_used = bool(op.outputs) and op.outputs[0] in self._used
        order = op.attrs.get("order")
        olds = []
        for off, value, mask in zip(offsets, values, masks):
            if mask is False:
                olds.append(Ref(UNDEFINED, frozenset({f"undefined:masked atomic@{op.node_id}"})))
                continue
            if mask is MAYBE:
                raise _NotEstablished("atomic with undecided mask")
            address = self._addresses(op, [off], buffer)[0]
            current = buffer[address]
            olds.append(current)
            add_op = Op("add", ("old", "a"), ("new",), {}, (), op.node_id)
            try:
                updated = float_rule(add_op, [current, value], self.mode)
            except _NotEstablished as exc:
                updated = Ref(UNDEFINED, current.taints | value.taints | {f"not_established:{exc}@{op.node_id}"})
            buffer[address] = _with_context(updated, context | off.taints)
            state.written.setdefault(name, set()).add(address)
        if op.outputs:
            if result_used and order is None:
                reason = f"not_established:atomic return value depends on an undeclared order@{op.node_id}"
                olds = [Ref(UNDEFINED, r.taints | {reason}) for r in olds]
            state.env[op.outputs[0]] = [_with_context(r, context) for r in olds]

    # Control flow -----------------------------------------------------------------

    def _op_if(self, op, args, state, context):
        (cond_block,) = args
        if len(cond_block) != 1:
            raise _NotEstablished("if condition is not a scalar")
        cond = cond_block[0]
        then_region, else_region = op.regions
        actual_cond = None
        if op.inputs[0] in state.actual:
            actual_cond = bool(state.actual[op.inputs[0]][0])
        ref_cond = cond.value
        if actual_cond is not None:
            state.events.append({"node": op.node_id, "reference": ref_cond, "actual": actual_cond,
                                 "flipped": ref_cond is MAYBE or ref_cond != actual_cond})
        branch_context = context | cond.taints
        if state.follow_actual_path:
            if actual_cond is None:
                raise KeyError(f"follow_actual_path needs the captured condition of {op.node_id}")
            if ref_cond is MAYBE or ref_cond != actual_cond:
                branch_context = branch_context | {f"actual_path:{op.node_id}"}
            self._run_region(op, then_region if actual_cond else else_region, state, branch_context)
            return
        if ref_cond is True or ref_cond is False:
            self._run_region(op, then_region if ref_cond else else_region, state, branch_context)
            return
        if ref_cond is not MAYBE:
            raise _NotEstablished("if condition is undefined")
        # Undecided: run both branches on copies and take the union.
        then_state = state.fork()
        else_state = state.fork()
        self._run_region(op, then_region, then_state, branch_context)
        self._run_region(op, else_region, else_state, branch_context)
        info = f"path_union:{op.node_id}"
        for name in op.outputs:
            state.env[name] = [_hull_refs(a, b, info) for a, b in zip(then_state.env[name], else_state.env[name])]
        for buffer in state.memory:
            for i, (a, b) in enumerate(zip(then_state.memory[buffer], else_state.memory[buffer])):
                if a is not b:
                    state.memory[buffer][i] = _hull_refs(a, b, info)
        for buffer, idx in list(then_state.written.items()) + list(else_state.written.items()):
            state.written.setdefault(buffer, set()).update(idx)
        state.events.extend(e for e in then_state.events + else_state.events if e not in state.events)

    def _run_region(self, op, region: Program, state, context):
        self._run(region, state, context)
        for target, source in zip(op.outputs, region.outputs):
            state.env[target] = [_with_context(r, context) for r in state.env[source]]

    def _op_for(self, op, args, state, context):
        lb, ub, step = (self._scalar_int(a) for a in args[:3])
        carried = args[3:]
        (body,) = op.regions
        iv_name, *carried_names = body.inputs
        i = lb
        while (step > 0 and i < ub) or (step < 0 and i > ub):
            state.env[iv_name] = [Ref(i, context)]
            for name, value in zip(carried_names, carried):
                state.env[name] = value
            self._run(body, state, context)
            carried = [state.env[name] for name in body.outputs]
            i += step
        for target, value in zip(op.outputs, carried):
            state.env[target] = value

    @staticmethod
    def _scalar_int(block) -> int:
        if len(block) != 1 or not isinstance(block[0].value, int) or isinstance(block[0].value, bool):
            raise _NotEstablished("loop bound is not a definite integer")
        return block[0].value


@dataclass
class _State:
    env: dict
    memory: dict
    written: dict
    actual: dict
    pin_loads: set
    follow_actual_path: bool
    events: list

    def fork(self) -> "_State":
        return _State(
            env=dict(self.env),
            memory={k: list(v) for k, v in self.memory.items()},
            written={k: set(v) for k, v in self.written.items()},
            actual=self.actual,
            pin_loads=self.pin_loads,
            follow_actual_path=self.follow_actual_path,
            events=list(self.events),
        )


def _broadcast_to(block: list, n: int) -> list:
    if len(block) == n:
        return block
    if len(block) == 1:
        return block * n
    raise _NotEstablished(f"cannot broadcast a block of {len(block)} to {n}")


def _with_context(ref: Ref, context: frozenset) -> Ref:
    if not context or context <= ref.taints:
        return ref
    return Ref(ref.value, ref.taints | context, ref.tangent)


def adjoint_residual(backward_dot_u: Sequence, v: Sequence, jvp: Sequence[Ref]) -> Interval:
    """``<B(x, v), u> - <v, J(x) u>`` with ``J(x) u`` from :meth:`derivative`.

    ``backward_dot_u`` is the list of products ``B(x, v)_i u_i`` (or the
    backward output and direction already multiplied); ``jvp`` holds the
    tangent references of the forward output.
    """

    left = Interval(0)
    for term in backward_dot_u:
        left = left + to_element(term)
    right = Interval(0)
    for vi, ref in zip(v, jvp):
        if ref.tangent is None:
            raise ValueError("jvp entry has no tangent")
        right = right + to_element(vi) * ref.tangent
    return left - right
