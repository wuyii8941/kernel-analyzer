"""Comparison of a reference result with an expected result (PLAN_v3 section 3).

Per scalar, with E the parsed expected value and R the reference value:
- exact R and exact E: equal is consistent, different is a contradiction;
- enclosure R and exact E: E inside R is consistent, outside is a contradiction;
- enclosure R and enclosure E (an irrational formula): disjoint is a contradiction, E inside R is consistent,
  anything else is not established;
- exact R and enclosure E: R outside E is a contradiction; R equal to a point E is consistent; otherwise not
  established (a rational cannot be certified equal to a formula that is only enclosed);
- allowed sets compare as exact sets.
There is no tolerance anywhere. Shape, key or type disagreements are structure errors.
"""
from __future__ import annotations

from fractions import Fraction

from . import expr
from .num import IV, ends

CONSISTENT, CONTRADICTION, NOT_ESTABLISHED, STRUCTURE = "consistent", "contradiction", "not_established", "structure_error"


def check_enclosed_output(reference, observed):
    """Membership of a complete point candidate in a single-valued extended-real task.

    This is deliberately stronger than development-example compatibility below:
    a nonpoint enclosure containing a candidate does not establish equality.
    Not applicable to allowed sets or correlated nondeterministic outputs.
    """
    import math
    from .num import Invalid, NotEstablished, rational
    from .real import R

    def point(value):
        value = value.v if isinstance(value, R) else value
        if isinstance(value, str) and value.strip() in ("inf", "+inf", "-inf"):
            return "-inf" if value.strip() == "-inf" else "+inf"
        if isinstance(value, float) and not math.isfinite(value):
            return "nan" if math.isnan(value) else "-inf" if value < 0 else "+inf"
        try:
            return rational(value, "observed output")
        except Invalid:
            return None  # An unsupported coordinate cannot hide other contradictions.

    def visit(ref, candidate):
        if ref is None:
            return candidate is None  # An explicitly absent optional output.
        if isinstance(ref, dict):
            if not isinstance(candidate, dict) or set(ref) != set(candidate):
                return False
            states = [visit(ref[k], candidate[k]) for k in ref]
        elif isinstance(ref, (list, tuple)):
            if not isinstance(candidate, (list, tuple)) or len(ref) != len(candidate):
                return False
            states = [visit(r, c) for r, c in zip(ref, candidate)]
        else:
            value = ref.v if isinstance(ref, R) else ref
            if isinstance(value, bool):
                return type(candidate) is bool and candidate == value
            candidate_point = point(candidate)
            if isinstance(value, str) and value in ("+inf", "-inf"):
                return None if candidate_point is None else candidate_point == value
            if not isinstance(value, (int, Fraction, IV)) or isinstance(value, bool):
                return None
            lo, hi = ends(value)
            if lo > hi:
                raise ValueError("inverted reference enclosure")
            if candidate_point is None:
                return None
            if isinstance(candidate_point, str):
                return False  # NaN and infinities cannot equal a finite real value.
            if candidate_point < lo or candidate_point > hi:
                return False
            return True if lo == hi == candidate_point else None
        if False in states:
            return False
        return None if None in states else True

    state = visit(reference, observed)
    if state is None:
        raise NotEstablished("nonpoint enclosure or unsupported coordinate; complete output equality is not established")
    return state


def validate_reference_values(fields, outputs):
    """Check result representations against guarantees, without claiming formula or shape correctness.

    Strict adapters return exact values or outward enclosures, never a floating-point approximation labelled strict.
    This check also applies on queries with no expected example to compare against.
    """
    from .real import R

    def visit(value, guarantee, kind, path):
        if isinstance(value, R):
            value = value.v
        if value is None:
            # Optional positions in structured outputs are represented explicitly.
            # Contract-specific complete-output checkers decide where null is legal.
            pass
        elif isinstance(value, dict):
            for k, item in value.items():
                visit(item, guarantee, kind, f"{path}.{k}")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(item, guarantee, kind, f"{path}[{i}]")
        elif isinstance(value, (set, frozenset)):
            # Some outputs are themselves exact sets (e.g. dropped expert IDs),
            # and an empty set can be a declared sentinel. Members must be exact.
            for item in value:
                visit(item, "exact", kind, f"{path}.member")
        elif isinstance(value, bool):
            if kind not in ("boolean", "discrete"):
                raise ValueError(f"{path}: boolean returned for a numeric output")
        elif kind == "boolean":
            raise ValueError(f"{path}: non-boolean returned for a boolean output")
        elif isinstance(value, IV):
            if guarantee == "exact":
                raise ValueError(f"{path}: {guarantee} output returned an enclosure")
            lo, hi = ends(value)
            if lo > hi:
                raise ValueError(f"{path}: inverted reference enclosure")
        elif isinstance(value, (int, Fraction)):
            pass
        elif isinstance(value, str) and value in ("+inf", "-inf"):
            pass
        elif isinstance(value, float) and guarantee == "declared_approximation":
            import math
            if not math.isfinite(value):
                raise ValueError(f"{path}: non-finite approximation")
        else:
            raise ValueError(f"{path}: {type(value).__name__} is not a {guarantee} reference value")

    for name, value in fields.items():
        kind = outputs[name].get("kind")
        if kind not in (None, "boolean", "discrete"):
            raise ValueError(f"{name}: unknown output kind {kind!r}")
        visit(value, outputs[name]["guarantee"], kind, name)


def _canon(x):
    """Canonical exact form of an allowed-set element."""
    if isinstance(x, dict) and "set" in x:
        return frozenset(_canon(e) for e in x["set"])
    if isinstance(x, (set, frozenset)):
        return frozenset(_canon(e) for e in x)
    if isinstance(x, (list, tuple)):
        return tuple(_canon(e) for e in x)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, Fraction)):
        return Fraction(x)
    if type(x).__name__ == "R":
        x = x.v
        if isinstance(x, str):
            return x
    if isinstance(x, str):
        v = expr.parse(x)
        if isinstance(v, IV):
            raise expr.ExpressionError(f"allowed-set element {x!r} is not exact")
        return v
    if isinstance(x, IV):
        raise expr.ExpressionError("allowed-set element is an enclosure, not an exact value")
    raise expr.ExpressionError(f"allowed-set element of type {type(x).__name__}")


def _scalar(ref, exp, guarantee, path):
    try:
        E = expr.parse(exp)
    except expr.ExpressionError as err:
        return [(path, STRUCTURE, f"expected value: {err}")]
    if hasattr(ref, "v") and type(ref).__name__ == "R":
        ref = ref.v
    if isinstance(E, str) or isinstance(ref, str):
        if not (isinstance(E, str) and isinstance(ref, str)):
            return [(path, CONTRADICTION if not isinstance(E, bool) and not isinstance(ref, bool) else STRUCTURE,
                     f"infinity against a finite value ({ref!r} vs {exp!r})")]
        return [(path, CONSISTENT if ref == E else CONTRADICTION, f"{ref} vs {E}")]
    if isinstance(E, bool) or isinstance(ref, bool):
        if type(ref) is not type(E):
            return [(path, STRUCTURE, f"boolean against non-boolean ({ref!r} vs {exp!r})")]
        return [(path, CONSISTENT if ref == E else CONTRADICTION, f"{ref} vs {E}")]
    if isinstance(ref, int):
        ref = Fraction(ref)
    if not isinstance(ref, (Fraction, IV)):
        return [(path, STRUCTURE, f"reference value of type {type(ref).__name__}")]
    if guarantee == "exact" and isinstance(ref, IV):
        return [(path, STRUCTURE, "output declared exact but the reference returned an enclosure")]
    try:
        r0, r1 = ends(ref)
        e0, e1 = ends(E)
    except ValueError as err:
        return [(path, STRUCTURE, str(err))]
    if r0 > r1:
        return [(path, STRUCTURE, "inverted reference enclosure")]
    if e1 < r0 or e0 > r1:
        return [(path, CONTRADICTION, f"expected [{float(e0):.17g}, {float(e1):.17g}] outside reference "
                                      f"[{float(r0):.17g}, {float(r1):.17g}]")]
    if r0 <= e0 and e1 <= r1:
        if isinstance(ref, Fraction) and e0 != e1:
            return [(path, NOT_ESTABLISHED, "exact reference inside an enclosed irrational expected value")]
        return [(path, CONSISTENT, "exact" if r0 == r1 == e0 == e1 else f"enclosed, width {float(r1 - r0):.3g}")]
    return [(path, NOT_ESTABLISHED, "enclosures overlap without containment")]


def compare(ref, exp, guarantee, path=""):
    """List of (path, verdict, detail) for one output field."""
    if exp is None or ref is None:
        return [(path, CONSISTENT if exp is None and ref is None else CONTRADICTION,
                 "both outputs absent" if exp is None and ref is None else "absent output against a present value")]
    if isinstance(exp, dict) and "set" in exp:
        if not isinstance(ref, (set, frozenset)):
            return [(path, STRUCTURE, "expected an allowed set")]
        try:
            ok = _canon(ref) == _canon(exp)
        except expr.ExpressionError as err:
            return [(path, STRUCTURE, str(err))]
        return [(path, CONSISTENT if ok else CONTRADICTION, "allowed set")]
    if isinstance(exp, dict):
        if not isinstance(ref, dict) or set(ref) != set(exp):
            return [(path, STRUCTURE, f"keys {sorted(ref) if isinstance(ref, dict) else type(ref).__name__} "
                                      f"vs {sorted(exp)}")]
        return [c for k in exp for c in compare(ref[k], exp[k], guarantee, f"{path}.{k}")]
    if isinstance(exp, list):
        if not isinstance(ref, (list, tuple)) or len(ref) != len(exp):
            return [(path, STRUCTURE, f"length {len(ref) if isinstance(ref, (list, tuple)) else type(ref).__name__} "
                                      f"vs {len(exp)}")]
        return [c for i, (a, b) in enumerate(zip(ref, exp)) for c in compare(a, b, guarantee, f"{path}[{i}]")]
    if isinstance(ref, (list, tuple, dict, set, frozenset)):
        return [(path, STRUCTURE, f"reference is a {type(ref).__name__}, expected a scalar")]
    return _scalar(ref, exp, guarantee, path)


def compare_result(res, expected: dict, outputs: dict):
    """Compare a Result with an expected {'outcome', 'fields'} block. Returns (verdict, checks)."""
    want = expected["outcome"]
    if res.outcome == "implementation_error":
        return STRUCTURE, [("", STRUCTURE, f"implementation error: {res.note}")]
    if res.outcome != want:
        return CONTRADICTION, [("", CONTRADICTION, f"outcome {res.outcome} ({res.note}), expected {want}")]
    if want != "value":
        return CONSISTENT, [("", CONSISTENT, want)]
    fields = res.fields or {}
    try:
        validate_reference_values(fields, outputs)
    except (ValueError, TypeError, KeyError) as error:
        return STRUCTURE, [("", STRUCTURE, f"reference guarantee: {error}")]
    wanted = expected.get("fields", {})
    if set(fields) != set(wanted):
        return STRUCTURE, [("", STRUCTURE, f"output fields {sorted(fields)} vs {sorted(wanted)}")]
    checks = [c for k in wanted for c in compare(fields[k], wanted[k], outputs[k]["guarantee"], k)]
    return aggregate([v for _, v, _ in checks]), checks


def aggregate(verdicts):
    for v in (STRUCTURE, CONTRADICTION, NOT_ESTABLISHED):
        if v in verdicts:
            return v
    return CONSISTENT
