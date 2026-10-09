"""Program-level certificates (external audit F03 follow-up): proofs that replace a premise the reference otherwise checks
only on evidence.

* ``scan_associativity``: the combine region of a ``tt.scan`` is associative for every argument, f(f(a, b), c) =
  f(a, f(b, c)), proved by z3 over the exact semantics the reference uses -- real numbers for float values (the
  reference evaluates float operations as real operations; rounding is part of K, not of K_R), bit-vectors of the
  declared width for integers (modular arithmetic), Booleans for i1.  With associativity proved, the sequential fold
  the reference computes equals every bracketing the hardware may use, so the scan result needs no premise.  The
  proof covers finite real arguments; callers apply it only when every scanned element is a finite established value.

Only a closed set of operations is translated (arithmetic without division, comparisons, select, min / max, integer
width changes, int-to-float conversion).  Anything else -- division (partial), control flow, calls, memory, special
float constants -- gives no certificate, and the premise stays.  z3 is optional: without it no certificate is issued.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

try:  # optional dependency (z3-solver)
    import z3
except ImportError:  # pragma: no cover - environment without z3
    z3 = None

FLOAT_ELEMS = {"f16", "bf16", "f32", "f64", "f8E5M2", "f8E4M3FN", "f8E4M3FNUZ", "f8E5M2FNUZ", "f8E4M3B11FNUZ"}
INT_WIDTH = {"i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64}
TIMEOUT_MS = 20000


class Unsupported(Exception):
    """An operation or value outside the translated set: no certificate."""


@dataclass(frozen=True)
class Certificate:
    proved: bool
    detail: str


_CACHE: dict = {}


def available() -> bool:
    return z3 is not None


def _sort_of(elem):
    if elem in FLOAT_ELEMS:
        return "real"
    if elem == "i1":
        return "bool"
    if elem in INT_WIDTH:
        return ("bv", INT_WIDTH[elem])
    raise Unsupported(f"element type {elem}")


def _var(name, elem):
    s = _sort_of(elem)
    if s == "real":
        return z3.Real(name)
    if s == "bool":
        return z3.Bool(name)
    return z3.BitVec(name, s[1])


def _const(elem, value):
    s = _sort_of(elem)
    if s == "real":
        if not math.isfinite(float(value)):
            raise Unsupported("special float constant")
        f = Fraction(float(value))
        return z3.RealVal(f"{f.numerator}/{f.denominator}")
    if s == "bool":
        return z3.BoolVal(bool(int(value) & 1))
    return z3.BitVecVal(int(value) % (1 << s[1]), s[1])


_CMPF = {"oeq": "eq", "ueq": "eq", "one": "ne", "une": "ne", "olt": "lt", "ult": "lt", "ole": "le", "ule": "le",
         "ogt": "gt", "ugt": "gt", "oge": "ge", "uge": "ge"}


def _cmp(kind, a, b):
    return {"eq": lambda: a == b, "ne": lambda: a != b, "lt": lambda: a < b, "le": lambda: a <= b,
            "gt": lambda: a > b, "ge": lambda: a >= b}[kind]()


def _translate(op, a, constant):
    """One operation on z3 terms (``a``: operand terms); ``constant(op)`` gives a literal's scalar value."""
    n = op.name
    rt = op.result_types[0].elem if op.result_types else None
    if n == "arith.constant":
        if op.result_types[0].shape:
            raise Unsupported("tensor constant in a combine region")
        return _const(rt, constant(op))
    if n in ("arith.addf", "arith.addi"):
        return a[0] + a[1]
    if n in ("arith.subf", "arith.subi"):
        return a[0] - a[1]
    if n in ("arith.mulf", "arith.muli"):
        return a[0] * a[1]
    if n == "arith.negf":
        return -a[0]
    if n in ("arith.maximumf", "arith.maxnumf", "arith.maxsi"):
        return z3.If(a[0] >= a[1], a[0], a[1])
    if n in ("arith.minimumf", "arith.minnumf", "arith.minsi"):
        return z3.If(a[0] <= a[1], a[0], a[1])
    if n == "arith.maxui":
        return z3.If(z3.UGE(a[0], a[1]), a[0], a[1])
    if n == "arith.minui":
        return z3.If(z3.ULE(a[0], a[1]), a[0], a[1])
    if n == "arith.cmpf":
        pred = op.attrs.get("predicate")
        if pred not in _CMPF:
            raise Unsupported(f"cmpf {pred}")
        return _cmp(_CMPF[pred], a[0], a[1])
    if n == "arith.cmpi":
        pred = op.attrs.get("predicate")
        if z3.is_bool(a[0]):
            if pred == "eq":
                return a[0] == a[1]
            if pred == "ne":
                return a[0] != a[1]
            raise Unsupported(f"cmpi {pred} on i1")
        signed = {"slt": "lt", "sle": "le", "sgt": "gt", "sge": "ge", "eq": "eq", "ne": "ne"}
        if pred in signed:
            return _cmp(signed[pred], a[0], a[1])
        unsigned = {"ult": z3.ULT, "ule": z3.ULE, "ugt": z3.UGT, "uge": z3.UGE}
        if pred in unsigned:
            return unsigned[pred](a[0], a[1])
        raise Unsupported(f"cmpi {pred}")
    if n == "arith.select":
        return z3.If(a[0], a[1], a[2])
    if n in ("arith.andi", "arith.ori", "arith.xori"):
        if z3.is_bool(a[0]):
            return {"arith.andi": z3.And, "arith.ori": z3.Or, "arith.xori": z3.Xor}[n](a[0], a[1])
        return {"arith.andi": lambda: a[0] & a[1], "arith.ori": lambda: a[0] | a[1],
                "arith.xori": lambda: a[0] ^ a[1]}[n]()
    if n in ("arith.extsi", "arith.extui", "arith.trunci"):
        w = INT_WIDTH[rt]
        x = a[0]
        if z3.is_bool(x):
            if n == "arith.trunci":
                raise Unsupported("trunci of i1")
            one = z3.BitVecVal(1, w)
            return z3.If(x, -one if n == "arith.extsi" else one, z3.BitVecVal(0, w))
        if rt == "i1":   # trunci to i1: the low bit
            return z3.Extract(0, 0, x) == z3.BitVecVal(1, 1)
        cur = x.size()
        if n == "arith.trunci":
            return z3.Extract(w - 1, 0, x)
        return (z3.SignExt if n == "arith.extsi" else z3.ZeroExt)(w - cur, x)
    if n in ("arith.sitofp", "arith.uitofp"):
        x = a[0]
        if z3.is_bool(x):
            raise Unsupported("int-to-float of i1")
        return z3.ToReal(z3.BV2Int(x, is_signed=(n == "arith.sitofp")))
    raise Unsupported(n)


def _apply(region, args, outer, constant):
    """The region's yielded terms for the argument terms ``args``; ``outer``: terms of values defined outside it."""
    block = region.entry
    if len(region.blocks) != 1 or len(block.args) != len(args):
        raise Unsupported("region shape")
    env = dict(outer)
    for (name, _), t in zip(block.args, args):
        env[name] = t
    for op in block.ops:
        if op.name == "tt.scan.return":
            return [env[v] for v in op.operands]
        if op.regions or len(op.results) != 1:
            raise Unsupported(f"{op.name} with regions or several results")
        try:
            operands = [env[v] for v in op.operands]
        except KeyError as exc:
            raise Unsupported(f"value {exc} not available") from exc
        env[op.results[0]] = _translate(op, operands, constant)
    raise Unsupported("no tt.scan.return")


def outer_names(region) -> list:
    """Values a region uses that it does not define (constants or values of the enclosing scope)."""
    block = region.entry
    defined = {name for name, _ in block.args}
    used = []
    for op in block.ops:
        for v in op.operands:
            if v not in defined and v not in used:
                used.append(v)
        defined.update(op.results)
    return used


def scan_associativity(region, outer: dict, constant, key=None) -> Certificate:
    """Certificate that the scan combine ``region`` is associative.  ``outer``: name -> ("const", elem, value) for a
    scalar value fixed in this execution, or ("var", elem) for a value the proof must cover for every choice.
    ``constant(op)`` returns the scalar of an ``arith.constant`` inside the region."""
    if z3 is None:
        return Certificate(False, "z3 not available")
    ck = (key, tuple(sorted((k, v) for k, v in outer.items()))) if key is not None else None
    if ck is not None and ck in _CACHE:
        return _CACHE[ck]
    try:
        block = region.entry
        k = len(block.args) // 2
        if k == 0 or len(block.args) != 2 * k:
            raise Unsupported("combine region arity")
        elems = [t.elem for _, t in block.args[:k]]
        if [t.elem for _, t in block.args[k:]] != elems:
            raise Unsupported("operand types differ between the two sides")
        terms = {}
        for name, spec in outer.items():
            terms[name] = _const(spec[1], spec[2]) if spec[0] == "const" else _var(f"outer_{name}", spec[1])
        A = [_var(f"a{j}", e) for j, e in enumerate(elems)]
        B = [_var(f"b{j}", e) for j, e in enumerate(elems)]
        C = [_var(f"c{j}", e) for j, e in enumerate(elems)]
        ab = _apply(region, A + B, terms, constant)
        bc = _apply(region, B + C, terms, constant)
        left = _apply(region, ab + C, terms, constant)
        right = _apply(region, A + bc, terms, constant)
        if len(left) != k or len(right) != k:
            raise Unsupported("yield arity")
        s = z3.Solver()
        s.set("timeout", TIMEOUT_MS)
        s.add(z3.Not(z3.And(*[lt == rt for lt, rt in zip(left, right)])))
        r = s.check()
        if r == z3.unsat:
            cert = Certificate(True, f"z3 {z3.get_version_string()}: f(f(a,b),c) = f(a,f(b,c)) for all arguments "
                                     f"({', '.join(elems)}; reals for floats, bit-vectors for integers)")
        elif r == z3.sat:
            cert = Certificate(False, "not associative for some arguments (z3 counterexample)")
        else:
            cert = Certificate(False, f"z3 gave {r} (no certificate)")
    except Unsupported as exc:
        cert = Certificate(False, f"not translated: {exc}")
    if ck is not None:
        _CACHE[ck] = cert
    return cert
