"""Reference interpreter for TTIR kernels (composed reference of one launch).

The interpreter executes the parsed TTIR of a captured launch on interval
tensors (float64 endpoints with rigorous directed rounding, see
``intervals.py``).  Kernel inputs are the captured operands; values inside the
kernel are not observable on the device, so the whole kernel is one region
whose reference is composed from its actual inputs.  This is the kernel-level
local reference; chaining several launches through the reference memory
(``evaluate_sequence``) gives the composed reference across kernels.

Rules that keep the reference honest (stage summary section 4):

* the evaluator keeps its own reference memory; loads read reference values;
* branches are decided by reference values; an undecided ``scf.if`` runs both
  branches and takes the union, an undecided loop condition aborts the
  program instance (its outputs are not established);
* program instances are evaluated one at a time; a load of an address written
  by another instance, or two instances writing different values to the same
  address, is a race whose result is not established;
* an atomic's returned old value depends on the undeclared order of the
  participating updates and is not established when it is used;
* every element carries a status (finite / NaN / +-inf / undefined / not
  established) and a "conditional" flag (a captured value or an actual path
  was used), so each output element falls into one of the three classes.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import intervals as iv
from .ttir_mapping import (LIBDEVICE, LIBDEVICE_ROUNDING, inline_asm_internal, inline_asm_rounding, match_welford,
                           recognize_combiner, rule_for)
from .ttir_parser import PtrType, TFunc, TModule, TOp, TRegion, TType, parse_ttir

ST_OK, ST_NAN, ST_PINF, ST_NINF, ST_UNDEF, ST_NE = 0, 1, 2, 3, 4, 5
ATOMIC_EVENT_BUDGET = 1_000_000  # atomic updates per launch for the two-pass returned-value evaluation
ATOMIC_ADDRESS_BUDGET = 4096  # updates of one address for the set bound of a returned value

# Trigger evidence (DSL v2 rc3 04 W1): with KA_TRIGGER_TRACE=<path>, every executed (operation, internal rule) and every
# reduction route is recorded with the running pytest test id and written to <path>.<pid> at exit.  Off by default.
_TRIGGER_PATH = __import__("os").environ.get("KA_TRIGGER_TRACE")
_TRIGGER_SEEN: set = set()


def _trace(signature: str) -> None:
    if _TRIGGER_PATH:
        import os
        test = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" ", 1)[0]
        _TRIGGER_SEEN.add((test, signature))


def _flush_trace() -> None:
    if _TRIGGER_PATH and _TRIGGER_SEEN:
        import os
        with open(f"{_TRIGGER_PATH}.{os.getpid()}", "a") as handle:
            for test, sig in sorted(_TRIGGER_SEEN):
                handle.write(f"{test}\t{sig}\n")


if _TRIGGER_PATH:
    __import__("atexit").register(_flush_trace)
_ROUNDING_MODES = {"rtne": "rtne", "rn": "rtne", "rtz": "rtz", "rz": "rtz", "rd": "rd", "rm": "rd",
                   "ru": "ru", "rp": "ru"}
MAYBE = 2

INT_WIDTH = {"i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64}
FLOAT_ELEMS = {"f16", "bf16", "f32", "f64", "f8E5M2", "f8E4M3FN", "f8E4M3FNUZ", "f8E5M2FNUZ",
               "f8E4M3B11FNUZ", "tf32"}
ELEM_SIZE = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "f8E5M2": 1, "f8E4M3FN": 1, "f8E4M3FNUZ": 1,
             "f8E5M2FNUZ": 1, "f8E4M3B11FNUZ": 1, "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8}
TORCH_TO_ELEM = {"float32": "f32", "float16": "f16", "bfloat16": "bf16", "float64": "f64",
                 "float8_e5m2": "f8E5M2", "float8_e4m3fn": "f8E4M3FN", "int8": "i8", "uint8": "i8",
                 "int16": "i16", "int32": "i32", "int64": "i64", "bool": "i1",
                 "uint16": "i16", "uint32": "i32", "uint64": "i64"}


class NumericMode:
    NUMERICAL_DIFFERENCE = "numerical_difference"
    ROUNDING_CHECK = "rounding_check"


class ProgramAbort(Exception):
    """The program instance cannot be evaluated further (reason recorded)."""


def kind_of(elem) -> str:
    if isinstance(elem, PtrType):
        return "p"
    if elem == "i1":
        return "b"
    if elem in INT_WIDTH:
        return "i"
    if elem in FLOAT_ELEMS:
        return "f"
    raise ProgramAbort(f"unsupported element type {elem}")


# ---------------------------------------------------------------------------
# Tensor values
# ---------------------------------------------------------------------------


@dataclass
class TV:
    kind: str  # "f" float interval, "i" integer, "b" tri-state boolean, "p" pointer
    elem: Any
    lo: np.ndarray  # f: lower bound; i: value; b: 0/1/2; p: byte offset
    hi: Optional[np.ndarray] = None  # f: upper bound
    base: Optional[np.ndarray] = None  # p: buffer id
    st: Optional[np.ndarray] = None
    cond: Optional[np.ndarray] = None
    reasons: frozenset = frozenset()
    d: Optional[tuple] = None  # forward-mode tangent interval (dlo, dhi); None means zero

    def __post_init__(self):
        self.lo = np.asarray(self.lo)
        if self.st is None:
            self.st = np.zeros(self.lo.shape, dtype=np.int8)
        if self.cond is None:
            self.cond = np.zeros(self.lo.shape, dtype=bool)

    @property
    def shape(self):
        return self.lo.shape

    def map(self, fn) -> "TV":
        """Apply a layout function to every per-element array."""

        return TV(self.kind, self.elem, fn(self.lo), None if self.hi is None else fn(self.hi),
                  None if self.base is None else fn(self.base), fn(self.st), fn(self.cond), self.reasons,
                  None if self.d is None else (fn(self.d[0]), fn(self.d[1])))

    def with_reason(self, reason: str) -> "TV":
        return TV(self.kind, self.elem, self.lo, self.hi, self.base, self.st, self.cond,
                  self.reasons | {reason}, self.d)

    def tangent(self):
        if self.d is None:
            z = np.zeros(self.shape)
            return z, z
        return self.d


def _lane_hull(t: TV) -> TV:
    """The results the lanes of one warp hold after a butterfly (each lane may be the consumer): the hull of their
    enclosures; not established where their status or integer value differ."""
    st0 = t.st[..., 0]
    same_st = np.all(t.st == st0[..., None], axis=-1)
    cond = np.any(t.cond, axis=-1)
    why = frozenset({"not_established:the lanes of a warp hold different results after the butterfly (which lane the "
                     "consumer reads is not modelled)"})
    if t.kind == "f":
        d = None if t.d is None else (np.min(t.d[0], axis=-1), np.max(t.d[1], axis=-1))
        return TV("f", t.elem, np.min(t.lo, axis=-1), np.max(t.hi, axis=-1), None,
                  np.where(same_st, st0, ST_NE).astype(np.int8), cond,
                  t.reasons | (why if not same_st.all() else frozenset()), d)
    v0 = t.lo[..., 0]
    same = same_st & np.all(t.lo == v0[..., None], axis=-1)
    return TV(t.kind, t.elem, v0, None, None if t.base is None else t.base[..., 0],
              np.where(same, st0, ST_NE).astype(np.int8), cond,
              t.reasons | (why if not same.all() else frozenset()))


def _merge_status(*vals: TV) -> np.ndarray:
    st = np.zeros(vals[0].shape, dtype=np.int8)
    for v in vals:
        bad = np.broadcast_to(v.st, st.shape)
        st = np.where(bad >= ST_UNDEF, np.maximum(st, bad), st)
    return st


def _merge_cond(*vals: TV) -> np.ndarray:
    out = np.zeros(vals[0].shape, dtype=bool)
    for v in vals:
        out = out | np.broadcast_to(v.cond, out.shape)
    return out


def _merge_reasons(*vals: TV) -> frozenset:
    out = frozenset()
    for v in vals:
        out = out | v.reasons
    return out


def _float_reps(v: TV, which: str) -> np.ndarray:
    """IEEE representatives: endpoint for finite elements, nan/inf for specials."""

    base = (v.lo if which == "lo" else v.hi).astype(np.float64)
    out = np.where(v.st == ST_NAN, np.nan, base)
    out = np.where(v.st == ST_PINF, np.inf, out)
    out = np.where(v.st == ST_NINF, -np.inf, out)
    return out


def _classify_floats(a: np.ndarray, b: np.ndarray):
    """Status + value from two IEEE evaluations (lo reps, hi reps)."""

    st = np.full(a.shape, ST_NE, dtype=np.int8)
    both_nan = np.isnan(a) & np.isnan(b)
    st = np.where(both_nan, ST_NAN, st)
    st = np.where((a == np.inf) & (b == np.inf), ST_PINF, st)
    st = np.where((a == -np.inf) & (b == -np.inf), ST_NINF, st)
    finite_equal = np.isfinite(a) & np.isfinite(b) & (a == b)
    st = np.where(finite_equal, ST_OK, st)
    val = np.where(finite_equal, a, 0.0)
    return st, val


def _ftv(elem, lo, hi, st, cond, reasons) -> TV:
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    st = np.asarray(st, dtype=np.int8)
    # A finite real value whose enclosure leaves the float64 range (e.g. exp of a huge argument) is not enclosed by
    # [inf, inf]: such a lane is not established, never complete (found by the per-signature boundary tests, 4.0).
    overflow = (st == ST_OK) & ~(np.isfinite(lo) & np.isfinite(hi))
    if overflow.any():
        st = np.where(overflow, ST_NE, st).astype(np.int8)
        reasons = reasons | {"not_established:enclosure leaves the float64 range"}
    return TV("f", elem, lo, hi, None, st, np.asarray(cond, dtype=bool), reasons)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _parse_scalar_literal(text: str, elem):
    text = text.strip()
    if text in ("true", "false"):
        return 1 if text == "true" else 0
    if text.startswith("0x") or text.startswith("-0x"):
        raw = int(text, 16)
        if elem in INT_WIDTH:
            return raw
        if elem == "f32":
            return struct.unpack("<f", struct.pack("<I", raw & 0xFFFFFFFF))[0]
        if elem == "f64":
            return struct.unpack("<d", struct.pack("<Q", raw & 0xFFFFFFFFFFFFFFFF))[0]
        if elem == "f16":
            return float(np.frombuffer(struct.pack("<H", raw & 0xFFFF), dtype=np.float16)[0])
        if elem == "bf16":
            return struct.unpack("<f", struct.pack("<I", (raw & 0xFFFF) << 16))[0]
        raise ProgramAbort(f"hex literal for {elem}")
    if elem in INT_WIDTH:
        return int(text)
    value = float(text)
    if elem in iv.FLOAT_FORMATS and elem != "f64" and math.isfinite(value):
        # MLIR prints round-trippable literals; rounding to the type recovers the value.
        value = float(iv.round_nearest_even(np.array([value]), elem)[0][0])
    return value


def _constant(op: TOp) -> TV:
    ttype: TType = op.result_types[0]
    text = op.attrs["value"].strip()
    if text.startswith("dense<"):
        inner = text[len("dense<"):-1]
        if inner.startswith("["):
            raise ProgramAbort("non-splat dense constants are not supported")
        text = inner
    elem = ttype.elem
    value = _parse_scalar_literal(text, elem)
    kind = kind_of(elem)
    shape = ttype.shape
    if kind == "f":
        st = ST_OK
        if isinstance(value, float) and math.isnan(value):
            st = ST_NAN
        elif value == math.inf:
            st = ST_PINF
        elif value == -math.inf:
            st = ST_NINF
        v = 0.0 if st != ST_OK else value
        return _ftv(elem, np.full(shape, v), np.full(shape, v), np.full(shape, st, dtype=np.int8),
                    np.zeros(shape, dtype=bool), frozenset())
    return TV(kind, elem, np.full(shape, value, dtype=np.int64 if kind == "i" else np.int8))


# ---------------------------------------------------------------------------
# Integer helpers
# ---------------------------------------------------------------------------


def _wrap(x: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    if width >= 64:
        return x
    mask = (1 << width) - 1
    x = x & mask
    return np.where(x >= (1 << (width - 1)), x - (1 << width), x).astype(np.int64)


def _unsigned(x: np.ndarray, width: int) -> np.ndarray:
    if width >= 64:
        return np.asarray(x, dtype=np.int64).view(np.uint64)
    return np.asarray(x, dtype=np.int64) & ((1 << width) - 1)


# ---- atomics (DSL v2 increment 4, rc3 02 5.3, 6.3, 6.9, 10) -------------------------------------------------------

_BIT_LAYOUT = {"f32": (np.float32, np.int32), "f64": (np.float64, np.int64), "f16": (np.float16, np.int16)}


def _scoped_used_atomics(module) -> list:
    """tt.atomic_rmw ops whose result is used inside its own block (and the regions nested in that block).  SSA names
    are reused across sibling regions (an unused exchange in an scf.if may share its name with a used CAS result in a
    while region), so a module-wide name set over-approximates (DSL v2 increment 7)."""
    from .ttir_parser import _walk_region

    out = []

    def block_uses(block):
        names = set()
        for o in block.ops:
            names.update(o.operands)
            for _, vals in o.successors:
                names.update(vals)
            for r in o.regions:
                for x in _walk_region(r):
                    names.update(x.operands)
                    for _, vals in x.successors:
                        names.update(vals)
        return names

    def visit(region):
        for block in region.blocks:
            used = None
            for o in block.ops:
                if o.name in ("tt.atomic_rmw", "amdg.buffer_atomic_rmw") and o.results:
                    used = block_uses(block) if used is None else used
                    if o.results[0] in used:
                        out.append(o)
                for r in o.regions:
                    visit(r)

    for fn in module.funcs.values():
        visit(fn.body)
    return out


def _ptr_like(t) -> bool:
    from .ttir_parser import PtrType, TType
    return isinstance(t, PtrType) or (isinstance(t, TType) and isinstance(t.elem, PtrType))


def _may_write_params(func):
    """Parameters of ``func`` whose memory the kernel may write: an over-approximating address flow (DSL v2
    increment 12).  A value derived from a parameter by any op except a memory read carries it; a global write op,
    a call or an inline asm with such an operand may write that parameter's buffer.  Returns (names, any_write);
    names is None (every parameter) when addresses escape through memory (tt.int_to_ptr, or a read whose result is a
    pointer)."""
    from .ttir_parser import _walk_region

    taint = {p[0]: 1 << i for i, p in enumerate(func.params)}
    written = 0
    any_write = False
    ops = list(func.walk())
    for o in ops:
        last = o.name.rsplit(".", 1)[-1]
        if o.name == "tt.int_to_ptr" or (("load" in last or "gather" in last) and any(map(_ptr_like, o.result_types))):
            return None, True
    changed = True
    while changed:
        changed = False

        def add(name, bits):
            nonlocal changed
            if bits and (taint.get(name, 0) | bits) != taint.get(name, 0):
                taint[name] = taint.get(name, 0) | bits
                changed = True

        for o in ops:
            last = o.name.rsplit(".", 1)[-1]
            bits = 0
            for v in o.operands:
                bits |= taint.get(v, 0)
            for _, vals in o.successors:
                for v in vals:
                    bits |= taint.get(v, 0)
            local = o.name.startswith(("ttg.local_", "ttng.tmem_"))
            writes = not local and (o.name in ("tt.call", "tt.elementwise_inline_asm", "ttg.inline_asm") or any(
                k in last for k in ("store", "atomic", "scatter", "local_to_global", "reduce_")) or
                o.name in ("ttng.async_tma_reduce", "tt.descriptor_reduce"))
            if writes:
                any_write = True
                if bits & ~written:
                    written |= bits
                    changed = True
            reads = "load" in last or last.startswith(("atomic", "buffer_atomic")) or last in ("descriptor_gather",
                                                                                            "local_gather")
            # terminators of nested regions carry values to the op's results and block arguments
            for r in o.regions:
                for x in _walk_region(r):
                    if not x.regions and not x.results:
                        for v in x.operands:
                            bits |= taint.get(v, 0)
            if not reads:
                for name in o.results:
                    add(name, bits)
            if bits:
                for r in o.regions:
                    for blk in r.blocks:
                        for name, _ in blk.args:
                            add(name, bits)
        # branch operands reach the block arguments of the function body
        for blk in func.body.blocks:
            for o in blk.ops:
                for _, vals in o.successors:
                    bits = 0
                    for v in vals:
                        bits |= taint.get(v, 0)
                    for b2 in func.body.blocks:
                        for name, _ in b2.args:
                            add(name, bits)
    return {p[0] for i, p in enumerate(func.params) if written >> i & 1}, any_write


def _float_bits(elem: str, lo, hi, st):
    """Bit patterns of float reference values and whether they are definite (the tt.bitcast policy): a point exactly
    representable in the format, or an infinity.  NaN payloads and non-points are not definite."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    st = np.asarray(st)
    point = (lo == hi) & (st == ST_OK)
    x = np.where(point, lo, 0.0)
    if elem == "bf16":
        f = x.astype(np.float32)
        b32 = f.view(np.int32).astype(np.int64)
        representable = (f.astype(np.float64) == x) & ((b32 & 0xFFFF) == 0)
        bits = b32 >> 16
        pinf, ninf = 0x7F80, 0xFF80 - 0x10000
    else:
        fdt, idt = _BIT_LAYOUT[elem]
        f = x.astype(fdt)
        representable = f.astype(np.float64) == x
        bits = f.view(idt).astype(np.int64)
        pinf = int(np.array([np.inf], fdt).view(idt)[0])
        ninf = int(np.array([-np.inf], fdt).view(idt)[0])
    bits = np.where(st == ST_PINF, pinf, np.where(st == ST_NINF, ninf, bits))
    definite = (point & representable) | (st == ST_PINF) | (st == ST_NINF)
    return bits.astype(np.int64), definite


def _f8_bits(elem: str, lo, hi, st):
    """Bit patterns of fp8 reference values (definite: a point exactly representable, or an infinity of e5m2)."""
    import torch

    lo = np.asarray(lo, dtype=np.float64)
    point = (lo == np.asarray(hi, dtype=np.float64)) & (np.asarray(st) == ST_OK)
    x = np.where(point, lo, 0.0)
    tdtype = torch.float8_e5m2 if elem == "f8E5M2" else torch.float8_e4m3fn
    t = torch.from_numpy(x.reshape(-1).copy()).to(tdtype)
    back = t.to(torch.float64).numpy().reshape(x.shape)
    bits = t.view(torch.uint8).numpy().astype(np.int64).reshape(x.shape)
    definite = point & (back == x)
    if elem == "f8E5M2":
        bits = np.where(np.asarray(st) == ST_PINF, 0x7C, np.where(np.asarray(st) == ST_NINF, 0xFC, bits))
        definite = definite | (np.asarray(st) == ST_PINF) | (np.asarray(st) == ST_NINF)
    return bits, definite


def _bits_float(elem: str, bits):
    """Values and status of float bit patterns (signed integer representation of the format's width)."""
    bits = np.asarray(bits, dtype=np.int64)
    if elem == "bf16":
        f = ((bits & 0xFFFF) << 16).astype(np.uint32).view(np.float32).astype(np.float64)
    else:
        fdt, idt = _BIT_LAYOUT[elem]
        f = bits.astype(idt).view(fdt).astype(np.float64)
    st = np.where(np.isnan(f), ST_NAN, np.where(f == np.inf, ST_PINF, np.where(f == -np.inf, ST_NINF, ST_OK)))
    return np.where(st == ST_OK, f, 0.0), st.astype(np.int8)


def _atomic_old(kind: str, is_float: bool, width: int, init: tuple, others: list):
    """Old value one atomic update returns.  ``init`` = (lo, hi, st) of the address before the launch, ``others`` the
    (kind, lo, hi, st) of every other update of the address in the launch.  Without others the update sees init.
    Otherwise every interleaving returns init combined with some subset of the others (program order only removes
    subsets), so the hull of those values encloses the returned value of the actual execution: a set target (L_E,
    rc3 02 5.3 / 10), never refined to a point by precision.  Integers are kept as points only.
    Returns (lo, hi, st, is_set, reason or None)."""
    i_lo, i_hi, i_st = init
    if not others:
        return i_lo, i_hi, i_st, False, None
    if any(o[0] != kind for o in others):
        return 0.0, 0.0, ST_NE, False, "mixed atomic kinds on the address"
    if i_st != ST_OK or any(o[3] != ST_OK for o in others):
        return 0.0, 0.0, ST_NE, False, "a value combined into the returned value is not established or not finite"
    los = [float(o[1]) for o in others] if is_float else [int(o[1]) for o in others]
    his = [float(o[2]) for o in others] if is_float else los
    if is_float:
        if kind == "fadd":
            neg = np.array([float(i_lo)] + [min(0.0, x) for x in los])
            pos = np.array([float(i_hi)] + [max(0.0, x) for x in his])
            lo, hi = float(iv.isum(neg, neg, axis=0)[0]), float(iv.isum(pos, pos, axis=0)[1])
        elif kind == "max":
            lo, hi = float(i_lo), max([float(i_hi)] + his)
        elif kind == "min":
            lo, hi = min([float(i_lo)] + los), float(i_hi)
        elif kind == "exch":
            lo, hi = min([float(i_lo)] + los), max([float(i_hi)] + his)
        else:
            return 0.0, 0.0, ST_NE, False, f"no set rule for a returned atomic {kind}"
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return 0.0, 0.0, ST_NE, False, "the set of returned values leaves the float64 range"
        return lo, hi, ST_OK, (lo, hi) != (float(i_lo), float(i_hi)), None
    v = int(i_lo)
    mask = (1 << width) - 1
    if width:   # DSL v2 increment 14: compare in the signed representation (values are kept modulo 2^width)
        half = 1 << (width - 1)
        v = ((v + half) & mask) - half
        los = [((x + half) & mask) - half for x in los]
    point = {"add": all(x == 0 for x in los), "max": all(x <= v for x in los), "min": all(x >= v for x in los),
             "umax": all((x & mask) <= (v & mask) for x in los), "umin": all((x & mask) >= (v & mask) for x in los),
             "and": all((v & x) == v for x in los), "or": all((v | x) == v for x in los),
             "xor": all(x == 0 for x in los), "exch": all(x == v for x in los)}.get(kind, False)
    if point:
        return v, v, ST_OK, False, None
    hull = _int_return_hull(kind, width, v, los)
    if hull is None:
        return 0, 0, ST_NE, False, "set-valued integer atomic return value (L_E): its hull is not representable"
    return hull[0], hull[1], ST_OK, hull[0] != hull[1], None


INT_SET_REACH_BUDGET = 4096


def _int_return_hull(kind: str, width: int, v: int, xs: list):
    """DSL v2 increment 14 (rc3 02 5.3 / 10): an integer interval (signed representation of the element width) that
    contains every value an integer atomic can return: init combined with some subset of the other updates.  None when
    no such interval is available (add wrapping the width, a reachable set above the budget, an unknown kind)."""
    if not width:
        return None
    lo_lim, hi_lim = -(1 << (width - 1)), (1 << (width - 1)) - 1

    def signed(x):   # integer values are kept modulo 2^width in either representation: use the signed one
        x = int(x) & ((1 << width) - 1)
        return x - (1 << width) if x > hi_lim else x
    v, xs = signed(v), [signed(x) for x in xs]
    if kind == "add":
        lo, hi = v + sum(x for x in xs if x < 0), v + sum(x for x in xs if x > 0)
        return (lo, hi) if lo_lim <= lo and hi <= hi_lim else None
    if kind == "max":
        return v, max([v] + xs)
    if kind == "min":
        return min([v] + xs), v
    if kind in ("umax", "umin", "exch"):   # the returned value is init or one of the contributions
        return min([v] + xs), max([v] + xs)
    fn = {"and": lambda a, b: a & b, "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b}.get(kind)
    if fn is None:
        return None
    reach = {v}
    for x in xs:
        reach |= {fn(r, x) for r in reach}
        if len(reach) > INT_SET_REACH_BUDGET:
            return None
    return min(reach), max(reach)


# operations that pass an integer set target on unchanged, and the operand positions that may carry one (increment 14)
_INT_SET_TRANSPARENT = {"store": (1,), "convert_layout": (0,), "in_thread_transpose": (0,), "splat": (0,),
                        "broadcast": (0,), "expand_dims": (0,), "reshape": (0,), "trans": (0,)}


def _is_int_set(v) -> bool:
    return isinstance(v, TV) and v.kind == "i" and v.hi is not None


def _drop_int_set(v: "TV", op) -> "TV":
    """An integer set target reaching an operation without a set rule: those lanes are not established."""
    lanes = (np.asarray(v.hi) != np.asarray(v.lo)) & (np.asarray(v.st) == ST_OK)
    reasons = v.reasons
    if lanes.any():
        reasons = reasons | {f"not_established:integer set target (L_E) used by {op.name}, which has no set "
                             f"rule@{op.node_id}"}
    return TV("i", v.elem, v.lo, None, v.base, np.where(lanes, ST_NE, v.st).astype(np.int8), v.cond, reasons, v.d)


_JOINT_ORDER_LAWS = (  # mixed kinds on one address whose composition is one commutative, associative operation
    # max by the float key of the bit pattern: signed max for non-negative patterns, unsigned min for negative ones
    ({"max", "umin"}, {"max": 1, "umin": -1}),
    # min by the float key: signed min for non-negative patterns, unsigned max for negative ones
    ({"min", "umax"}, {"min": 1, "umax": -1}),
)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class Buffer:
    ident: int
    elem: str
    kind: str
    lo: np.ndarray
    hi: Optional[np.ndarray]
    st: np.ndarray
    cond: np.ndarray
    writer: np.ndarray  # -1 none, >=0 program index, -2 atomic updates
    written: np.ndarray
    actual_after: Optional[np.ndarray] = None
    actual_after_st: Optional[np.ndarray] = None
    name: str = ""
    index: Optional[np.ndarray] = None  # storage-relative element indices of a windowed capture
    after_raw: Any = None  # captured bytes after the launch (to detect writes between launches)
    dtype: str = ""  # torch storage dtype; integers are kept in its representation
    d: Optional[tuple] = None  # tangent of the memory contents (dlo, dhi)
    # DSL v2 increment 14: integer set targets (L_E) established at the end of the launch; inside the evaluator these
    # elements are not established (st), so no later read, atomic or check uses them as values
    iset: Optional[np.ndarray] = None
    iset_lo: Optional[np.ndarray] = None
    iset_hi: Optional[np.ndarray] = None

    def copy(self) -> "Buffer":
        return Buffer(self.ident, self.elem, self.kind, self.lo.copy(),
                      None if self.hi is None else self.hi.copy(), self.st.copy(), self.cond.copy(),
                      self.writer.copy(), self.written.copy(), self.actual_after, self.actual_after_st,
                      self.name, self.index, self.after_raw, self.dtype,
                      None if self.d is None else (self.d[0].copy(), self.d[1].copy()),
                      None if self.iset is None else self.iset.copy(),
                      None if self.iset_lo is None else self.iset_lo.copy(),
                      None if self.iset_hi is None else self.iset_hi.copy())

    def global_indices(self) -> np.ndarray:
        return np.arange(self.st.size) if self.index is None else self.index


def decode_storage(raw: np.ndarray, dtype: str):
    """Decode a uint8 storage copy into (float64 or int64 values, status)."""

    raw = np.asarray(raw, dtype=np.uint8)
    elem = TORCH_TO_ELEM[dtype]
    size = ELEM_SIZE[elem]
    raw = raw[: (raw.size // size) * size]
    if elem in ("f32", "f16", "f64"):
        vals = raw.view({"f32": np.float32, "f16": np.float16, "f64": np.float64}[elem]).astype(np.float64)
    elif elem == "bf16":
        vals = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    elif elem.startswith("f8"):
        import torch

        tdtype = {"f8E5M2": torch.float8_e5m2, "f8E4M3FN": torch.float8_e4m3fn}[elem]
        vals = torch.from_numpy(raw.copy()).view(tdtype).to(torch.float64).numpy()
    else:
        # uint64 (pointer tables) is read as int64: device addresses are below 2**63
        np_dtype = {"int8": np.int8, "uint8": np.uint8, "int16": np.int16, "uint16": np.uint16, "int32": np.int32,
                    "uint32": np.uint32, "int64": np.int64, "uint64": np.int64, "bool": np.uint8}[dtype]
        vals = raw.view(np_dtype).astype(np.int64)
        return vals, np.zeros(vals.shape, dtype=np.int8)
    st = np.zeros(vals.shape, dtype=np.int8)
    st = np.where(np.isnan(vals), ST_NAN, st)
    st = np.where(vals == np.inf, ST_PINF, st)
    st = np.where(vals == -np.inf, ST_NINF, st)
    vals = np.where(st == ST_OK, vals, 0.0)
    return vals, st.astype(np.int8)


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------


@dataclass
class SharedBuf:
    """Shared memory of one program (DSL v2 increment 10): values, status, written, and the async copy (pending >= 0)
    that has not been observed complete."""
    elem: str
    kind: str
    lo: np.ndarray
    hi: np.ndarray
    st: np.ndarray
    cond: np.ndarray
    written: np.ndarray
    pending: np.ndarray
    base: Optional[np.ndarray] = None  # pointer elements: the buffer each points into


@dataclass
class ProgramState:
    pid: tuple
    pid_index: int
    grid: tuple
    memory: dict  # ident -> Buffer
    epoch: int = 0  # barrier phase inside the program (gpu.barrier orders the threads of one program)
    occ: dict = field(default_factory=dict)  # atomic node -> executions so far (keys of atomic events)
    rel_epoch: int = 0  # release operations performed so far (writes before release e carry epoch <= e)
    loop_depth: int = 0  # enclosing scf.for / scf.while (trigger evidence of nested control, increment 11)
    # DSL v2 increment 10 (TTGIR): the program's shared memory, its outstanding async copies and mbarriers
    shared: dict = field(default_factory=dict)  # id -> SharedBuf
    async_open: list = field(default_factory=list)  # (shared id, element indices) issued, not yet attached / committed
    async_groups: list = field(default_factory=list)  # committed groups (cp.async)
    mbars: dict = field(default_factory=dict)  # (shared id, element) -> barrier state
    # Control dependence: conditions (conditional flag, reasons) that decided the current path.
    ctrl: list = field(default_factory=list)
    sticky_cond: bool = False  # set by unstructured branches / loop exits, kept to the program end
    sticky_reasons: frozenset = frozenset()

    def control(self):
        flag = self.sticky_cond or any(c for c, _ in self.ctrl)
        reasons = self.sticky_reasons.union(*(r for _, r in self.ctrl)) if self.ctrl else self.sticky_reasons
        return flag, reasons


@dataclass
class KernelReference:
    kernel: str
    mode: str
    buffers: dict
    programs: list
    aborted: dict
    notes: list = field(default_factory=list)
    reasons: dict = field(default_factory=dict)
    rules: dict = field(default_factory=dict)  # element/event counts per reference rule
    loaded: set = field(default_factory=set)  # storages whose captured initial values the reference loaded
    loaded_any: set = field(default_factory=set)  # storages the reference loaded from at all
    stored: set = field(default_factory=set)  # storages this launch's reference wrote (stores, atomics, poisoned targets)

    def element_classes(self, ident: int):
        """Per element: complete_composed / conditional_local / not_established / set_target / not_written."""

        buf = self.buffers[ident]
        cls = np.full(buf.st.shape, "complete_composed", dtype=object)
        cls = np.where(buf.cond, "conditional_local", cls)
        cls = np.where(buf.st >= ST_UNDEF, "not_established", cls)
        if buf.iset is not None:   # DSL v2 increment 14: an integer set target [iset_lo, iset_hi] (L_E)
            cls = np.where(buf.iset, "set_target", cls)
        return np.where(buf.written, cls, "not_written")

    def established(self, ident: int) -> np.ndarray:
        """Written elements with an established reference: a point, a special value of its class, or an integer set
        target (increment 14)."""
        buf = self.buffers[ident]
        ok = np.asarray(buf.st) <= ST_NINF
        if buf.iset is not None:
            ok = ok | buf.iset
        return np.asarray(buf.written) & ok

    def int_bounds(self, ident: int):
        """(lo, hi) of an integer buffer's reference values, the set target's hull where there is one."""
        buf = self.buffers[ident]
        lo = np.asarray(buf.lo, dtype=np.int64)
        if buf.iset is None:
            return lo, lo
        return np.where(buf.iset, buf.iset_lo, lo), np.where(buf.iset, buf.iset_hi, lo)

    def compare(self) -> dict:
        """Per output buffer: classes, residual ``actual - reference`` and widths."""

        report = {}
        if self.aborted:
            # Outputs an aborted program would have written are unknown; never report the kernel as clean.
            report["_aborted_programs"] = {"count": len(self.aborted),
                                           "reasons": sorted(set(self.aborted.values()))[:5]}
        for ident, buf in self.buffers.items():
            mask = buf.written
            if not mask.any():
                continue
            cls = self.element_classes(ident)[mask]
            entry = {
                "name": buf.name, "elem": buf.elem, "written": int(mask.sum()),
                "classes": {c: int((cls == c).sum()) for c in ("complete_composed", "conditional_local",
                                                               "not_established", "set_target")},
            }
            if buf.kind == "f" and buf.actual_after is not None:
                act = buf.actual_after[mask]
                act_st = buf.actual_after_st[mask]
                ref_st = buf.st[mask]
                lo, hi = buf.lo[mask], buf.hi[mask]
                finite = (ref_st == ST_OK) & (act_st == ST_OK)
                r_lo = iv.add_bounds(act, -hi)[0]
                r_hi = iv.add_bounds(act, -lo)[1]
                positive = finite & (r_lo > 0)
                negative = finite & (r_hi < 0)
                special_match = (ref_st == act_st) & (ref_st > ST_OK) & (ref_st < ST_UNDEF)
                entry.update({
                    "finite_compared": int(finite.sum()),
                    "residual_positive": int(positive.sum()),
                    "residual_negative": int(negative.sum()),
                    "residual_contains_zero": int((finite & ~positive & ~negative).sum()),
                    "special_matches": int(special_match.sum()),
                    "special_mismatches": int(((ref_st != act_st) & (ref_st < ST_UNDEF)).sum()),
                    "max_reference_width": float(np.max(hi[finite] - lo[finite])) if finite.any() else 0.0,
                    "max_abs_residual": float(np.max(np.maximum(np.abs(r_lo[finite]), np.abs(r_hi[finite]))))
                    if finite.any() else 0.0,
                    "mean_residual_bounds": [float(np.mean(r_lo[finite])), float(np.mean(r_hi[finite]))]
                    if finite.any() else None,
                })
            elif buf.kind in ("i", "b") and buf.actual_after is not None:
                act = buf.actual_after[mask]
                ok = buf.st[mask] == ST_OK
                entry["integer_matches"] = int((ok & (buf.lo[mask] == act)).sum())
                entry["integer_mismatches"] = int((ok & (buf.lo[mask] != act)).sum())
                if buf.iset is not None and buf.iset[mask].any():
                    # increment 14: the actual value against the set target, both in the signed representation
                    width = INT_WIDTH.get(buf.elem, 64)
                    a = _wrap(np.asarray(act, dtype=np.int64), width)
                    inside = (buf.iset_lo[mask] <= a) & (a <= buf.iset_hi[mask])
                    entry["integer_in_set"] = int((buf.iset[mask] & inside).sum())
                    entry["integer_outside_set"] = int((buf.iset[mask] & ~inside).sum())
            report[buf.name or str(ident)] = entry
        return report

    def residual_arrays(self, ident: int):
        """(mask, actual, ref_lo, ref_hi, status, cond) over the written elements."""

        buf = self.buffers[ident]
        m = buf.written
        return m, buf.actual_after[m], buf.lo[m], buf.hi[m], buf.st[m], buf.cond[m]


class KernelReferenceEvaluator:
    def __init__(self, module: TModule, mode: str = NumericMode.NUMERICAL_DIFFERENCE,
                 func_name: Optional[str] = None, masked_fill_zero: bool = False, ttgir: Optional[str] = None):
        """``masked_fill_zero``: treat masked-off lanes of loads without ``other`` as 0 instead of undefined.  TTIR
        leaves them undefined; use only with a lowering checked to zero-fill them (see ``ptx_zero_fills``).
        ``ttgir``: the captured TTGIR of the same compilation; its layouts give the combination order of reductions
        whose combine region has no order-free fast path, and the threads of loads and stores (``layouts.py``)."""
        self.module = module
        self.func = module.entry(func_name)
        self.mode = mode
        self.masked_fill_zero = masked_fill_zero
        self.ttgir = ttgir
        self._layout_cache = None
        self._uses = self._collect_uses()
        self._has_cas = any(op.name in ("tt.atomic_cas", "amdg.buffer_atomic_cas")
                            for fn in module.funcs.values() for op in fn.walk())
        self._returning_atomics = any(op.name in ("tt.atomic_load", "tt.atomic_poll")
                                      for fn in module.funcs.values() for op in fn.walk()) or \
            bool(_scoped_used_atomics(module))

    def _ttgir_layouts(self) -> tuple:
        """(layouts, reduce node -> axis parameters, load/store node -> pointer layout, num_warps); empty without a
        TTGIR."""
        if self._layout_cache is None:
            if not self.ttgir:
                self._layout_cache = ({}, {}, {}, None)
            else:
                import re

                from . import layouts as lay
                m = re.search(r'"ttg\.num-warps"\s*=\s*(\d+)', self.ttgir)
                self._layout_cache = (lay.parse_layouts(self.ttgir), lay.reduce_layouts(self.module, self.ttgir),
                                      lay.access_layouts(self.module, self.ttgir), int(m.group(1)) if m else None)
        return self._layout_cache

    @classmethod
    def from_text(cls, ttir: str, **kwargs) -> "KernelReferenceEvaluator":
        return cls(parse_ttir(ttir), **kwargs)

    def _collect_uses(self) -> set:
        used = set()
        for fn in self.module.funcs.values():
            for op in fn.walk():
                used.update(op.operands)
                for _, vals in op.successors:
                    used.update(vals)
        return used

    # -- binding -------------------------------------------------------------

    def bind(self, launch, memory: Optional[dict] = None) -> tuple:
        """Bind TTIR parameters to captured operands; build the reference memory."""

        params = self.func.params
        captured = launch.ttir_params()
        if len(params) != len(captured):
            raise ValueError(f"{len(params)} TTIR parameters but {len(captured)} non-constexpr arguments")
        memory = {} if memory is None else memory
        bindings = {}
        for (name, ttype, _attrs), arg in zip(params, captured):
            if ttype.is_ptr:
                if arg.kind != "tensor":
                    raise ValueError(f"{name}: pointer parameter bound to {arg.kind}")
                ident = arg.storage_ptr
                if ident not in memory:
                    memory[ident] = self._buffer_from_capture(arg)
                else:
                    # Reference state carries over; the actual side is this launch's capture.
                    after = arg.after.numpy() if hasattr(arg.after, "numpy") else arg.after
                    buf = memory[ident]
                    buf.actual_after, buf.actual_after_st = decode_storage(after, arg.dtype)
                    buf.after_raw = np.asarray(after).copy()
                buf = memory[ident]
                if buf.elem != ttype.elem.pointee and not (buf.elem == "i8" and ttype.elem.pointee == "i1"):
                    buf.name = buf.name or arg.name
                byte = arg.data_ptr - arg.storage_ptr
                bindings[name] = TV("p", ttype.elem, np.array(byte, dtype=np.int64), None,
                                    np.array(ident, dtype=np.int64))
                if not buf.name:
                    buf.name = arg.name
            else:
                kind = kind_of(ttype.elem)
                if kind == "f":
                    # A Python float argument is converted to the declared parameter type.
                    v = float(arg.value)
                    if ttype.elem in iv.FLOAT_FORMATS and math.isfinite(v):
                        v = float(iv.round_nearest_even(np.array([v]), ttype.elem)[0][0])
                    bindings[name] = _ftv(ttype.elem, np.array(v), np.array(v), np.array(ST_OK, dtype=np.int8),
                                          np.array(False), frozenset())
                else:
                    # two's complement of the parameter width (a uint64 2**64 - 1 arrives as a Python int; DSL v2
                    # increment 5, official test_value_specialization_overflow)
                    v, w = int(arg.value), INT_WIDTH.get(ttype.elem, 64)
                    v = ((v + (1 << (w - 1))) % (1 << w)) - (1 << (w - 1)) if ttype.elem != "i1" else v
                    bindings[name] = TV(kind, ttype.elem, np.array(v, dtype=np.int64))
        for arg in getattr(launch, "implicit", []) or []:  # reachable through raw addresses only
            ident = arg.storage_ptr
            if ident not in memory:
                memory[ident] = self._buffer_from_capture(arg)
            elif arg.after is not None:
                after = arg.after.numpy() if hasattr(arg.after, "numpy") else arg.after
                buf = memory[ident]
                buf.actual_after, buf.actual_after_st = decode_storage(after, arg.dtype)
                buf.after_raw = np.asarray(after).copy()
            if not memory[ident].name:
                memory[ident].name = arg.name
        return bindings, memory

    def _buffer_from_capture(self, arg) -> Buffer:
        elem = TORCH_TO_ELEM[arg.dtype]
        kind = "f" if elem in FLOAT_ELEMS else ("b" if elem == "i1" else "i")
        before = arg.before.numpy() if hasattr(arg.before, "numpy") else arg.before
        after = arg.after.numpy() if hasattr(arg.after, "numpy") else arg.after
        vals, st = decode_storage(before, arg.dtype)
        after_vals, after_st = decode_storage(after, arg.dtype)
        n = vals.size
        index = None if arg.window is None else np.asarray(arg.window, dtype=np.int64)
        return Buffer(arg.storage_ptr, elem, kind, vals.copy(), vals.copy() if kind == "f" else None,
                      st, np.zeros(n, dtype=bool), np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=bool),
                      after_vals, after_st, arg.name, index, np.asarray(after).copy(), arg.dtype)

    # -- evaluation -----------------------------------------------------------

    def evaluate(self, launch, programs: Optional[list] = None, pin_loads: tuple = (),
                 memory: Optional[dict] = None, tangents: Optional[dict] = None) -> KernelReference:
        """Composed reference of one launch.

        ``tangents`` (derivative reference) maps argument names to the tangent
        direction u of a float operand: an array over the buffer (window or
        storage) for pointer arguments, a number for float scalars.  Output
        buffers then carry J(x) u in ``Buffer.d``.
        """

        bindings, memory = self.bind(launch, memory)
        self._derivative = bool(tangents)
        if tangents:
            by_name = {a.name: a for a in launch.ttir_params()}
            for name, value in tangents.items():
                arg = by_name[name]
                if arg.kind == "tensor":
                    buf = memory[arg.storage_ptr]
                    t = np.broadcast_to(np.asarray(value, dtype=np.float64), buf.st.shape).copy()
                    buf.d = (t, t.copy())
                else:
                    param = next(p[0] for p, a in zip(self.func.params, launch.ttir_params()) if a.name == name)
                    v = float(value)
                    bindings[param].d = (np.array(v), np.array(v))
        grid = tuple(launch.grid)
        if programs is None:
            programs = [(x, y, z) for z in range(grid[2]) for y in range(grid[1]) for x in range(grid[0])]
        self._pin = set(pin_loads)
        self._atomic_pass, self._atomic_table, self._atomic_init, self._plain_prev = 0, None, None, None
        self._cyclic = set()
        self._cas_serial = self._has_cas and not self._returning_atomics
        cas_snapshot = {k: b.copy() for k, b in memory.items()} if self._cas_serial else None
        snapshot = table = None
        if self._returning_atomics:
            # DSL v2 increment 4: returned atomic values need every update of their address in this launch.  Pass 1
            # records the updates (returned values not established); pass 2 restarts from the memory before the
            # launch and returns the value (or set) determined by pass 1's table.
            snapshot = {k: b.copy() for k, b in memory.items()}
            self._atomic_pass = 1
            self._run_programs(programs, grid, bindings, memory)
            order = None
            if sum(e[1].size for e in self._events) <= ATOMIC_EVENT_BUDGET:
                table = self._event_table()
                self._plain_prev = self._plain_stores
                order, self._cyclic = self._poll_order(len(programs), table, self._polls)
                for k in list(memory):
                    memory[k] = snapshot[k].copy()
                self._atomic_pass, self._atomic_table, self._atomic_init = 2, table, snapshot
        if self._atomic_pass == 1:
            # over the event budget: pass 1 stands (returned values not established)
            aborted, reasons = self._last_run
            self._rules["atomic.two_pass_skipped_event_budget"] += 1
        else:
            aborted, reasons = self._run_programs(programs, grid, bindings, memory,
                                                  order if self._atomic_pass == 2 else None)
        if self._atomic_pass == 2:
            self._check_two_pass(table, memory, reasons)
        iset_valid = True
        if self._cas_serial and self._cas_contended and len(programs) > 1:
            aborted, reasons = self._cas_reverse_order(programs, grid, bindings, memory, cas_snapshot, aborted,
                                                       reasons)
            iset_valid = False   # two serializations were merged: no integer set target record stands
        if aborted:
            # An aborted program may stop before some of its stores: elements the kernel changed but the reference
            # did not write in this launch keep their earlier reference value, which a later launch would read as
            # established.  They are not established.
            names, any_write = _may_write_params(self.func)
            params = list(zip(self.func.params, launch.ttir_params()))
            may_write = {a.storage_ptr for p, a in params if a.kind == "tensor" and (names is None or p[0] in names)}
            if any_write:   # buffers reachable through raw addresses only
                may_write |= {a.storage_ptr for a in getattr(launch, "implicit", []) or []}
            for arg in list(launch.args) + list(getattr(launch, "implicit", []) or []):
                if arg.kind != "tensor" or arg.storage_ptr not in memory or arg.before is None:
                    continue
                buf = memory[arg.storage_ptr]
                if buf.after_raw is None:
                    continue
                b = np.asarray(arg.before.numpy() if hasattr(arg.before, "numpy") else arg.before)
                a = np.asarray(buf.after_raw)
                n = buf.st.size
                if b.shape != a.shape or b.size % n:
                    continue
                changed = (b != a).reshape(n, -1).any(axis=1)
                hit = changed & (buf.writer == -1)
                if hit.any():
                    buf.st = np.where(hit, ST_NE, buf.st).astype(np.int8)
                    reasons["not_established:changed by the kernel, not written by the aborted reference"] += \
                        int(hit.sum())
                # DSL v2 increment 12 (defect found by the TTIR / TTGIR cross-check): an aborted program may have
                # rewritten an element with identical bytes; in a buffer the kernel writes, no element the reference
                # did not write keeps the captured value as an established reference
                if changed.any() or np.asarray(buf.written).any() or arg.storage_ptr in may_write:
                    rest = (buf.writer == -1) & ~changed & (buf.st == ST_OK)
                    if rest.any():
                        buf.st = np.where(rest, ST_NE, buf.st).astype(np.int8)
                        reasons["not_established:in a buffer the kernel writes, not written by the aborted reference"] \
                            += int(rest.sum())
        self._finalize_int_sets(memory, programs, grid, aborted, reasons, iset_valid)
        notes = []
        if self._outside_window:
            notes.append("some accesses fell outside the captured windows; those lanes are not established")
        if len(programs) < grid[0] * grid[1] * grid[2]:
            notes.append(f"evaluated {len(programs)} of {grid[0] * grid[1] * grid[2]} program instances; "
                         "races with unevaluated instances are not checked")
        if aborted:
            self._rules["path.program_aborted"] += len(aborted)
        return KernelReference(self.func.name, self.mode, memory, [tuple(p) for p in programs], aborted,
                               notes, dict(reasons), dict(self._rules), set(self._loaded), set(self._loaded_any),
                               set(self._stored))

    def _cas_reverse_order(self, programs, grid, bindings, memory, snapshot, aborted, reasons):
        """DSL v2 increment 7 (rc3 02 6.9, declared order + evidence): the launch was evaluated as one serialization of
        its contended compare-and-swap operations (program order).  Re-evaluate it in reverse program order from the
        memory before the launch and keep, per element, only what both orders establish and agree on (the hull when
        the enclosures overlap); elements the two orders disagree on are not established.  Agreement of two orders is
        evidence, not a proof over every serialization: recorded as a declared premise."""
        first = {k: b.copy() for k, b in memory.items()}
        first_rules = collections.Counter(self._rules)
        for k in list(memory):
            memory[k] = snapshot[k].copy()
        aborted2, reasons2 = self._run_programs(programs, grid, bindings, memory,
                                                list(range(len(programs)))[::-1])
        differ = 0
        for k, b2 in memory.items():
            b1 = first[k]
            w = np.asarray(b1.written) | np.asarray(b2.written)
            if not w.any():
                continue
            both_ok = (b1.st == ST_OK) & (b2.st == ST_OK)
            if b1.kind == "f":
                overlap = both_ok & (b1.lo <= b2.hi) & (b2.lo <= b1.hi)
                b2.lo = np.where(overlap, np.minimum(b1.lo, b2.lo), b2.lo)
                b2.hi = np.where(overlap, np.maximum(b1.hi, b2.hi), b2.hi)
            else:
                overlap = both_ok & (b1.lo == b2.lo)
            same_special = (b1.st == b2.st) & (b1.st != ST_OK) & (b1.st < ST_UNDEF) & \
                ((b1.kind != "f") | True)
            agree = overlap | same_special
            bad = w & ~agree & ((b1.st == ST_OK) | (b2.st == ST_OK) | (b1.st != b2.st))
            if bad.any():
                differ += int(bad.sum())
                b2.st = np.where(bad, ST_NE, b2.st).astype(np.int8)
            b2.written = w
        merged = collections.Counter(reasons)
        merged.update(reasons2)
        self._rules.update(first_rules)
        self._rules["execution.cas_two_order_launches"] += 1
        if differ:
            merged["not_established:the launch result depends on the order of its contended compare-and-swap "
                   "operations (program order and reverse order differ)"] += differ
            self._rules["execution.cas_order_dependent_elements"] += differ
        merged["assumed:the launch result does not depend on the order of its contended compare-and-swap operations "
               "(program order and reverse order agree; a declared premise, not a proof over every order)"] += 1
        all_aborted = dict(aborted)
        all_aborted.update(aborted2)
        return all_aborted, merged

    def _run_programs(self, programs, grid, bindings, memory, order=None) -> tuple:
        """Run every program instance of the launch once on ``memory`` (in ``order``: indices into ``programs``, by
        default their order; the program index stays its position in ``programs``); launch-level execution checks at
        the end."""
        self._rules = collections.Counter()
        self._outside_window = False
        self._loaded = set()
        self._loaded_any = set()
        self._stored = set()
        self._poisoned = set()
        # execution validity (DSL v2 rc3 02 6.3 / 04 W4): who read each address in this launch, and which store,
        # lane position and barrier phase wrote it last
        self._readers = {}  # ident -> int64 array: -1 not read, >= 0 the one program that read it, -3 several
        self._reader_programs = collections.defaultdict(set)
        self._race_programs = set()
        self._wstore = {}  # ident -> (store node index, lane position in the store, barrier epoch) per element
        self._store_nodes = []
        self._thread_cache = {}
        self._events = []  # atomic updates: (ident, idx, kind, lo, hi, st, pid index, node, occurrence, lanes)
        self._plain_stores = {}  # ident -> bool array: written by a non-atomic store in this launch
        self._consumers = collections.defaultdict(set)  # (ident, address) -> programs that used a returned value
        # happens-before (DSL v2 increment 5): release epochs of plain writes, releases per address, acquired edges
        self._wepoch = {}  # ident -> int64 array: writer's release epoch at the write
        self._releases = collections.defaultdict(list)  # (ident, address) -> [(program, epoch, value)]
        self._hb = collections.defaultdict(dict)  # program -> {writer program: highest acquired release epoch}
        self._polls = []  # (program, ident, address, expected) recorded by atomic_poll
        self._awriter = {}  # ident -> (program, release epoch) of the last atomic write per element
        self._read_chain = {}  # ident -> (last reader, its epoch, every earlier reader happens before it)
        self._cas_contended = False
        self._two_pass_bad = set()
        # DSL v2 increment 14: integer set targets stored in this run, ident -> {element: (lo, hi, program)}; set
        # targets of earlier launches are carried as records of program -1
        self._iset_records = {}
        for buf in memory.values():
            if buf.iset is not None and buf.iset.any():
                self._iset_records[buf.ident] = {int(e): (int(buf.iset_lo[e]), int(buf.iset_hi[e]), -1)
                                                 for e in np.flatnonzero(buf.iset)}
        for buf in memory.values():
            buf.writer[:] = -1  # kernel boundaries order all earlier writes
        aborted = {}
        reasons = collections.Counter()
        for index in (order if order is not None else range(len(programs))):
            pid = programs[index]
            state = ProgramState(tuple(pid), index, grid, memory)
            self._reasons = reasons
            try:
                env = collections.ChainMap(dict(bindings))
                self._call(self.func, [bindings[p[0]] for p in self.func.params], state, env_override=env)
            except ProgramAbort as exc:
                aborted[tuple(pid)] = str(exc)
                for buf in memory.values():
                    hit = buf.writer == index
                    buf.st = np.where(hit, ST_NE, buf.st).astype(np.int8)
        if self._race_programs:
            # a program read an address that another program of the same launch writes: what it read depends on
            # the schedule, so nothing it wrote keeps an established reference value
            hits = 0
            for buf in memory.values():
                hit = np.isin(buf.writer, sorted(self._race_programs))
                if hit.any():
                    buf.st = np.where(hit, ST_NE, buf.st).astype(np.int8)
                    hits += int(hit.sum())
            reasons["not_established:execution race: an address one program read is written by another program "
                    "of the same launch"] += hits
            self._rules["execution.cross_program_read_write_race_programs"] += len(self._race_programs)
        for ident in self._poisoned:
            tb = memory.get(ident)
            if tb is not None:
                tb.st = np.full(tb.st.shape, ST_NE, dtype=np.int8)
                tb.written[:] = True
        self._atomic_final_laws(memory, reasons)
        self._last_run = (aborted, reasons)
        return aborted, reasons

    def _finalize_int_sets(self, memory, programs, grid, aborted, reasons, valid: bool):
        """DSL v2 increment 14: the integer set targets that stand at the end of the launch.  A record stands only if
        the element is still what that store left (value, writer, not-established status), its program did not
        abort, race or see a two-pass mismatch, and the buffer was not poisoned by a store through an undefined
        address.  A record carried from an earlier launch (program -1) stands only if this launch did not write the
        element, no program aborted and every program instance was evaluated."""
        bad = {i for i, pid in enumerate(programs) if tuple(pid) in aborted} | set(self._race_programs) | \
            set(getattr(self, "_two_pass_bad", set()))
        full = len(programs) == grid[0] * grid[1] * grid[2]
        for buf in memory.values():
            buf.iset = buf.iset_lo = buf.iset_hi = None
        kept = dropped = 0
        for ident, recs in self._iset_records.items():
            buf = memory.get(ident)
            if buf is None:
                continue
            for e, (lo, hi, pid) in recs.items():
                ok = valid and ident not in self._poisoned and buf.st[e] == ST_NE and int(buf.lo[e]) == lo
                if pid == -1:
                    ok = ok and buf.writer[e] == -1 and not aborted and full
                else:
                    ok = ok and buf.writer[e] == pid and pid not in bad
                if not ok:
                    dropped += 1
                    continue
                if buf.iset is None:
                    n = buf.st.size
                    buf.iset, buf.iset_lo, buf.iset_hi = (np.zeros(n, dtype=bool), np.zeros(n, dtype=np.int64),
                                                          np.zeros(n, dtype=np.int64))
                buf.iset[e], buf.iset_lo[e], buf.iset_hi[e] = True, lo, hi
                kept += 1
        if kept:
            reasons["set:integer set target stored (L_E)"] += kept
        if dropped:
            reasons["not_established:integer set target record invalidated (later write, abort, race, poisoned "
                    "buffer or order merge)"] += dropped
        self._rules["atomic.integer_set_elements"] += kept

    # -- functions and regions ----------------------------------------------------

    def _call(self, func: TFunc, args: list, state: ProgramState, env_override=None) -> list:
        if _TRIGGER_PATH:
            _trace("tt.func:func")
        env = env_override if env_override is not None else collections.ChainMap({})
        entry = func.body.blocks[0]
        for (name, _), value in zip(entry.args, args):
            env[name] = value
        return self._run_cfg(func.body, env, state)

    def _run_cfg(self, region: TRegion, env, state: ProgramState) -> list:
        blocks = {b.label: b for b in region.blocks}
        block = region.blocks[0]
        steps = 0
        while True:
            steps += 1
            if steps > 100000:
                raise ProgramAbort("control-flow graph did not terminate")
            for op in block.ops:
                if op.name in ("tt.return", "cf.br", "cf.cond_br") and _TRIGGER_PATH:
                    _trace(f"{op.name}:{rule_for(op.name).internal}")  # DSL v2 increment 8: terminators traced
                if op.name == "tt.return":
                    return [env[v] for v in op.operands]
                if op.name in ("cf.br", "cf.cond_br"):
                    if op.name == "cf.br":
                        label, vals = op.successors[0]
                    else:
                        c = env[op.operands[0]]
                        if c.st.max() >= ST_UNDEF or int(c.lo) == MAYBE:
                            raise ProgramAbort(f"undecided branch condition at {op.node_id}")
                        label, vals = op.successors[0] if int(c.lo) == 1 else op.successors[1]
                        self._rules["path.branch_decided_by_reference"] += 1
                        if c.cond.any():
                            state.sticky_cond = True
                            state.sticky_reasons = state.sticky_reasons | c.reasons
                    target = blocks[label]
                    values = [env[v] for v in vals]
                    for (name, _), value in zip(target.args, values):
                        env[name] = value
                    block = target
                    break
                self._exec(op, env, state)
            else:
                return []

    def _run_region(self, region: TRegion, env, state: ProgramState, args: list) -> tuple:
        """Run a single-block region; returns (terminator op, yielded values)."""

        scope = env.new_child()
        block = region.entry
        for (name, _), value in zip(block.args, args):
            scope[name] = value
        for op in block.ops:
            if op.name in ("scf.yield", "tt.reduce.return", "tt.scan.return", "scf.condition"):
                if _TRIGGER_PATH:
                    _trace(f"{op.name}:{rule_for(op.name).internal}")
                return op, [scope[v] for v in op.operands]
            self._exec(op, scope, state)
        return None, []

    # -- dispatch -------------------------------------------------------------------

    def _exec(self, op: TOp, env, state: ProgramState):
        rule = rule_for(op.name)
        if rule is None or rule.status != "SUPPORTED":
            raise ProgramAbort(f"{op.node_id}: {op.name} has no reference rule"
                               + (f" ({rule.reason})" if rule else ""))
        handler = getattr(self, f"_op_{rule.internal}", None)
        if _TRIGGER_PATH:
            _trace(f"{op.name}:{rule.internal}")
        args = [env[v] for v in op.operands] if rule.internal not in ("for", "if", "while") else None
        if args and any(_is_int_set(a) for a in args):
            if rule.internal == "buffer_store":
                keep = (op.attrs["roles"].split(",").index("value"),)
            else:
                keep = _INT_SET_TRANSPARENT.get(rule.internal, ())
            args = [_drop_int_set(a, op) if _is_int_set(a) and k not in keep else a for k, a in enumerate(args)]
        if handler is not None:
            out = handler(op, args, env, state)
        else:
            out = self._elementwise(rule.internal, op, args)
        if out is None:
            return
        if isinstance(out, TV):
            out = [out]
        if getattr(self, "_derivative", False) and args and any(
                isinstance(a, TV) and a.kind == "f" and a.d is not None for a in args):
            if any(isinstance(v, TV) and v.kind == "f" and v.d is None for v in out) \
                    and rule.internal not in _ZERO_TANGENT_OK:
                raise ProgramAbort(f"{op.node_id}: no derivative rule for {op.name}")
        for name, value in zip(op.results, out):
            env[name] = value

    # ---- layout / addressing --------------------------------------------------

    def _op_constant(self, op, args, env, state):
        return _constant(op)

    def _op_make_range(self, op, args, env, state):
        start = int(op.attrs["start"].split(":")[0])
        end = int(op.attrs["end"].split(":")[0])
        return TV("i", "i32", np.arange(start, end, dtype=np.int64))

    def _op_program_id(self, op, args, env, state):
        axis = {"x": 0, "y": 1, "z": 2}[op.attrs["axis"]]
        return TV("i", "i32", np.array(state.pid[axis], dtype=np.int64))

    def _op_num_programs(self, op, args, env, state):
        axis = {"x": 0, "y": 1, "z": 2}[op.attrs["axis"]]
        return TV("i", "i32", np.array(state.grid[axis], dtype=np.int64))

    def _op_splat(self, op, args, env, state):
        shape = op.result_types[0].shape
        return args[0].map(lambda a: np.broadcast_to(a, shape).copy())

    def _op_unsplat(self, op, args, env, state):
        return args[0].map(lambda a: np.asarray(a).reshape(()).copy())

    def _op_broadcast(self, op, args, env, state):
        shape = op.result_types[0].shape
        return args[0].map(lambda a: np.broadcast_to(a, shape).copy())

    def _op_expand_dims(self, op, args, env, state):
        axis = int(op.attrs["axis"].split(":")[0])
        return args[0].map(lambda a: np.expand_dims(a, axis))

    def _op_reshape(self, op, args, env, state):
        shape = op.result_types[0].shape
        out = args[0].map(lambda a: np.reshape(a, shape))
        if "allow_reorder" in op.attrs:
            uses = self._users(op.results[0])
            if not uses or not all(u.name == "tt.reduce" for u in uses):
                raise ProgramAbort(f"{op.node_id}: reshape with allow_reorder feeds an order-sensitive use")
        return out

    def _users(self, value: str) -> list:
        return [o for fn in self.module.funcs.values() for o in fn.walk() if value in o.operands]

    def _op_trans(self, op, args, env, state):
        order = [int(x) for x in op.attrs["order"].split(":")[1].strip(" >").split(",")]
        return args[0].map(lambda a: np.transpose(a, order).copy())

    def _op_join(self, op, args, env, state):
        a, b = args
        out = TV(a.kind, a.elem, np.stack([a.lo, b.lo], axis=-1),
                 None if a.hi is None else np.stack([a.hi, b.hi], axis=-1),
                 None if a.base is None else np.stack([a.base, b.base], axis=-1),
                 np.stack([a.st, b.st], axis=-1), np.stack([a.cond, b.cond], axis=-1), a.reasons | b.reasons)
        if a.kind == "f" and (a.d is not None or b.d is not None):
            ta, tb = a.tangent(), b.tangent()
            out.d = (np.stack([ta[0], tb[0]], axis=-1), np.stack([ta[1], tb[1]], axis=-1))
        return out

    def _op_split(self, op, args, env, state):
        (a,) = args
        return [a.map(lambda x: x[..., 0].copy()), a.map(lambda x: x[..., 1].copy())]

    def _op_gather(self, op, args, env, state):
        src, idx = args
        axis = int(op.attrs.get("axis", "0").split(":")[0])
        if idx.st.max() >= ST_UNDEF:
            raise ProgramAbort(f"{op.node_id}: gather index not established")
        index = idx.lo.astype(np.int64)
        oob = (index < 0) | (index >= src.shape[axis])
        out = src.map(lambda a: np.take_along_axis(a, np.clip(index, 0, src.shape[axis] - 1), axis=axis))
        if oob.any():
            # DSL v2 increment 8 (rc3 02 6.6, per-sample index check): an index outside the source has no value
            out = TV(out.kind, out.elem, out.lo, out.hi, out.base, np.where(oob, ST_NE, out.st).astype(np.int8),
                     out.cond, out.reasons | {f"not_established:gather index out of range@{op.node_id}"}, out.d)
        return out

    def _op_addptr(self, op, args, env, state):
        ptr, off = args
        pointee = ptr.elem.pointee if isinstance(ptr.elem, PtrType) else None
        size = ELEM_SIZE.get(pointee)
        if size is None:
            raise ProgramAbort(f"{op.node_id}: pointer to {pointee}")
        lo = ptr.lo + off.lo.astype(np.int64) * size
        st = _merge_status(ptr, off)
        return TV("p", ptr.elem, lo, None, np.broadcast_to(ptr.base, lo.shape).copy(), st,
                  _merge_cond(ptr, off), _merge_reasons(ptr, off))

    def _op_int_to_ptr(self, op, args, env, state):
        """A raw address back to a pointer: the captured storage (operand or registered implicit tensor) whose
        byte range holds it, plus the byte offset.  Addresses outside every captured storage are not established."""
        x = args[0]
        out_elem = op.result_types[0].elem
        addr = np.asarray(x.lo, dtype=np.int64)
        idents = np.array(sorted(state.memory), dtype=np.int64)
        ends = np.array([i + state.memory[int(i)].st.size * ELEM_SIZE[state.memory[int(i)].elem] for i in idents],
                        dtype=np.int64)
        base = np.zeros(addr.shape, dtype=np.int64)
        found = np.zeros(addr.shape, dtype=bool)
        if idents.size:
            k = np.searchsorted(idents, addr, side="right") - 1
            kc = np.clip(k, 0, idents.size - 1)
            found = (k >= 0) & (addr < ends[kc])
            base = np.where(found, idents[kc], 0)
        reasons = set(x.reasons)
        st = np.where(found, x.st, ST_NE).astype(np.int8)
        if (~found & (x.st == ST_OK)).any():
            reasons.add(f"not_established:address outside every captured storage@{op.node_id}")
        return TV("p", out_elem, np.where(found, addr - base, 0), None, base, st, x.cond, reasons)

    def _op_ptr_to_int(self, op, args, env, state):
        p = args[0]
        out_elem = op.result_types[0].elem
        return TV("i", out_elem, (np.asarray(p.base, dtype=np.int64) + np.asarray(p.lo, dtype=np.int64)), None, None,
                  p.st, p.cond, p.reasons)

    def _op_poison(self, op, args, env, state):
        ttype = op.result_types[0]
        kind = kind_of(ttype.elem)
        lo = np.zeros(ttype.shape, dtype=np.float64 if kind == "f" else np.int64)
        return TV(kind, ttype.elem, lo, lo.copy() if kind == "f" else None,
                  np.zeros(ttype.shape, dtype=np.int64) if kind == "p" else None,
                  np.full(ttype.shape, ST_UNDEF, dtype=np.int8))

    def _op_nop(self, op, args, env, state):
        return None

    def _op_barrier(self, op, args, env, state):
        state.epoch += 1
        return None

    def _op_assume(self, op, args, env, state):
        """llvm.intr.assume (DSL v2 rc3 02 12): a compilation assumption is a premise of the program, checked on the
        reference path of this sample.  True: no effect.  False or undecided: the reference of this program is not
        established (an assumption the reference path violates leaves the program's meaning undefined)."""
        cond = args[0]
        v = np.asarray(cond.lo)
        if (v == 1).all() and (np.asarray(cond.st) == ST_OK).all():
            self._rules["premise.assume_held"] += 1
            return None
        why = "violated" if ((v == 0) & (np.asarray(cond.st) == ST_OK)).any() else "undecided"
        self._rules[f"premise.assume_{why}"] += 1
        raise ProgramAbort(f"{op.node_id}: not_established: compilation assumption {why} on the reference path")

    # ---- execution validity: conflicting accesses ----------------------------------

    def _track_read(self, buf, idx, state):
        r = self._readers.get(buf.ident)
        if r is None:
            r = self._readers[buf.ident] = np.full(buf.writer.shape, -1, dtype=np.int64)
        cur = r[idx]
        r[idx] = np.where((cur == -1) | (cur == state.pid_index), state.pid_index, -3)
        self._reader_programs[buf.ident].add(state.pid_index)
        # DSL v2 increment 7: the last reader of each element and whether every earlier reader happens before it
        chain = self._read_chain.get(buf.ident)
        if chain is None:
            n = buf.writer.shape
            chain = self._read_chain[buf.ident] = (np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64),
                                                   np.ones(n, dtype=bool))
        last, epoch, ok = chain
        prev = last[idx]
        covered = (prev == -1) | (prev == state.pid_index) | self._hb_covers(state, prev, epoch[idx])
        ok[idx] = ok[idx] & covered
        last[idx], epoch[idx] = state.pid_index, state.rel_epoch

    def _hb_covers(self, state, programs, epochs):
        """Elementwise: program p's actions up to epoch e happen before this program's next action."""
        edges = self._hb.get(state.pid_index)
        if not edges:
            return np.zeros(np.shape(programs), dtype=bool)
        out = [p >= 0 and edges.get(int(p), -1) >= int(e) for p, e in zip(np.ravel(programs).tolist(),
                                                                         np.ravel(epochs).tolist())]
        return np.asarray(out, dtype=bool).reshape(np.shape(programs))

    def _check_read_then_write(self, buf, idx, state):
        """A write (store or atomic) to addresses another program of this launch already read: that program's reads
        depend on the schedule, unless every earlier read happens before this write (DSL v2 increment 7: reads ordered
        by acquired releases, e.g. inside a lock)."""
        r = self._readers.get(buf.ident)
        if r is None or not idx.size:
            return
        others = r[idx]
        clash = (others != -1) & (others != state.pid_index)
        chain = self._read_chain.get(buf.ident)
        if clash.any() and chain is not None and self._hb.get(state.pid_index):
            last, epoch, ok = chain
            ordered = ok[idx] & ((last[idx] == state.pid_index) | self._hb_covers(state, last[idx], epoch[idx]))
            clash = clash & ~ordered
        if not clash.any():
            return
        self._race_programs.update(int(p) for p in np.unique(others[clash & (others >= 0)]))
        if (others[clash] == -3).any():
            self._race_programs.update(p for p in self._reader_programs[buf.ident] if p != state.pid_index)
        self._rules["execution.cross_program_read_then_write_lanes"] += int(clash.sum())

    def _mark_plain(self, buf, idx):
        """Addresses written by a non-atomic access in this launch (a returned atomic value after one is not the value
        before the launch)."""
        mark = self._plain_stores.get(buf.ident)
        if mark is None:
            mark = self._plain_stores[buf.ident] = np.zeros(buf.writer.shape, dtype=bool)
        mark[idx] = True

    def _record_store(self, op, buf, idx, positions, state):
        ws = self._wstore.get(buf.ident)
        if ws is None:
            n = buf.writer.shape[0]
            ws = self._wstore[buf.ident] = tuple(np.full(n, -1, dtype=np.int64) for _ in range(3))
        if op.node_id not in self._store_nodes:
            self._store_nodes.append(op.node_id)
        ws[0][idx] = self._store_nodes.index(op.node_id)
        ws[1][idx] = positions
        ws[2][idx] = state.epoch

    def _threads(self, info):
        """Threads holding each element of an access (layouts.element_threads), cached; None if unknown."""
        layouts, _, _, nw = self._ttgir_layouts()
        if info is None or nw is None:
            return None
        key = (info["layout"], info["shape"])
        if key not in self._thread_cache:
            from .layouts import element_threads
            self._thread_cache[key] = element_threads(layouts, info["layout"], tuple(info["shape"]), nw)
        return self._thread_cache[key]

    def _same_program_write_read(self, op, buf, idx, safe, state):
        """Lanes of a load that read a value this program stored in the same barrier phase.  Same thread (the load
        and the store hold the element in the same single thread): ordered, allowed.  Another thread: an execution
        race (no happens-before without a barrier).  Thread mapping unknown: execution validity not established.
        A replicated stored element is written by its lowest thread id (Triton 3.6.0 redundant-thread predicate)."""
        ws = self._wstore.get(buf.ident)
        if ws is None:
            return None, ()
        flat = np.asarray(safe).reshape(-1)
        lanes = np.flatnonzero(flat)
        addr = np.asarray(idx).reshape(-1)[lanes]
        own = (buf.writer[addr] == state.pid_index) & (ws[2][addr] == state.epoch) & (ws[0][addr] >= 0)
        if not own.any():
            return None, ()
        access = self._ttgir_layouts()[2]
        th_load = self._threads(access.get(op.node_id))
        verdict = np.zeros(lanes.size, dtype=np.int8)  # 1 same thread, 2 another thread, 3 unknown
        for node in np.unique(ws[0][addr[own]]):
            sel = own & (ws[0][addr] == node)
            th_store = self._threads(access.get(self._store_nodes[int(node)]))
            if th_store is None or th_load is None:
                verdict[sel] = 3
                continue
            writer = th_store[ws[1][addr[sel]]].min(axis=1)
            readers = th_load[lanes[sel]]
            verdict[sel] = np.where(np.all(readers == writer[:, None], axis=1), 1, 2)
        self._rules["execution.same_program_write_read_same_thread_lanes"] += int((verdict == 1).sum())
        why = []
        if (verdict == 2).any():
            self._rules["execution.intra_program_cross_thread_race_lanes"] += int((verdict == 2).sum())
            why.append(f"not_established:execution race: read by another thread of the same program after a store "
                       f"without a barrier@{op.node_id}")
        if (verdict == 3).any():
            self._rules["execution.same_program_write_read_unknown_threads_lanes"] += int((verdict == 3).sum())
            why.append(f"not_established:execution validity: same-program store then load without a barrier, thread "
                       f"mapping not available@{op.node_id}")
        if not why:
            return None, ()
        bad = np.zeros(flat.size, dtype=bool)
        bad[lanes[verdict >= 2]] = True
        return bad.reshape(np.asarray(safe).shape), why

    def _op_assert(self, op, args, env, state):
        cond = args[0]
        if (cond.lo == 0).any() and (cond.st == ST_OK).any():
            self._reasons[f"reference violates assert at {op.node_id}"] += 1
        return None

    def _op_return(self, op, args, env, state):
        return None

    # ---- memory -------------------------------------------------------------------

    def _addresses(self, op, ptr: TV, state: ProgramState, bitview: bool = False):
        """``bitview``: an integer pointer may address a float buffer of the same width (atomics on bit patterns)."""
        pointee = ptr.elem.pointee
        size = ELEM_SIZE[pointee]
        idents = np.unique(ptr.base[ptr.st == ST_OK]) if ptr.base is not None else []
        if len(idents) > 1:
            raise ProgramAbort(f"{op.node_id}: pointer tensor spans several buffers")
        if len(idents) == 0:
            return None, None, None
        buf = state.memory[int(idents[0])]
        view_ok = bitview and kind_of(pointee) == "i" and buf.kind == "f" and buf.elem in ("f16", "bf16", "f32", "f64")
        if ELEM_SIZE[buf.elem] != size or (kind_of(pointee) != buf.kind and not view_ok and not
                                           ((pointee, buf.elem) in (("i1", "i8"), ("i8", "i1")))):
            raise ProgramAbort(f"{op.node_id}: {pointee} access to a {buf.elem} buffer (reinterpretation)")
        index = ptr.lo // size
        aligned = (ptr.lo % size) == 0
        if buf.index is None:
            in_range = (index >= 0) & (index < buf.st.size) & aligned
            return buf, index, in_range
        # Windowed capture: map storage indices to window positions; others are not captured.
        pos = np.searchsorted(buf.index, index)
        pos_c = np.minimum(pos, buf.index.size - 1)
        present = (buf.index[pos_c] == index) & aligned
        if (~present & (ptr.st == ST_OK)).any():
            self._outside_window = True
        return buf, np.where(present, pos_c, 0), present

    def _op_load(self, op, args, env, state):
        ptr = args[0]
        mask = args[1] if len(args) > 1 else None
        other = args[2] if len(args) > 2 else None
        pointee = ptr.elem.pointee
        kind = kind_of(pointee)
        shape = ptr.shape
        buf, index, in_range = self._addresses(op, ptr, state)
        m = np.ones(shape, dtype=np.int8) if mask is None else mask.lo.astype(np.int8)
        m_st = np.zeros(shape, dtype=np.int8) if mask is None else mask.st
        active = (m != 0) & (m_st == ST_OK)
        pinned = op.node_id in self._pin or (op.results and op.results[0] in self._pin)
        lo = np.zeros(shape)
        hi = np.zeros(shape)
        st = np.full(shape, ST_UNDEF, dtype=np.int8)
        cond = np.zeros(shape, dtype=bool)
        dlo = dhi = None
        reasons = set(ptr.reasons)
        if buf is not None:
            safe = active & in_range & (ptr.st == ST_OK)
            idx = np.where(safe, index, 0).astype(np.int64)
            if safe.any():
                self._loaded_any.add(buf.ident)
            if (safe & ~buf.written[idx]).any():
                self._loaded.add(buf.ident)  # lanes read the captured initial value (not a reference write)
            if pinned:
                src_lo = buf.actual_after
                src_hi = buf.actual_after if kind == "f" else None
                src_st = buf.actual_after_st
                src_cond = np.ones(buf.st.shape, dtype=bool)
                reasons.add(f"pinned_load:{op.node_id}")
            else:
                src_lo, src_hi, src_st, src_cond = buf.lo, buf.hi, buf.st, buf.cond
            lo = np.where(safe, src_lo[idx], 0.0 if kind == "f" else 0)
            if kind == "f":
                hi = np.where(safe, src_hi[idx], 0.0)
            st = np.where(safe, src_st[idx], st)
            if kind == "i" and buf.kind == "b":  # int8 view of a bool buffer: an undecided bool is not a byte
                st = np.where(safe & (np.asarray(lo) == MAYBE), ST_NE, st)
            cond = np.where(safe, src_cond[idx], False)
            if kind == "f" and buf.d is not None and not pinned:
                dlo = np.where(safe, buf.d[0][idx], 0.0)
                dhi = np.where(safe, buf.d[1][idx], 0.0)
            carried = safe & buf.written[idx]
            if carried.any() and not pinned:
                self._rules["memory.load_reads_reference_value"] += int(carried.sum())
                if _TRIGGER_PATH:   # DSL v2 increment 11: the composite rule "one storage, the reference value"
                    _trace("tt.load/tt.store (alias):memory")
            if pinned:
                self._rules["observability.pinned_load_lanes"] += int(safe.sum())
            race = safe & (buf.writer[idx] != -1) & (buf.writer[idx] != state.pid_index)
            if race.any() and self._hb.get(state.pid_index):
                race = race & ~self._ordered_by_hb(buf, idx, state)
            if race.any():
                self._rules["memory.cross_program_race_lanes"] += int(race.sum())
                st = np.where(race, ST_NE, st)
                reasons.add(f"not_established:cross-program race on load@{op.node_id}")
            if not pinned and safe.any() and not getattr(self, "_atomic_read", False):
                self._track_read(buf, idx[safe], state)
            if not pinned and safe.any():
                bad, why = self._same_program_write_read(op, buf, idx, safe, state)
                if bad is not None:
                    st = np.where(bad, ST_NE, st)
                    reasons.update(why)
            oob = active & ~in_range & (ptr.st == ST_OK)
            if oob.any():
                st = np.where(oob, ST_NE, st)
                reasons.add(f"not_established:out-of-bounds load@{op.node_id}")
        st = np.where(active & (ptr.st >= ST_UNDEF), ST_NE, st)
        inactive = (m == 0) & (m_st == ST_OK)
        if other is not None:
            ob = np.broadcast_to
            lo = np.where(inactive, ob(other.lo, shape), lo)
            if kind == "f":
                hi = np.where(inactive, ob(other.hi, shape), hi)
            st = np.where(inactive, ob(other.st, shape), st)
            cond = np.where(inactive, ob(other.cond, shape), cond)
            reasons |= other.reasons
            if kind == "f" and other.d is not None:
                dlo = np.where(inactive, ob(other.d[0], shape), 0.0 if dlo is None else dlo)
                dhi = np.where(inactive, ob(other.d[1], shape), 0.0 if dhi is None else dhi)
        elif inactive.any() and self.masked_fill_zero:
            # Declared lowering assumption (verified on the kernel's PTX by the caller): masked-off lanes of a load
            # without `other` hold 0, as Triton's NVIDIA backend zero-initializes the destination registers.
            lo = np.where(inactive, 0.0 if kind == "f" else 0, lo)
            if kind == "f":
                hi = np.where(inactive, 0.0, hi)
            st = np.where(inactive, ST_OK, st)
            reasons.add(f"assumed:masked lanes zero-filled by the lowering@{op.node_id}")
            self._rules["memory.masked_load_zero_filled_lanes"] += int(inactive.sum())
        elif inactive.any():
            reasons.add(f"undefined:masked load without other@{op.node_id}")
            self._rules["memory.masked_load_undefined_lanes"] += int(inactive.sum())
        maybe = (m == MAYBE) | (m_st >= ST_UNDEF)
        if maybe.any():
            st = np.where(maybe, ST_NE, st)
            reasons.add(f"not_established:undecided load mask@{op.node_id}")
        if mask is not None:
            cond = cond | mask.cond
            reasons |= mask.reasons
        if kind == "f":
            out = _ftv(pointee, lo, hi, st, cond, frozenset(reasons))
            if dlo is not None:
                out.d = (np.where(st == ST_OK, dlo, 0.0), np.where(st == ST_OK, dhi, 0.0))
            return out
        if kind == "i":
            # Registers hold iN values in two's-complement signed form; unsigned ops reinterpret.
            return TV(kind, pointee, _wrap(np.asarray(lo).astype(np.int64), INT_WIDTH[pointee]), None, None,
                      st.astype(np.int8), cond, frozenset(reasons))
        return TV(kind, pointee, np.asarray(lo).astype(np.int8), None, None, st.astype(np.int8), cond,
                  frozenset(reasons))

    def _op_store(self, op, args, env, state):
        ptr, value = args[0], args[1]
        mask = args[2] if len(args) > 2 else None
        shape = ptr.shape
        m = np.ones(shape, dtype=np.int8) if mask is None else mask.lo.astype(np.int8)
        m_st = np.zeros(shape, dtype=np.int8) if mask is None else mask.st
        active = (m != 0)
        p_st = np.broadcast_to(ptr.st, shape)
        blind = (active | (m == MAYBE) | (m_st >= ST_UNDEF)) & (p_st != ST_OK)
        if blind.any():
            # A lane stores through an address that is not established: any element of the target buffer may have
            # been written, so none of them keeps an established reference value.
            targets = np.unique(np.broadcast_to(ptr.base, shape)[blind]) if ptr.base is not None else []
            if len(targets) == 0:
                raise ProgramAbort(f"{op.node_id}: store through an address of unknown buffer")
            self._poisoned.update(int(i) for i in targets)  # applied again at the end of the launch
            self._stored.update(int(i) for i in targets)
            # DSL v2 increment 11 (defect found by the W1 alias test): the store may have written any element, so a
            # later load in this program must not read the old value as established; and it counts as a write to
            # every element against reads of other programs
            for t in targets:
                tb = state.memory.get(int(t))
                if tb is not None:
                    tb.st = np.full(tb.st.shape, ST_NE, dtype=np.int8)
                    tb.written[:] = True
                    self._check_read_then_write(tb, np.arange(tb.st.size), state)
            self._reasons[f"not_established:store through a not-established address@{op.node_id}"] += int(blind.sum())
            active = active & ~blind
        buf, index, in_range = self._addresses(op, ptr, state)
        if buf is None:
            return None
        v_lo = np.broadcast_to(value.lo, shape)
        v_hi = np.broadcast_to(value.hi, shape) if value.kind == "f" else None
        v_st = np.broadcast_to(value.st, shape).copy()
        # DSL v2 increment 14: lanes holding an integer set target [lo, s_hi]
        s_hi = np.broadcast_to(value.hi, shape) if _is_int_set(value) else None
        ctrl_flag, ctrl_reasons = state.control()
        v_cond = np.broadcast_to(value.cond, shape) | (False if mask is None else mask.cond) | ctrl_flag
        for r in ctrl_reasons:
            self._reasons[r] += 1
        v_st = np.where(ptr.st >= ST_UNDEF, ST_NE, v_st)
        if mask is not None:
            v_st = np.where((m == MAYBE) | (m_st >= ST_UNDEF), ST_NE, v_st)
        oob = active & ~in_range
        if (oob & (ptr.st == ST_OK)).any():
            self._reasons[f"out-of-bounds store at {op.node_id}"] += 1
        sel = active & in_range
        idx = index[sel].astype(np.int64)
        lo_w = v_lo[sel]
        if buf.kind == "b" and value.kind == "i":
            # A torch.bool storage written through an int8 pointer (Inductor casts bool outputs to int8):
            # the byte holds 0 or 1.
            lo_w = (np.asarray(lo_w, dtype=np.int64) != 0).astype(np.int8)
        elif buf.kind == "i" and buf.dtype in ("uint8", "bool"):
            lo_w = np.asarray(lo_w, dtype=np.int64) & 0xFF
        elif buf.kind == "i":
            lo_w = _wrap(np.asarray(lo_w, dtype=np.int64), INT_WIDTH[buf.elem])
        hi_w = v_hi[sel] if v_hi is not None else None
        st_w = v_st[sel]
        cond_w = v_cond[sel]
        set_w = np.zeros(idx.size, dtype=bool)
        if s_hi is not None:
            sh = np.asarray(s_hi[sel], dtype=np.int64)
            set_w = (sh != np.asarray(v_lo[sel], dtype=np.int64)) & (st_w == ST_OK)
            if buf.kind != "i" or buf.dtype in ("uint8", "bool"):
                # the byte representation of a signed interval is not an interval: no set target
                st_w = np.where(set_w, ST_NE, st_w)
                if set_w.any():
                    self._reasons[f"not_established:integer set target stored into a byte or bool storage"
                                  f"@{op.node_id}"] += 1
                set_w[:] = False
        # Two lanes writing one address in the same store: order is unspecified.
        if idx.size:
            order = np.argsort(idx, kind="stable")
            sidx = idx[order]
            dup = np.zeros(idx.size, dtype=bool)
            same = sidx[1:] == sidx[:-1]
            if same.any():
                differ = same & ((lo_w[order][1:] != lo_w[order][:-1]) |
                                 (hi_w is not None and False) | (st_w[order][1:] != st_w[order][:-1]))
                if hi_w is not None:
                    differ = differ | (same & (hi_w[order][1:] != hi_w[order][:-1]))
                if s_hi is not None:
                    shs = np.where(set_w, sh, np.asarray(lo_w, dtype=np.int64))[order]
                    differ = differ | (same & ((shs[1:] != shs[:-1]) | (set_w[order][1:] != set_w[order][:-1])))
                bad_addr = set(sidx[1:][differ].tolist())
                if bad_addr:
                    dup = np.isin(idx, list(bad_addr))
                    self._reasons[f"conflicting lanes in one store at {op.node_id}"] += 1
            st_w = np.where(dup, ST_NE, st_w)
            self._check_read_then_write(buf, idx, state)
            other_writer = (buf.writer[idx] != -1) & (buf.writer[idx] != state.pid_index)
            if other_writer.any() and self._hb.get(state.pid_index):
                other_writer = other_writer & ~self._ordered_by_hb(buf, idx, state)
            if other_writer.any():
                old_differs = (buf.lo[idx] != lo_w) | (buf.st[idx] != st_w)
                if hi_w is not None:
                    old_differs = old_differs | (buf.hi[idx] != hi_w)
                race = other_writer & old_differs
                self._rules["memory.cross_program_race_lanes"] += int(race.sum())
                st_w = np.where(race, ST_NE, st_w)
                if race.any():
                    self._reasons[f"cross-program write race at {op.node_id}"] += 1
            if buf.kind == "f" and (value.d is not None or buf.d is not None):
                if buf.d is None:
                    buf.d = (np.zeros(buf.st.shape), np.zeros(buf.st.shape))
                t = value.tangent()
                buf.d[0][idx] = np.broadcast_to(t[0], shape)[sel]
                buf.d[1][idx] = np.broadcast_to(t[1], shape)[sel]
            # integer set targets: not established inside the evaluator; a record, checked at the end of the launch
            set_w = set_w & (st_w == ST_OK)
            recs = self._iset_records.setdefault(buf.ident, {})
            for j, e in enumerate(idx.tolist()):
                if set_w[j]:
                    recs[e] = (int(lo_w[j]), int(sh[j]), state.pid_index)
                else:
                    recs.pop(e, None)
            st_w = np.where(set_w, ST_NE, st_w)
            buf.lo[idx] = lo_w
            if hi_w is not None:
                buf.hi[idx] = hi_w
            buf.st[idx] = st_w
            buf.cond[idx] = cond_w
            buf.writer[idx] = state.pid_index
            buf.written[idx] = True
            self._mark_plain(buf, idx)
            we = self._wepoch.get(buf.ident)
            if we is None:
                we = self._wepoch[buf.ident] = np.full(buf.writer.shape, -1, dtype=np.int64)
            we[idx] = state.rel_epoch
            self._record_store(op, buf, idx, np.flatnonzero(np.asarray(sel).reshape(-1)), state)
            self._stored.add(buf.ident)
        for r in value.reasons:
            self._reasons[r] += 1
        return None

    def _op_atomic_store(self, op, args, env, state):
        """tt.atomic_store (DSL v2 increment 5): an exchange whose returned value is not used."""
        return self._op_atomic_rmw(op, args, env, state, kind_override="exch")

    def _op_atomic_rmw(self, op, args, env, state, kind_override=None):
        ptr, value = args[0], args[1]
        mask = args[2] if len(args) > 2 else None
        kind_name = kind_override or op.attrs["rmw_op"]
        shape = ptr.shape
        buf, index, in_range = self._addresses(op, ptr, state, bitview=True)
        m = np.ones(shape, dtype=np.int8) if mask is None else mask.lo.astype(np.int8)
        if mask is not None and ((m == MAYBE) | (mask.st >= ST_UNDEF)).any():
            raise ProgramAbort(f"{op.node_id}: atomic with undecided mask")
        if buf is None:
            if (np.broadcast_to(m, shape) == 1).any():  # an active lane updates an address the reference cannot place
                raise ProgramAbort(f"{op.node_id}: atomic through a not-established address")
            index, in_range = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=bool)
        active = (m == 1) & in_range
        idx = index[active].astype(np.int64)
        result_used = op.results and op.results[0] in self._uses
        occ = state.occ[op.node_id] = state.occ.get(op.node_id, -1) + 1
        if _TRIGGER_PATH:
            _trace(f"{op.name}/return {'used' if result_used else 'unused'}")
        # DSL v2 increment 4: an integer pointer onto a float buffer of the same width (the frontend's float max/min)
        # works on the bit patterns of the stored values
        bitview = buf is not None and buf.kind == "f" and kind_of(ptr.elem.pointee) == "i"
        # masked-off lanes return no value (undefined); active lanes before pass 2: not established
        olds = TV(value.kind, value.elem, np.zeros(shape), np.zeros(shape) if value.kind == "f" else None,
                  None, np.where(np.broadcast_to(m, shape) == 1, ST_NE, ST_UNDEF).astype(np.int8),
                  np.zeros(shape, dtype=bool),
                  frozenset({f"not_established:atomic return value depends on an undeclared order@{op.node_id}"}
                            if idx.size else ()))
        if idx.size == 0:
            return olds if op.results else None
        self._check_read_then_write(buf, idx, state)
        plain = (buf.writer[idx] >= 0) & (buf.writer[idx] != state.pid_index)
        contrib_lo = np.broadcast_to(value.lo, shape)[active]
        contrib_hi = np.broadcast_to(value.hi, shape)[active] if value.kind == "f" else None
        contrib_st = np.broadcast_to(value.st, shape)[active]
        lanes = np.flatnonzero(np.asarray(active).reshape(-1))
        self._events.append((buf.ident, idx, kind_name, contrib_lo, contrib_hi, contrib_st, state.pid_index,
                             op.node_id, occ, lanes))
        if result_used and self._atomic_pass == 2:
            olds = self._atomic_returns(op, buf, idx, lanes, occ, shape, active, value, bitview, state)
            olds.st = np.where(np.broadcast_to(m, shape) == 1, olds.st, ST_UNDEF).astype(np.int8)
        elif result_used:
            self._rules["atomic.return_value_not_established_lanes"] += int(idx.size)
        else:
            self._rules["atomic.folded_order_free_lanes"] += int(idx.size)
        uniq, inverse = np.unique(idx, return_inverse=True)
        if kind_name == "fadd" and self.mode == NumericMode.ROUNDING_CHECK:
            raise ProgramAbort(f"{op.node_id}: atomic accumulation order is not declared (rounding-check mode)")
        if kind_name == "fadd" and value.d is not None:
            if buf.d is None:
                buf.d = (np.zeros(buf.st.shape), np.zeros(buf.st.shape))
            t = value.tangent()
            t_lo = np.broadcast_to(t[0], shape)[active]
            t_hi = np.broadcast_to(t[1], shape)[active]
            for j, address in enumerate(uniq):
                sel = inverse == j
                c_lo, c_hi = iv.isum(t_lo[sel], t_hi[sel], axis=0) if sel.sum() > 1 else (t_lo[sel][0], t_hi[sel][0])
                n_lo, n_hi = iv.iadd(np.array(buf.d[0][address]), np.array(buf.d[1][address]),
                                     np.array(c_lo), np.array(c_hi))
                buf.d[0][address], buf.d[1][address] = float(n_lo), float(n_hi)
        elif kind_name != "fadd" and value.d is not None:
            raise ProgramAbort(f"{op.node_id}: no derivative rule for atomic {kind_name}")
        if bitview:
            width = INT_WIDTH[ptr.elem.pointee]
            tgt, definite = _float_bits(buf.elem, buf.lo, buf.hi, buf.st)
            tgt_kind = "i"
        else:
            width = INT_WIDTH.get(buf.elem, 0)
            tgt, definite, tgt_kind = buf.lo, None, buf.kind
        if kind_name == "fadd":
            for j, address in enumerate(uniq):
                sel = inverse == j
                c_lo, c_hi = iv.isum(contrib_lo[sel], contrib_hi[sel], axis=0) if sel.sum() > 1 else \
                    (contrib_lo[sel][0], contrib_hi[sel][0])
                bad = contrib_st[sel].max() != ST_OK or buf.st[address] != ST_OK
                new_lo, new_hi = iv.iadd(np.array(buf.lo[address]), np.array(buf.hi[address]),
                                         np.array(c_lo), np.array(c_hi))
                buf.lo[address], buf.hi[address] = float(new_lo), float(new_hi)
                if bad:
                    buf.st[address] = ST_NE
        elif kind_name == "exch":
            # the last exchange in evaluation order; contended addresses are widened at the end of the launch
            if tgt_kind == "f":
                buf.lo[idx], buf.hi[idx], buf.st[idx] = contrib_lo, contrib_hi, contrib_st
            else:
                tgt[idx] = _wrap(np.asarray(contrib_lo, dtype=np.int64), width)
        elif kind_name in ("add",):
            np.add.at(tgt, idx, contrib_lo.astype(np.int64))
            tgt[uniq] = _wrap(tgt[uniq], width)
        elif kind_name in ("max", "min", "umax", "umin", "and", "or", "xor"):
            if tgt_kind == "f":
                fn = np.maximum if kind_name == "max" else np.minimum
                fn.at(buf.lo, idx, contrib_lo)
                fn.at(buf.hi, idx, contrib_hi)
            elif kind_name in ("umax", "umin"):  # DSL v2 increment 3: unsigned comparison, stored back as two's complement
                tmp = _unsigned(tgt, width).copy()
                (np.maximum if kind_name == "umax" else np.minimum).at(tmp, idx, _unsigned(contrib_lo, width))
                tgt[:] = tmp.view(np.int64) if width >= 64 else _wrap(tmp.astype(np.int64), width)
            else:
                fn = {"max": np.maximum, "min": np.minimum, "and": np.bitwise_and, "or": np.bitwise_or,
                      "xor": np.bitwise_xor}.get(kind_name)
                if fn is None:
                    raise ProgramAbort(f"{op.node_id}: atomic {kind_name}")
                fn.at(tgt, idx, contrib_lo.astype(np.int64))
        else:
            raise ProgramAbort(f"{op.node_id}: atomic {kind_name} is order dependent")
        if bitview:
            vals, vst = _bits_float(buf.elem, tgt[uniq])
            buf.lo[uniq], buf.hi[uniq] = vals, vals
            buf.st[uniq] = np.where(definite[uniq], vst, ST_NE)
            if not definite[uniq].all():
                self._reasons[f"not_established:bit-level reinterpretation without definite bits@{op.node_id}"] += 1
            self._rules["atomic.bitview_lanes"] += int(idx.size)
        # DSL v2 increment 4 (defect fix): a contribution without an established value leaves the address not
        # established for every kind (only fadd checked it before)
        bad_addr = np.unique(idx[np.asarray(contrib_st) != ST_OK])
        if bad_addr.size:
            buf.st[bad_addr] = ST_NE
        buf.st[idx] = np.where(plain, ST_NE, buf.st[idx])
        ctrl_flag, _ = state.control()
        if ctrl_flag or value.cond.any() or (mask is not None and mask.cond.any()):
            extra = np.broadcast_to(value.cond, shape)[active] | ctrl_flag
            if mask is not None:
                extra = extra | np.broadcast_to(mask.cond, shape)[active]
            buf.cond[idx] = buf.cond[idx] | extra
        buf.writer[idx] = -2
        buf.written[idx] = True
        self._stored.add(buf.ident)
        aw = self._awriter.get(buf.ident)
        if aw is None:
            aw = self._awriter[buf.ident] = (np.full(buf.writer.shape, -1, dtype=np.int64),
                                             np.full(buf.writer.shape, -1, dtype=np.int64))
        aw[0][idx], aw[1][idx] = state.pid_index, state.rel_epoch
        if op.attrs.get("sem") in ("release", "acq_rel"):
            # a release: the program's earlier writes (epoch <= rel_epoch) happen before an acquire that reads it
            vals = np.asarray(contrib_lo)
            snap = dict(self._hb.get(state.pid_index, {}))
            for j, a in enumerate(idx.tolist()):
                self._releases[(buf.ident, a)].append((state.pid_index, state.rel_epoch, vals[j].item(), snap))
            state.rel_epoch += 1
        return olds if op.results else None

    def _acquire(self, state, release):
        """Merge a release (program, epoch, value, happens-before snapshot) into this program's edges (DSL v2
        increments 5 / 7): the releasing program's writes up to that epoch, and everything that program had acquired
        before releasing (transitivity), happen before this program's later accesses."""
        w, epoch, _, snap = release
        if w == state.pid_index:
            return
        edges = self._hb[state.pid_index]
        edges[w] = max(edges.get(w, -1), epoch)
        for v, ev in snap.items():
            if v != state.pid_index:
                edges[v] = max(edges.get(v, -1), ev)
        self._rules["execution.happens_before_edges"] += 1

    def _ordered_by_hb(self, buf, idx, state):
        """Lanes whose last writer's write happens before this program's access through an acquired release (plain
        writers from the write epochs, atomic writers from the atomic-writer table)."""
        edges = self._hb.get(state.pid_index, {})
        we = self._wepoch.get(buf.ident)
        writers = np.array(buf.writer[idx], copy=True)
        if not edges:
            return np.zeros(writers.shape, dtype=bool)
        epochs = we[idx] if we is not None else np.full(writers.shape, -1, dtype=np.int64)
        aw = self._awriter.get(buf.ident)
        if aw is not None:
            atomic = writers == -2
            writers = np.where(atomic, aw[0][idx], writers)
            epochs = np.where(atomic, aw[1][idx], epochs)
        ok = [w >= 0 and edges.get(int(w), -1) >= int(e) for w, e in zip(np.ravel(writers).tolist(),
                                                                          np.ravel(epochs).tolist())]
        return np.asarray(ok, dtype=bool).reshape(np.shape(writers))

    def _poll_order(self, n, table, polls):
        """Pass-2 program order (DSL v2 increment 5): a program that writes the value a poll waits for runs before the
        polling program; otherwise program order.  Programs on a dependency cycle are returned as cyclic."""
        import heapq
        deps = collections.defaultdict(set)
        for c, ident, a, expected in polls:
            for e in table.get((ident, a), []):
                w = e[0][0]
                if w != c and e[1] == "exch" and e[2] == expected:
                    deps[c].add(w)
        if not deps:
            return None, set()
        users = collections.defaultdict(set)
        indeg = [0] * n
        for c, ws in deps.items():
            indeg[c] = len(ws)
            for w in ws:
                users[w].add(c)
        ready = [i for i in range(n) if indeg[i] == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            i = heapq.heappop(ready)
            order.append(i)
            for c in users[i]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    heapq.heappush(ready, c)
        cyclic = {i for i in range(n) if indeg[i] > 0}
        self._rules["atomic.poll_reordered_launch"] += 1
        return order + sorted(cyclic), cyclic

    def _op_atomic_load(self, op, args, env, state):
        """tt.atomic_load (DSL v2 increment 5): a load whose address no atomic update of the launch reaches is an
        ordinary load (a point).  Otherwise every interleaving reads the value before the launch combined with some
        subset of those updates: the set target of ``_atomic_old`` (pass 2), not established before.  Atomic reads
        are not registered as plain reads, so they do not race with atomic writes."""
        self._atomic_read = True
        try:
            base = self._op_load(op, args[:2], env, state)
        finally:
            self._atomic_read = False
        ptr = args[0]
        mask = args[1] if len(args) > 1 else None
        shape = ptr.shape
        buf, index, in_range = self._addresses(op, ptr, state)
        if buf is None:
            return base
        m = np.ones(shape, dtype=np.int8) if mask is None else np.asarray(mask.lo).astype(np.int8)
        active = (m == 1) & in_range & (np.broadcast_to(ptr.st, shape) == ST_OK)
        idx = np.asarray(index)[active].astype(np.int64)
        if not idx.size:
            return base
        is_float = base.kind == "f"
        lo = np.array(base.lo, dtype=np.float64 if is_float else np.int64, copy=True)
        # DSL v2 increment 14: an integer load gets an upper bound for its set-valued lanes
        hi = np.array(base.hi, dtype=np.float64, copy=True) if is_float else np.array(lo, copy=True)
        st = np.array(base.st, dtype=np.int8, copy=True)
        reasons = set(base.reasons)
        if self._atomic_pass == 2 and not ((buf.writer[idx] >= 0) & (buf.writer[idx] != state.pid_index)).any():
            # the ordinary load flags addresses an earlier program updated atomically; those lanes get the set below
            reasons -= {r for r in base.reasons if "cross-program race on load" in r}
        flat_pos = np.flatnonzero(active.reshape(-1))
        why = collections.Counter()
        n_set = 0
        width = INT_WIDTH.get(base.elem, 0)
        table = self._atomic_table if self._atomic_pass == 2 else None
        lo_f, st_f = lo.reshape(-1), st.reshape(-1)
        hi_f = hi.reshape(-1) if hi is not None else None
        for pos, a in zip(flat_pos.tolist(), idx.tolist()):
            if table is None:
                if self._atomic_pass == 1:  # pass 1: the updates of later programs are not known yet
                    st_f[pos] = ST_NE
                    why["atomic load before the second pass"] += 1
                continue
            self._consumers[(buf.ident, a)].add(state.pid_index)
            evs = table.get((buf.ident, a), [])
            if not evs:
                continue  # no atomic update of this address in the launch: the ordinary load stands
            if self._plain_prev is not None and self._plain_prev.get(buf.ident) is not None \
                    and self._plain_prev[buf.ident][a]:
                st_f[pos] = ST_NE
                why["atomic load of an address with non-atomic writes in the same launch"] += 1
                continue
            kinds = {e[1] for e in evs}
            init_buf = self._atomic_init[buf.ident]
            init = ((float(init_buf.lo[a]), float(init_buf.hi[a]), int(init_buf.st[a])) if is_float else
                    (int(init_buf.lo[a]), int(init_buf.lo[a]), int(init_buf.st[a])))
            if len(kinds) != 1:
                st_f[pos] = ST_NE
                why["mixed atomic kinds on the address"] += 1
                continue
            r_lo, r_hi, r_st, is_set, reason = _atomic_old(kinds.pop(), is_float, width, init, [e[1:] for e in evs])
            if reason:
                why[reason] += 1
            lo_f[pos], st_f[pos] = r_lo, r_st
            if hi_f is not None:
                hi_f[pos] = r_hi
            if is_set and r_st == ST_OK and base.kind == "b":   # a bool set is not an interval of the format
                st_f[pos] = ST_NE
                why["set-valued boolean atomic load"] += 1
                continue
            n_set += bool(is_set and r_st == ST_OK)
        for r in why:
            reasons.add(f"not_established:{r}@{op.node_id}")
        if n_set:
            reasons.add(f"set:atomic load over all interleavings (L_E)@{op.node_id}")
        self._rules["atomic.load_set_lanes"] += n_set
        if is_float:
            return _ftv(base.elem, lo, hi, st, base.cond, frozenset(reasons))
        return TV(base.kind, base.elem, lo.astype(np.int64) if base.kind == "i" else lo.astype(np.int8),
                  hi.astype(np.int64) if (n_set and base.kind == "i") else None, None, st, base.cond,
                  frozenset(reasons))

    def _op_atomic_poll(self, op, args, env, state):
        """tt.atomic_poll (DSL v2 increment 5, rc3 02 6.9): pass 1 records the polled addresses; pass 2 decides each
        element from pass 1's updates of the address.  No other writer: (value before the launch == expected); without
        a timeout an unequal value never terminates on the reference.  Another program exchanges the expected value
        in: true without a timeout under the termination premise (fair scheduling), {false, true} (MAYBE) with one.
        A successful acquire poll whose value comes from one program's release orders that program's earlier writes
        before this program's later loads (happens-before)."""
        ptr, expected = args[0], args[1]
        timeout = op.attrs.get("timeout") == "1"
        shape = ptr.shape
        buf, index, in_range = self._addresses(op, ptr, state)
        if buf is None:
            raise ProgramAbort(f"{op.node_id}: atomic_poll through an address of unknown buffer")
        active = in_range & (np.broadcast_to(ptr.st, shape) == ST_OK)
        idx = np.asarray(index)[active].astype(np.int64)
        exp_lo = np.broadcast_to(expected.lo, shape)[active]
        exp_st = np.broadcast_to(expected.st, shape)[active]
        res = np.zeros(shape, dtype=np.int8)
        st = np.full(shape, ST_NE, dtype=np.int8)
        reasons = set()
        if self._atomic_pass != 2:
            for a, x in zip(idx.tolist(), exp_lo.tolist()):
                self._polls.append((state.pid_index, buf.ident, a, int(x)))
            reasons.add(f"not_established:atomic poll before the second pass@{op.node_id}")
            return TV("b", "i1", res, None, None, st, np.zeros(shape, dtype=bool), frozenset(reasons))
        r_out = np.zeros(idx.size, dtype=np.int8)
        s_out = np.full(idx.size, ST_NE, dtype=np.int8)
        why = collections.Counter()
        init_buf = self._atomic_init[buf.ident]
        plain = self._plain_prev.get(buf.ident) if self._plain_prev is not None else None
        for j, (a, x) in enumerate(zip(idx.tolist(), exp_lo.tolist())):
            self._consumers[(buf.ident, a)].add(state.pid_index)
            if state.pid_index in self._cyclic:
                why["poll dependency cycle: progress not proven"] += 1
                continue
            if exp_st[j] != ST_OK or init_buf.st[a] != ST_OK:
                why["polled or expected value not established"] += 1
                continue
            if plain is not None and plain[a]:
                why["non-atomic write to the polled address in the same launch"] += 1
                continue
            evs = [e for e in self._atomic_table.get((buf.ident, a), []) if e[0][0] != state.pid_index]
            if any(e[1] != "exch" for e in evs):
                why["accumulating atomic updates of the polled address"] += 1
                continue
            init, x = int(init_buf.lo[a]), int(x)
            produced = [e for e in evs if e[2] == x and e[4] == ST_OK]
            if not evs:
                if init != x and not timeout:
                    raise ProgramAbort(f"{op.node_id}: not_established: atomic_poll cannot terminate on the reference "
                                       f"(no update of the address to the expected value)")
                r_out[j], s_out[j] = int(init == x), ST_OK
            elif timeout:
                r_out[j], s_out[j] = (MAYBE if (produced or init == x) else 0), ST_OK
                if produced or init == x:
                    reasons.add(f"set:atomic poll with a timeout: {{false, true}}@{op.node_id}")
            elif produced:
                r_out[j], s_out[j] = 1, ST_OK
                reasons.add(f"assumed:atomic_poll terminates (fair scheduling: the writing program makes "
                            f"progress)@{op.node_id}")
                writers = {e[0][0] for e in produced}
                if op.attrs.get("sem") == "acquire" and init != x and len(writers) == 1:
                    w = writers.pop()
                    rel = [r for r in self._releases.get((buf.ident, a), []) if r[0] == w and r[2] == x]
                    if rel:
                        self._acquire(state, min(rel, key=lambda r: r[1]))
            elif init == x:
                why["the expected value may be overwritten before the first load: progress not proven"] += 1
            else:
                raise ProgramAbort(f"{op.node_id}: not_established: atomic_poll cannot terminate on the reference "
                                   f"(no update of the address to the expected value)")
        if why and not timeout:
            # without a timeout an element whose termination is not shown may never let the program continue
            raise ProgramAbort(f"{op.node_id}: not_established: {sorted(why)[0]}")
        for r in why:
            reasons.add(f"not_established:{r}@{op.node_id}")
        for r in reasons:
            if r.startswith(("assumed:", "set:")):  # premises and set targets hold for the launch, used or not
                self._reasons[r] += 1
        res[active], st[active] = r_out, s_out
        self._rules["atomic.poll_decided_lanes"] += int((s_out == ST_OK).sum())
        return TV("b", "i1", res, None, None, st, np.zeros(shape, dtype=bool), frozenset(reasons))

    def _op_map_elementwise(self, op, args, env, state):
        """tt.map_elementwise (DSL v2 increment 6, official main): the region is a pure scalar function applied to
        each group of ``pack`` consecutive elements; its arguments are, per input, the group's elements in order, its
        results, per output, the group's elements.  The region (which may branch: cf.br / cf.cond_br) is interpreted
        group by group.  A group whose branch the reference cannot decide is not established (the region has no
        memory effects, so the program continues)."""
        if any(a.d is not None for a in args):
            raise ProgramAbort(f"{op.node_id}: no derivative rule for map_elementwise")
        pack = int(str(op.attrs.get("pack", "1")).split(":")[0].strip())
        region = op.regions[0]
        shape = np.broadcast_shapes(*[a.shape for a in args])
        flat = [a.map(lambda x, s=shape: np.broadcast_to(np.asarray(x), s).reshape(-1)) for a in args]
        n = int(np.prod(shape)) if shape else 1
        if n % pack:
            raise ProgramAbort(f"{op.node_id}: map_elementwise pack does not divide the tensor")
        n_out = len(op.result_types)
        outs = [[None] * n for _ in range(n_out)]
        failed = collections.Counter()
        for g in range(n // pack):
            region_args = [t.map(lambda x, e=g * pack + k: x[e]) for t in flat for k in range(pack)]
            try:
                vals = self._run_map_region(region, env, state, region_args)
            except ProgramAbort as exc:
                failed[str(exc).split("@")[0][:80]] += 1
                continue
            for j in range(n_out):
                for k in range(pack):
                    outs[j][g * pack + k] = vals[j * pack + k]
        results = []
        for j, rtype in enumerate(op.result_types):
            elems = outs[j]
            proto = next((v for v in elems if v is not None), None)
            kind = proto.kind if proto is not None else kind_of(rtype.elem)
            lo = np.array([0 if v is None else np.asarray(v.lo).item() for v in elems])
            hi = np.array([0.0 if v is None or v.hi is None else np.asarray(v.hi).item() for v in elems])
            st = np.array([ST_NE if v is None else int(np.asarray(v.st).item()) for v in elems], dtype=np.int8)
            cond = np.array([False if v is None else bool(np.asarray(v.cond).item()) for v in elems])
            reasons = frozenset().union(*(v.reasons for v in elems if v is not None)) if proto is not None \
                else frozenset()
            reasons = reasons | {f"not_established:map_elementwise region not decided ({r})@{op.node_id}"
                                 for r in failed}
            lo, hi, st, cond = (x.reshape(shape) for x in (lo, hi, st, cond))
            if kind == "f":
                results.append(_ftv(rtype.elem, lo.astype(np.float64), hi, st, cond, reasons))
            else:
                results.append(TV(kind, rtype.elem, lo.astype(np.int64) if kind == "i" else lo.astype(np.int8),
                                  None, None, st, cond, reasons))
        self._rules["map_elementwise.groups"] += n // pack
        return results

    def _run_map_region(self, region, env, state, args):
        """The CFG of a map_elementwise region from its entry block to tt.map_elementwise.return."""
        scope = env.new_child()
        blocks = {b.label: b for b in region.blocks}
        block = region.blocks[0]
        for (name, _), value in zip(block.args, args):
            scope[name] = value
        for _ in range(100000):
            for op in block.ops:
                if op.name in ("tt.map_elementwise.return", "cf.br", "cf.cond_br") and _TRIGGER_PATH:
                    _trace(f"{op.name}:{rule_for(op.name).internal}")
                if op.name == "tt.map_elementwise.return":
                    return [scope[v] for v in op.operands]
                if op.name in ("cf.br", "cf.cond_br"):
                    if op.name == "cf.br":
                        label, vals = op.successors[0]
                    else:
                        c = scope[op.operands[0]]
                        if np.asarray(c.st).max() >= ST_UNDEF or int(np.asarray(c.lo)) == MAYBE:
                            raise ProgramAbort(f"undecided branch condition@{op.node_id}")
                        label, vals = op.successors[0] if int(np.asarray(c.lo)) == 1 else op.successors[1]
                    values = [scope[v] for v in vals]
                    block = blocks[label]
                    for (name, _), value in zip(block.args, values):
                        scope[name] = value
                    break
                self._exec(op, scope, state)
            else:
                raise ProgramAbort("map_elementwise region ended without a return")
        raise ProgramAbort("map_elementwise region did not terminate")

    # ---- TTGIR: layouts and shared memory (DSL v2 increment 10) -----------------------------------------------

    def _op_convert_layout(self, op, args, env, state):
        """ttg.convert_layout changes only the distribution of a tensor over threads: the values are unchanged."""
        return args[0]

    def _shared_note(self):
        self._reasons["assumed:shared memory accesses of one program are ordered by the compiler's barrier insertion "
                      "(Membar, triton/backends/nvidia/compiler.py make_llir)"] += 1

    def _memdesc(self, buf_id: int, index: np.ndarray, elem) -> TV:
        index = np.asarray(index, dtype=np.int64)
        return TV("m", elem, index, None, np.full(index.shape, buf_id, dtype=np.int64))

    def _op_local_alloc(self, op, args, env, state):
        rt = op.result_types[0]
        n = int(np.prod(rt.shape)) if rt.shape else 1
        kind = kind_of(rt.elem)
        sid = len(state.shared)
        state.shared[sid] = SharedBuf(rt.elem, kind, np.zeros(n), np.zeros(n), np.full(n, ST_UNDEF, dtype=np.int8),
                                      np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.full(n, -1, dtype=np.int64),
                                      np.full(n, -1, dtype=np.int64))
        view = self._memdesc(sid, np.arange(n).reshape(rt.shape), rt.elem)
        self._shared_note()
        if args:
            self._shared_write(state, view, args[0])
        return view

    def _shared_write(self, state, view: TV, val: TV, pending: int = -1, idx_override=None):
        buf = state.shared[int(np.asarray(view.base).reshape(-1)[0])]
        idx = (np.asarray(view.lo) if idx_override is None else idx_override).reshape(-1)
        shape = np.asarray(view.lo).shape if idx_override is None else idx_override.shape
        v = val.map(lambda a, s=shape: np.broadcast_to(np.asarray(a), s).reshape(-1))
        buf.lo[idx] = v.lo
        buf.hi[idx] = v.hi if v.hi is not None else v.lo
        if v.base is not None:
            buf.base[idx] = v.base
        buf.st[idx] = v.st
        buf.cond[idx] = v.cond
        buf.written[idx] = True
        buf.pending[idx] = pending
        return buf, idx

    def _shared_read(self, op, state, view: TV, rtype, idx_override=None) -> TV:
        buf = state.shared[int(np.asarray(view.base).reshape(-1)[0])]
        idx = np.asarray(view.lo) if idx_override is None else idx_override
        lo, hi, st = buf.lo[idx], buf.hi[idx], buf.st[idx].copy()
        reasons = set()
        pending = buf.pending[idx] >= 0
        if pending.any():
            st = np.where(pending, ST_NE, st).astype(np.int8)
            reasons.add(f"not_established:shared memory read before its asynchronous copy is observed complete@"
                        f"{op.node_id}")
            self._rules["shared.read_pending_async_lanes"] += int(pending.sum())
        cond = buf.cond[idx]
        if isinstance(rtype.elem, PtrType):
            return TV("p", rtype.elem, lo.astype(np.int64), None, buf.base[idx], st, cond, frozenset(reasons))
        if kind_of(rtype.elem) == "f":
            ok = st == ST_OK
            return _ftv(rtype.elem, np.where(ok, lo, 0.0), np.where(ok, hi, 0.0), st, cond, frozenset(reasons))
        k = "b" if rtype.elem == "i1" else "i"
        return TV(k, rtype.elem, lo.astype(np.int64) if k == "i" else lo.astype(np.int8), None, None, st, cond,
                  frozenset(reasons))

    def _op_local_store(self, op, args, env, state):
        val, view = args[0], args[1]
        self._shared_write(state, view, val)
        return None

    def _op_local_load(self, op, args, env, state):
        return self._shared_read(op, state, args[0], op.result_types[0])

    def _op_memdesc_subslice(self, op, args, env, state):
        view = args[0]
        rt = op.result_types[0]
        offs = op.attrs.get("offsets")
        if offs is None:
            raise ProgramAbort(f"{op.node_id}: memdesc_subslice with dynamic offsets is not modelled")
        sl = tuple(slice(o, o + d) for o, d in zip(offs, rt.shape))
        return self._memdesc(int(np.asarray(view.base).reshape(-1)[0]), np.asarray(view.lo)[sl], rt.elem)

    def _op_memdesc_index(self, op, args, env, state):
        view, i = args[0], args[1]
        if np.asarray(i.st).max() != ST_OK:
            raise ProgramAbort(f"{op.node_id}: memdesc index not established")
        k = int(np.asarray(i.lo))
        lo = np.asarray(view.lo)
        if not 0 <= k < lo.shape[0]:
            raise ProgramAbort(f"{op.node_id}: memdesc index {k} out of range")
        return self._memdesc(int(np.asarray(view.base).reshape(-1)[0]), lo[k], op.result_types[0].elem)

    def _op_memdesc_reshape(self, op, args, env, state):
        view = args[0]
        rt = op.result_types[0]
        return self._memdesc(int(np.asarray(view.base).reshape(-1)[0]), np.asarray(view.lo).reshape(rt.shape), rt.elem)

    def _op_memdesc_reinterpret(self, op, args, env, state):
        view = args[0]
        rt = op.result_types[0]
        if rt.elem != view.elem or int(np.prod(rt.shape)) != np.asarray(view.lo).size:
            raise ProgramAbort(f"{op.node_id}: memdesc_reinterpret to another element type (a bit-level view) is not "
                               f"modelled")
        return self._memdesc(int(np.asarray(view.base).reshape(-1)[0]), np.asarray(view.lo).reshape(rt.shape), rt.elem)

    def _gather_index(self, op, view: TV, idx: TV, axis: int):
        """Storage indices of a gather / scatter along ``axis`` and the lanes whose index is outside the view."""
        lo = np.asarray(view.lo)
        k = np.asarray(idx.lo, dtype=np.int64)
        bad = (k < 0) | (k >= lo.shape[axis]) | (np.asarray(idx.st) != ST_OK)
        storage = np.take_along_axis(lo, np.clip(k, 0, lo.shape[axis] - 1), axis=axis)
        return storage, bad

    def _op_local_gather(self, op, args, env, state):
        view, idx = args[0], args[1]
        axis = int(str(op.attrs.get("axis", "0")).split(":")[0])
        storage, bad = self._gather_index(op, view, idx, axis)
        out = self._shared_read(op, state, view, op.result_types[0], idx_override=storage)
        if bad.any():
            out = TV(out.kind, out.elem, out.lo, out.hi, out.base, np.where(bad, ST_NE, out.st).astype(np.int8),
                     out.cond, out.reasons | {f"not_established:local_gather index out of range@{op.node_id}"})
        return out

    def _op_local_scatter(self, op, args, env, state):
        view, idx, val = args[0], args[1], args[2]
        axis = int(str(op.attrs.get("axis", "0")).split(":")[0])
        storage, bad = self._gather_index(op, view, idx, axis)
        flat = storage.reshape(-1)
        v = val.map(lambda a, s=storage.shape: np.broadcast_to(np.asarray(a), s))
        st = np.asarray(v.st).copy()
        uniq, counts = np.unique(flat[~bad.reshape(-1)], return_counts=True)
        dup = np.isin(storage, uniq[counts > 1])
        if dup.any():   # two lanes write one element in one scatter: unspecified which wins
            st = np.where(dup, ST_NE, st)
            self._reasons[f"not_established:local_scatter writes one element from several lanes@{op.node_id}"] += 1
        keep = ~bad
        buf = state.shared[int(np.asarray(view.base).reshape(-1)[0])]
        sel = keep.reshape(-1)
        buf.lo[flat[sel]] = np.asarray(v.lo).reshape(-1)[sel]
        buf.hi[flat[sel]] = (np.asarray(v.hi) if v.hi is not None else np.asarray(v.lo)).reshape(-1)[sel]
        if v.base is not None:
            buf.base[flat[sel]] = np.asarray(v.base).reshape(-1)[sel]
        buf.st[flat[sel]] = st.reshape(-1)[sel]
        buf.written[flat[sel]] = True
        buf.pending[flat[sel]] = -1
        if bad.any():
            self._reasons[f"not_established:local_scatter index out of range@{op.node_id}"] += 1
        return None

    def _op_local_atomic_scatter_rmw(self, op, args, env, state):
        """Atomic read-modify-write into shared memory along an axis (official Gluon): each lane returns the element's
        value before the update; lanes of one operation that share an element see an unspecified order (their returned
        values not established unless every order agrees, increment 4's rule); the update folds with the kind's law."""
        view, idx, val = args[0], args[1], args[2]
        kind = op.attrs.get("rmw_op", "")
        axis = int(str(op.attrs.get("axis", "0")).split(":")[0])
        storage, bad = self._gather_index(op, view, idx, axis)
        off = np.zeros(storage.shape, dtype=bool)
        if len(args) > 3:   # optional mask: masked-off lanes neither update nor return a value
            mk = np.broadcast_to(np.asarray(args[3].lo), storage.shape)
            if ((mk == MAYBE) | (np.broadcast_to(args[3].st, storage.shape) != ST_OK)).any():
                raise ProgramAbort(f"{op.node_id}: local atomic with an undecided mask")
            off = mk == 0
            bad = bad & ~off
        buf = state.shared[int(np.asarray(view.base).reshape(-1)[0])]
        rt = op.result_types[0]
        flat = storage.reshape(-1)
        v = val.map(lambda a, s=storage.shape: np.broadcast_to(np.asarray(a), s).reshape(-1))
        is_f = kind_of(rt.elem) == "f"
        width = INT_WIDTH.get(rt.elem, 0)
        n = flat.size
        odt = np.float64 if is_f else np.int64   # integers keep int64 precision (DSL v2 increment 14)
        old_lo, old_hi, old_st = np.zeros(n, dtype=odt), np.zeros(n, dtype=odt), np.full(n, ST_NE, dtype=np.int8)
        n_set = 0
        badf = bad.reshape(-1)
        offf = off.reshape(-1)
        groups = collections.defaultdict(list)
        for j in range(n):
            if not badf[j] and not offf[j]:
                groups[int(flat[j])].append(j)
        laws = {"add": lambda a, b: a + b, "max": max, "min": min, "and": lambda a, b: a & b,
                "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b}
        if kind not in laws and kind != "exch" and not (kind in ("fadd",) and is_f):
            raise ProgramAbort(f"{op.node_id}: local atomic {kind} is not modelled")
        why = collections.Counter()
        for a, lanes in groups.items():
            init = (float(buf.lo[a]), float(buf.hi[a]), int(buf.st[a])) if is_f else \
                (int(buf.lo[a]), int(buf.lo[a]), int(buf.st[a]))
            contribs = [(kind, (float if is_f else int)(np.asarray(v.lo)[j]),
                         (float if is_f else int)((np.asarray(v.hi) if v.hi is not None else np.asarray(v.lo))[j]),
                         int(np.asarray(v.st)[j])) for j in lanes]
            for t, j in enumerate(lanes):
                r_lo, r_hi, r_st, is_set, reason = _atomic_old(kind, is_f, width, init,
                                                                contribs[:t] + contribs[t + 1:])
                old_lo[j], old_hi[j], old_st[j] = r_lo, r_hi, r_st
                n_set += bool(is_set and r_st == ST_OK)
                if reason:
                    why[reason] += 1
            ok = init[2] == ST_OK and all(c[3] == ST_OK for c in contribs)
            if kind == "exch":
                # the last exchange of the element wins: a set over the candidates when several lanes share it
                lo_c, hi_c = [c[1] for c in contribs], [c[2] for c in contribs]
                if is_f:
                    buf.lo[a], buf.hi[a] = min(lo_c), max(hi_c)
                else:
                    buf.lo[a] = buf.hi[a] = lo_c[0]
                    ok = ok and len(set(lo_c)) == 1
                    if len(set(lo_c)) > 1:
                        why["contended integer exchange: final value is set-valued (L_E)"] += 1
                buf.st[a] = ST_OK if ok else ST_NE
                buf.written[a] = True
                continue
            if is_f:
                total_lo = [init[0]] + [c[1] for c in contribs]
                total_hi = [init[1]] + [c[2] for c in contribs]
                if kind == "fadd":
                    buf.lo[a] = float(iv.isum(np.array(total_lo), np.array(total_lo), axis=0)[0])
                    buf.hi[a] = float(iv.isum(np.array(total_hi), np.array(total_hi), axis=0)[1])
                else:
                    f = max if kind == "max" else min
                    buf.lo[a], buf.hi[a] = f(total_lo), f(total_hi)
            else:
                acc = init[0]
                for c in contribs:
                    acc = laws[kind](acc, c[1])
                buf.lo[a] = _wrap(np.array(acc, dtype=object).astype(np.int64) if abs(acc) < 2 ** 63 else
                                  np.array(acc % (1 << 64), dtype=np.uint64).view(np.int64), width) if width else acc
                buf.hi[a] = buf.lo[a]
            buf.st[a] = ST_OK if ok else ST_NE
            buf.written[a] = True
        shape = storage.shape
        old_st = np.where(offf, ST_UNDEF, old_st).astype(np.int8)
        reasons = {f"not_established:local atomic index out of range@{op.node_id}"} if bad.any() else set()
        reasons |= {f"not_established:{r}@{op.node_id}" for r in why}
        if offf.any():
            reasons.add(f"not_established:masked-off lanes of a local atomic return no value@{op.node_id}")
        if n_set:
            reasons.add(f"set:local atomic return value over all interleavings (L_E)@{op.node_id}")
        reasons = frozenset(reasons)
        cond = np.zeros(shape, dtype=bool)
        if is_f:
            return _ftv(rt.elem, old_lo.reshape(shape), old_hi.reshape(shape), old_st.reshape(shape), cond, reasons)
        return TV("i", rt.elem, old_lo.reshape(shape), old_hi.reshape(shape) if n_set else None, None,
                  old_st.reshape(shape), cond, reasons)

    # ---- TTGIR: asynchronous copies and mbarriers (DSL v2 increment 10) -------------------------------------------

    def _op_async_copy_global_to_local(self, op, args, env, state):
        """cp.async into shared memory: the values are those of tt.load with the same operands (masked-off lanes take
        ``other``, or zero as the official test asserts), but the elements stay pending until a completion is
        observed (async_wait or an mbarrier wait), and a read before that is not established."""
        ptrs, view = args[0], args[1]
        k = 2
        mask = other = None
        if op.attrs.get("has_mask"):
            mask, k = args[k], k + 1
        if op.attrs.get("has_other"):
            other = args[k]
        # the operands are those of tt.load (official TritonGPUOps.td); a masked-off lane without other is zero-filled:
        # the official test_async_copy_mbarrier (python/test/gluon/test_core.py) asserts zeros there over a shared
        # buffer initialized to 7 (the cp.async src-size-0 lowering)
        if mask is not None and other is None:
            elem = ptrs.elem.pointee
            other = (_ftv(elem, np.zeros(ptrs.shape), np.zeros(ptrs.shape), np.zeros(ptrs.shape, dtype=np.int8),
                          np.zeros(ptrs.shape, dtype=bool), frozenset()) if kind_of(elem) == "f" else
                     TV("i", elem, np.zeros(ptrs.shape, dtype=np.int64)))
            self._reasons["assumed:masked-off lanes of an async copy are zero-filled (official test_async_copy_mbarrier)"] += 1
        load_args = [ptrs] + ([mask] if mask is not None else []) + ([other] if other is not None else [])
        val = self._op_load(op, load_args, env, state)
        tag = len(state.async_open) + 1000 * len(state.async_groups) + 1
        buf, idx = self._shared_write(state, view, val, pending=tag)
        state.async_open.append((int(np.asarray(view.base).reshape(-1)[0]), idx))
        self._rules["shared.async_copies"] += 1
        return TV("i", "i32", np.array(0, dtype=np.int64))

    # ---- AMD target ops (DSL v2 increment 12; official TritonAMDGPUOps.td at e50b186e8bd2) ----

    def _buffer_operands(self, op, args):
        """Operands of an amdg buffer op by role (parser: attrs["roles"]) and the address tensor ptr + offsets: the
        offsets are element offsets of the pointee type (the lowering multiplies them by the element size).  stride,
        cachePolicy and contiguity are performance hints without a value effect."""
        roles = dict(zip(op.attrs["roles"].split(","), args))
        ptr, off = roles["ptr"], roles["offsets"]
        shape = off.shape
        splat = ptr.map(lambda a: np.broadcast_to(a, shape).copy())
        return roles, self._op_addptr(op, [splat, off], None, None)

    def _true_mask(self, shape):
        return TV("b", "i1", np.ones(shape, dtype=np.int8))

    def _op_buffer_load(self, op, args, env, state):
        """amdg.buffer_load: tt.load at ptr + offsets with mask / other.  A masked-off lane without other is
        undefined as for tt.load: the official lowering returns 0 there, the operation's description does not say."""
        roles, addr = self._buffer_operands(op, args)
        mask, other = roles.get("mask"), roles.get("other")
        if other is not None and mask is None:
            mask = self._true_mask(addr.shape)
        self._rules["amd.buffer_load"] += 1
        return self._op_load(op, [addr] + ([mask] if mask is not None else []) + ([other] if other is not None else []),
                             env, state)

    def _op_buffer_store(self, op, args, env, state):
        roles, addr = self._buffer_operands(op, args)
        self._rules["amd.buffer_store"] += 1
        return self._op_store(op, [addr, roles["value"]] + ([roles["mask"]] if "mask" in roles else []), env, state)

    def _op_buffer_atomic_rmw(self, op, args, env, state):
        roles, addr = self._buffer_operands(op, args)
        self._rules["amd.buffer_atomic_rmw"] += 1
        return self._op_atomic_rmw(op, [addr, roles["value"]] + ([roles["mask"]] if "mask" in roles else []), env,
                                   state)

    def _op_buffer_atomic_cas(self, op, args, env, state):
        roles, addr = self._buffer_operands(op, args)
        self._rules["amd.buffer_atomic_cas"] += 1
        return self._op_atomic_cas(op, [addr, roles["cmp"], roles["value"]], env, state)

    def _op_buffer_load_to_local(self, op, args, env, state):
        """amdg.buffer_load_to_local: the asynchronous copy of increment 10 (ttg.async_copy_global_to_local) with the
        buffer address ptr + offsets."""
        roles, addr = self._buffer_operands(op, args)
        copy_args = [addr, roles["dest"]] + [roles[k] for k in ("mask", "other") if k in roles]
        self._rules["amd.buffer_load_to_local"] += 1
        return self._op_async_copy_global_to_local(op, copy_args, env, state)

    def _op_in_thread_transpose(self, op, args, env, state):
        """amdg.in_thread_transpose: a register layout change inside each thread, a special case of ttg.convert_layout
        (official description): the values are unchanged."""
        return args[0]

    def _op_sched_hint(self, op, args, env, state):
        """rocdl.s.setprio (wave priority) and rocdl.sched.barrier (compiler scheduling barrier): no value or memory
        effect."""
        self._rules["amd.scheduling_hint"] += 1
        return None

    def _e8m0_exponent(self, op, scale):
        """The E8M0 exponent of an amdg.scaled_upcast scale: the exponent field of a BF16 carrier's bit pattern, or the
        byte of an i8 scale.  Returns (e, established)."""
        if scale.kind == "f":
            bits, definite = _float_bits(scale.elem, scale.lo, scale.hi, scale.st)
            e = (np.asarray(bits, dtype=np.int64) >> 7) & 0xFF
            return e, np.asarray(definite, dtype=bool)
        return np.asarray(scale.lo, dtype=np.int64) & 0xFF, np.asarray(scale.st) == ST_OK

    def _scaled_upcast(self, op, vals, x_st, x_cond, x_reasons, scale):
        rt = op.result_types[0]
        e, ok = self._e8m0_exponent(op, scale)
        e, ok = np.broadcast_to(e, rt.shape), np.broadcast_to(ok, rt.shape)
        nan_scale = ok & (e == 255)
        good = ok & ~nan_scale
        factor = np.ldexp(1.0, np.where(good, e, 127) - 127)
        x_st = np.broadcast_to(x_st, rt.shape)
        st = np.where(good, x_st, ST_NE).astype(np.int8)
        prod = np.where(st == ST_OK, vals * factor, 0.0)
        reasons = set(x_reasons | scale.reasons)
        if nan_scale.any():
            reasons.add(f"not_established:E8M0 scale 0xFF (NaN marker, replaced by the compiler's maskNan)@{op.node_id}")
        if (~ok).any():
            reasons.add(f"not_established:scale without definite bits@{op.node_id}")
        self._rules[f"amd.{op.name.split('.')[-1]}"] += 1
        out = _ftv(rt.elem, prod, prod.copy(), st, np.broadcast_to(x_cond, rt.shape) | np.broadcast_to(scale.cond, rt.shape),
                   frozenset(reasons))
        return self._maybe_round(out, op, rt.elem, False)

    def _op_scaled_upcast_fp4(self, op, args, env, state):
        """amdg.scaled_upcast_fp4: e2m1 pairs (low nibble first, along ``axis``) decoded exactly as ttg.fp4_to_fp, times
        2^(e - 127) of the E8M0 scale; the result format's rounding is checked only in the rounding-check mode."""
        x, scale = args
        rt = op.result_types[0]
        axis = int(str(op.attrs.get("axis", "0")).split(":")[0])
        raw = np.asarray(x.lo, dtype=np.int64) & 0xFF
        vals = np.stack([self._E2M1[raw & 0xF], self._E2M1[raw >> 4]], axis=axis + 1).reshape(rt.shape)
        st = np.repeat(np.asarray(x.st), 2, axis=axis).reshape(rt.shape)
        cond = np.repeat(np.broadcast_to(x.cond, np.shape(x.lo)), 2, axis=axis).reshape(rt.shape)
        return self._scaled_upcast(op, vals, st, cond, x.reasons, scale)

    def _op_scaled_upcast_fp8(self, op, args, env, state):
        """amdg.scaled_upcast_fp8: an fp8 (e4m3fn / e5m2) value times 2^(e - 127) of the E8M0 scale."""
        x, scale = args
        return self._scaled_upcast(op, np.asarray(x.lo, dtype=np.float64), x.st, x.cond, x.reasons, scale)

    def _retire(self, state, entries):
        for sid, idx in entries:
            state.shared[sid].pending[idx] = -1

    def _op_async_commit_group(self, op, args, env, state):
        state.async_groups.append(list(state.async_open))
        state.async_open.clear()
        return TV("i", "i32", np.array(0, dtype=np.int64))

    def _op_async_wait(self, op, args, env, state):
        num = int(str(op.attrs.get("num", "0")).split(":")[0])
        keep = len(state.async_groups) - num
        for g in state.async_groups[:max(0, keep)]:
            self._retire(state, g)
        state.async_groups = state.async_groups[max(0, keep):]
        return TV("i", "i32", np.array(0, dtype=np.int64))

    def _mbar(self, op, state, bar: TV):
        key = (int(np.asarray(bar.base).reshape(-1)[0]), int(np.asarray(bar.lo).reshape(-1)[0]))
        if key not in state.mbars:
            raise ProgramAbort(f"{op.node_id}: mbarrier used before init_barrier")
        return state.mbars[key]

    def _op_mbar_init(self, op, args, env, state):
        bar = args[0]
        key = (int(np.asarray(bar.base).reshape(-1)[0]), int(np.asarray(bar.lo).reshape(-1)[0]))
        count = int(op.attrs.get("count", 1))
        state.mbars[key] = {"count": count, "pending": count, "completed": 0, "attached": [], "phase_copies": []}
        return None

    def _arrive(self, m, n):
        """n arrivals on the current phase; the phase completes when its pending count reaches 0 (PTX mbarrier)."""
        m["pending"] -= n
        while m["pending"] <= 0:
            m["phase_copies"].append(list(m["attached"]))  # the copies tracked by the phase complete with it
            m["attached"] = []
            m["completed"] += 1
            m["pending"] += m["count"]

    def _op_mbar_arrive(self, op, args, env, state):
        pred = args[1] if len(args) > 1 else None
        if pred is not None and (np.asarray(pred.st).max() != ST_OK or int(np.asarray(pred.lo)) == MAYBE):
            raise ProgramAbort(f"{op.node_id}: mbarrier arrive with an undecided predicate")
        if pred is None or int(np.asarray(pred.lo)) == 1:
            self._arrive(self._mbar(op, state, args[0]), int(op.attrs.get("count", 1)))
        return None

    def _op_mbar_async_arrive(self, op, args, env, state):
        """cp.async.mbarrier.arrive (official TritonNvidiaGPUOps.td): the barrier tracks the program's earlier async
        copies.  Without noIncrement the pending count is raised by one before the asynchronous arrival (net zero), so
        the copies only gate the completion of the current phase; with noIncrement the arrival also counts."""
        m = self._mbar(op, state, args[0])
        m["attached"].extend(state.async_open)
        state.async_open.clear()
        if "noIncrement" in op.attrs:
            self._arrive(m, 1)
        return None

    def _op_mbar_wait(self, op, args, env, state):
        """mbarrier.try_wait.parity: passes when the phase of the given parity has completed, i.e. the current
        phase's parity differs from it (the operand names the current or the immediately preceding phase).  Passing
        retires the copies of every completed phase.  When the current phase has that parity, the wait needs arrivals
        that come later in program order or from other warps: not established (progress not proven)."""
        m = self._mbar(op, state, args[0])
        par = args[1]
        n_main = int(op.attrs.get("n_main", len(args)))
        pred = args[2] if n_main > 2 else None   # operands after the main ones are dependencies ("deps")
        if pred is not None and int(np.asarray(pred.lo)) == 0:
            return None
        if np.asarray(par.st).max() != ST_OK:
            raise ProgramAbort(f"{op.node_id}: mbarrier wait parity not established")
        p = int(np.asarray(par.lo)) & 1
        if m["completed"] % 2 == p:
            raise ProgramAbort(f"{op.node_id}: not_established: mbarrier phase of parity {p} does not complete within "
                               f"the program (progress not proven)")
        for copies in m["phase_copies"]:
            self._retire(state, copies)
        m["phase_copies"] = [[] for _ in m["phase_copies"]]
        return None

    # ---- TTGIR: NVIDIA Hopper / Blackwell modules (DSL v2 increment 11; device validation pending) ------------------

    _PRECISION = {"0": "tf32", "1": "tf32x3", "2": "ieee", "3": "bf16x3", "4": "bf16x6"}

    def _as_value(self, op, state, v: TV, rtype=None) -> TV:
        """A matrix operand as a value: a register tensor as is, a shared-memory descriptor read in program order."""
        if v.kind != "m":
            return v
        from .ttir_parser import TType
        return self._shared_read(op, state, v, rtype or TType(np.asarray(v.lo).shape, v.elem))

    def _dot_values(self, op, a, b, c):
        import dataclasses
        prec = str(op.attrs.get("inputPrecision", "2")).split(":")[0].strip()
        dop = dataclasses.replace(op, attrs={**op.attrs, "inputPrecision": self._PRECISION.get(prec, prec)})
        return self._op_dot(dop, [a, b, c], None, None)

    def _op_warp_group_dot(self, op, args, env, state):
        """ttng.warp_group_dot (Hopper wgmma): d = a.b + c, the tt.dot rule on the operands' values (a register tensor
        or shared memory, b shared memory read in program order; the pipeliner keeps an asynchronously read buffer
        unchanged until warp_group_dot_wait)."""
        a, b, c = (self._as_value(op, state, x) for x in args[:3])
        self._rules["nvidia.warp_group_dot"] += 1
        return self._dot_values(op, a, b, c)

    def _op_warp_group_dot_wait(self, op, args, env, state):
        """Completion of asynchronous wgmma: its results are its operands."""
        return list(args) if len(args) > 1 else args[0]

    def _op_fence_async_shared(self, op, args, env, state):
        """Orders generic-proxy shared-memory writes before async-proxy reads (wgmma, tcgen05, TMA); in the program-
        order model of shared memory it has no value effect."""
        return None

    def _op_mbar_inval(self, op, args, env, state):
        bar = args[0]
        key = (int(np.asarray(bar.base).reshape(-1)[0]), int(np.asarray(bar.lo).reshape(-1)[0]))
        state.mbars.pop(key, None)   # a later use without init_barrier is not established (_mbar)
        return None

    def _op_tmem_store(self, op, args, env, state):
        val, view = args[0], args[1]
        pred = args[2] if len(args) > 2 else None
        if pred is not None:
            if np.asarray(pred.st).max() != ST_OK or int(np.asarray(pred.lo)) == MAYBE:
                raise ProgramAbort(f"{op.node_id}: tensor-memory store with an undecided predicate")
            if int(np.asarray(pred.lo)) == 0:
                return None
        self._shared_write(state, view, val)
        return None

    def _single_cta_mma(self, op):
        """two_ctas (operands distributed over a CTA pair) and multicast tcgen05 MMAs are not modelled: the reference
        evaluates one program at a time (DSL v2 increment 13)."""
        for key in ("two_ctas", "multicast"):
            if key in op.attrs:
                raise ProgramAbort(f"{op.node_id}: tcgen05 MMA with {key} (CTA-pair data distribution) is not modelled")

    def _op_tc_gen5_mma(self, op, args, env, state):
        """ttng.tc_gen5_mma (Blackwell tcgen05, official TritonNvidiaGPUOps.td): D += A.B, D = A.B when useD is false,
        nothing when pred is false.  Asynchronous: the result is safe to read after a wait on one of its barriers,
        so D stays pending until such a wait observes the completion (later MMAs on D are ordered by the hardware)."""
        a_m, b_m, d_m, use_d, pred = args[:5]
        self._single_cta_mma(op)
        for flag in (use_d, pred):
            if np.asarray(flag.st).max() != ST_OK or int(np.asarray(flag.lo)) == MAYBE:
                raise ProgramAbort(f"{op.node_id}: tcgen05 MMA with an undecided flag")
        if int(np.asarray(pred.lo)) == 0:
            return None
        from .ttir_parser import TType
        a = self._as_value(op, state, a_m)
        b = self._as_value(op, state, b_m)
        dshape = np.asarray(d_m.lo).shape
        buf = state.shared[int(np.asarray(d_m.base).reshape(-1)[0])]
        if int(np.asarray(use_d.lo)) == 1:
            idx = np.asarray(d_m.lo)
            c = _ftv(d_m.elem, buf.lo[idx], buf.hi[idx], buf.st[idx], buf.cond[idx], frozenset()) \
                if kind_of(d_m.elem) == "f" else TV("i", d_m.elem, buf.lo[idx].astype(np.int64), None, None, buf.st[idx])
        else:
            c = _ftv(d_m.elem, np.zeros(dshape), np.zeros(dshape), np.zeros(dshape, dtype=np.int8),
                     np.zeros(dshape, dtype=bool), frozenset()) if kind_of(d_m.elem) == "f" else \
                TV("i", d_m.elem, np.zeros(dshape, dtype=np.int64))
        d = self._dot_values(op, a, b, c)
        is_async = "is_async" in op.attrs
        nb = int(op.attrs.get("n_barriers", 0))
        tag = 1 if is_async else -1
        _, idx = self._shared_write(state, d_m, d, pending=tag)
        sid = int(np.asarray(d_m.base).reshape(-1)[0])
        for k in range(nb):
            bar, bpred = args[5 + 2 * k], args[6 + 2 * k]
            if int(np.asarray(bpred.lo)) == 1:
                m = self._mbar(op, state, bar)
                m["attached"].append((sid, idx))
                self._arrive(m, 1)   # tcgen05.commit arrives once the MMA completes
        self._rules["nvidia.tc_gen5_mma"] += 1
        return None

    def _op_tc_gen5_mma_scaled(self, op, args, env, state):
        """ttng.tc_gen5_mma_scaled (DSL v2 increment 13, official TritonNvidiaGPUOps.td): D += scale(A, a_scale) .
        scale(B, b_scale), decoded as tt.dot_scaled; useD, pred, barriers and completion as ttng.tc_gen5_mma."""
        a_m, b_m, d_m, sa_m, sb_m, use_d, pred = args[:7]
        self._single_cta_mma(op)
        for flag in (use_d, pred):
            if np.asarray(flag.st).max() != ST_OK or int(np.asarray(flag.lo)) == MAYBE:
                raise ProgramAbort(f"{op.node_id}: tcgen05 MMA with an undecided flag")
        if int(np.asarray(pred.lo)) == 0:
            return None
        a, b = self._as_value(op, state, a_m), self._as_value(op, state, b_m)
        sa, sb = self._as_value(op, state, sa_m), self._as_value(op, state, sb_m)
        dshape = np.asarray(d_m.lo).shape
        buf = state.shared[int(np.asarray(d_m.base).reshape(-1)[0])]
        if int(np.asarray(use_d.lo)) == 1:
            idx = np.asarray(d_m.lo)
            c = _ftv(d_m.elem, buf.lo[idx], buf.hi[idx], buf.st[idx], buf.cond[idx], frozenset())
        else:
            c = _ftv(d_m.elem, np.zeros(dshape), np.zeros(dshape), np.zeros(dshape, dtype=np.int8),
                     np.zeros(dshape, dtype=bool), frozenset())
        d = self._scaled_dot(op, a, sa, b, sb, c, op.attrs.get("lhs"), op.attrs.get("rhs"))
        tag = 1 if "is_async" in op.attrs else -1
        _, idx = self._shared_write(state, d_m, d, pending=tag)
        sid = int(np.asarray(d_m.base).reshape(-1)[0])
        for k in range(int(op.attrs.get("n_barriers", 0))):
            bar, bpred = args[7 + 2 * k], args[8 + 2 * k]
            if int(np.asarray(bpred.lo)) == 1:
                m = self._mbar(op, state, bar)
                m["attached"].append((sid, idx))
                self._arrive(m, 1)   # tcgen05.commit arrives once the MMA completes
        self._rules["nvidia.tc_gen5_mma_scaled"] += 1
        return None

    def _op_memdesc_trans(self, op, args, env, state):
        view = args[0]
        text = str(op.attrs.get("order", ""))
        mm = re.search(r"array<i\d+\s*:\s*([^>]*)>", text)
        order = [int(x) for x in re.findall(r"-?\d+", mm.group(1) if mm else text)]
        lo = np.asarray(view.lo)
        if sorted(order) != list(range(lo.ndim)):
            raise ProgramAbort(f"{op.node_id}: memdesc_trans order {order} not understood")
        return self._memdesc(int(np.asarray(view.base).reshape(-1)[0]), np.transpose(lo, order), view.elem)

    _E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

    def _op_fp4_to_fp(self, op, args, env, state):
        """ttg.fp4_to_fp: two e2m1 values per byte, low nibble first, along ``axis``; decoded exactly."""
        x = args[0]
        rt = op.result_types[0]
        axis = int(str(op.attrs.get("axis", "0")).split(":")[0])
        raw = np.asarray(x.lo, dtype=np.int64) & 0xFF
        lo_n, hi_n = self._E2M1[raw & 0xF], self._E2M1[raw >> 4]
        vals = np.stack([lo_n, hi_n], axis=axis + 1).reshape(rt.shape)
        st = np.repeat(np.asarray(x.st), 2, axis=axis).reshape(rt.shape)
        return _ftv(rt.elem, vals, vals.copy(), st.astype(np.int8), np.repeat(np.asarray(x.cond), 2, axis=axis).reshape(rt.shape),
                    x.reasons)

    def _op_histogram(self, op, args, env, state):
        """tt.histogram (DSL v2 increment 5, rc3 02 6.8): exact counts in bins of width 1 starting at 0.  Inputs
        outside [0, bins) are dropped (official contract: python/triton/runtime/interpreter.py "The GPU drops every
        out-of-range value", and test_histogram_out_of_range); masked-off inputs are not counted.  A counted position
        whose input has no established value (its bin, or whether it is dropped, is unknown) leaves every bin not
        established."""
        src = args[0]
        mask = args[1] if len(args) > 1 else None
        out = op.result_types[0]
        nb = int(out.shape[0])
        vals = np.asarray(src.lo, dtype=np.int64).reshape(-1)
        s_st = np.broadcast_to(src.st, np.shape(src.lo)).reshape(-1)
        if mask is not None:
            m = np.broadcast_to(mask.lo, np.shape(src.lo)).reshape(-1)
            m_st = np.broadcast_to(mask.st, np.shape(src.lo)).reshape(-1)
            undecided = ((m == MAYBE) | (m_st != ST_OK)).any()
        else:
            m, undecided = np.ones(vals.shape, dtype=np.int8), False
        bad = (m == 1) & (s_st != ST_OK)
        counted = (m == 1) & (s_st == ST_OK) & (vals >= 0) & (vals < nb)
        if ((m == 1) & (s_st == ST_OK) & ~counted).any():
            self._rules["histogram.dropped_out_of_range_inputs"] += int(((m == 1) & (s_st == ST_OK) & ~counted).sum())
        reasons = set(src.reasons | (mask.reasons if mask is not None else frozenset()))
        cond = np.full(nb, bool(np.any(src.cond) or (mask is not None and np.any(mask.cond))))
        if bad.any() or undecided:
            reasons.add(f"not_established:histogram input without an established value or undecided mask@{op.node_id}")
            return TV("i", out.elem, np.zeros(nb, dtype=np.int64), None, None, np.full(nb, ST_NE, dtype=np.int8),
                      cond, frozenset(reasons))
        counts = np.bincount(vals[counted], minlength=nb).astype(np.int64)
        self._rules["histogram.exact_counts"] += 1
        return TV("i", out.elem, _wrap(counts, INT_WIDTH[out.elem]), None, None, np.zeros(nb, dtype=np.int8), cond,
                  frozenset(reasons))

    def _op_math_clampf(self, op, args, env, state):
        """math.clampf (DSL v2 increment 8): clampf(v, min, max) = maxf(minf(v, max), min), poison when min > max
        (MLIR MathOps.td).  The NaN behaviour of its maxf / minf is not verified here, so a NaN operand is not
        established; min > max (poison) is not established; otherwise the clamp is unambiguous."""
        x, lo_b, hi_b = args
        out = _clamp(dataclasses.replace(op, attrs={**op.attrs, "propagateNan": "none"}), args)
        nan = (x.st == ST_NAN) | (lo_b.st == ST_NAN) | (hi_b.st == ST_NAN)
        poison = (lo_b.st == ST_OK) & (hi_b.st == ST_OK) & (np.asarray(lo_b.lo) > np.asarray(hi_b.hi))
        undecided = (lo_b.st == ST_OK) & (hi_b.st == ST_OK) & ~poison & (np.asarray(lo_b.hi) > np.asarray(hi_b.lo))
        bad = np.broadcast_to(nan | poison | undecided, out.shape)
        if bad.any():
            out = TV(out.kind, out.elem, out.lo, out.hi, out.base, np.where(bad, ST_NE, out.st).astype(np.int8),
                     out.cond, out.reasons | {f"not_established:math.clampf with NaN or min > max (poison)@{op.node_id}"})
        return out

    def _op_approx_div(self, op, args, env, state):
        """tt.approx_divf and inline asm div.full.f32 (DSL v2 increment 5, rc3 02 6.5): an approximate instruction
        with a documented target function is that function in the numerical-difference mode (x / y, the arith.divf
        rule); the rounding-check mode needs its error contract, not implemented."""
        if self.mode == NumericMode.ROUNDING_CHECK:
            raise ProgramAbort(f"{op.node_id}: approximate division has no implemented error contract (rounding-check "
                               f"mode)")
        return self._float_op("div", op, args, op.result_types[0].elem)

    def _atomic_returns(self, op, buf, idx, lanes, occ, shape, active, value, bitview, state):
        """Pass 2: the old value each lane of this atomic returns, from pass 1's table of every update of the address
        in the launch and the memory before the launch (``_atomic_old``)."""
        kind_name = op.attrs["rmw_op"]
        is_float = value.kind == "f" and not bitview
        width = INT_WIDTH.get(value.elem, 0)
        init_buf = self._atomic_init[buf.ident]
        if bitview:
            init_bits, init_def = _float_bits(init_buf.elem, init_buf.lo, init_buf.hi, init_buf.st)
        plain_prev = self._plain_prev.get(buf.ident)
        rdt = np.float64 if is_float else np.int64   # integers keep int64 precision (DSL v2 increment 14)
        lo = np.zeros(idx.size, dtype=rdt)
        hi = np.zeros(idx.size, dtype=rdt)
        st = np.full(idx.size, ST_NE, dtype=np.int8)
        why = collections.Counter()
        n_set = 0
        for j, (a, lane) in enumerate(zip(idx.tolist(), lanes.tolist())):
            self._consumers[(buf.ident, a)].add(state.pid_index)
            evs = self._atomic_table.get((buf.ident, a), [])
            key = (state.pid_index, op.node_id, occ, lane)
            if not any(e[0] == key for e in evs):
                why["the update is missing from the first pass"] += 1
                continue
            if plain_prev is not None and plain_prev[a]:
                why["returned value after a non-atomic write of the same launch"] += 1
                continue
            if len(evs) > ATOMIC_ADDRESS_BUDGET:
                why["too many updates of one address for the set bound (budget)"] += 1
                continue
            others = [e[1:] for e in evs if e[0] != key]
            if bitview:
                init = (int(init_bits[a]), int(init_bits[a]), ST_OK if init_def[a] else ST_NE)
            elif is_float:
                init = (float(init_buf.lo[a]), float(init_buf.hi[a]), int(init_buf.st[a]))
            else:
                init = (int(init_buf.lo[a]), int(init_buf.lo[a]), int(init_buf.st[a]))
            r_lo, r_hi, r_st, is_set, reason = _atomic_old(kind_name, is_float, width, init, others)
            if reason:
                why[reason] += 1
            lo[j], hi[j], st[j] = r_lo, r_hi, r_st
            n_set += bool(is_set and r_st == ST_OK)
        reasons = set(value.reasons) - {r for r in value.reasons if "atomic return value" in r}
        for r in why:
            reasons.add(f"not_established:{r}@{op.node_id}")
        if n_set:
            reasons.add(f"set:atomic return value over all interleavings (L_E)@{op.node_id}")
        self._rules["atomic.return_set_lanes"] += n_set
        self._rules["atomic.return_point_lanes"] += int((st == ST_OK).sum()) - n_set
        self._rules["atomic.return_value_not_established_lanes"] += int((st != ST_OK).sum())
        out_lo = np.zeros(shape, dtype=rdt)
        out_hi = np.zeros(shape, dtype=rdt)
        out_st = np.full(shape, ST_NE, dtype=np.int8)
        out_lo[active], out_hi[active], out_st[active] = lo, hi, st
        cond = np.zeros(shape, dtype=bool)
        if is_float:
            return _ftv(value.elem, out_lo, out_hi, out_st, cond, frozenset(reasons))
        return TV("i", value.elem, out_lo, out_hi if n_set else None, None, out_st, cond, frozenset(reasons))

    def _event_table(self) -> dict:
        """(ident, address) -> sorted [(key, kind, lo, hi, st)] of the atomic updates recorded in this run."""
        table = collections.defaultdict(list)
        for ident, idx, kind, c_lo, c_hi, c_st, pid, node, occ, lanes in self._events:
            c_lo = np.asarray(c_lo)
            for j, a in enumerate(idx.tolist()):
                lo = c_lo[j].item()
                hi = lo if c_hi is None else np.asarray(c_hi)[j].item()
                table[(ident, a)].append(((pid, node, occ, int(lanes[j])), kind, lo, hi, int(c_st[j])))
        for v in table.values():
            v.sort(key=lambda e: e[0])
        return dict(table)

    def _check_two_pass(self, table1, memory, reasons):
        """Returned values are valid only if pass 2 performed exactly the updates of pass 1 on their addresses;
        otherwise the programs that used them and every atomically written address lose their reference value."""
        table2 = self._event_table()
        bad = set()
        for key, programs in self._consumers.items():
            if table1.get(key) != table2.get(key):
                bad |= programs
        self._rules["atomic.two_pass_launch"] += 1
        self._two_pass_bad = set(bad)
        if not bad:
            return
        hits = 0
        for buf in memory.values():
            hit = np.isin(buf.writer, sorted(bad)) | (buf.writer == -2)
            if hit.any():
                buf.st = np.where(hit, ST_NE, buf.st).astype(np.int8)
                hits += int(hit.sum())
        reasons["not_established:atomic updates differ between the two evaluation passes"] += hits
        self._rules["atomic.two_pass_mismatch_programs"] += len(bad)

    def _atomic_final_laws(self, memory, reasons):
        """rc3 02 6.3: the final value of an address several atomic updates of the launch reach is a fold only under an
        order law: a single commutative kind, or a mixed pair with a checked joint law (_JOINT_ORDER_LAWS).  Contended
        exchanges keep the set of candidates (L_E).  Anything else is not established."""
        by_ident = collections.defaultdict(list)
        for ev in self._events:
            by_ident[ev[0]].append(ev)
        for ident, evs in by_ident.items():
            if all(e[2] != "exch" for e in evs) and len({e[2] for e in evs}) == 1:
                continue
            buf = memory[ident]
            idx = np.concatenate([e[1] for e in evs])
            kinds = np.concatenate([np.full(e[1].size, e[2], dtype=object) for e in evs])
            is_f = [buf.kind == "f" and e[4] is not None for e in evs]
            c_lo = np.concatenate([np.asarray(e[3], dtype=np.float64 if f else np.int64).reshape(-1)
                                   for e, f in zip(evs, is_f)])
            c_hi = np.concatenate([np.asarray(e[4] if f else e[3], dtype=np.float64 if f else np.int64).reshape(-1)
                                   for e, f in zip(evs, is_f)])
            float_c = np.concatenate([np.full(e[1].size, f) for e, f in zip(evs, is_f)])
            c_st = np.concatenate([np.asarray(e[5]).reshape(-1) for e in evs])
            order = np.argsort(idx, kind="stable")
            idx, kinds, c_lo, c_hi, float_c, c_st = (x[order] for x in (idx, kinds, c_lo, c_hi, float_c, c_st))
            starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
            ends = np.r_[starts[1:], idx.size]
            for a0, a1 in zip(starts.tolist(), ends.tolist()):
                ks = set(kinds[a0:a1].tolist())
                if len(ks) == 1 and ("exch" not in ks or a1 - a0 == 1):
                    continue
                a = int(idx[a0])
                if ks == {"exch"}:
                    if (c_st[a0:a1] != ST_OK).any():
                        buf.st[a] = ST_NE
                    elif float_c[a0:a1].all():
                        lo, hi = float(c_lo[a0:a1].min()), float(c_hi[a0:a1].max())
                        buf.lo[a], buf.hi[a], buf.st[a] = lo, hi, ST_OK
                        if lo != hi:
                            reasons["set:atomic exchange final value over all interleavings (L_E)"] += 1
                            self._rules["atomic.exchange_set_addresses"] += 1
                    elif len(set(c_lo[a0:a1].tolist())) > 1:
                        buf.st[a] = ST_NE
                        reasons["not_established:contended integer exchange: final value is set-valued (L_E)"] += 1
                    continue
                law = next((signs for kset, signs in _JOINT_ORDER_LAWS if ks == kset), None)
                ok = law is not None and (c_st[a0:a1] == ST_OK).all() and not float_c[a0:a1].any()
                if ok:
                    k_seg, c_seg = kinds[a0:a1], c_lo[a0:a1]
                    ok = all(((c_seg[k_seg == k] >= 0) if sign > 0 else (c_seg[k_seg == k] < 0)).all()
                             for k, sign in law.items())
                if ok:
                    self._rules["atomic.joint_order_law_addresses"] += 1
                else:
                    buf.st[a] = ST_NE
                    reasons["not_established:mixed atomic kinds on one address without a joint order law"] += 1

    # ---- control flow -----------------------------------------------------------

    def _op_for(self, op, args, env, state):
        bounds = [env[v] for v in op.operands[:3]]
        lb, ub, step = (self._scalar_int(b, op) for b in bounds)
        carried = [env[v] for v in op.operands[3:]]
        region = op.regions[0]
        bound_flag = any(b.cond.any() for b in bounds)
        bound_reasons = frozenset().union(*(b.reasons for b in bounds))
        state.ctrl.append((bound_flag, bound_reasons))
        state.loop_depth += 1
        try:
            i = lb
            count = 0
            while (step > 0 and i < ub) or (step < 0 and i > ub):
                iv_value = TV("i", op.attrs["iv_type"].elem, np.array(i, dtype=np.int64))
                _, carried = self._run_region(region, env, state, [iv_value] + carried)
                i += step
                count += 1
                if count > 10_000_000:
                    raise ProgramAbort("loop bound too large")
        finally:
            state.ctrl.pop()
            state.loop_depth -= 1
        if bound_flag:
            carried = [TV(v.kind, v.elem, v.lo, v.hi, v.base, v.st, v.cond | True, v.reasons | bound_reasons, v.d)
                       for v in carried]
        return carried

    def _scalar_int(self, value: TV, op) -> int:
        if value.st.max() >= ST_UNDEF or value.kind not in ("i", "b"):
            raise ProgramAbort(f"{op.node_id}: loop bound not established")
        if value.hi is not None and np.any(np.asarray(value.hi) != np.asarray(value.lo)):
            raise ProgramAbort(f"{op.node_id}: loop bound is an integer set target (L_E)")
        return int(value.lo)

    def _op_if(self, op, args, env, state):
        cond = env[op.operands[0]]
        if _TRIGGER_PATH and state.loop_depth > 0:
            _trace("scf.for/scf.if (nested):control")
        if cond.st.max() >= ST_UNDEF:
            raise ProgramAbort(f"{op.node_id}: undefined branch condition")
        c = int(cond.lo)
        if c in (0, 1):
            self._rules["path.branch_decided_by_reference"] += 1
            region = op.regions[0] if c == 1 else (op.regions[1] if len(op.regions) > 1 else None)
            if region is None:
                return []
            state.ctrl.append((bool(cond.cond.any()), cond.reasons))
            try:
                _, out = self._run_region(region, env, state, [])
            finally:
                state.ctrl.pop()
            return [self._with_cond(v, cond) for v in out]
        # Undecided: run both branches on copies of the memory and take the union.
        self._rules["path.branch_union"] += 1
        saved = {k: b.copy() for k, b in state.memory.items()}
        saved_recs = {k: dict(v) for k, v in self._iset_records.items()}
        state.ctrl.append((bool(cond.cond.any()), cond.reasons))
        try:
            _, then_out = self._run_region(op.regions[0], env, state, [])
            then_mem = state.memory
            then_recs = self._iset_records
            self._iset_records = {k: dict(v) for k, v in saved_recs.items()}
            state.memory = {k: b.copy() for k, b in saved.items()}
            else_out = []
            if len(op.regions) > 1:
                _, else_out = self._run_region(op.regions[1], env, state, [])
            else_mem = state.memory
        finally:
            state.ctrl.pop()
        # integer set targets (increment 14): a record stands only if both branches leave the same record
        else_recs = self._iset_records
        self._iset_records = {}
        for ident in set(then_recs) | set(else_recs):
            a, b = then_recs.get(ident, {}), else_recs.get(ident, {})
            kept = {e: r for e, r in a.items() if b.get(e) == r}
            if kept:
                self._iset_records[ident] = kept
        state.memory = then_mem
        for k in state.memory:
            state.memory[k] = _hull_buffers(then_mem[k], else_mem[k])
        info = f"path_union:{op.node_id}"
        return [self._with_cond(_hull_tv(a, b).with_reason(info), cond) for a, b in zip(then_out, else_out)]

    def _with_cond(self, value: TV, cond: TV) -> TV:
        if not cond.cond.any() and not cond.reasons:
            return value
        return TV(value.kind, value.elem, value.lo, value.hi, value.base, value.st,
                  value.cond | bool(cond.cond.any()), value.reasons | cond.reasons)

    def _op_while(self, op, args, env, state):
        carried = [env[v] for v in op.operands]
        before, after = op.regions
        from .ttir_parser import _walk_region
        spin = any(o.name in ("tt.atomic_cas", "tt.atomic_load", "amdg.buffer_atomic_cas")
                   for r in op.regions for o in _walk_region(r))
        for it in range(10_000_000):
            if spin and it >= 10_000:
                # DSL v2 increment 7: in the serialization the earlier programs have finished; a spin loop that has
                # not acquired by now waits for a program that never releases in this order: progress not proven
                raise ProgramAbort(f"{op.node_id}: not_established: spin loop does not acquire in the reference "
                                   f"serialization (progress not proven)")
            term, vals = self._run_region(before, env, state, carried)
            c = vals[0]
            if c.st.max() >= ST_UNDEF or int(c.lo) == MAYBE:
                raise ProgramAbort(f"{op.node_id}: undecided loop condition")
            if c.cond.any():
                state.sticky_cond = True
                state.sticky_reasons = state.sticky_reasons | c.reasons
            if int(c.lo) == 0:
                if state.sticky_cond:
                    return [TV(v.kind, v.elem, v.lo, v.hi, v.base, v.st, v.cond | True, v.reasons, v.d)
                            for v in vals[1:]]
                return vals[1:]
            _, carried = self._run_region(after, env, state, vals[1:])
        raise ProgramAbort("while loop did not terminate")

    def _op_call(self, op, args, env, state):
        callee = self.module.funcs.get(op.attrs["callee"])
        if callee is None:
            raise ProgramAbort(f"{op.node_id}: unknown callee {op.attrs['callee']}")
        return self._call(callee, args, state)

    def _op_condition(self, op, args, env, state):  # handled by _run_region
        return None

    def _op_yield(self, op, args, env, state):
        return None

    def _op_region_return(self, op, args, env, state):
        return None

    # ---- reductions -------------------------------------------------------------

    def _op_reduce(self, op, args, env, state):
        axis = int(op.attrs["axis"].split(":")[0])
        combiner = recognize_combiner(op)
        if _TRIGGER_PATH:
            _trace(f"tt.reduce/{combiner or 'other'}")
        if combiner is None:  # any combine region: interpret it along the lowering's combination order
            return self._tree_reduce(op, args, axis, env, state, "no order-free fast path matches the region")
        if combiner in ("argmax", "argmin"):
            # DSL v2 increment 6: with NaN (or another non-finite value) the official combine (v1 > v2, a where) is
            # not order free -- NaN on the left is dropped, on the right it propagates; the lowering tree decides
            nonfinite = any((np.asarray(a.st) != ST_OK).any() for a in args)
            if nonfinite and op.node_id in self._ttgir_layouts()[1]:
                return self._tree_reduce(op, args, axis, env, state,
                                         "argmax / argmin over non-finite values: the combine is order dependent")
            return self._arg_reduce(op, args, axis, combiner)
        if combiner == "welford":
            out = self._welford_reduce(op, args, axis, env)
            # the closed form (Phi certificate) is the fast path; rows it leaves open because the result depends on
            # the merge tree (zero weights) are defined by the actual tree when the TTGIR gives it
            open_rows = any(("depends on the merge tree" in r or "unguarded Welford ratio" in r)
                            for r in out[0].reasons | out[1].reasons)
            if open_rows and op.node_id in self._ttgir_layouts()[1]:
                return self._tree_reduce(op, args, axis, env, state, "Welford closed form depends on the merge tree")
            return out
        (x,) = args
        st_red = np.max(np.where(x.st >= ST_UNDEF, x.st, 0), axis=axis).astype(np.int8)
        cond = np.any(x.cond, axis=axis)
        reasons = x.reasons
        if x.kind == "f":
            finite = x.st == ST_OK
            lo0 = np.where(finite, x.lo, 0.0)
            hi0 = np.where(finite, x.hi, 0.0)
            any_nan = np.any(x.st == ST_NAN, axis=axis)
            any_pinf = np.any(x.st == ST_PINF, axis=axis)
            any_ninf = np.any(x.st == ST_NINF, axis=axis)
            if combiner in ("sum", "prod") and self.mode == NumericMode.ROUNDING_CHECK:
                raise ProgramAbort(f"{op.node_id}: reduction order is not declared (rounding-check mode)")
            if combiner == "sum":
                lo, hi = iv.isum(lo0, hi0, axis)
                special = np.where(any_nan | (any_pinf & any_ninf), ST_NAN,
                                   np.where(any_pinf, ST_PINF, np.where(any_ninf, ST_NINF, ST_OK)))
            elif combiner == "prod":
                lo, hi = np.take(lo0, 0, axis=axis), np.take(hi0, 0, axis=axis)
                for k in range(1, x.shape[axis]):
                    lo, hi = iv.imul(lo, hi, np.take(lo0, k, axis=axis), np.take(hi0, k, axis=axis))
                special = np.where(any_nan | any_pinf | any_ninf, ST_NE, ST_OK)
            elif combiner.startswith(("max", "min")):
                is_max = combiner.startswith("max")
                big = np.inf if is_max else -np.inf
                lo_x = np.where(x.st == ST_PINF, np.inf, np.where(x.st == ST_NINF, -np.inf, lo0))
                hi_x = np.where(x.st == ST_PINF, np.inf, np.where(x.st == ST_NINF, -np.inf, hi0))
                fill = -big
                lo_x = np.where(x.st == ST_NAN, fill, lo_x)
                hi_x = np.where(x.st == ST_NAN, fill, hi_x)
                red = np.max if is_max else np.min
                lo, hi = red(lo_x, axis=axis), red(hi_x, axis=axis)
                special = np.where(lo == np.inf, ST_PINF, np.where(hi == -np.inf, ST_NINF, ST_OK))
                special = np.where((lo == np.inf) != (hi == np.inf), ST_NE, special)
                special = np.where((lo == -np.inf) != (hi == -np.inf), ST_NE, special)
                all_nan = np.all(x.st == ST_NAN, axis=axis)
                if combiner.endswith("_nan"):
                    special = np.where(any_nan, ST_NAN, special)
                elif combiner.endswith("_num"):
                    special = np.where(all_nan, ST_NAN, special)
                else:  # select-based: NaN handling depends on the combination order
                    special = np.where(any_nan, ST_NE, special)
                lo = np.where(np.isfinite(lo), lo, 0.0)
                hi = np.where(np.isfinite(hi), hi, 0.0)
            else:
                raise ProgramAbort(f"{op.node_id}: float combiner {combiner}")
            st = np.where(st_red >= ST_UNDEF, st_red, special).astype(np.int8)
            out = _ftv(x.elem, lo, hi, st, cond, reasons)
            if x.d is not None:
                t_lo, t_hi = (np.where(finite, t, 0.0) for t in x.tangent())
                if combiner == "sum":
                    d = iv.isum(t_lo, t_hi, axis)
                elif combiner.startswith(("max", "min")):
                    # Tangent of the attaining element; hull over every element that may attain it.
                    lo_x = np.where(finite, x.lo, -np.inf if combiner.startswith("max") else np.inf)
                    hi_x = np.where(finite, x.hi, -np.inf if combiner.startswith("max") else np.inf)
                    if combiner.startswith("max"):
                        cand = hi_x >= np.expand_dims(np.max(lo_x, axis=axis), axis)
                    else:
                        cand = lo_x <= np.expand_dims(np.min(hi_x, axis=axis), axis)
                    d = (np.min(np.where(cand, t_lo, np.inf), axis=axis), np.max(np.where(cand, t_hi, -np.inf), axis=axis))
                    if (np.sum(cand, axis=axis) > 1).any():
                        out.reasons = out.reasons | {f"nonsmooth_union:{op.node_id}"}
                else:
                    raise ProgramAbort(f"{op.node_id}: no derivative rule for {combiner} reduction")
                out.d = (np.where(st == ST_OK, d[0], 0.0), np.where(st == ST_OK, d[1], 0.0))
            return out
        width = INT_WIDTH[x.elem]
        vals = x.lo.astype(np.int64)
        if x.kind == "b":
            if combiner == "and":
                v = np.where(np.all(vals == 1, axis=axis), 1, np.where(np.any(vals == 0, axis=axis), 0, MAYBE))
            elif combiner == "or":
                v = np.where(np.any(vals == 1, axis=axis), 1, np.where(np.all(vals == 0, axis=axis), 0, MAYBE))
            else:
                if (vals == MAYBE).any():
                    raise ProgramAbort(f"{op.node_id}: xor of undecided booleans")
                v = np.bitwise_xor.reduce(vals, axis=axis)
            return TV("b", "i1", v.astype(np.int8), None, None, st_red, cond, reasons)
        fn = {"sum_int": np.sum, "prod_int": np.prod, "max_int": np.max, "min_int": np.min,
              "and": np.bitwise_and.reduce, "or": np.bitwise_or.reduce, "xor": np.bitwise_xor.reduce}.get(combiner)
        if combiner in ("max_uint", "min_uint"):
            u = _unsigned(vals, width)
            v = (np.max if combiner == "max_uint" else np.min)(u, axis=axis).astype(np.int64)
            return TV("i", x.elem, _wrap(v, width), None, None, st_red, cond, reasons)
        if fn is None:
            raise ProgramAbort(f"{op.node_id}: integer combiner {combiner}")
        v = fn(vals, axis=axis)
        return TV("i", x.elem, _wrap(v, width), None, None, st_red, cond, reasons)

    def _tree_reduce(self, op, args, axis, env, state, why):
        """A reduction with an arbitrary combine region, interpreted step by step along the combination order of the
        locked lowering (DSL v2 rc3 02 6.2: with a trusted order the target is order-specific).  Order: sequential
        within a thread in register order, butterfly over the lanes along the axis, then over the warps (layout from
        the captured TTGIR; the same model the bitwise emulator checks against the device).  A warp-synchronous
        result (one warp along the axis) is held by every lane: its enclosure is the hull over the lanes."""

        if self.mode == NumericMode.ROUNDING_CHECK:
            raise ProgramAbort(f"{op.node_id}: combine region order is not modelled in rounding-check mode")
        lay = self._ttgir_layouts()[1].get(op.node_id)
        if lay is None:
            raise ProgramAbort(f"{op.node_id}: unrecognized reduction combiner ({why}) and no trusted order: the reduce "
                               "layout is not available from the TTGIR")
        shape = args[0].shape
        n = shape[axis]
        spt = min(lay["spt"], n)
        tpw = min(lay["tpw"], max(1, n // spt))
        wpc = min(lay["wpc"], max(1, n // (spt * tpw)))
        tile = spt * tpw * wpc
        if n % tile:
            raise ProgramAbort(f"{op.node_id}: reduce layout does not tile the reduced axis ({n} vs {tile})")
        reps = n // tile
        region = op.regions[0]
        full = [x.map(lambda a, s=x.shape: np.moveaxis(np.broadcast_to(np.asarray(a), s), axis, -1)) for x in args]
        idx = (np.arange(reps)[:, None, None, None] * tile + np.arange(wpc)[None, :, None, None] * (spt * tpw)
               + np.arange(tpw)[None, None, :, None] * spt + np.arange(spt)[None, None, None, :])
        order = [idx[r, :, :, j] for r in range(reps) for j in range(spt)]

        def pick(t, pos):
            return t.map(lambda a: a[..., pos])

        def combine(a, b):
            _, out = self._run_region(region, env, state, list(a) + list(b))
            return out

        acc = [pick(t, order[0]) for t in full]          # [..., warp, lane]
        for pos in order[1:]:
            acc = combine(acc, [pick(t, pos) for t in full])
        lanes = np.arange(tpw)
        stride = tpw // 2
        while stride >= 1:
            acc = combine(acc, [pick(t, lanes ^ stride) for t in acc])
            stride //= 2
        if wpc > 1:
            part = [t.map(lambda a: a[..., 0]) for t in acc]   # lane 0 of each warp writes the warp partial
            warps = np.arange(wpc)
            stride = wpc // 2
            while stride >= 1:
                part = combine(part, [pick(t, warps ^ stride) for t in part])
                stride //= 2
            res = [t.map(lambda a: a[..., 0]) for t in part]   # broadcast through shared memory
        else:
            res = [_lane_hull(t.map(lambda a: a[..., 0, :])) for t in acc]
        note = (f"assumed:combine order = Triton 3.6.0 reduce lowering tree from the TTGIR layout (spt={spt}, "
                f"tpw={tpw}, wpc={wpc}; model checked bit-exactly for float sums)@{op.node_id}")
        self._rules["reduce.generic_region_lowering_order"] += int(np.prod(res[0].shape)) if res[0].shape else 1
        res = [r.with_reason(note) for r in res]
        return res if len(res) > 1 else res[0]

    def _welford_reduce(self, op, args, axis, env):
        """Reduction with the Welford merge of (mean, M2, weight) triples (``ttir_mapping.match_welford``).

        Real semantics (rule ``reduce.welford``): for weights w_i >= 0 with W = sum w_i > 0 the merge of any tree
        equals the closed form  mean = sum w_i m_i / W,  M2 = sum s_i + sum w_i (m_i - mean)^2,  weight = W  (the
        pairwise merge is the exact parallel-axis identity, and a zero-weight triple only adds its s).  The tree
        Triton uses is therefore irrelevant.  Rows with W = 0: the merged mean depends on the tree -> not
        established; M2 = sum s_i.  Rows whose weights may be negative or whose W may be zero -> not established.
        Enclosure: interval arithmetic on the closed form, M2 in the centred form sum w d^2 - (sum w d)^2 / W with
        d = m - c around a float estimate c of the mean (an exact identity for every c)."""

        if self.mode == NumericMode.ROUNDING_CHECK:
            raise ProgramAbort(f"{op.node_id}: Welford merge order is not declared (rounding-check mode)")
        info = match_welford(op)
        guarded = bool(info["zero_constants"])
        for name in info["zero_constants"]:
            z = env.get(name)
            if z is None or z.kind != "f" or not (np.all(z.lo == 0.0) and np.all(z.hi == 0.0)) or (z.st != ST_OK).any():
                raise ProgramAbort(f"{op.node_id}: Welford guard constant {name} is not exactly 0.0")
        m, s2, w = args
        if any(v.d is not None for v in args):
            raise ProgramAbort(f"{op.node_id}: no derivative rule for the Welford reduction")
        ok = (m.st == ST_OK) & (s2.st == ST_OK) & (w.st == ST_OK)
        row_bad = np.any(~ok, axis=axis)
        z = lambda t: np.where(ok, t, 0.0)  # noqa: E731
        m_lo, m_hi, s_lo, s_hi, w_lo, w_hi = z(m.lo), z(m.hi), z(s2.lo), z(s2.hi), z(w.lo), z(w.hi)
        neg_w = np.any(w_lo < 0, axis=axis)
        # decisions on the exact input weights (the enclosure of an all-zero sum is not the point 0)
        all_zero_in = np.all(ok & (w_lo == 0) & (w_hi == 0), axis=axis)
        pos_in = np.any(ok & (w_lo > 0), axis=axis) & ~neg_w
        # unguarded ratio r = w_b / W: two weights that may be zero can meet in some merge tree -> 0 / 0
        unguarded_zero = (np.sum(w_lo <= 0, axis=axis) >= 2) if not guarded else np.zeros_like(neg_w)
        W_lo, W_hi = iv.isum(w_lo, w_hi, axis)
        S_lo, S_hi = iv.isum(s_lo, s_hi, axis)
        p_lo, p_hi = iv.imul(w_lo, w_hi, m_lo, m_hi)
        S1_lo, S1_hi = iv.isum(p_lo, p_hi, axis)
        pos = pos_in & (W_lo > 0)
        safe_W_lo = np.where(pos, W_lo, 1.0)
        safe_W_hi = np.where(pos, W_hi, 1.0)
        mean_lo, mean_hi = iv.idiv(S1_lo, S1_hi, safe_W_lo, safe_W_hi)
        c = np.expand_dims(0.5 * (mean_lo + mean_hi), axis)
        d_lo, d_hi = iv.isub(m_lo, m_hi, np.broadcast_to(c, m_lo.shape), np.broadcast_to(c, m_lo.shape))
        d2_lo, d2_hi = iv.isquare(d_lo, d_hi)
        a_lo, a_hi = iv.imul(w_lo, w_hi, d2_lo, d2_hi)
        A_lo, A_hi = iv.isum(a_lo, a_hi, axis)
        b_lo, b_hi = iv.imul(w_lo, w_hi, d_lo, d_hi)
        B_lo, B_hi = iv.isum(b_lo, b_hi, axis)
        B2_lo, B2_hi = iv.isquare(B_lo, B_hi)
        q_lo, q_hi = iv.idiv(B2_lo, B2_hi, safe_W_lo, safe_W_hi)
        v_lo, v_hi = iv.isub(A_lo, A_hi, q_lo, q_hi)
        v_lo = np.maximum(v_lo, 0.0)                     # sum w (m - mean)^2 >= 0 for w >= 0
        M2_lo, M2_hi = iv.iadd(S_lo, S_hi, v_lo, v_hi)
        all_zero = all_zero_in
        undecided = ~pos & ~all_zero
        bad = row_bad | neg_w | unguarded_zero
        st_mean = np.where(bad | undecided | all_zero, ST_NE, ST_OK).astype(np.int8)
        st_m2 = np.where(bad | undecided, ST_NE, ST_OK).astype(np.int8)
        M2_lo = np.where(all_zero, S_lo, M2_lo)
        M2_hi = np.where(all_zero, S_hi, M2_hi)
        W_lo = np.where(all_zero, 0.0, W_lo)
        W_hi = np.where(all_zero, 0.0, W_hi)
        st_w = np.where(row_bad, ST_NE, ST_OK).astype(np.int8)
        cond = np.any(m.cond | s2.cond | w.cond, axis=axis)
        reasons = m.reasons | s2.reasons | w.reasons
        extra = set()
        if neg_w.any():
            extra.add(f"not_established:Welford weight may be negative@{op.node_id}")
        if undecided.any():
            extra.add(f"not_established:Welford total weight may be zero@{op.node_id}")
        if all_zero.any():
            extra.add(f"not_established:Welford mean of zero total weight depends on the merge tree@{op.node_id}")
        if np.any(unguarded_zero):
            extra.add(f"not_established:unguarded Welford ratio with zero weights (0 / 0 in some merge tree)@{op.node_id}")
        self._rules["reduce.welford"] += int(np.size(W_lo))
        return [_ftv(m.elem, np.where(st_mean == ST_OK, mean_lo, 0.0), np.where(st_mean == ST_OK, mean_hi, 0.0),
                     st_mean, cond, reasons | extra),
                _ftv(s2.elem, np.where(st_m2 == ST_OK, M2_lo, 0.0), np.where(st_m2 == ST_OK, M2_hi, 0.0),
                     st_m2, cond, reasons | extra),
                _ftv(w.elem, np.where(st_w == ST_OK, W_lo, 0.0), np.where(st_w == ST_OK, W_hi, 0.0), st_w, cond,
                     reasons)]

    def _arg_reduce(self, op, args, axis, combiner):
        val, idx = args
        if val.d is not None:
            raise ProgramAbort(f"{op.node_id}: no derivative rule for {combiner}")
        if (val.st != ST_OK).any() or (idx.st != ST_OK).any():
            raise ProgramAbort(f"{op.node_id}: argmax over non-finite or undefined values")
        is_max = combiner == "argmax"
        lo, hi = val.lo, val.hi
        best_lo = np.max(lo, axis=axis) if is_max else np.min(lo, axis=axis)
        best_hi = np.max(hi, axis=axis) if is_max else np.min(hi, axis=axis)
        bl = np.expand_dims(best_lo, axis)
        bh = np.expand_dims(best_hi, axis)
        candidate = (hi >= bl) if is_max else (lo <= bh)
        n_cand = np.sum(candidate, axis=axis)
        point = (lo == hi)
        all_points_equal = np.all(~candidate | (point & (lo == bl) & (bl == bh)), axis=axis)
        big = np.iinfo(np.int64).max
        index_vals = np.where(candidate, idx.lo, big)
        chosen = np.min(index_vals, axis=axis)
        definite = (n_cand == 1) | all_points_equal
        st_i = np.where(definite, ST_OK, ST_NE).astype(np.int8)
        reasons = val.reasons | idx.reasons
        if not definite.all():
            reasons = reasons | {f"not_established:argmax candidates overlap@{op.node_id}"}
        cond = np.any(val.cond | idx.cond, axis=axis)
        v_out = _ftv(val.elem, best_lo, best_hi, np.zeros(best_lo.shape, dtype=np.int8), cond, val.reasons)
        i_out = TV("i", idx.elem, chosen, None, None, st_i, cond, reasons)
        return [v_out, i_out]

    def _op_scan(self, op, args, env, state):
        axis = int(op.attrs["axis"].split(":")[0])
        reverse = op.attrs.get("reverse", "false").startswith("true")
        combiner = recognize_combiner(op)
        if _TRIGGER_PATH:
            _trace(f"tt.scan/{combiner if combiner in ('sum', 'sum_int') else ('generic fold (several operands)' if len(args) > 1 else 'generic fold (one operand)')}")
        if len(args) > 1:  # several operands (e.g. cummax values + indices): only the generic fold applies
            if self.mode == NumericMode.ROUNDING_CHECK:
                raise ProgramAbort(f"{op.node_id}: multi-operand scan in rounding-check mode")
            return self._generic_scan(op, args, axis, reverse, env, state)
        (x,) = args
        if combiner == "sum" and x.kind == "f" and self.mode == NumericMode.ROUNDING_CHECK:
            raise ProgramAbort(f"{op.node_id}: scan order is not declared (rounding-check mode)")
        if combiner == "sum" and x.kind == "f":
            finite = x.st == ST_OK
            lo, hi = iv.icumsum(np.where(finite, x.lo, 0.0), np.where(finite, x.hi, 0.0), axis, reverse)
            bad = np.where(x.st != ST_OK, 1, 0)
            bad = np.cumsum(np.flip(bad, axis) if reverse else bad, axis=axis)
            if reverse:
                bad = np.flip(bad, axis)
            st = np.where(bad > 0, ST_NE, ST_OK).astype(np.int8)
            out = _ftv(x.elem, lo, hi, st, x.cond, x.reasons)
            if x.d is not None:
                d = iv.icumsum(*(np.where(finite, t, 0.0) for t in x.tangent()), axis, reverse)
                out.d = (np.where(st == ST_OK, d[0], 0.0), np.where(st == ST_OK, d[1], 0.0))
            return out
        if combiner == "sum_int" and x.kind == "i":
            vals = np.flip(x.lo, axis) if reverse else x.lo
            out = np.cumsum(vals, axis=axis)
            out = np.flip(out, axis) if reverse else out
            return TV("i", x.elem, _wrap(out, INT_WIDTH[x.elem]), None, None, x.st, x.cond, x.reasons)
        if self.mode != NumericMode.ROUNDING_CHECK:
            # any other combiner (an unrecognized region, prod, max/min): fold its region in scan order, exact in real
            # arithmetic for the associative combiner a Triton scan requires
            return self._generic_scan(op, args, axis, reverse, env, state)
        raise ProgramAbort(f"{op.node_id}: scan combiner {combiner} not supported")

    def _generic_scan(self, op, args, axis, reverse, env, state):
        """A scan with its own combine region (e.g. Inductor's logcumsumexp), folded sequentially in scan order.
        Exact in real arithmetic for an associative combiner, which a Triton scan requires (the hardware order
        is a tree); the interval of each prefix encloses its real value."""

        n = args[0].shape[axis]
        order = list(range(n - 1, -1, -1)) if reverse else list(range(n))
        take = lambda t, i: t.map(lambda a: np.take(a, i, axis=axis))  # noqa: E731
        carry = [take(x, order[0]) for x in args]
        outs = [{order[0]: c} for c in carry]
        for i in order[1:]:
            _, carry = self._run_region(op.regions[0], env, state, carry + [take(x, i) for x in args])
            for j, c in enumerate(carry):
                outs[j][i] = c
        result = []
        for j, x in enumerate(args):
            parts = [outs[j][i] for i in range(n)]
            st = lambda f: np.stack([f(p) for p in parts], axis=axis)  # noqa: E731
            d = None
            if any(p.d is not None for p in parts):
                d = (st(lambda p: p.tangent()[0]), st(lambda p: p.tangent()[1]))
            result.append(TV(parts[0].kind, parts[0].elem, st(lambda p: p.lo),
                             None if parts[0].hi is None else st(lambda p: p.hi),
                             None if parts[0].base is None else st(lambda p: p.base),
                             st(lambda p: np.broadcast_to(p.st, p.shape)), st(lambda p: np.broadcast_to(p.cond, p.shape)),
                             frozenset().union(*(p.reasons for p in parts)), d))
        if n > 1 and not any(r.d is not None for r in result):
            result = self._check_scan_bracketing(op, args, axis, reverse, env, state, result)
        return result if len(result) > 1 else result[0]

    def _check_scan_bracketing(self, op, args, axis, reverse, env, state, seq):
        """DSL v2 increment 8: a Triton scan requires an associative combine (the tl.associative_scan precondition)
        and the hardware brackets it as a tree.  The sequential fold is checked on these inputs against a second
        bracketing (Hillis-Steele doubling: prefix_i <- combine(prefix_{i - 2^k}, prefix_i)); prefixes the two agree
        on keep a value (the hull of both enclosures), the others are not established.  Agreement is evidence on these
        inputs, recorded as the premise "combine associative"."""
        flip = (lambda t: t.map(lambda a: np.flip(a, axis))) if reverse else (lambda t: t)
        cur = [flip(x) for x in args]
        n = args[0].shape[axis]
        take = lambda t, sl: t.map(lambda a: np.take(a, sl, axis=axis))  # noqa: E731
        shift = 1
        while shift < n:
            left = [take(t, np.arange(0, n - shift)) for t in cur]
            right = [take(t, np.arange(shift, n)) for t in cur]
            _, comb = self._run_region(op.regions[0], env, state, left + right)
            new = []
            for t, c in zip(cur, comb):
                head = take(t, np.arange(0, shift))
                cat = lambda f, h=head, c=c: np.concatenate([np.broadcast_to(f(h), h.shape),  # noqa: E731
                                                             np.broadcast_to(f(c), c.shape)], axis=axis)
                new.append(TV(t.kind, t.elem, cat(lambda v: v.lo), None if t.hi is None else cat(lambda v: v.hi),
                              None if t.base is None else cat(lambda v: v.base), cat(lambda v: v.st),
                              cat(lambda v: v.cond), t.reasons | c.reasons))
            cur = new
            shift *= 2
        alt = [flip(x) for x in cur]
        out = []
        disagree_any = False
        for a, b in zip(seq, alt):
            both = (a.st == ST_OK) & (b.st == ST_OK)
            if a.kind == "f":
                agree = both & (a.lo <= b.hi) & (b.lo <= a.hi)
                lo, hi = np.where(agree, np.minimum(a.lo, b.lo), a.lo), np.where(agree, np.maximum(a.hi, b.hi), a.hi)
            else:
                agree = both & (a.lo == b.lo)
                lo, hi = a.lo, a.hi
            same_special = (a.st == b.st) & (a.st != ST_OK)
            bad = ~(agree | same_special)
            disagree_any |= bool(bad.any())
            reasons = a.reasons | {f"assumed:scan combine associative (the tl.associative_scan precondition; two "
                                   f"bracketings agree on these inputs)@{op.node_id}"}
            if bad.any():
                reasons = reasons | {f"not_established:scan combine not associative on these inputs (two bracketings "
                                     f"differ)@{op.node_id}"}
            out.append(TV(a.kind, a.elem, lo, hi, a.base, np.where(bad, ST_NE, a.st).astype(np.int8), a.cond, reasons))
        self._rules["scan.bracketing_checked"] += 1
        if disagree_any:
            self._rules["scan.bracketing_disagreement"] += 1
        return out

    def _op_dot(self, op, args, env, state):
        a, b, c = args
        if any((v.st != ST_OK).any() for v in (a, b, c)):
            st_bad = True
        else:
            st_bad = False
        if a.kind == "i" or b.kind == "i" or c.kind == "i":
            return self._int_dot(op, a, b, c)
        if self.mode == NumericMode.ROUNDING_CHECK:
            raise ProgramAbort(f"{op.node_id}: dot accumulation order is not declared (rounding-check mode)")
        lo, hi = iv.idot(np.where(a.st == ST_OK, a.lo, 0.0), np.where(a.st == ST_OK, a.hi, 0.0),
                         np.where(b.st == ST_OK, b.lo, 0.0), np.where(b.st == ST_OK, b.hi, 0.0))
        lo, hi = iv.iadd(lo, hi, np.where(c.st == ST_OK, c.lo, 0.0), np.where(c.st == ST_OK, c.hi, 0.0))
        shape = lo.shape
        st = np.zeros(shape, dtype=np.int8)
        if st_bad:
            row_bad = np.any(a.st != ST_OK, axis=-1)[..., :, None]
            col_bad = np.any(b.st != ST_OK, axis=-2)[..., None, :]
            st = np.where(row_bad | col_bad | (c.st != ST_OK), ST_NE, st).astype(np.int8)
        cond = np.any(a.cond, axis=-1)[..., :, None] | np.any(b.cond, axis=-2)[..., None, :] | c.cond
        precision = op.attrs.get("inputPrecision", "ieee").strip()  # the default (ieee) is not printed
        out = _ftv(c.elem, lo, hi, st, cond, a.reasons | b.reasons | c.reasons | {f"dot_input_precision:{precision}"})
        if a.d is not None or b.d is not None or c.d is not None:
            ok = lambda v, t: np.where(v.st == ST_OK, t, 0.0)  # noqa: E731
            ta, tb, tc = a.tangent(), b.tangent(), c.tangent()
            da = iv.idot(ok(a, ta[0]), ok(a, ta[1]), ok(b, b.lo), ok(b, b.hi))
            db = iv.idot(ok(a, a.lo), ok(a, a.hi), ok(b, tb[0]), ok(b, tb[1]))
            d = iv.iadd(*iv.iadd(*da, *db), np.broadcast_to(tc[0], shape), np.broadcast_to(tc[1], shape))
            out.d = (np.where(st == ST_OK, d[0], 0.0), np.where(st == ST_OK, d[1], 0.0))
        return out

    def _int_dot(self, op, a, b, c):
        """Integer tt.dot (DSL v2 increment 4): d = matmul(a, b) + c computed exactly.  The op contract does not fix
        overflow (the NVIDIA MMA path saturates with .satfinite, the FMA path wraps) or the operand signedness; the
        operands are read as signed (the official NVIDIA/AMD lowerings use s8), recorded as an assumption, and a result
        outside the accumulator range is not established."""
        if not (a.kind == b.kind == c.kind == "i"):
            raise ProgramAbort(f"{op.node_id}: dot mixing integer and float operands")
        wa, wb, wc = INT_WIDTH[a.elem], INT_WIDTH[b.elem], INT_WIDTH[c.elem]
        A = _wrap(np.asarray(a.lo, dtype=np.int64), wa).astype(object)
        B = _wrap(np.asarray(b.lo, dtype=np.int64), wb).astype(object)
        exact = np.matmul(A, B) + np.asarray(c.lo, dtype=np.int64).astype(object)
        limit = 1 << (wc - 1)
        fits = np.vectorize(lambda x: -limit <= x < limit, otypes=[bool])(exact)
        v = np.where(fits, exact, 0).astype(np.int64)
        row_bad = np.any(a.st != ST_OK, axis=-1)[..., :, None]
        col_bad = np.any(b.st != ST_OK, axis=-2)[..., None, :]
        st = np.where(row_bad | col_bad | (c.st != ST_OK) | ~fits, ST_NE, ST_OK).astype(np.int8)
        reasons = set(a.reasons | b.reasons | c.reasons)
        reasons.add(f"assumed:integer dot operands signed (official NVIDIA/AMD lowerings)@{op.node_id}")
        if not fits.all():
            reasons.add(f"not_established:integer dot leaves the accumulator range (saturation or wrap is lowering "
                        f"dependent)@{op.node_id}")
        cond = np.any(a.cond, axis=-1)[..., :, None] | np.any(b.cond, axis=-2)[..., None, :] | c.cond
        self._rules["dot.integer_exact_lanes"] += int((st == ST_OK).sum())
        return TV("i", c.elem, v, None, None, st, cond, frozenset(reasons))

    def _op_atomic_cas(self, op, args, env, state):
        """tt.atomic_cas (DSL v2 increment 3, rc3 02 6.9 partial): with a single program per address in the launch the
        result is determined: old = mem; mem = val where old == cmp; returns old.  Several programs on one address make
        the outcome depend on the interleaving: an execution race (every program involved is not established) --
        the finite-interleaving relation is a later increment.  Duplicate addresses inside one CAS, a non-point
        reference value (equality not decidable) or a non-point operand: not established."""
        ptr, cmp, val = args[0], args[1], args[2]
        shape = ptr.shape
        buf, index, in_range = self._addresses(op, ptr, state)
        if buf is None:
            raise ProgramAbort(f"{op.node_id}: atomic_cas through an address of unknown buffer")
        active = in_range & (np.broadcast_to(ptr.st, shape) == ST_OK)
        idx = np.asarray(index)[active].astype(np.int64)
        old_lo = np.zeros(shape)
        old_st = np.full(shape, ST_NE, dtype=np.int8)
        reasons = set(ptr.reasons | cmp.reasons | val.reasons)
        if idx.size:
            owner = self._readers.setdefault(("cas", buf.ident), np.full(buf.writer.shape, -1, dtype=np.int64))
            others = owner[idx]
            clash = (others >= 0) & (others != state.pid_index)
            if clash.any() and self._cas_serial:
                # DSL v2 increment 7: one serialization of the contended CAS (program order); evaluate() checks the
                # reverse order and keeps only what both orders agree on
                if _TRIGGER_PATH:
                    _trace("tt.atomic_cas:cas_serialization")
                self._cas_contended = True
                self._rules["execution.atomic_cas_serialized_lanes"] += int(clash.sum())
                clash = np.zeros(clash.shape, dtype=bool)
            elif clash.any():
                self._race_programs.update(int(p) for p in np.unique(others[clash]))
                self._race_programs.add(state.pid_index)
                self._rules["execution.atomic_cas_contention_lanes"] += int(clash.sum())
                reasons.add(f"not_established:execution race: compare-and-swap on one address by several programs "
                            f"(interleaving relation not modelled yet)@{op.node_id}")
            owner[idx] = state.pid_index
            self._check_read_then_write(buf, idx, state)
            if not self._cas_serial:
                self._track_read(buf, idx, state)  # the CAS reads the old value (an atomic read in serialization)
            uniq, counts = np.unique(idx, return_counts=True)
            dup = np.isin(idx, uniq[counts > 1])
            lo_m = buf.lo[idx].astype(np.float64)
            hi_m = (buf.hi[idx] if buf.hi is not None else buf.lo[idx]).astype(np.float64)
            c_lo = np.broadcast_to(cmp.lo, shape)[active].astype(np.float64)
            c_hi = np.broadcast_to(cmp.hi if cmp.hi is not None else cmp.lo, shape)[active].astype(np.float64)
            v_lo = np.broadcast_to(val.lo, shape)[active]
            v_hi = np.broadcast_to(val.hi if val.hi is not None else val.lo, shape)[active]
            point = (lo_m == hi_m) & (c_lo == c_hi) & (buf.st[idx] == ST_OK) & \
                (np.broadcast_to(cmp.st, shape)[active] == ST_OK) & (np.broadcast_to(val.st, shape)[active] == ST_OK)
            decided = point & ~dup & ~clash
            swap = decided & (lo_m == c_lo)
            res_lo = old_lo[active]
            res_lo[:] = lo_m
            old_lo[active] = res_lo
            st_a = np.where(decided, ST_OK, ST_NE).astype(np.int8)
            old_st[active] = st_a
            new_lo = np.where(swap, v_lo, buf.lo[idx])
            buf.lo[idx] = new_lo
            if buf.hi is not None:
                buf.hi[idx] = np.where(swap, v_hi, buf.hi[idx])
            buf.st[idx] = np.where(decided, buf.st[idx], ST_NE)
            if self._cas_serial and op.attrs.get("sem") in ("acquire", "acq_rel"):
                # the value read was written by the latest release of that value at the address (program order)
                for a, v, ok in zip(idx.tolist(), lo_m.tolist(), decided.tolist()):
                    rel = [r for r in self._releases.get((buf.ident, a), []) if r[2] == v]
                    if ok and rel:
                        self._acquire(state, rel[-1])
            buf.writer[idx] = state.pid_index
            self._mark_plain(buf, idx)
            we = self._wepoch.get(buf.ident)
            if we is None:
                we = self._wepoch[buf.ident] = np.full(buf.writer.shape, -1, dtype=np.int64)
            we[idx] = state.rel_epoch
            if self._cas_serial and op.attrs.get("sem") in ("release", "acq_rel"):
                snap = dict(self._hb.get(state.pid_index, {}))
                new_vals = np.asarray(new_lo)
                for j, a in enumerate(idx.tolist()):
                    self._releases[(buf.ident, a)].append((state.pid_index, state.rel_epoch, new_vals[j].item(), snap))
                state.rel_epoch += 1
            buf.written[idx] = True
            self._stored.add(buf.ident)
            if (~decided).any():
                reasons.add(f"not_established:compare-and-swap not decidable (duplicate address, non-point value or "
                            f"contention)@{op.node_id}")
            self._rules["atomic.cas_decided_lanes"] += int(decided.sum())
        kind = "f" if ptr.elem.pointee in FLOAT_ELEMS else "i"
        if kind == "f":
            return _ftv(ptr.elem.pointee, old_lo, old_lo.copy(), old_st, np.zeros(shape, dtype=bool), frozenset(reasons))
        return TV("i", ptr.elem.pointee, old_lo.astype(np.int64), None, None, old_st, np.zeros(shape, dtype=bool),
                  frozenset(reasons))

    def _op_dot_scaled(self, op, args, env, state):
        """tt.dot_scaled (DSL v2 rc3 02 6.5, format bridge): the operands are decoded exactly (fp8 e4m3 / e5m2 as
        loaded, fp4 e2m1 packed two per byte along K, low nibble first), each block of 32 along K multiplied by its
        e8m0 scale 2^(e - 127) (e = 255: NaN), and the real dot product is enclosed by DotK.  The products are exact in
        float64 (|fp8| <= 57344, 2^(+-127)), so the only rounding is in the enclosed sum."""
        if self.mode == NumericMode.ROUNDING_CHECK:
            raise ProgramAbort(f"{op.node_id}: scaled dot accumulation order is not declared (rounding-check mode)")
        fmt_a, fmt_b = op.attrs.get("lhs"), op.attrs.get("rhs")
        if len(args) == 5:
            a, sa, b, sb, c = args
        elif len(args) == 3:
            (a, b, c), sa, sb = args, None, None
        else:
            raise ProgramAbort(f"{op.node_id}: dot_scaled with {len(args)} operands")
        return self._scaled_dot(op, a, sa, b, sb, c, fmt_a, fmt_b)

    def _scaled_dot(self, op, a, sa, b, sb, c, fmt_a, fmt_b):
        """The exact MX decoding and enclosed dot product of tt.dot_scaled, shared with ttng.tc_gen5_mma_scaled (DSL v2
        increment 13): a scale is [M, K/32] for a, [N, K/32] for b."""
        ok_fmt = {"e4m3", "e5m2", "e2m1", "bf16", "fp16"}
        if fmt_a not in ok_fmt or fmt_b not in ok_fmt:
            raise ProgramAbort(f"{op.node_id}: dot_scaled format {fmt_a} / {fmt_b} has no declared semantics")

        def values(x, fmt, k_axis):
            if fmt != "e2m1":
                if x.kind != "f":
                    raise ProgramAbort(f"{op.node_id}: dot_scaled {fmt} operand is not a float tensor")
                return np.where(x.st == ST_OK, x.lo, np.nan).astype(np.float64)
            byte = np.asarray(x.lo).astype(np.int64) & 0xFF
            lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
            lo_n, hi_n = lut[byte & 0xF], lut[(byte >> 4) & 0xF]
            stacked = np.stack([lo_n, hi_n], axis=k_axis + 1 if k_axis >= 0 else k_axis)
            shape = list(byte.shape)
            shape[k_axis] *= 2
            v = stacked.reshape(shape)
            bad = np.repeat(np.asarray(x.st) != ST_OK, 2, axis=k_axis)
            return np.where(bad, np.nan, v)

        def scaled(v, s, k_axis):
            if s is None:
                return v
            e = np.asarray(s.lo).astype(np.int64) & 0xFF
            f = np.where(e == 255, np.nan, np.ldexp(1.0, (e - 127).astype(np.int32)))
            f = np.where(np.asarray(s.st) == ST_OK, f, np.nan)
            reps = v.shape[k_axis] // f.shape[-1]
            f = np.repeat(f, reps, axis=-1)
            return v * (f if k_axis == v.ndim - 1 else np.swapaxes(f, -1, -2))
        av = scaled(values(a, fmt_a, a.lo.ndim - 1), sa, a.lo.ndim - 1)
        bv = scaled(values(b, fmt_b, b.lo.ndim - 2), sb, b.lo.ndim - 2)
        bad_a, bad_b = np.isnan(av), np.isnan(bv)
        lo, hi = iv.idot(np.where(bad_a, 0.0, av), np.where(bad_a, 0.0, av), np.where(bad_b, 0.0, bv),
                         np.where(bad_b, 0.0, bv))
        lo, hi = iv.iadd(lo, hi, np.where(c.st == ST_OK, c.lo, 0.0), np.where(c.st == ST_OK, c.hi, 0.0))
        row_bad = np.any(bad_a, axis=-1)[..., :, None]
        col_bad = np.any(bad_b, axis=-2)[..., None, :]
        st = np.where(row_bad | col_bad | (c.st != ST_OK), ST_NE, ST_OK).astype(np.int8)
        cond = np.any(a.cond, axis=-1)[..., :, None] | np.any(b.cond, axis=-2)[..., None, :] | c.cond
        self._rules["dot_scaled.exact_decode"] += int(st.size)
        reasons = a.reasons | b.reasons | c.reasons | {f"dot_scaled:{fmt_a}x{fmt_b}"}
        for s_ in (sa, sb):
            if s_ is not None:
                reasons = reasons | s_.reasons
        return _ftv(c.elem, lo, hi, st, cond, reasons)

    # ---- elementwise --------------------------------------------------------------

    def _op_extern(self, op, args, env, state):
        symbol = op.attrs.get("symbol", "").strip('"')
        if _TRIGGER_PATH:
            _trace(f"tt.extern_elementwise/symbol={symbol}")
        internal = LIBDEVICE.get(symbol)
        if internal is None:
            raise ProgramAbort(f"{op.node_id}: libdevice {symbol} has no declared semantics")
        if any(a.kind != "f" for a in args) and internal not in ("isnan", "isinf", "isfinite", "signbit"):
            # integer operands (e.g. __nv_abs on int32): the integer rule, never the float one (4.0: the float rule
            # returned 0 for every lane, found by the per-signature tests)
            if internal == "abs" and all(a.kind == "i" for a in args):
                return _int_op("absi", op, args)
            raise ProgramAbort(f"{op.node_id}: libdevice {symbol} with integer operands has no declared semantics")
        mode = LIBDEVICE_ROUNDING.get(symbol)
        if mode is not None:
            # A rounding-suffixed libdevice call declares the real operation; the suffix is used
            # only in rounding-check mode.
            import dataclasses

            op = dataclasses.replace(op, attrs={**op.attrs, "rounding": mode})
        return self._elementwise(internal, op, args)

    def _op_inline_asm(self, op, args, env, state):
        internal = inline_asm_internal(op.attrs.get("asm", ""))
        if _TRIGGER_PATH:
            from .ttir_mapping import _normalize_asm
            _trace(f"tt.elementwise_inline_asm/asm={_normalize_asm(op.attrs.get('asm', ''))}")
        if internal is None:
            program = parse_ptx_program(op.attrs.get("asm", ""))
            if program is None or len(op.results) != 1 or \
                    op.attrs.get("packed_element", "1 : i32").split(":")[0].strip() != "1":
                return self._run_bit_ptx(op, args)
            if self.mode == NumericMode.ROUNDING_CHECK and any(ins[2] in _PTX_APPROX for ins in program):
                raise ProgramAbort(f"{op.node_id}: approximate PTX instruction in rounding-check mode")
            return self._run_ptx_program(op, program, args)
        if internal == "approx_div":
            return self._op_approx_div(op, args, env, None)
        if internal == "tf32_round":
            # DSL v2 increment 11: numerical-difference mode keeps a tf32 dot's operand rounding as a recorded precision
            # property (the TTIR level's dot_input_precision); rounding-check mode would need the exact rna rounding
            if self.mode == NumericMode.ROUNDING_CHECK:
                raise ProgramAbort(f"{op.node_id}: tf32 rounding (cvt.rna.tf32.f32) is not modelled in rounding-check mode")
            return args[0].with_reason("dot_input_precision:tf32 (cvt.rna.tf32.f32 inserted by the lowering)")
        mode = inline_asm_rounding(op.attrs.get("asm", ""))
        if mode is not None:  # used only in rounding-check mode, like a rounding-suffixed libdevice call
            import dataclasses

            op = dataclasses.replace(op, attrs={**op.attrs, "rounding": mode})
        return self._elementwise(internal, op, args)

    def _run_bit_ptx(self, op, args):
        """Inline asm interpreted instruction by instruction on bit patterns and real values (ptx_bits, DSL v2
        increment 9, rc3 02 6.10).  An environment read (clock, SM id) leaves the results without a reference value;
        an instruction outside the subset or a memory effect inside the asm aborts the program."""
        from . import ptx_bits
        asm = op.attrs.get("asm", "")
        cons = op.attrs.get("constraints", "").strip().strip('"')
        pack = int(str(op.attrs.get("packed_element", "1 : i32")).split(":")[0].strip())
        shape = np.broadcast_shapes(*[a.shape for a in args]) if args else op.result_types[0].shape
        widths = {"i1": 8, "i8": 8, "i16": 16, "i32": 32, "i64": 64, "f16": 16, "f32": 32, "bf16": 16, "f64": 64}
        inputs = []
        for a in args:
            if a.kind == "p":
                raise ProgramAbort(f"{op.node_id}: inline asm with pointer operands (memory effects) is not modelled")
            w = widths.get(a.elem)
            if w is None:
                raise ProgramAbort(f"{op.node_id}: inline asm operand type {a.elem} is not modelled")
            flat = a.map(lambda x, s=shape: np.broadcast_to(np.asarray(x), s).reshape(-1))
            ok = np.asarray(flat.st) == ST_OK
            if a.kind == "f" and a.elem in ("f16", "f32"):
                inputs.append(("real", w, None, None, np.where(ok, flat.lo, 0.0), np.where(ok, flat.hi, 0.0)))
                if not ok.all():
                    raise ProgramAbort(f"{op.node_id}: inline asm operand without a finite established value")
            elif a.kind == "f":
                bits, definite = (_float_bits(a.elem, flat.lo, flat.hi, flat.st) if a.elem in _BIT_LAYOUT or
                                  a.elem == "bf16" else (None, None))
                if bits is None:
                    raise ProgramAbort(f"{op.node_id}: inline asm operand type {a.elem} is not modelled")
                inputs.append(("bits", w, bits.astype(np.uint64) & np.uint64((1 << w) - 1), definite, None, None))
            else:
                v = np.asarray(flat.lo, dtype=np.int64)
                inputs.append(("bits", w, (v.astype(np.uint64) & np.uint64((1 << w) - 1 if w < 64 else -1 & (2**64 - 1))),
                               ok, None, None))
        outs = []
        for rt in op.result_types:
            w = widths.get(rt.elem)
            if w is None or rt.elem in ("bf16", "f64"):
                raise ProgramAbort(f"{op.node_id}: inline asm result type {rt.elem} is not modelled")
            outs.append(("real" if rt.elem in ("f16", "f32") else "bits", w))
        try:
            res = ptx_bits.run(asm, cons, pack, inputs, outs, total=int(np.prod(op.result_types[0].shape)))
        except ptx_bits.EnvironmentRead as exc:
            why = frozenset({f"not_established:inline asm {exc}@{op.node_id}"})
            out = []
            for rt in op.result_types:
                z = np.zeros(rt.shape)
                st = np.full(rt.shape, ST_NE, dtype=np.int8)
                out.append(_ftv(rt.elem, z, z.copy(), st, np.zeros(rt.shape, dtype=bool), why) if kind_of(rt.elem) == "f"
                           else TV("i", rt.elem, z.astype(np.int64), None, None, st, np.zeros(rt.shape, dtype=bool), why))
            self._rules["inline_asm.environment_reads"] += 1
            return out if len(out) > 1 else out[0]
        except ptx_bits.Unsupported as exc:
            raise ProgramAbort(f"{op.node_id}: inline asm not modelled ({exc})")
        cond = np.zeros(shape, dtype=bool)
        for a in args:
            cond = cond | np.broadcast_to(a.cond, shape)
        reasons = frozenset().union(*(a.reasons for a in args)) if args else frozenset()
        out = []
        for rt, r in zip(op.result_types, res):
            if kind_of(rt.elem) == "f":
                lo, hi, ok = r
                st = np.where(ok, ST_OK, np.where(np.isnan(lo), ST_NAN, np.where(lo == np.inf, ST_PINF,
                                                                                np.where(lo == -np.inf, ST_NINF, ST_NE))))
                lo = np.where(ok, lo, 0.0)
                hi = np.where(ok, hi, 0.0)
                out.append(_ftv(rt.elem, lo.reshape(rt.shape), hi.reshape(rt.shape), st.reshape(rt.shape).astype(np.int8),
                                cond.reshape(rt.shape), reasons))
            else:
                bits, definite = r
                w = widths[rt.elem]
                v = _wrap(np.asarray(bits, dtype=np.uint64).astype(np.int64), w) if w < 64 else \
                    np.asarray(bits, dtype=np.uint64).view(np.int64)
                st = np.where(definite, ST_OK, ST_NE).astype(np.int8)
                out.append(TV("i", rt.elem, v.reshape(rt.shape), None, None, st.reshape(rt.shape), cond.reshape(rt.shape),
                              reasons | (frozenset({f"not_established:inline asm on operands without definite bits@"
                                                    f"{op.node_id}"}) if not definite.all() else frozenset())))
        self._rules["inline_asm.bit_level_programs"] += 1
        return out if len(out) > 1 else out[0]

    def _run_ptx_program(self, op, program, args):
        """Lane-wise reference of a straight-line PTX snippet (see parse_ptx_program).  Arithmetic is exact real
        arithmetic and the .approx functions are the exact functions (their approximation error is a node
        computation error, measured by K - K_R); a predicated instruction is r := select(p, value, r)."""
        import dataclasses

        shape = op.result_types[0].shape
        out_elem = op.result_types[0].elem
        n_out = len(op.results)
        regs, preds = {}, {}
        for i, a in enumerate(args):
            regs[f"${n_out + i}"] = a if a.shape == shape else a.map(lambda x: np.broadcast_to(x, shape))
        undefined = _ftv(out_elem, np.zeros(shape), np.zeros(shape), np.full(shape, ST_UNDEF, dtype=np.int8),
                         np.zeros(shape, dtype=bool), frozenset())

        def value(tok):
            if tok in regs:
                return regs[tok]
            if tok.startswith("$"):
                return regs.setdefault(tok, undefined)
            if tok.lower().startswith("0f"):
                v = float(np.frombuffer(int(tok[2:], 16).to_bytes(4, "little"), dtype=np.float32)[0])
            else:
                v = float(iv.round_nearest_even(np.array([float(tok)]), "f32")[0][0])
            return _ftv("f32", np.full(shape, v), np.full(shape, v), np.zeros(shape, dtype=np.int8),
                        np.zeros(shape, dtype=bool), frozenset())

        def fake(name, n, attrs=None):
            return dataclasses.replace(op, name=name, operands=[f"%ptx{i}" for i in range(n)], attrs=attrs or {},
                                       results=["%ptx_out"], result_types=[op.result_types[0]])

        for pred, neg, kind, dst, srcs in program:
            if kind == "setp":
                cmp, a, b = srcs
                preds[dst] = _cmpf(fake("arith.cmpf", 2, {"predicate": cmp}), [value(a), value(b)])
                continue
            if kind == "mov":
                v = value(srcs[0])
            else:
                v = self._float_op(kind, fake(kind, len(srcs)), [value(s) for s in srcs], out_elem)
            if pred is not None:
                p = preds[pred]
                if neg:
                    p = TV("b", "i1", np.where(p.lo == MAYBE, MAYBE, 1 - p.lo).astype(np.int8), None, None,
                           p.st, p.cond, p.reasons)
                v = _select(fake("arith.select", 3), [p, v, value(dst)])
            regs[dst] = v
        return regs["$0"]

    def _elementwise(self, name: str, op: TOp, args: list) -> TV:
        out_type = op.result_types[0]
        out_elem = out_type.elem
        if name in _INT_OPS:
            return _int_op(name, op, args)
        if name == "cmpi":
            return _cmpi(op, args)
        if name == "cmpf":
            return _cmpf(op, args)
        if name == "select":
            undecided = int((np.asarray(args[0].lo) == MAYBE).sum())
            if undecided:
                self._rules["path.select_union_lanes"] += undecided
            return _select(op, args)
        if name in ("isnan", "isinf", "isfinite", "signbit"):
            return _float_test(name, args[0])
        if name in ("sitofp", "uitofp"):
            return self._int_to_float(name, op, args[0], out_elem)
        if name in ("fptosi", "fptoui"):
            return _float_to_int(name, op, args[0], out_elem)
        if name == "bitcast":
            return _bitcast(op, args[0], out_type)
        if name == "cast" and out_type.elem in INT_WIDTH:
            raise ProgramAbort(f"{op.node_id}: unexpected integer cast")
        return self._float_op(name, op, args, out_elem)

    def _int_to_float(self, name, op, x: TV, out_elem) -> TV:
        width = INT_WIDTH[x.elem]
        ints = _unsigned(x.lo, width) if name == "uitofp" else x.lo.astype(np.int64)
        f = ints.astype(np.float64)
        lo, hi = f.copy(), f.copy()
        # Exactness is decided on the original integer: |x| <= 2**53 always converts exactly;
        # beyond that the converted value is compared with the integer to widen on the right side.
        limit = 1 << 53
        big = (ints > limit) if ints.dtype == np.uint64 else ((ints > limit) | (ints < -limit))
        for i in np.flatnonzero(big.reshape(-1)):
            fi, xi = int(f.reshape(-1)[i]), int(ints.reshape(-1)[i])
            if fi > xi:
                lo.reshape(-1)[i] = math.nextafter(f.reshape(-1)[i], -math.inf)
            elif fi < xi:
                hi.reshape(-1)[i] = math.nextafter(f.reshape(-1)[i], math.inf)
        out = _ftv(out_elem, lo, hi, x.st, x.cond, x.reasons)
        return self._maybe_round(out, op, out_elem, declared=False)

    def _maybe_round(self, v: TV, op: TOp, fmt, declared: bool) -> TV:
        rounds = declared or self.mode == NumericMode.ROUNDING_CHECK
        if not rounds:
            return v
        if fmt not in iv.FLOAT_FORMATS:
            st = np.where(v.st == ST_OK, ST_NE, v.st).astype(np.int8)
            return TV("f", v.elem, v.lo, v.hi, None, st, v.cond,
                      v.reasons | {f"not_established:no rounding model for {fmt}@{op.node_id}"})
        mode = _ROUNDING_MODES.get(op.attrs.get("rounding", "rtne"))
        if mode is None:
            st = np.where(v.st == ST_OK, ST_NE, v.st).astype(np.int8)
            return TV("f", v.elem, v.lo, v.hi, None, st, v.cond,
                      v.reasons | {f"not_established:rounding mode {op.attrs.get('rounding')}@{op.node_id}"})
        if v.d is not None and op.attrs.get("surrogate_gradient") != "identity":
            raise ProgramAbort(f"{op.node_id}: derivative of a rounding node needs a declared surrogate gradient")
        # Rounding is monotone in every IEEE mode, so the endpoints suffice.
        lo, lo_pinf, lo_ninf = iv.round_directed(v.lo, fmt, mode)
        hi, hi_pinf, hi_ninf = iv.round_directed(v.hi, fmt, mode)
        st = v.st.copy()
        ok = v.st == ST_OK
        reasons = v.reasons
        if iv.FLOAT_FORMATS[fmt][3]:
            st = np.where(ok & lo_pinf & hi_pinf, ST_PINF, st)
            st = np.where(ok & lo_ninf & hi_ninf, ST_NINF, st)
            st = np.where(ok & ((lo_pinf != hi_pinf) | (lo_ninf != hi_ninf)), ST_NE, st)
        else:  # no infinity in the format: saturation or NaN on overflow is the implementation's choice
            over = ok & (lo_pinf | hi_pinf | lo_ninf | hi_ninf)
            st = np.where(over, ST_NE, st)
            if over.any():
                reasons = reasons | {f"not_established:overflow in {fmt} (no infinity; saturation or NaN "
                                     f"is implementation-defined)@{op.node_id}"}
        lo = np.where(st == ST_OK, lo, 0.0)
        hi = np.where(st == ST_OK, hi, 0.0)
        out = _ftv(v.elem, lo, hi, st, v.cond, reasons)
        out.d = v.d
        return out

    def _float_op(self, name: str, op: TOp, args: list, out_elem) -> TV:
        if any(a.kind != "f" for a in args):
            raise ProgramAbort(f"{op.node_id}: {name} on non-float operands")
        shape = np.broadcast_shapes(*(a.shape for a in args))
        args = [a if a.shape == shape else a.map(lambda x: np.broadcast_to(x, shape)) for a in args]
        finite = np.ones(shape, dtype=bool)
        for a in args:
            finite &= a.st == ST_OK
        los = [np.where(finite, a.lo, 1.0) for a in args]
        his = [np.where(finite, a.hi, 1.0) for a in args]
        ok = np.ones(shape, dtype=bool)
        same = len(op.operands) == 2 and op.operands[0] == op.operands[1]
        if name in ("add", "sub", "mul"):
            fn = {"add": iv.iadd, "sub": iv.isub}.get(name)
            lo, hi = (iv.imul(los[0], his[0], los[1], his[1], same=same) if name == "mul"
                      else fn(los[0], his[0], los[1], his[1]))
            if name == "sub" and same:                       # one variable: x - x = 0 exactly (tool 3.0)
                lo, hi = np.zeros(shape), np.zeros(shape)
        elif name in ("div", "rcp"):
            num_lo, num_hi = (los[0], his[0]) if name == "div" else (np.ones(shape), np.ones(shape))
            den_lo, den_hi = (los[1], his[1]) if name == "div" else (los[0], his[0])
            ok = ~((den_lo <= 0) & (den_hi >= 0))
            safe_lo = np.where(ok, den_lo, 1.0)
            safe_hi = np.where(ok, den_hi, 1.0)
            lo, hi = iv.idiv(num_lo, num_hi, safe_lo, safe_hi)
            if name == "div" and same:                       # one variable: x / x = 1 where x != 0 (tool 3.0)
                lo, hi = np.ones(shape), np.ones(shape)
        elif name == "neg":
            lo, hi = -his[0], -los[0]
        elif name == "abs":
            a, b = los[0], his[0]
            lo = np.where(a >= 0, a, np.where(b <= 0, -b, 0.0))
            hi = np.where(a >= 0, b, np.where(b <= 0, -a, np.maximum(-a, b)))
        elif name == "fma":
            lo, hi = iv.ifma(los[0], his[0], los[1], his[1], los[2], his[2])
        elif name == "sqrt":
            ok = los[0] >= 0
            lo, hi = iv.isqrt(np.where(ok, los[0], 1.0), np.where(ok, his[0], 1.0))
        elif name in ("floor", "ceil", "trunc", "round", "roundeven"):
            fn = {"floor": np.floor, "ceil": np.ceil, "trunc": np.trunc, "roundeven": np.rint,
                  "round": _round_half_away}[name]
            lo, hi = fn(los[0]), fn(his[0])
        elif name in ("maxnum", "minnum", "maximum", "minimum"):
            return _minmax(name, op, args)
        elif name == "clamp":
            return _clamp(op, args)
        elif name == "copysign":
            sign_neg, sign_pos = _sign_classes(los[1], his[1])
            mag_lo = np.where(los[0] >= 0, los[0], np.where(his[0] <= 0, -his[0], 0.0))
            mag_hi = np.maximum(np.abs(los[0]), np.abs(his[0]))
            ok = sign_neg | sign_pos
            lo = np.where(sign_neg, -mag_hi, mag_lo)
            hi = np.where(sign_neg, -mag_lo, mag_hi)
        elif name == "remf":
            point = (los[0] == his[0]) & (los[1] == his[1]) & (los[1] != 0)
            ok = point
            lo = np.where(point, np.fmod(los[0], np.where(point, los[1], 1.0)), 0.0)
            hi = lo
        elif name == "saturate":
            lo, hi = np.clip(los[0], 0.0, 1.0), np.clip(his[0], 0.0, 1.0)
        elif name == "identity":
            lo, hi = los[0], his[0]
        elif name == "cast":
            lo, hi = los[0], his[0]
        elif name == "pow":
            lo, hi, ok = iv.pow_bounds(los[0], his[0], los[1], his[1])
        elif name == "exp10":
            lo, hi, ok = iv.elementary_bounds("exp10", los[0], his[0])
        elif name in iv._MONOTONE or name in ("sin", "cos", "tan", "cosh"):
            lo, hi, ok = iv.elementary_bounds(name, los[0], his[0])
        elif name in iv.BESSEL:  # DSL v2 increment 6
            lo, hi, ok = iv.bessel_bounds(name, los[0], his[0])
        else:
            raise ProgramAbort(f"{op.node_id}: no reference rule for {name}")
        st = np.where(finite & ~ok, ST_NE, ST_OK).astype(np.int8)
        reasons = _merge_reasons(*args)
        if (finite & ~ok).any():
            reasons = reasons | {f"not_established:domain of {name}@{op.node_id}"}
        # Special values (NaN / inf) follow IEEE on representatives.
        special = ~finite & np.all([a.st < ST_UNDEF for a in args], axis=0)
        if special.any():
            reps_lo = [_float_reps(a, "lo") for a in args]
            reps_hi = [_float_reps(a, "hi") for a in args]
            f_lo = _ieee_eval(name, reps_lo)
            f_hi = _ieee_eval(name, reps_hi)
            s_st, s_val = _classify_floats(f_lo, f_hi)
            # a finite limit is a point only when it is exactly representable (0, +-1, 2, ...): otherwise (atan(inf) =
            # pi / 2) the float64 result is the rounded limit, enclosed one ulp outward (4.0, per-signature tests)
            exact_limit = np.isin(s_val, (0.0, 1.0, -1.0, 2.0)) | (s_st != ST_OK)
            s_lo = np.where(exact_limit, s_val, np.nextafter(s_val, -np.inf))
            s_hi = np.where(exact_limit, s_val, np.nextafter(s_val, np.inf))
            st = np.where(special, s_st, st)
            lo = np.where(special, s_lo, lo)
            hi = np.where(special, s_hi, hi)
        if name == "copysign":
            # The sign of a NaN is not tracked, so copysign(x, NaN) is not established.
            st = np.where(np.broadcast_to(args[1].st, st.shape) == ST_NAN, ST_NE, st).astype(np.int8)
        undef = _merge_status(*args)
        st = np.where(undef >= ST_UNDEF, undef, st).astype(np.int8)
        lo = np.where(st == ST_OK, lo, 0.0)
        hi = np.where(st == ST_OK, hi, 0.0)
        out = _ftv(out_elem, lo, hi, st, _merge_cond(*args), reasons)
        if any(a.d is not None for a in args):
            t = _tangent_rule(name, op, args, los, his, lo, hi, same)
            ok_t = st == ST_OK
            out.d = (np.where(ok_t, t[0], 0.0), np.where(ok_t, t[1], 0.0))
        declared = op.attrs.get("declared_quantization") is not None
        return self._maybe_round(out, op, out_elem, declared)


# ---------------------------------------------------------------------------
# Elementwise helpers (pure functions)
# ---------------------------------------------------------------------------

_CONSTS = {}


def _const_interval(name):
    """Rigorous float64 enclosure of ln 2 or 2/sqrt(pi)."""

    if name not in _CONSTS:
        import gmpy2

        expr = (lambda: gmpy2.log(gmpy2.mpfr(2))) if name == "ln2" else \
            (lambda: 2 / gmpy2.sqrt(gmpy2.const_pi()))
        with gmpy2.context(precision=53, round=gmpy2.RoundDown):
            lo = float(expr())
        with gmpy2.context(precision=53, round=gmpy2.RoundUp):
            hi = float(expr())
        _CONSTS[name] = (lo, hi)
    return _CONSTS[name]


def _full(shape, pair):
    return np.full(shape, pair[0]), np.full(shape, pair[1])


def _tangent_rule(name, op, args, los, his, lo, hi, same):
    """Interval tangent of an elementwise float op (forward mode)."""

    shape = lo.shape
    ts = [a.tangent() for a in args]
    ts = [(np.broadcast_to(t[0], shape), np.broadcast_to(t[1], shape)) for t in ts]
    t0 = ts[0]
    one = (np.ones(shape), np.ones(shape))
    if name == "add":
        return iv.iadd(*t0, *ts[1])
    if name == "sub":
        return iv.isub(*t0, *ts[1])
    if name == "neg":
        return -t0[1], -t0[0]
    if name in ("cast", "identity"):
        return t0
    if name == "mul":
        if same:
            return iv.imul(*iv.imul(*_full(shape, (2.0, 2.0)), los[0], his[0]), *t0)
        return iv.iadd(*iv.imul(los[0], his[0], *ts[1]), *iv.imul(*t0, los[1], his[1]))
    if name == "div":
        num = iv.isub(*iv.imul(*t0, los[1], his[1]), *iv.imul(los[0], his[0], *ts[1]))
        den = iv.isquare(los[1], his[1])
        if not (den[0] > 0).all():
            raise ProgramAbort(f"{op.node_id}: derivative of division at a zero divisor")
        return iv.idiv(*num, *den)
    if name == "rcp":
        den = iv.isquare(los[0], his[0])
        if not (den[0] > 0).all():
            raise ProgramAbort(f"{op.node_id}: derivative of reciprocal at zero")
        q = iv.idiv(*t0, *den)
        return -q[1], -q[0]
    if name == "fma":
        a = iv.iadd(*iv.imul(*t0, los[1], his[1]), *iv.imul(los[0], his[0], *ts[1]))
        return iv.iadd(*a, *ts[2])
    if name == "abs":
        pos, neg = los[0] >= 0, his[0] <= 0
        dlo = np.where(pos, t0[0], np.where(neg, -t0[1], np.minimum(t0[0], -t0[1])))
        dhi = np.where(pos, t0[1], np.where(neg, -t0[0], np.maximum(t0[1], -t0[0])))
        return dlo, dhi
    if name == "exp":
        return iv.imul(lo, hi, *t0)
    if name == "exp2":
        return iv.imul(*iv.imul(lo, hi, *_full(shape, _const_interval("ln2"))), *t0)
    if name == "log":
        return iv.idiv(*t0, los[0], his[0])
    if name == "log2":
        return iv.idiv(*t0, *iv.imul(los[0], his[0], *_full(shape, _const_interval("ln2"))))
    if name == "log1p":
        return iv.idiv(*t0, *iv.iadd(los[0], his[0], *one))
    if name == "sqrt":
        if not (los[0] > 0).all():
            raise ProgramAbort(f"{op.node_id}: derivative of sqrt at zero")
        return iv.idiv(*iv.imul(*t0, *_full(shape, (0.5, 0.5))), lo, hi)
    if name == "rsqrt":
        r3 = iv.imul(*iv.isquare(lo, hi), lo, hi)
        return iv.imul(*iv.imul(*r3, *_full(shape, (-0.5, -0.5))), *t0)
    if name == "tanh":
        return iv.imul(*iv.isub(*one, *iv.isquare(lo, hi)), *t0)
    if name in ("sin", "cos"):
        clo, chi, _ = iv.elementary_bounds("cos" if name == "sin" else "sin", los[0], his[0])
        if name == "cos":
            clo, chi = -chi, -clo
        return iv.imul(clo, chi, *t0)
    if name == "erf":
        sq = iv.isquare(los[0], his[0])
        elo, ehi, _ = iv.elementary_bounds("exp", -sq[1], -sq[0])
        return iv.imul(*iv.imul(elo, ehi, *_full(shape, _const_interval("two_over_sqrt_pi"))), *t0)
    if name == "atan":
        return iv.idiv(*t0, *iv.iadd(*one, *iv.isquare(los[0], his[0])))
    if name in ("floor", "ceil", "trunc", "round", "roundeven"):
        if not np.array_equal(lo, hi):
            raise ProgramAbort(f"{op.node_id}: {name} may jump inside the interval; derivative undefined")
        z = np.zeros(shape)
        return z, z
    raise ProgramAbort(f"{op.node_id}: no derivative rule for {name}")



def _ieee_eval(name, reps):
    with np.errstate(all="ignore"):
        a = reps[0]
        if name == "add":
            return a + reps[1]
        if name == "sub":
            return a - reps[1]
        if name == "mul":
            return a * reps[1]
        if name == "div":
            return a / reps[1]
        if name == "rcp":
            return 1.0 / a
        if name == "neg":
            return -a
        if name == "abs":
            return np.abs(a)
        if name == "fma":
            return a * reps[1] + reps[2]
        if name in ("cast", "identity"):
            return a
        if name in ("floor", "ceil", "trunc", "roundeven", "round"):
            return np.floor(a) if name == "floor" else np.ceil(a) if name == "ceil" else np.trunc(a)
        fn = {"exp": np.exp, "exp2": np.exp2, "log": np.log, "log2": np.log2, "log1p": np.log1p,
              "sqrt": np.sqrt, "tanh": np.tanh, "erf": None, "sin": np.sin, "cos": np.cos,
              "expm1": np.expm1, "log10": np.log10, "atan": np.arctan, "sinh": np.sinh, "cosh": np.cosh}.get(name)
        if name == "rsqrt":
            return 1.0 / np.sqrt(a)
        if name in ("bessel_j0", "bessel_j1"):  # -> 0 at +-inf
            return np.where(np.isnan(a), np.nan, 0.0)
        if name in ("bessel_y0", "bessel_y1"):  # -> 0 at +inf; undefined for x < 0
            return np.where(a == np.inf, 0.0, np.nan)
        if name == "bessel_i0":  # even, -> +inf at +-inf
            return np.where(np.isnan(a), np.nan, np.inf)
        if name == "bessel_i1":  # odd, -> +-inf
            return np.where(np.isnan(a), np.nan, np.sign(a) * np.inf)
        if name == "erf":
            return np.where(np.isnan(a), np.nan, np.sign(a))
        if name == "erfc":   # erfc(+inf) = 0, erfc(-inf) = 2
            return np.where(np.isnan(a), np.nan, 1.0 - np.sign(a))
        if name == "pow":
            return np.power(a, reps[1])
        # limits at +-inf (C99 Annex F); 4.0, found by the per-signature boundary tests
        if name == "asinh":
            return np.arcsinh(a)
        if name == "acosh":
            return np.arccosh(a)
        if name == "cbrt":
            return np.cbrt(a)
        if name == "exp10":
            return np.power(10.0, a)
        if name == "saturate":
            return np.where(np.isnan(a), np.nan, np.clip(a, 0.0, 1.0))
        if name == "tan":
            return np.tan(a)
        if fn is None:
            return np.full(a.shape, np.nan) * np.where(np.isnan(a), 1, np.nan)
        return fn(a)


def _minmax(name, op, args) -> TV:
    a, b = args
    is_max = name in ("maxnum", "maximum")
    propagate = name in ("maximum", "minimum")
    shape = a.shape
    st = np.zeros(shape, dtype=np.int8)
    a_nan, b_nan = a.st == ST_NAN, b.st == ST_NAN
    a_lo, a_hi = _float_reps(a, "lo"), _float_reps(a, "hi")
    b_lo, b_hi = _float_reps(b, "lo"), _float_reps(b, "hi")
    fn = np.maximum if is_max else np.minimum
    with np.errstate(invalid="ignore"):
        lo = fn(np.where(a_nan, b_lo, a_lo), np.where(b_nan, a_lo, b_lo))
        hi = fn(np.where(a_nan, b_hi, a_hi), np.where(b_nan, a_hi, b_hi))
    both_nan = a_nan & b_nan
    if propagate:
        st = np.where(a_nan | b_nan, ST_NAN, st)
    else:
        st = np.where(both_nan, ST_NAN, st)
    st = np.where((st == ST_OK) & (lo == np.inf) & (hi == np.inf), ST_PINF, st)
    st = np.where((st == ST_OK) & (lo == -np.inf) & (hi == -np.inf), ST_NINF, st)
    st = np.where((st == ST_OK) & ~(np.isfinite(lo) & np.isfinite(hi)), ST_NE, st)
    undef = _merge_status(a, b)
    st = np.where(undef >= ST_UNDEF, undef, st).astype(np.int8)
    lo = np.where(st == ST_OK, lo, 0.0)
    hi = np.where(st == ST_OK, hi, 0.0)
    out = _ftv(op.result_types[0].elem, lo, hi, st, _merge_cond(a, b), _merge_reasons(a, b))
    if a.d is not None or b.d is not None:
        ta, tb = a.tangent(), b.tangent()
        ta = (np.broadcast_to(ta[0], shape), np.broadcast_to(ta[1], shape))
        tb = (np.broadcast_to(tb[0], shape), np.broadcast_to(tb[1], shape))
        with np.errstate(invalid="ignore"):
            a_wins = (a_lo > b_hi) if is_max else (a_hi < b_lo)
            b_wins = (b_lo > a_hi) if is_max else (b_hi < a_lo)
        a_wins = (a_wins & ~a_nan) | (b_nan & ~a_nan)
        b_wins = (b_wins & ~b_nan) | (a_nan & ~b_nan)
        dlo = np.where(a_wins, ta[0], np.where(b_wins, tb[0], np.minimum(ta[0], tb[0])))
        dhi = np.where(a_wins, ta[1], np.where(b_wins, tb[1], np.maximum(ta[1], tb[1])))
        out.d = (np.where(st == ST_OK, dlo, 0.0), np.where(st == ST_OK, dhi, 0.0))
        if not (a_wins | b_wins | (st != ST_OK)).all():
            out.reasons = out.reasons | {f"nonsmooth_union:{op.node_id}"}
    return out


def _clamp(op, args) -> TV:
    x, lo_b, hi_b = args
    propagate = op.attrs.get("propagateNan", "none") != "none"
    mx = _minmax("maximum" if propagate else "maxnum", op, [x, lo_b])
    return _minmax("minimum" if propagate else "minnum", op, [mx, hi_b])


_PRED = {"oeq": ("eq", False), "ogt": ("gt", False), "oge": ("ge", False), "olt": ("lt", False),
         "ole": ("le", False), "one": ("ne", False), "ueq": ("eq", True), "ugt": ("gt", True),
         "uge": ("ge", True), "ult": ("lt", True), "ule": ("le", True), "une": ("ne", True)}


def _cmpf(op, args) -> TV:
    a, b = args
    pred = op.attrs["predicate"]
    shape = np.broadcast_shapes(a.shape, b.shape)
    a_lo, a_hi = _float_reps(a, "lo"), _float_reps(a, "hi")
    b_lo, b_hi = _float_reps(b, "lo"), _float_reps(b, "hi")
    nan = (a.st == ST_NAN) | (b.st == ST_NAN)
    if a is b and pred in _PRED:
        # The same SSA value on both sides (e.g. the NaN test x != x in Inductor's maximum / clamp): the two
        # sides are one real number, not two independent members of the interval, so only NaN decides.
        base, unordered = _PRED[pred]
        v = np.where(nan, 1 if unordered else 0, 1 if base in ("eq", "ge", "le") else 0)
        return TV("b", "i1", np.broadcast_to(v, shape).astype(np.int8), None, None, _merge_status(a, b),
                  _merge_cond(a, b), _merge_reasons(a, b))
    if pred in ("ord", "uno", "true", "false"):
        val = {"ord": ~nan, "uno": nan, "true": np.ones(shape, bool), "false": np.zeros(shape, bool)}[pred]
        v = val.astype(np.int8)
    else:
        base, unordered = _PRED[pred]
        with np.errstate(invalid="ignore"):
            if base == "gt":
                sure, poss = a_lo > b_hi, a_hi > b_lo
            elif base == "ge":
                sure, poss = a_lo >= b_hi, a_hi >= b_lo
            elif base == "lt":
                sure, poss = a_hi < b_lo, a_lo < b_hi
            elif base == "le":
                sure, poss = a_hi <= b_lo, a_lo <= b_hi
            elif base == "eq":
                sure = (a_lo == a_hi) & (b_lo == b_hi) & (a_lo == b_lo)
                poss = (a_lo <= b_hi) & (b_lo <= a_hi)
            else:  # ne
                eq_sure = (a_lo == a_hi) & (b_lo == b_hi) & (a_lo == b_lo)
                eq_poss = (a_lo <= b_hi) & (b_lo <= a_hi)
                sure, poss = ~eq_poss, ~eq_sure
        v = np.where(sure, 1, np.where(poss, MAYBE, 0)).astype(np.int8)
        v = np.where(nan, 1 if unordered else 0, v).astype(np.int8)
    st = _merge_status(a, b)
    return TV("b", "i1", v, None, None, st, _merge_cond(a, b), _merge_reasons(a, b))


def _float_test(name, x: TV) -> TV:
    if name == "isnan":
        v = x.st == ST_NAN
    elif name == "isinf":
        v = (x.st == ST_PINF) | (x.st == ST_NINF)
    elif name == "isfinite":
        v = x.st == ST_OK
    else:  # signbit
        neg, pos = _sign_classes(x.lo, x.hi)
        v = np.where(neg, 1, np.where(pos, 0, MAYBE))
        v = np.where(x.st == ST_NINF, 1, np.where(x.st == ST_PINF, 0, v))
        st = np.where(x.st == ST_NAN, ST_NE, _merge_status(x)).astype(np.int8)  # NaN sign is not tracked
        return TV("b", "i1", v.astype(np.int8), None, None, st, x.cond, x.reasons)
    return TV("b", "i1", v.astype(np.int8), None, None, _merge_status(x), x.cond, x.reasons)


def _sign_classes(lo, hi):
    """(definitely negative, definitely positive) including signed zeros.

    A zero endpoint contributes its IEEE sign bit; an interval whose members
    may carry both signs is in neither class.
    """

    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    lo_neg = (lo < 0) | ((lo == 0) & np.signbit(lo))
    hi_neg = (hi < 0) | ((hi == 0) & np.signbit(hi))
    neg = hi_neg & lo_neg
    pos = ~lo_neg & ~hi_neg
    return neg, pos


def _round_half_away(x):
    a = np.abs(x)
    f = np.floor(a)
    r = np.where(a - f >= 0.5, f + 1.0, f)  # a - f is exact
    return np.copysign(r, x)


def _hull_tv(a: TV, b: TV) -> TV:
    st = np.where(a.st == b.st, a.st, ST_NE).astype(np.int8)
    if a.kind == "f":
        lo = np.minimum(a.lo, b.lo)
        hi = np.maximum(a.hi, b.hi)
        out = _ftv(a.elem, np.where(st == ST_OK, lo, 0.0), np.where(st == ST_OK, hi, 0.0), st,
                   a.cond | b.cond, a.reasons | b.reasons)
        if a.d is not None or b.d is not None:
            ta, tb = a.tangent(), b.tangent()
            out.d = (np.minimum(ta[0], tb[0]), np.maximum(ta[1], tb[1]))
        return out
    if a.kind == "b":
        v = np.where(a.lo == b.lo, a.lo, MAYBE).astype(np.int8)
        return TV("b", a.elem, v, None, None, st, a.cond | b.cond, a.reasons | b.reasons)
    same = (a.lo == b.lo) & ((a.base == b.base) if a.kind == "p" else True)
    hi = None
    if a.kind == "i" and (a.hi is not None or b.hi is not None):   # integer set targets (increment 14)
        a_hi = a.hi if a.hi is not None else a.lo
        b_hi = b.hi if b.hi is not None else b.lo
        same = same & (np.asarray(a_hi) == np.asarray(b_hi))
        hi = a_hi
    st = np.where(same, st, ST_NE).astype(np.int8)
    return TV(a.kind, a.elem, a.lo, hi, a.base, st, a.cond | b.cond, a.reasons | b.reasons)


def _hull_buffers(a: Buffer, b: Buffer) -> Buffer:
    out = a.copy()
    differs = (a.lo != b.lo) | (a.st != b.st)
    if a.hi is not None:
        differs |= a.hi != b.hi
    if differs.any():
        if a.kind == "f":
            out.lo = np.minimum(a.lo, b.lo)
            out.hi = np.maximum(a.hi, b.hi)
            out.st = np.where(a.st == b.st, a.st, ST_NE).astype(np.int8)
        else:
            out.st = np.where(differs, ST_NE, a.st).astype(np.int8)
    out.cond = a.cond | b.cond
    if a.d is not None or b.d is not None:
        z = np.zeros(a.st.shape)
        ta = a.d if a.d is not None else (z, z)
        tb = b.d if b.d is not None else (z, z)
        out.d = (np.minimum(ta[0], tb[0]), np.maximum(ta[1], tb[1]))
    out.written = a.written | b.written
    out.writer = np.where(a.writer == b.writer, a.writer, np.maximum(a.writer, b.writer))
    return out


def _select(op, args) -> TV:
    c, a, b = args
    shape = np.broadcast_shapes(c.shape, a.shape, b.shape)
    cv = np.broadcast_to(c.lo, shape)

    def pick(x, y):
        if x is None:
            return None
        return np.where(cv == 1, np.broadcast_to(x, shape), np.broadcast_to(y, shape))

    out = TV(a.kind, a.elem, pick(a.lo, b.lo), pick(a.hi, b.hi), pick(a.base, b.base),
             pick(a.st, b.st), pick(a.cond, b.cond) | np.broadcast_to(c.cond, shape),
             a.reasons | b.reasons | c.reasons)
    if a.kind == "f" and (a.d is not None or b.d is not None):
        ta, tb = a.tangent(), b.tangent()
        out.d = (pick(ta[0], tb[0]), pick(ta[1], tb[1]))
    maybe = cv == MAYBE
    if maybe.any():
        hull = _hull_tv(a.map(lambda x: np.broadcast_to(x, shape)), b.map(lambda x: np.broadcast_to(x, shape)))
        out.lo = np.where(maybe, hull.lo, out.lo)
        if out.hi is not None:
            out.hi = np.where(maybe, hull.hi, out.hi)
        if out.d is not None:
            out.d = (np.where(maybe, hull.d[0], out.d[0]), np.where(maybe, hull.d[1], out.d[1]))
        out.st = np.where(maybe, hull.st, out.st)
        out.reasons = out.reasons | {f"path_union:{op.node_id}"}
    out.st = np.where(np.broadcast_to(c.st, shape) >= ST_UNDEF, np.broadcast_to(c.st, shape), out.st).astype(np.int8)
    return out


# Ops whose float outputs legitimately have zero tangent even when an input has one.
_ZERO_TANGENT_OK = {"cmpf", "isnan", "isinf", "isfinite", "signbit", "fptosi", "fptoui"}

_INT_OPS = {"addi", "subi", "muli", "divsi", "divui", "remsi", "remui", "andi", "ori", "xori", "shli",
            "shrsi", "shrui", "maxsi", "minsi", "maxui", "minui", "extsi", "extui", "trunci", "ceildivsi",
            "ceildivui", "floordivsi", "mulhiui", "absi"}


def _int_op(name, op, args) -> TV:
    out_elem = op.result_types[0].elem
    width = INT_WIDTH[out_elem]
    st = _merge_status(*args)
    cond = _merge_cond(*args)
    reasons = _merge_reasons(*args)
    if out_elem == "i1" and name in ("andi", "ori", "xori"):
        a, b = (x.lo.astype(np.int8) for x in args)
        if name == "andi":
            v = np.where((a == 0) | (b == 0), 0, np.where((a == 1) & (b == 1), 1, MAYBE))
        elif name == "ori":
            v = np.where((a == 1) | (b == 1), 1, np.where((a == 0) & (b == 0), 0, MAYBE))
        else:
            v = np.where((a == MAYBE) | (b == MAYBE), MAYBE, a ^ b)
        return TV("b", "i1", v.astype(np.int8), None, None, st, cond, reasons)
    if name in ("extsi", "extui", "trunci"):
        x = args[0]
        in_w = INT_WIDTH[x.elem]
        vals = x.lo.astype(np.int64)
        if x.kind == "b":
            if (vals == MAYBE).any():
                st = np.where(vals == MAYBE, ST_NE, st).astype(np.int8)
            vals = np.where(vals == MAYBE, 0, vals)
            if name == "extsi":
                vals = -vals
        elif name == "extui":
            vals = _unsigned(vals, in_w).astype(np.int64)
        v = _wrap(vals, width)
        kind = "b" if out_elem == "i1" else "i"
        if kind == "b":
            v = (v & 1).astype(np.int8)
        return TV(kind, out_elem, v, None, None, st, cond, reasons)
    vals = [x.lo.astype(np.int64) for x in args]
    with np.errstate(all="ignore"):
        if name == "addi":
            v = vals[0] + vals[1]
        elif name == "subi":
            v = vals[0] - vals[1]
        elif name == "muli":
            v = vals[0] * vals[1]
        elif name in ("divsi", "remsi", "ceildivsi", "floordivsi"):
            a, b = vals
            zero = (b == 0) | ((a == -(1 << (width - 1))) & (b == -1))  # division by zero, INT_MIN / -1
            st = np.where(zero, ST_NE, st).astype(np.int8)
            bb = np.where(zero, 1, b)
            q = np.abs(a) // np.abs(bb) * np.where((a >= 0) == (bb >= 0), 1, -1)
            if name == "divsi":
                v = q
            elif name == "remsi":
                v = a - q * bb
            elif name == "ceildivsi":
                v = -((-a) // bb)
            else:
                v = a // bb
        elif name in ("divui", "remui", "ceildivui"):
            a, b = (_unsigned(x, width).astype(np.int64) for x in vals)
            zero = b == 0
            st = np.where(zero, ST_NE, st).astype(np.int8)
            bb = np.where(zero, 1, b)
            v = a // bb if name == "divui" else (a % bb if name == "remui" else -((-a) // bb))
        elif name == "andi":
            v = vals[0] & vals[1]
        elif name == "ori":
            v = vals[0] | vals[1]
        elif name == "xori":
            v = vals[0] ^ vals[1]
        elif name in ("shli", "shrsi", "shrui"):
            a, s = vals
            bad = (s < 0) | (s >= width)
            st = np.where(bad, ST_NE, st).astype(np.int8)
            s = np.where(bad, 0, s)
            if name == "shli":
                v = a << s
            elif name == "shrsi":
                v = a >> s
            else:
                v = (_unsigned(a, width).astype(np.int64) if width < 64 else a) >> s
        elif name in ("maxsi", "minsi"):
            v = (np.maximum if name == "maxsi" else np.minimum)(vals[0], vals[1])
        elif name in ("maxui", "minui"):
            ua, ub = (_unsigned(x, width).astype(np.int64) for x in vals)
            v = (np.maximum if name == "maxui" else np.minimum)(ua, ub)
        elif name == "mulhiui":
            if width == 32:
                ua, ub = (_unsigned(x, 32).astype(np.uint64) for x in vals)
                v = ((ua * ub) >> np.uint64(32)).astype(np.int64)
            elif width == 64:  # DSL v2 increment 3: the high 64 bits of the exact 128-bit product (Python integers)
                ua, ub = (_unsigned(x, 64) for x in vals)
                ua, ub = np.broadcast_arrays(ua, ub)
                hi = [(int(p) * int(q)) >> 64 for p, q in zip(ua.reshape(-1).tolist(), ub.reshape(-1).tolist())]
                v = np.array(hi, dtype=np.uint64).reshape(ua.shape).view(np.int64)
            else:
                raise ProgramAbort(f"{op.node_id}: mulhiui width {width}")
        elif name == "absi":
            v = np.abs(vals[0])
        else:
            raise ProgramAbort(f"{op.node_id}: integer op {name}")
    if name in ("muli", "andi"):
        # An established exact zero absorbs the other operand: x * 0 = x & 0 = 0 for every integer x, so a lane
        # whose other operand is undefined (e.g. a masked load without `other`) is still determined.
        a, b = args
        zero = ((a.st == ST_OK) & (vals[0] == 0)) | ((b.st == ST_OK) & (vals[1] == 0))
        if zero.any():
            st = np.where(zero, ST_OK, st).astype(np.int8)
            v = np.where(zero, 0, v)
    return TV("i", out_elem, _wrap(v, width), None, None, st, cond, reasons)


def _cmpi(op, args) -> TV:
    a, b = args
    pred = op.attrs["predicate"]
    width = INT_WIDTH.get(a.elem, 64) if not isinstance(a.elem, PtrType) else 64
    st = _merge_status(a, b)
    if a.kind == "b":
        av, bv = a.lo.astype(np.int64), b.lo.astype(np.int64)
        maybe = (av == MAYBE) | (bv == MAYBE)
    else:
        av, bv = a.lo.astype(np.int64), b.lo.astype(np.int64)
        maybe = np.zeros(np.broadcast_shapes(av.shape, bv.shape), dtype=bool)
    if pred.startswith("u"):
        av, bv = _unsigned(av, width).astype(np.int64), _unsigned(bv, width).astype(np.int64)
    fn = {"eq": np.equal, "ne": np.not_equal, "slt": np.less, "sle": np.less_equal, "sgt": np.greater,
          "sge": np.greater_equal, "ult": np.less, "ule": np.less_equal, "ugt": np.greater,
          "uge": np.greater_equal}[pred]
    v = np.where(maybe, MAYBE, fn(av, bv)).astype(np.int8)
    return TV("b", "i1", v, None, None, st, _merge_cond(a, b), _merge_reasons(a, b))


def _float_to_int(name, op, x: TV, out_elem) -> TV:
    width = INT_WIDTH[out_elem]
    t_lo, t_hi = np.trunc(x.lo), np.trunc(x.hi)
    ok = (t_lo == t_hi) & (x.st == ST_OK)
    limit = 2.0 ** (width - 1) if name == "fptosi" else 2.0 ** width
    lower = -limit if name == "fptosi" else 0.0
    ok &= (t_lo >= lower) & (t_lo < limit)
    st = np.where(ok, ST_OK, ST_NE).astype(np.int8)
    st = np.where(x.st >= ST_UNDEF, x.st, st).astype(np.int8)
    v = np.where(ok, t_lo, 0).astype(np.int64)
    reasons = x.reasons
    if not ok.all():
        reasons = reasons | {f"not_established:float-to-int conversion is discontinuous or out of range@{op.node_id}"}
    kind = "b" if out_elem == "i1" else "i"
    return TV(kind, out_elem, v if kind == "i" else v.astype(np.int8), None, None, st, x.cond, reasons)


def _bitcast(op, x: TV, out_type: TType) -> TV:
    out_elem = out_type.elem
    if x.kind == "p" or isinstance(out_elem, PtrType):
        if x.kind == "p" and isinstance(out_elem, PtrType):
            return TV("p", out_elem, x.lo, None, x.base, x.st, x.cond, x.reasons)
        raise ProgramAbort(f"{op.node_id}: pointer/integer bitcast")
    src, dst = x.elem, out_elem
    layouts = {"f32": (np.float32, np.int32), "f64": (np.float64, np.int64), "f16": (np.float16, np.int16)}
    if x.kind == "f" and dst in INT_WIDTH and src in ("bf16", "f8E5M2", "f8E4M3FN"):
        # DSL v2 increment 6: narrow formats, the same definite-bits policy (a point exactly representable)
        if src == "bf16":
            bits, definite = _float_bits("bf16", x.lo, x.hi, x.st)
        else:
            bits, definite = _f8_bits(src, x.lo, x.hi, x.st)
        st = np.where(definite, ST_OK, ST_NE).astype(np.int8)
        st = np.where(x.st >= ST_UNDEF, x.st, st).astype(np.int8)
        reasons = x.reasons
        if not definite.all():
            reasons = reasons | {f"not_established:bit-level reinterpretation without definite bits@{op.node_id}"}
        return TV("i", dst, _wrap(bits, INT_WIDTH[dst]), None, None, st, x.cond, reasons)
    if x.kind == "f" and dst in INT_WIDTH:
        if src not in layouts:
            raise ProgramAbort(f"{op.node_id}: bitcast from {src}")
        fdt, idt = layouts[src]
        point = (x.lo == x.hi) & (x.st == ST_OK)
        vals = np.where(point, x.lo, 0.0).astype(fdt)
        representable = vals.astype(np.float64) == np.where(point, x.lo, 0.0)
        definite = point & representable
        specials = np.where(x.st == ST_NAN, np.nan, np.where(x.st == ST_PINF, np.inf, -np.inf)).astype(fdt)
        vals = np.where(x.st == ST_OK, vals, specials)
        bits = vals.view(idt).astype(np.int64)
        definite = definite | ((x.st >= ST_NAN) & (x.st <= ST_NINF))
        st = np.where(definite, ST_OK, ST_NE).astype(np.int8)
        st = np.where(x.st >= ST_UNDEF, x.st, st).astype(np.int8)
        reasons = x.reasons
        if not definite.all():
            reasons = reasons | {f"not_established:bit-level reinterpretation without definite bits@{op.node_id}"}
        return TV("i", dst, bits, None, None, st, x.cond, reasons)
    if x.kind == "i" and dst in ("bf16", "f8E5M2", "f8E4M3FN"):
        raw = np.asarray(x.lo, dtype=np.int64)
        if dst == "bf16":
            vals, st = _bits_float("bf16", raw)
        else:
            from_bytes = (raw & 0xFF).astype(np.uint8)
            vals, st = decode_storage(from_bytes.reshape(-1), "float8_e5m2" if dst == "f8E5M2" else "float8_e4m3fn")
            vals, st = vals.reshape(raw.shape), st.reshape(raw.shape)
        st = np.where(x.st >= ST_UNDEF, x.st, st).astype(np.int8)
        vals = np.where(st == ST_OK, vals, 0.0)
        return _ftv(dst, vals, vals, st, x.cond, x.reasons | {f"bit_level:{op.node_id}"})
    if x.kind == "i" and dst in layouts:
        fdt, idt = layouts[dst]
        vals = x.lo.astype(idt).view(fdt).astype(np.float64)
        st = np.where(np.isnan(vals), ST_NAN, np.where(vals == np.inf, ST_PINF,
                                                       np.where(vals == -np.inf, ST_NINF, ST_OK)))
        st = np.where(x.st >= ST_UNDEF, x.st, st).astype(np.int8)
        vals = np.where(st == ST_OK, vals, 0.0)
        return _ftv(dst, vals, vals, st, x.cond, x.reasons | {f"bit_level:{op.node_id}"})
    if x.kind == "i" and dst in INT_WIDTH:
        return TV("i", dst, _wrap(x.lo, INT_WIDTH[dst]), None, None, x.st, x.cond, x.reasons)
    raise ProgramAbort(f"{op.node_id}: bitcast {src} -> {dst}")


# ---------------------------------------------------------------------------
# Composed reference across launches
# ---------------------------------------------------------------------------


@dataclass
class SequenceReference:
    launches: list  # KernelReference per launch (sharing the final memory state)
    external_writes: list  # buffers rewritten outside the evaluated launches
    memory: dict


def _window_of(arg):
    return None if arg.window is None else np.asarray(arg.window, dtype=np.int64)


_PTX_APPROX = {"exp2", "log2", "rcp", "rsqrt", "sqrt", "tanh", "sin", "cos"}
_PTX_SETP = {"eq": "oeq", "ne": "one", "lt": "olt", "le": "ole", "gt": "ogt", "ge": "oge", "equ": "ueq",
             "neu": "une", "ltu": "ult", "leu": "ule", "gtu": "ugt", "geu": "uge"}
_PTX_FN = {"ex2": "exp2", "lg2": "log2", "rcp": "rcp", "rsqrt": "rsqrt", "sqrt": "sqrt", "tanh": "tanh",
           "sin": "sin", "cos": "cos"}


def parse_ptx_program(asm: str):
    """Parse a straight-line f32 PTX snippet of inline asm into [(pred, negated, kind, dst, srcs)], or None when an
    instruction is outside the supported subset: .reg .pred declarations; setp.<cmp>.f32; mov.f32 / mov.b32;
    add / sub / mul (any rounding / ftz modifier); fma.rn; neg / abs; ex2 / lg2 / rcp / rsqrt / sqrt / tanh / sin
    / cos (.approx or .rn, optional .ftz); each optionally predicated with @p or @!p."""
    import re

    text = asm.replace("\\n", " ").replace("\\t", " ").replace("{", " ").replace("}", " ")
    program = []
    for stmt in (s.strip() for s in text.split(";")):
        if not stmt:
            continue
        if stmt.startswith(".reg"):
            if not re.fullmatch(r"\.reg\s+\.pred\s+[\w, ]+", stmt):
                return None
            continue
        m = re.fullmatch(r"(?:@(!?)(\w+)\s+)?([\w.]+)\s+(.+)", stmt)
        if not m:
            return None
        neg, pred, opcode, rest = m.group(1) == "!", m.group(2), m.group(3), m.group(4)
        ops = [t.strip() for t in rest.split(",")]
        parts = opcode.split(".")
        base, mods = parts[0], parts[1:]
        if not mods or mods[-1] not in ("f32", "b32"):
            return None
        if base == "setp" and len(mods) == 2 and mods[0] in _PTX_SETP and len(ops) == 3:
            program.append((pred, neg, "setp", ops[0], [_PTX_SETP[mods[0]], ops[1], ops[2]]))
        elif base == "mov" and len(ops) == 2:
            program.append((pred, neg, "mov", ops[0], [ops[1]]))
        elif base in ("add", "sub", "mul") and len(ops) == 3 and set(mods[:-1]) <= {"rn", "rz", "rm", "rp", "ftz", "sat"} \
                and "sat" not in mods:
            program.append((pred, neg, base, ops[0], ops[1:]))
        elif base == "fma" and len(ops) == 4 and set(mods[:-1]) <= {"rn", "rz", "rm", "rp", "ftz"}:
            program.append((pred, neg, "fma", ops[0], ops[1:]))
        elif base in ("neg", "abs") and len(ops) == 2 and set(mods[:-1]) <= {"ftz"}:
            program.append((pred, neg, base, ops[0], ops[1:]))
        elif base in _PTX_FN and len(ops) == 2 and set(mods[:-1]) <= {"approx", "rn", "ftz", "full"}:
            program.append((pred, neg, _PTX_FN[base], ops[0], ops[1:]))
        else:
            return None
    return program or None


def ptx_zero_fills(ptx: str) -> bool:
    """True when every predicated global load in the PTX writes registers that were set to 0 just before (directly,
    or through a register holding the literal 0): the lowering then defines masked-off lanes as 0."""
    import re

    lines = ptx.splitlines()
    zero = set(re.findall(r"mov\.(?:u32|b32|u16|b16|u64|b64)\s+(%r[sd]?\d+),\s*0(?:x0)?;", ptx))
    for i, line in enumerate(lines):
        m = re.search(r"@!?%p\d+\s+ld\.global[\w.:]*\s+\{?\s*([^}\]]*?)\s*\}?,\s*\[", line)
        if not m:
            continue
        regs = [r.strip() for r in m.group(1).split(",") if r.strip()]
        window = "\n".join(lines[max(0, i - 4 * len(regs) - 2):i])
        for r in regs:
            mv = re.findall(rf"mov\.\w+\s+{re.escape(r)},\s*([^;]+);", window)
            if not mv or not (mv[-1].strip() in ("0", "0x0") or mv[-1].strip() in zero):
                return False
    return True


def evaluate_sequence(launches: list, mode: str = NumericMode.NUMERICAL_DIFFERENCE,
                      programs_for=None, pin_loads: tuple = (), masked_fill_zero: bool = False) -> SequenceReference:
    """Chain launches through one reference memory (composed reference).

    Before each launch every operand buffer is checked: if its captured bytes
    before this launch equal its captured bytes after the previous launch that
    touched it, the reference state carries over; otherwise something outside
    the evaluated launches wrote it, so it re-enters as an external input
    (captured value) and the event is recorded.
    """

    memory: dict = {}
    external = []
    results = []
    modules: dict = {}
    for position, launch in enumerate(launches):
        ttir = launch.asm.get("ttir") or launch.asm.get("ttgir")  # Gluon kernels have no TTIR (increment 10)
        module = modules.get(ttir)
        if module is None:
            module = modules[ttir] = parse_ttir(ttir)
        for arg in launch.args:
            if arg.kind != "tensor" or arg.storage_ptr not in memory:
                continue
            if _TRIGGER_PATH:   # DSL v2 increment 11: a later launch reads the composed reference memory
                _trace("launch sequence (cross-launch state):composed")
            buf = memory[arg.storage_ptr]
            before = arg.before.numpy() if hasattr(arg.before, "numpy") else arg.before
            win = _window_of(arg)
            same_window = (buf.index is None and win is None) or (
                buf.index is not None and win is not None and np.array_equal(buf.index, win))
            if not (same_window and buf.after_raw is not None and np.array_equal(buf.after_raw, before)):
                external.append({"launch": position, "buffer": arg.name, "storage": arg.storage_ptr})
                del memory[arg.storage_ptr]
        evaluator = KernelReferenceEvaluator(module, mode=mode, masked_fill_zero=masked_fill_zero,
                                             ttgir=launch.asm.get("ttgir"))
        programs = programs_for(launch) if programs_for is not None else None
        results.append(evaluator.evaluate(launch, programs=programs, pin_loads=pin_loads, memory=memory))
    return SequenceReference(results, external, memory)
