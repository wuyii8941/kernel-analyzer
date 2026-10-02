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
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import intervals as iv
from .ttir_mapping import LIBDEVICE, inline_asm_internal, recognize_combiner, rule_for
from .ttir_parser import PtrType, TFunc, TModule, TOp, TRegion, TType, parse_ttir

ST_OK, ST_NAN, ST_PINF, ST_NINF, ST_UNDEF, ST_NE = 0, 1, 2, 3, 4, 5
MAYBE = 2

INT_WIDTH = {"i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64}
FLOAT_ELEMS = {"f16", "bf16", "f32", "f64", "f8E5M2", "f8E4M3FN", "f8E4M3FNUZ", "f8E5M2FNUZ",
               "f8E4M3B11FNUZ", "tf32"}
ELEM_SIZE = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "f8E5M2": 1, "f8E4M3FN": 1, "f8E4M3FNUZ": 1,
             "f8E5M2FNUZ": 1, "f8E4M3B11FNUZ": 1, "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8}
TORCH_TO_ELEM = {"float32": "f32", "float16": "f16", "bfloat16": "bf16", "float64": "f64",
                 "float8_e5m2": "f8E5M2", "float8_e4m3fn": "f8E4M3FN", "int8": "i8", "uint8": "i8",
                 "int16": "i16", "int32": "i32", "int64": "i64", "bool": "i1"}


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
    return TV("f", elem, np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64), None,
              np.asarray(st, dtype=np.int8), np.asarray(cond, dtype=bool), reasons)


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

    def copy(self) -> "Buffer":
        return Buffer(self.ident, self.elem, self.kind, self.lo.copy(),
                      None if self.hi is None else self.hi.copy(), self.st.copy(), self.cond.copy(),
                      self.writer.copy(), self.written.copy(), self.actual_after, self.actual_after_st,
                      self.name, self.index, self.after_raw, self.dtype,
                      None if self.d is None else (self.d[0].copy(), self.d[1].copy()))

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
        np_dtype = {"int8": np.int8, "uint8": np.uint8, "int16": np.int16, "int32": np.int32,
                    "int64": np.int64, "bool": np.uint8}[dtype]
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
class ProgramState:
    pid: tuple
    pid_index: int
    grid: tuple
    memory: dict  # ident -> Buffer


@dataclass
class KernelReference:
    kernel: str
    mode: str
    buffers: dict
    programs: list
    aborted: dict
    notes: list = field(default_factory=list)
    reasons: dict = field(default_factory=dict)

    def element_classes(self, ident: int):
        buf = self.buffers[ident]
        cls = np.full(buf.st.shape, "complete_composed", dtype=object)
        cls = np.where(buf.cond, "conditional_local", cls)
        cls = np.where(buf.st >= ST_UNDEF, "not_established", cls)
        return cls

    def compare(self) -> dict:
        """Per output buffer: classes, residual ``actual - reference`` and widths."""

        report = {}
        for ident, buf in self.buffers.items():
            mask = buf.written
            if not mask.any():
                continue
            cls = self.element_classes(ident)[mask]
            entry = {
                "name": buf.name, "elem": buf.elem, "written": int(mask.sum()),
                "classes": {c: int((cls == c).sum()) for c in ("complete_composed", "conditional_local",
                                                               "not_established")},
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
            report[buf.name or str(ident)] = entry
        return report

    def residual_arrays(self, ident: int):
        """(mask, actual, ref_lo, ref_hi, status, cond) over the written elements."""

        buf = self.buffers[ident]
        m = buf.written
        return m, buf.actual_after[m], buf.lo[m], buf.hi[m], buf.st[m], buf.cond[m]


class KernelReferenceEvaluator:
    def __init__(self, module: TModule, mode: str = NumericMode.NUMERICAL_DIFFERENCE,
                 func_name: Optional[str] = None):
        self.module = module
        self.func = module.entry(func_name)
        self.mode = mode
        self._uses = self._collect_uses()

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
                    bindings[name] = TV(kind, ttype.elem, np.array(int(arg.value), dtype=np.int64))
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
        self._outside_window = False
        for buf in memory.values():
            buf.writer[:] = -1  # kernel boundaries order all earlier writes
        aborted = {}
        reasons = collections.Counter()
        for index, pid in enumerate(programs):
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
        notes = []
        if self._outside_window:
            notes.append("some accesses fell outside the captured windows; those lanes are not established")
        if len(programs) < grid[0] * grid[1] * grid[2]:
            notes.append(f"evaluated {len(programs)} of {grid[0] * grid[1] * grid[2]} program instances; "
                         "races with unevaluated instances are not checked")
        return KernelReference(self.func.name, self.mode, memory, [tuple(p) for p in programs], aborted,
                               notes, dict(reasons))

    # -- functions and regions ----------------------------------------------------

    def _call(self, func: TFunc, args: list, state: ProgramState, env_override=None) -> list:
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
        args = [env[v] for v in op.operands] if rule.internal not in ("for", "if", "while") else None
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
        if index.min() < 0 or index.max() >= src.shape[axis]:
            raise ProgramAbort(f"{op.node_id}: gather index out of range")
        return src.map(lambda a: np.take_along_axis(a, index, axis=axis))

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

    def _op_poison(self, op, args, env, state):
        ttype = op.result_types[0]
        kind = kind_of(ttype.elem)
        lo = np.zeros(ttype.shape, dtype=np.float64 if kind == "f" else np.int64)
        return TV(kind, ttype.elem, lo, lo.copy() if kind == "f" else None,
                  np.zeros(ttype.shape, dtype=np.int64) if kind == "p" else None,
                  np.full(ttype.shape, ST_UNDEF, dtype=np.int8))

    def _op_nop(self, op, args, env, state):
        return None

    def _op_assert(self, op, args, env, state):
        cond = args[0]
        if (cond.lo == 0).any() and (cond.st == ST_OK).any():
            self._reasons[f"reference violates assert at {op.node_id}"] += 1
        return None

    def _op_return(self, op, args, env, state):
        return None

    # ---- memory -------------------------------------------------------------------

    def _addresses(self, op, ptr: TV, state: ProgramState):
        pointee = ptr.elem.pointee
        size = ELEM_SIZE[pointee]
        idents = np.unique(ptr.base[ptr.st == ST_OK]) if ptr.base is not None else []
        if len(idents) > 1:
            raise ProgramAbort(f"{op.node_id}: pointer tensor spans several buffers")
        if len(idents) == 0:
            return None, None, None
        buf = state.memory[int(idents[0])]
        if ELEM_SIZE[buf.elem] != size or (kind_of(pointee) != buf.kind and not
                                           (pointee == "i1" and buf.elem == "i8")):
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
            cond = np.where(safe, src_cond[idx], False)
            if kind == "f" and buf.d is not None and not pinned:
                dlo = np.where(safe, buf.d[0][idx], 0.0)
                dhi = np.where(safe, buf.d[1][idx], 0.0)
            race = safe & (buf.writer[idx] != -1) & (buf.writer[idx] != state.pid_index)
            if race.any():
                st = np.where(race, ST_NE, st)
                reasons.add(f"not_established:cross-program race on load@{op.node_id}")
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
        elif inactive.any():
            reasons.add(f"undefined:masked load without other@{op.node_id}")
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
        buf, index, in_range = self._addresses(op, ptr, state)
        if buf is None:
            return None
        m = np.ones(shape, dtype=np.int8) if mask is None else mask.lo.astype(np.int8)
        m_st = np.zeros(shape, dtype=np.int8) if mask is None else mask.st
        active = (m != 0)
        v_lo = np.broadcast_to(value.lo, shape)
        v_hi = np.broadcast_to(value.hi, shape) if value.kind == "f" else None
        v_st = np.broadcast_to(value.st, shape).copy()
        v_cond = np.broadcast_to(value.cond, shape) | (False if mask is None else mask.cond)
        v_st = np.where(ptr.st >= ST_UNDEF, ST_NE, v_st)
        if mask is not None:
            v_st = np.where((m == MAYBE) | (m_st >= ST_UNDEF), ST_NE, v_st)
        oob = active & ~in_range
        if (oob & (ptr.st == ST_OK)).any():
            self._reasons[f"out-of-bounds store at {op.node_id}"] += 1
        sel = active & in_range
        idx = index[sel].astype(np.int64)
        lo_w = v_lo[sel]
        if buf.kind == "i" and buf.dtype in ("uint8", "bool"):
            lo_w = np.asarray(lo_w, dtype=np.int64) & 0xFF
        elif buf.kind == "i":
            lo_w = _wrap(np.asarray(lo_w, dtype=np.int64), INT_WIDTH[buf.elem])
        hi_w = v_hi[sel] if v_hi is not None else None
        st_w = v_st[sel]
        cond_w = v_cond[sel]
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
                bad_addr = set(sidx[1:][differ].tolist())
                if bad_addr:
                    dup = np.isin(idx, list(bad_addr))
                    self._reasons[f"conflicting lanes in one store at {op.node_id}"] += 1
            st_w = np.where(dup, ST_NE, st_w)
            other_writer = (buf.writer[idx] != -1) & (buf.writer[idx] != state.pid_index)
            if other_writer.any():
                old_differs = (buf.lo[idx] != lo_w) | (buf.st[idx] != st_w)
                if hi_w is not None:
                    old_differs = old_differs | (buf.hi[idx] != hi_w)
                race = other_writer & old_differs
                st_w = np.where(race, ST_NE, st_w)
                if race.any():
                    self._reasons[f"cross-program write race at {op.node_id}"] += 1
            if buf.kind == "f" and (value.d is not None or buf.d is not None):
                if buf.d is None:
                    buf.d = (np.zeros(buf.st.shape), np.zeros(buf.st.shape))
                t = value.tangent()
                buf.d[0][idx] = np.broadcast_to(t[0], shape)[sel]
                buf.d[1][idx] = np.broadcast_to(t[1], shape)[sel]
            buf.lo[idx] = lo_w
            if hi_w is not None:
                buf.hi[idx] = hi_w
            buf.st[idx] = st_w
            buf.cond[idx] = cond_w
            buf.writer[idx] = state.pid_index
            buf.written[idx] = True
        for r in value.reasons:
            self._reasons[r] += 1
        return None

    def _op_atomic_rmw(self, op, args, env, state):
        ptr, value = args[0], args[1]
        mask = args[2] if len(args) > 2 else None
        kind_name = op.attrs["rmw_op"]
        shape = ptr.shape
        buf, index, in_range = self._addresses(op, ptr, state)
        m = np.ones(shape, dtype=np.int8) if mask is None else mask.lo.astype(np.int8)
        if mask is not None and ((m == MAYBE) | (mask.st >= ST_UNDEF)).any():
            raise ProgramAbort(f"{op.node_id}: atomic with undecided mask")
        active = (m == 1) & in_range
        idx = index[active].astype(np.int64)
        result_used = op.results and op.results[0] in self._uses
        olds = TV(value.kind, value.elem, np.zeros(shape), np.zeros(shape) if value.kind == "f" else None,
                  None, np.full(shape, ST_NE, dtype=np.int8), np.zeros(shape, dtype=bool),
                  frozenset({f"not_established:atomic return value depends on an undeclared order@{op.node_id}"}))
        if idx.size == 0:
            return olds if op.results else None
        plain = (buf.writer[idx] >= 0) & (buf.writer[idx] != state.pid_index)
        contrib_lo = np.broadcast_to(value.lo, shape)[active]
        contrib_hi = np.broadcast_to(value.hi, shape)[active] if value.kind == "f" else None
        contrib_st = np.broadcast_to(value.st, shape)[active]
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
        elif kind_name in ("add",):
            np.add.at(buf.lo, idx, contrib_lo.astype(np.int64))
            buf.lo[uniq] = _wrap(buf.lo[uniq], INT_WIDTH[buf.elem])
        elif kind_name in ("max", "min", "umax", "umin", "and", "or", "xor"):
            if buf.kind == "f":
                fn = np.maximum if kind_name == "max" else np.minimum
                fn.at(buf.lo, idx, contrib_lo)
                fn.at(buf.hi, idx, contrib_hi)
            else:
                fn = {"max": np.maximum, "min": np.minimum, "and": np.bitwise_and, "or": np.bitwise_or,
                      "xor": np.bitwise_xor}.get(kind_name)
                if fn is None:
                    raise ProgramAbort(f"{op.node_id}: atomic {kind_name}")
                fn.at(buf.lo, idx, contrib_lo.astype(np.int64))
        else:
            raise ProgramAbort(f"{op.node_id}: atomic {kind_name} is order dependent")
        buf.st[idx] = np.where(plain, ST_NE, buf.st[idx])
        buf.writer[idx] = -2
        buf.written[idx] = True
        return olds if op.results else None

    # ---- control flow -----------------------------------------------------------

    def _op_for(self, op, args, env, state):
        lb, ub, step = (self._scalar_int(env[v], op) for v in op.operands[:3])
        carried = [env[v] for v in op.operands[3:]]
        region = op.regions[0]
        i = lb
        count = 0
        while (step > 0 and i < ub) or (step < 0 and i > ub):
            iv_value = TV("i", op.attrs["iv_type"].elem, np.array(i, dtype=np.int64))
            _, carried = self._run_region(region, env, state, [iv_value] + carried)
            i += step
            count += 1
            if count > 10_000_000:
                raise ProgramAbort("loop bound too large")
        return carried

    def _scalar_int(self, value: TV, op) -> int:
        if value.st.max() >= ST_UNDEF or value.kind not in ("i", "b"):
            raise ProgramAbort(f"{op.node_id}: loop bound not established")
        return int(value.lo)

    def _op_if(self, op, args, env, state):
        cond = env[op.operands[0]]
        if cond.st.max() >= ST_UNDEF:
            raise ProgramAbort(f"{op.node_id}: undefined branch condition")
        c = int(cond.lo)
        if c in (0, 1):
            region = op.regions[0] if c == 1 else (op.regions[1] if len(op.regions) > 1 else None)
            if region is None:
                return []
            _, out = self._run_region(region, env, state, [])
            return [self._with_cond(v, cond) for v in out]
        # Undecided: run both branches on copies of the memory and take the union.
        saved = {k: b.copy() for k, b in state.memory.items()}
        _, then_out = self._run_region(op.regions[0], env, state, [])
        then_mem = state.memory
        state.memory = {k: b.copy() for k, b in saved.items()}
        else_out = []
        if len(op.regions) > 1:
            _, else_out = self._run_region(op.regions[1], env, state, [])
        else_mem = state.memory
        state.memory = then_mem
        for k in state.memory:
            state.memory[k] = _hull_buffers(then_mem[k], else_mem[k])
        info = f"path_union:{op.node_id}"
        return [_hull_tv(a, b).with_reason(info) for a, b in zip(then_out, else_out)]

    def _with_cond(self, value: TV, cond: TV) -> TV:
        if not cond.cond.any() and not cond.reasons:
            return value
        return TV(value.kind, value.elem, value.lo, value.hi, value.base, value.st,
                  value.cond | bool(cond.cond.any()), value.reasons | cond.reasons)

    def _op_while(self, op, args, env, state):
        carried = [env[v] for v in op.operands]
        before, after = op.regions
        for _ in range(10_000_000):
            term, vals = self._run_region(before, env, state, carried)
            c = vals[0]
            if c.st.max() >= ST_UNDEF or int(c.lo) == MAYBE:
                raise ProgramAbort(f"{op.node_id}: undecided loop condition")
            if int(c.lo) == 0:
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
        if combiner is None:
            raise ProgramAbort(f"{op.node_id}: unrecognized reduction combiner")
        if combiner in ("argmax", "argmin"):
            return self._arg_reduce(op, args, axis, combiner)
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
        raise ProgramAbort(f"{op.node_id}: scan combiner {combiner} not supported")

    def _op_dot(self, op, args, env, state):
        a, b, c = args
        if any((v.st != ST_OK).any() for v in (a, b, c)):
            st_bad = True
        else:
            st_bad = False
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
        precision = op.attrs.get("inputPrecision", "tf32" if a.elem == "f32" else "ieee")
        out = _ftv(c.elem, lo, hi, st, cond, a.reasons | b.reasons | c.reasons | {f"dot_input_precision:{precision}"})
        if a.d is not None or b.d is not None or c.d is not None:
            ok = lambda v, t: np.where(v.st == ST_OK, t, 0.0)  # noqa: E731
            ta, tb, tc = a.tangent(), b.tangent(), c.tangent()
            da = iv.idot(ok(a, ta[0]), ok(a, ta[1]), ok(b, b.lo), ok(b, b.hi))
            db = iv.idot(ok(a, a.lo), ok(a, a.hi), ok(b, tb[0]), ok(b, tb[1]))
            d = iv.iadd(*iv.iadd(*da, *db), np.broadcast_to(tc[0], shape), np.broadcast_to(tc[1], shape))
            out.d = (np.where(st == ST_OK, d[0], 0.0), np.where(st == ST_OK, d[1], 0.0))
        return out

    # ---- elementwise --------------------------------------------------------------

    def _op_extern(self, op, args, env, state):
        symbol = op.attrs.get("symbol", "").strip('"')
        internal = LIBDEVICE.get(symbol)
        if internal is None:
            raise ProgramAbort(f"{op.node_id}: libdevice {symbol} has no declared semantics")
        return self._elementwise(internal, op, args)

    def _op_inline_asm(self, op, args, env, state):
        internal = inline_asm_internal(op.attrs.get("asm", ""))
        if internal is None:
            raise ProgramAbort(f"{op.node_id}: inline asm has no declared semantics")
        return self._elementwise(internal, op, args)

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
        vals = _unsigned(x.lo, width).astype(np.float64) if name == "uitofp" else x.lo.astype(np.float64)
        exact = np.abs(vals) <= 2.0 ** 53
        lo = np.where(exact, vals, iv.down(vals))
        hi = np.where(exact, vals, iv.up(vals))
        out = _ftv(out_elem, lo, hi, x.st, x.cond, x.reasons)
        return self._maybe_round(out, op, out_elem, declared=False)

    def _maybe_round(self, v: TV, op: TOp, fmt, declared: bool) -> TV:
        rounds = declared or self.mode == NumericMode.ROUNDING_CHECK
        if not rounds:
            return v
        if fmt not in iv.FLOAT_FORMATS:
            return v.with_reason(f"not_established:no rounding model for {fmt}@{op.node_id}")
        if op.attrs.get("rounding", "rtne") not in ("rtne",):
            st = np.where(v.st == ST_OK, ST_NE, v.st).astype(np.int8)
            return TV("f", v.elem, v.lo, v.hi, None, st, v.cond, v.reasons)
        if v.d is not None and op.attrs.get("surrogate_gradient") != "identity":
            raise ProgramAbort(f"{op.node_id}: derivative of a rounding node needs a declared surrogate gradient")
        lo, of_lo = iv.round_nearest_even(v.lo, fmt)
        hi, of_hi = iv.round_nearest_even(v.hi, fmt)
        st = v.st.copy()
        both = of_lo & of_hi & (v.st == ST_OK)
        st = np.where(both & (lo > 0), ST_PINF, st)
        st = np.where(both & (lo < 0), ST_NINF, st)
        st = np.where((of_lo != of_hi) & (v.st == ST_OK), ST_NE, st)
        lo = np.where(st == ST_OK, lo, 0.0)
        hi = np.where(st == ST_OK, hi, 0.0)
        out = _ftv(v.elem, lo, hi, st, v.cond, v.reasons)
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
        elif name in ("div", "rcp"):
            num_lo, num_hi = (los[0], his[0]) if name == "div" else (np.ones(shape), np.ones(shape))
            den_lo, den_hi = (los[1], his[1]) if name == "div" else (los[0], his[0])
            ok = ~((den_lo <= 0) & (den_hi >= 0))
            safe_lo = np.where(ok, den_lo, 1.0)
            safe_hi = np.where(ok, den_hi, 1.0)
            lo, hi = iv.idiv(num_lo, num_hi, safe_lo, safe_hi)
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
                  "round": lambda x: np.sign(x) * np.floor(np.abs(x) + 0.5)}[name]
            lo, hi = fn(los[0]), fn(his[0])
        elif name in ("maxnum", "minnum", "maximum", "minimum"):
            return _minmax(name, op, args)
        elif name == "clamp":
            return _clamp(op, args)
        elif name == "copysign":
            sign_neg = his[1] < 0
            sign_pos = los[1] >= 0
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
            st = np.where(special, s_st, st)
            lo = np.where(special, s_val, lo)
            hi = np.where(special, s_val, hi)
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
        if name == "erf":
            return np.where(np.isnan(a), np.nan, np.sign(a))
        if name == "pow":
            return np.power(a, reps[1])
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
        v = np.where(x.hi < 0, 1, np.where(x.lo >= 0, 0, MAYBE))
        v = np.where(x.st == ST_NINF, 1, np.where(x.st == ST_PINF, 0, v))
        return TV("b", "i1", v.astype(np.int8), None, None, _merge_status(x), x.cond, x.reasons)
    return TV("b", "i1", v.astype(np.int8), None, None, _merge_status(x), x.cond, x.reasons)


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
    st = np.where(same, st, ST_NE).astype(np.int8)
    return TV(a.kind, a.elem, a.lo, None, a.base, st, a.cond | b.cond, a.reasons | b.reasons)


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
            zero = b == 0
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
            if width != 32:
                raise ProgramAbort(f"{op.node_id}: mulhiui width {width}")
            ua, ub = (_unsigned(x, 32).astype(np.uint64) for x in vals)
            v = ((ua * ub) >> np.uint64(32)).astype(np.int64)
        elif name == "absi":
            v = np.abs(vals[0])
        else:
            raise ProgramAbort(f"{op.node_id}: integer op {name}")
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


def evaluate_sequence(launches: list, mode: str = NumericMode.NUMERICAL_DIFFERENCE,
                      programs_for=None, pin_loads: tuple = ()) -> SequenceReference:
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
        ttir = launch.asm["ttir"]
        module = modules.get(ttir)
        if module is None:
            module = modules[ttir] = parse_ttir(ttir)
        for arg in launch.args:
            if arg.kind != "tensor" or arg.storage_ptr not in memory:
                continue
            buf = memory[arg.storage_ptr]
            before = arg.before.numpy() if hasattr(arg.before, "numpy") else arg.before
            win = _window_of(arg)
            same_window = (buf.index is None and win is None) or (
                buf.index is not None and win is not None and np.array_equal(buf.index, win))
            if not (same_window and buf.after_raw is not None and np.array_equal(buf.after_raw, before)):
                external.append({"launch": position, "buffer": arg.name, "storage": arg.storage_ptr})
                del memory[arg.storage_ptr]
        evaluator = KernelReferenceEvaluator(module, mode=mode)
        programs = programs_for(launch) if programs_for is not None else None
        results.append(evaluator.evaluate(launch, programs=programs, pin_loads=pin_loads, memory=memory))
    return SequenceReference(results, external, memory)
