"""Commutativity certificate for CAS spin locks (external audit F03 follow-up).

The reference evaluates a launch with contended compare-and-swap operations as serializations of the programs and
keeps what the evaluated orders agree on; that agreement is evidence, not a proof over every allowed execution.  This
module proves, for the common lock pattern, that every serialization gives the same memory:

* pattern (static, on the entry function): ``scf.while`` whose before-region is only ``tt.atomic_cas(L, c, v)``, a
  comparison of the returned value and ``scf.condition`` (failed attempts have no effect), an empty do-region, and
  later in the same block ``tt.atomic_rmw exch L, c`` (the release); the lock pointer L is used by nothing else; the
  spin exits exactly when the CAS succeeded; values made inside the critical section (the ops between the two) are
  not used after the release; every ``tt.atomic_cas`` of the module is such an acquire;
* run (from the evaluated launch): every program passes each acquire and release exactly once, no program aborted,
  every program instance was evaluated, no race or validity finding, no atomic update outside the critical sections
  touches what a section reads or writes, and the lock initially holds one of the two lock values;
* proof: each program's critical section is replayed symbolically -- memory read in it becomes a symbol of its
  location (the state at section entry), float values it takes from outside become universally quantified reals,
  integers and pointers keep the concrete values of the evaluated program (addresses must stay concrete) -- giving a
  transformation T_p of memory.  For every pair of programs whose footprints overlap, z3 proves T_p(T_q(M)) =
  T_q(T_p(M)) for all memory M and all quantified values.  Programs whose concrete section inputs coincide share one
  replay; a pair of them is checked once with two independent instances.  Pairwise commuting transformations give one
  result for every serialization, so the evaluated result holds for every allowed execution: no premise.

Anything outside the translated operations, a symbolic address, or a check over the budget gives no certificate (the
premise stays).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import certificates as C

MAX_CLASSES = 256
MAX_PAIR_CHECKS = 400
MAX_LANES = 1 << 14

_SECTION_OPS = {"arith.constant", "tt.splat", "tt.broadcast", "tt.expand_dims", "tt.make_range", "tt.addptr",
                "tt.load", "tt.store", "gpu.barrier", "tt.atomic_rmw", "scf.if", "scf.yield"}


class Unsupported(Exception):
    pass


@dataclass
class LockSection:
    while_node: str
    release_node: str
    lock: str                       # SSA name of the lock pointer
    cmp_value: int
    val_value: int
    exit_on: str                    # "ne_cmp" (exit iff old == cmp) or "eq_val" (exit iff old != val: needs values in {cmp, val})
    ops: list                       # the critical section
    outer: list                     # values the section takes from outside
    nodes: set = field(default_factory=set)
    cas_node: str = ""
    lock_root: str = ""             # kernel parameter the lock pointer is derived from (by tt.addptr)


# ---------------------------------------------------------------------------------------------------- static pattern

def _walk(ops):
    for op in ops:
        yield op
        for r in op.regions:
            for b in r.blocks:
                yield from _walk(b.ops)


def _defs(func):
    out = {}
    for b in func.body.blocks:
        for op in _walk(b.ops):
            for r in op.results:
                out[r] = op
    return out


def _const_int(name, defs):
    op = defs.get(name)
    if op is None or op.name != "arith.constant" or op.result_types[0].shape:
        return None
    text = op.attrs.get("value", "").split(":")[0].strip()
    if text in ("true", "false"):
        return int(text == "true")
    try:
        return int(text)
    except ValueError:
        return None


def _outer_names(ops) -> list:
    defined, used = set(), []

    def visit(op_list):
        for op in op_list:
            for v in op.operands:
                if v not in defined and v not in used:
                    used.append(v)
            for r in op.regions:
                for b in r.blocks:
                    for name, _ in b.args:
                        defined.add(name)
                    visit(b.ops)
            defined.update(op.results)
    visit(ops)
    return used


def find_sections(module, func) -> tuple:
    """(sections by while node id, every CAS of the module is in a section)."""
    defs = _defs(func)
    uses = {}
    for fn in module.funcs.values():
        for b in fn.body.blocks:
            for op in _walk(b.ops):
                for v in op.operands:
                    uses[v] = uses.get(v, 0) + 1
    sections, cas_in = {}, set()
    for block in func.body.blocks:
        ops = block.ops
        for i, w in enumerate(ops):
            if w.name != "scf.while" or w.operands or len(w.regions) != 2:
                continue
            before = w.regions[0].entry.ops
            after = w.regions[1].entry.ops
            cas = [o for o in before if o.name == "tt.atomic_cas"]
            if len(cas) != 1 or any(o.name not in ("tt.atomic_cas", "arith.cmpi", "arith.constant", "scf.condition")
                                    for o in before) or any(o.name != "scf.yield" for o in after):
                continue
            cas = cas[0]
            lock, c_ssa, v_ssa = cas.operands[:3]
            cmp_value, val_value = _const_int(c_ssa, defs), _const_int(v_ssa, defs)
            if cmp_value is None or val_value is None or cmp_value == val_value or uses.get(lock, 0) != 2:
                continue
            cond = before[-1]
            cmpi = [o for o in before if o.name == "arith.cmpi"]
            if cond.name != "scf.condition" or len(cmpi) != 1 or cond.operands[0] != cmpi[0].results[0]:
                continue
            x = [v for v in cmpi[0].operands if v != cas.results[0]]
            if len(x) != 1 or cas.results[0] not in cmpi[0].operands:
                continue
            xv = _const_int(x[0], defs)
            pred = cmpi[0].attrs.get("predicate")
            if pred == "ne" and xv == cmp_value:
                exit_on = "ne_cmp"
            elif pred == "eq" and xv == val_value:
                exit_on = "eq_val"
            else:
                continue
            rel = None
            for j in range(i + 1, len(ops)):
                r = ops[j]
                if r.name == "tt.atomic_rmw" and r.attrs.get("rmw_op") == "exch" and r.operands[0] == lock:
                    rel = j
                    break
            if rel is None or _const_int(ops[rel].operands[1], defs) != cmp_value or \
                    (len(ops[rel].operands) > 2 and _const_int(ops[rel].operands[2], defs) != 1):
                continue
            section = ops[i + 1:rel]
            if any(o.name in ("tt.return", "cf.br", "cf.cond_br") for o in section):
                continue
            made = {r_ for o in _walk(section) for r_ in o.results}
            later = [v for o in _walk(ops[rel + 1:]) for v in o.operands]
            if made & set(later):
                continue
            root = lock
            while root in defs and defs[root].name == "tt.addptr":
                root = defs[root].operands[0]
            if root in defs or root not in {name for name, *_ in func.params}:
                continue                                     # the lock storage must be a kernel parameter's
            sec = LockSection(w.node_id, ops[rel].node_id, lock, cmp_value, val_value, exit_on, section,
                              _outer_names(section), {o.node_id for o in _walk(section)}, cas.node_id, root)
            sections[w.node_id] = sec
            cas_in.add(cas.node_id)
    all_cas = {o.node_id for fn in module.funcs.values() for b in fn.body.blocks for o in _walk(b.ops)
               if o.name in ("tt.atomic_cas", "amdg.buffer_atomic_cas")}
    return sections, bool(all_cas) and all_cas <= cas_in


# ---------------------------------------------------------------------------------------------------- symbolic replay

def _elem_sort(elem):
    s = C._sort_of(elem)
    if s == "bool":
        raise Unsupported("i1 memory")
    return s


class _State:
    """Symbolic memory: written locations -> terms; reads of unwritten locations give the symbol of the location."""

    def __init__(self, elems, base=None):
        self.elems = elems              # ident -> element type of the buffer
        self.mem = dict(base.mem) if base is not None else {}
        self.reads, self.writes = set(), set()

    def symbol(self, loc):
        s = _elem_sort(self.elems[loc[0]])
        name = f"m_{loc[0]}_{loc[1]}"
        return C.z3.Real(name) if s == "real" else C.z3.BitVec(name, s[1])

    def read(self, loc):
        if loc not in self.mem:
            self.reads.add(loc)
            return self.symbol(loc)
        return self.mem[loc]

    def write(self, loc, term):
        self.writes.add(loc)
        self.mem[loc] = term


def _arr(shape, fill=None):
    a = np.empty(shape, dtype=object)
    a.reshape(-1)[:] = [fill] * max(1, int(np.prod(shape))) if shape else [fill]
    return a


def _const_term(elem, value):
    return C._const(elem, value)


def _numeral(t) -> int:
    t = C.z3.simplify(t)
    if C.z3.is_bv_value(t):
        return t.as_signed_long()
    raise Unsupported("symbolic address or mask")


def _concrete_bool(t):
    t = C.z3.simplify(t)
    if C.z3.is_true(t):
        return True
    if C.z3.is_false(t):
        return False
    return None


class _Replay:
    def __init__(self, elems, elem_size, ptr_elem_size, constant):
        self.elems, self.elem_size, self.ptr_elem_size, self.constant = elems, elem_size, ptr_elem_size, constant

    def outer_value(self, name, tv, prefix):
        """A TV of the evaluated program as a symbolic value: pointers and integers concrete, floats quantified."""
        shape = np.shape(tv.lo)
        if int(np.prod(shape) if shape else 1) > MAX_LANES:
            raise Unsupported("tensor too large")
        st = np.asarray(tv.st)
        if (st != 0).any():
            raise Unsupported(f"{name}: value not established")
        if tv.kind == "p":
            a = np.empty(shape, dtype=object)
            base = np.broadcast_to(np.asarray(tv.base), shape)
            lo = np.broadcast_to(np.asarray(tv.lo), shape)
            for idx in np.ndindex(*shape) if shape else [()]:
                a[idx] = ("ptr", int(base[idx]), int(lo[idx]))
            return a
        if tv.kind == "i":
            if tv.hi is not None:
                raise Unsupported(f"{name}: integer set target")
            w = C.INT_WIDTH[tv.elem]
            a = np.empty(shape, dtype=object)
            lo = np.asarray(tv.lo)
            for idx in np.ndindex(*shape) if shape else [()]:
                a[idx] = C.z3.BitVecVal(int(lo[idx]) % (1 << w), w)
            return a
        if tv.kind == "b":
            lo = np.asarray(tv.lo)
            if (lo > 1).any():
                raise Unsupported(f"{name}: undecided Boolean")
            a = np.empty(shape, dtype=object)
            for idx in np.ndindex(*shape) if shape else [()]:
                a[idx] = C.z3.BoolVal(bool(lo[idx]))
            return a
        if tv.kind == "f":
            a = np.empty(shape, dtype=object)
            for k, idx in enumerate(np.ndindex(*shape) if shape else [()]):
                a[idx] = C.z3.Real(f"{prefix}_{name.lstrip('%')}_{k}")
            return a
        raise Unsupported(f"{name}: kind {tv.kind}")

    def loc(self, p):
        if not (isinstance(p, tuple) and p[0] == "ptr"):
            raise Unsupported("not a concrete pointer")
        _, ident, byte = p
        if ident not in self.elems:
            raise Unsupported("pointer into an unknown buffer")
        size = self.elem_size[ident]
        if byte % size:
            raise Unsupported("unaligned element")
        return ident, byte // size

    def run(self, ops, env, state):
        for op in ops:
            if op.name not in _SECTION_OPS and not op.name.startswith("arith."):
                raise Unsupported(op.name)
            if op.name == "scf.yield":
                return [env[v] for v in op.operands]
            out = self.op(op, env, state)
            if out is not None:
                for r, v in zip(op.results, out):
                    env[r] = v
        return []

    def op(self, op, env, state):
        n = op.name
        a = [env[v] for v in op.operands]
        rt = op.result_types[0] if op.result_types else None
        if n == "arith.constant":
            text = op.attrs["value"].strip()
            if text.startswith("dense<"):
                text = text[len("dense<"):-1]
                if text.startswith("["):
                    raise Unsupported("non-splat dense constant")
            from .ttir_eval import _parse_scalar_literal
            value = _parse_scalar_literal(text, rt.elem)
            return [_arr(tuple(rt.shape), _const_term(rt.elem, value))]
        if n == "tt.splat":
            return [_arr(tuple(rt.shape), a[0].reshape(-1)[0])]
        if n == "tt.broadcast":
            return [np.broadcast_to(a[0], tuple(rt.shape)).copy()]
        if n == "tt.expand_dims":
            return [np.expand_dims(a[0], int(op.attrs["axis"].split(":")[0]))]
        if n == "tt.make_range":
            s, e = int(op.attrs["start"].split(":")[0]), int(op.attrs["end"].split(":")[0])
            arr = np.empty((e - s,), dtype=object)
            for k in range(e - s):
                arr[k] = C.z3.BitVecVal(s + k, 32)
            return [arr]
        if n == "tt.addptr":
            size = self.ptr_elem_size(rt)
            p, off = np.broadcast_arrays(a[0], a[1])
            out = np.empty(p.shape, dtype=object)
            for idx in np.ndindex(*p.shape) if p.shape else [()]:
                q = p[idx]
                if not (isinstance(q, tuple) and q[0] == "ptr"):
                    raise Unsupported("addptr on a symbolic pointer")
                out[idx] = ("ptr", q[1], q[2] + _numeral(off[idx]) * size)
            return [out]
        if n == "gpu.barrier":
            return None
        if n == "tt.load":
            ptr = a[0]
            mask = a[1] if len(a) > 1 else None
            other = a[2] if len(a) > 2 else None
            out = np.empty(ptr.shape, dtype=object)
            for idx in np.ndindex(*ptr.shape) if ptr.shape else [()]:
                m = True if mask is None else _concrete_bool(mask[idx])
                if m is True:
                    out[idx] = state.read(self.loc(ptr[idx]))
                elif m is False:
                    if other is None:
                        raise Unsupported("masked load without other")
                    out[idx] = other[idx]
                else:
                    if other is None:
                        raise Unsupported("masked load without other")
                    out[idx] = C.z3.If(mask[idx], state.read(self.loc(ptr[idx])), other[idx])
            return [out]
        if n == "tt.store":
            ptr, val = a[0], a[1]
            mask = a[2] if len(a) > 2 else None
            seen = set()
            for idx in np.ndindex(*ptr.shape) if ptr.shape else [()]:
                m = True if mask is None else _concrete_bool(mask[idx])
                if m is False:
                    continue
                loc = self.loc(ptr[idx])
                if loc in seen:
                    raise Unsupported("one store writes an address from several lanes")
                seen.add(loc)
                v = val[idx] if m is True else C.z3.If(mask[idx], val[idx], state.read(loc))
                state.write(loc, v)
            return None
        if n == "tt.atomic_rmw":
            kind = op.attrs.get("rmw_op")
            if kind not in ("exch", "add", "fadd"):
                raise Unsupported(f"atomic {kind} in a critical section")
            ptr, val = a[0], a[1]
            mask = a[2] if len(a) > 2 else None
            out = np.empty(ptr.shape, dtype=object)
            seen = set()
            for idx in np.ndindex(*ptr.shape) if ptr.shape else [()]:
                m = True if mask is None else _concrete_bool(mask[idx])
                if m is None:
                    raise Unsupported("symbolic atomic mask")
                if m is False:
                    out[idx] = None
                    continue
                loc = self.loc(ptr[idx])
                if loc in seen:
                    raise Unsupported("atomic on one address from several lanes")
                seen.add(loc)
                old = state.read(loc)
                state.write(loc, val[idx] if kind == "exch" else old + val[idx])
                out[idx] = old
            return [out]
        if n == "scf.if":
            c = a[0].reshape(-1)[0] if isinstance(a[0], np.ndarray) else a[0]
            cb = _concrete_bool(c)
            regions = op.regions
            if cb is not None:
                region = regions[0] if cb else (regions[1] if len(regions) > 1 else None)
                if region is None or not region.blocks:
                    return [] if not op.results else None
                return self.run(region.entry.ops, dict(env), state) or None
            s_then = _State(state.elems, state)
            y_then = self.run(regions[0].entry.ops, dict(env), s_then) if regions[0].blocks else []
            s_else = _State(state.elems, state)
            y_else = self.run(regions[1].entry.ops, dict(env), s_else) if len(regions) > 1 and regions[1].blocks \
                else []
            for loc in s_then.writes | s_else.writes:
                state.write(loc, C.z3.If(c, s_then.mem.get(loc, state.read(loc)), s_else.mem.get(loc, state.read(loc))))
            state.reads |= s_then.reads | s_else.reads
            if not op.results:
                return None
            out = []
            for yt, ye in zip(y_then, y_else):
                arr = np.empty(np.shape(yt), dtype=object)
                for idx in np.ndindex(*arr.shape) if arr.shape else [()]:
                    arr[idx] = C.z3.If(c, yt[idx], ye[idx])
                out.append(arr)
            return out
        if n.startswith("arith."):
            if any(isinstance(x.reshape(-1)[0], tuple) for x in a if x.size):
                raise Unsupported(f"{n} on pointers")
            shape = np.broadcast_shapes(*(x.shape for x in a)) if a else ()
            xs = [np.broadcast_to(x, shape) for x in a]
            out = np.empty(shape, dtype=object)
            for idx in np.ndindex(*shape) if shape else [()]:
                try:
                    out[idx] = C._translate(op, [x[idx] for x in xs], self.constant)
                except C.Unsupported as exc:
                    raise Unsupported(str(exc)) from exc
            return [out]
        raise Unsupported(n)


# ---------------------------------------------------------------------------------------------------- certificate

@dataclass
class LockCertificate:
    proved: bool
    detail: str
    footprint: set = field(default_factory=set)   # (storage, element) read or written by some critical section


def _class_key(snapshot):
    key = []
    for name in sorted(snapshot):
        tv = snapshot[name]
        if tv.kind == "f":
            key.append((name, "f", tuple(np.shape(tv.lo))))
        else:
            key.append((name, tv.kind, np.asarray(tv.lo).tobytes(),
                        None if tv.base is None else np.asarray(tv.base).tobytes(), tuple(np.shape(tv.lo))))
    return tuple(key)


def prove(evaluator, section: LockSection, records: dict, memory) -> LockCertificate:
    """``records``: program index -> values of ``section.outer`` at section entry (the evaluated program order)."""
    if not C.available():
        return LockCertificate(False, "z3 not available")
    from .ttir_eval import ELEM_SIZE, _constant
    elems = {k: b.elem for k, b in memory.items()}
    elem_size = {k: ELEM_SIZE[b.elem] for k, b in memory.items() if b.elem in ELEM_SIZE}

    def ptr_size(rt):
        pointee = rt.elem.pointee if hasattr(rt.elem, "pointee") else None
        if pointee is None or pointee not in ELEM_SIZE:
            raise Unsupported("pointer element type")
        return ELEM_SIZE[pointee]

    replay = _Replay(elems, elem_size, ptr_size, lambda o: np.asarray(_constant(o).lo).reshape(-1)[0].item())
    try:
        classes = {}
        for pid, snap in records.items():
            classes.setdefault(_class_key(snap), []).append(pid)
        if len(classes) > MAX_CLASSES:
            raise Unsupported(f"{len(classes)} program classes")

        def transform(cls_pids, prefix, start=None):
            snap = records[cls_pids[0]]
            env = {name: replay.outer_value(name, snap[name], prefix) for name in section.outer}
            st = _State(elems, start)
            replay.run(section.ops, env, st)
            return st

        reps = {k: transform(p, "P") for k, p in classes.items()}
        keys = list(classes)
        checks = 0
        for i, ka in enumerate(keys):
            for kb in keys[i:]:
                if ka == kb and len(classes[ka]) < 2:
                    continue
                fa, fb = reps[ka], reps[kb]
                if not ((fa.reads | fa.writes) & fb.writes or fa.writes & (fb.reads | fb.writes)):
                    continue
                checks += 1
                if checks > MAX_PAIR_CHECKS:
                    raise Unsupported("too many overlapping program pairs")
                ab = transform(classes[kb], "Q", transform(classes[ka], "P"))
                ba = transform(classes[ka], "P", transform(classes[kb], "Q"))
                locs = set(ab.mem) | set(ba.mem)
                s = C.z3.Solver()
                s.set("timeout", C.TIMEOUT_MS)
                s.add(C.z3.Not(C.z3.And(*[ab.read(l) == ba.read(l) for l in locs]) if locs else C.z3.BoolVal(True)))
                r = s.check()
                if r != C.z3.unsat:
                    return LockCertificate(False, f"critical sections do not commute ({'counterexample' if r == C.z3.sat else r})")
        lock_locs = getattr(evaluator, "_lock_locs", set())
        footprint = set().union(*(f.reads | f.writes for f in reps.values())) if reps else set()
        if footprint & lock_locs:
            raise Unsupported("a critical section touches the lock")
        return LockCertificate(True, f"z3 {C.z3.get_version_string()}: the critical sections of all {len(records)} "
                                     f"programs commute pairwise ({len(classes)} program classes, {checks} pair checks)",
                               footprint)
    except Unsupported as exc:
        return LockCertificate(False, f"not translated: {exc}")
