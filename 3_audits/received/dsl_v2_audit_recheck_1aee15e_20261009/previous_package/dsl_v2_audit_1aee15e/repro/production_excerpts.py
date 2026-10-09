"""Manually transcribed function excerpts, not a replacement evaluator.

Origin: wuyii8941/kernel-analyzer
Commit: 1aee15e9df7a45f434699b7016056e6650fdae99
File: src/kernel_analyzer/reference_eval/ttir_eval.py
Git blob: f44d5f8211fba046439e282a52c766e4f12dc233

Function bodies below were read using the GitHub connector. No GPU execution or
complete repository installation is represented by these excerpts. The audit
harness supplies minimal state objects and, for _scaled_dot, a separately
specified elementary backend. --repo can check the AST against a local checkout.
"""

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
            self._track_read(buf, idx, state)
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
