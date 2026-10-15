"""Textbook premises: the same kernel, on the same inputs, executed under the assumptions an error analysis on paper
makes instead of the lowering rules read from the compiled kernel.

The emulator (emulate.GpuEmulator) reproduces the device's bits from the facts of the locked build: which instruction
an operation lowers to, the order of a reduction, which multiply-adds are contracted, how a tensor-core dot converts
and accumulates, where subnormals are flushed.  Each fact has a textbook counterpart; TextbookEmulator replaces any
subset of them:

- "functions": an approximate instruction (div.full, ex2.approx, lg2.approx, rsqrt.approx, sqrt.approx, ...) is the
  correctly rounded IEEE operation in its declared format;
- "fusion": no multiply-add contraction -- every operation rounds on its own, as written;
- "order": a sum or prefix sum adds left to right in the declared format;
- "tensor_core": a tensor-core dot rounds tf32 operands to nearest, forms exact products and accumulates them in fp32
  left to right with round to nearest (no truncation);
- "subnormals": libdevice's flush-to-zero paths keep subnormals.

With every premise replaced the emulator is a numerical analysis on paper carried out exactly on the actual data: it
keeps the declared program, its types and its roundings to nearest, and nothing it would have to read from the
compiled kernel.  With none replaced it is the emulator.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from . import emulate as E
from . import intervals as iv
from .ttir_eval import ST_NE, ST_OK, KernelReferenceEvaluator
from .ttir_mapping import recognize_combiner

PREMISES = ("functions", "fusion", "order", "tensor_core", "subnormals")


class TextbookEmulator(E.GpuEmulator):
    def __init__(self, module, ttgir: str, premises: Iterable[str] = PREMISES, enable_fp_fusion: bool = True,
                 enable_reflect_ftz: bool = True, **kw):
        self.premises = frozenset(premises)
        unknown = self.premises - set(PREMISES)
        if unknown:
            raise ValueError(f"unknown premises {sorted(unknown)}")
        super().__init__(module, ttgir, enable_fp_fusion=enable_fp_fusion and "fusion" not in self.premises,
                         enable_reflect_ftz=enable_reflect_ftz and "subnormals" not in self.premises, **kw)

    # -- functions ----------------------------------------------------------------

    def _correctly_rounded(self, name, op, args, out_elem):
        """RN of the declared operation's exact result in out_elem; not established where the enclosure's endpoints
        round apart."""
        out = self._unrounded(KernelReferenceEvaluator._float_op.__get__(self), name, op, args, out_elem)
        r_lo, r_hi = iv.round_nearest_even(out.lo, out_elem)[0], iv.round_nearest_even(out.hi, out_elem)[0]
        st = np.where((r_lo != r_hi) & (out.st == ST_OK), ST_NE, out.st).astype(np.int8)
        v = np.where((r_lo == r_hi) & (st == ST_OK), r_lo, 0.0)
        return E._ftv(out_elem, v, v.copy(), st, out.cond, out.reasons)

    def _float_op(self, name, op, args, out_elem):
        if ("functions" in self.premises and self._is_oracle(op) and op.node_id not in self.exact_nodes
                and out_elem in iv.FLOAT_FORMATS and all(a.kind == "f" for a in args)):
            self.trace[(op.node_id, "textbook_ieee")] += 1
            return self._correctly_rounded(name, op, args, out_elem)
        return super()._float_op(name, op, args, out_elem)

    def _group_oracle(self, op, args, out_elem):
        if "functions" in self.premises:     # no contraction of an approximate instruction's expansion into an add
            return None
        return super()._group_oracle(op, args, out_elem)

    # -- order ----------------------------------------------------------------------

    @staticmethod
    def _rn_add(a, b, fmt):
        lo, hi = iv.iadd(a, a, b, b)
        (r_lo, o_lo), (r_hi, o_hi) = iv.round_nearest_even(lo, fmt), iv.round_nearest_even(hi, fmt)
        r = np.where((r_lo == r_hi) & ~o_lo & ~o_hi & np.isfinite(lo) & np.isfinite(hi), r_lo, np.nan)
        return np.where(r == 0, np.copysign(0.0, a + b), r)

    def _op_reduce(self, op, args, env, state):
        if "order" not in self.premises or len(args) != 1 or args[0].kind != "f" \
                or recognize_combiner(op) != "sum" or op.node_id in self.exact_nodes:
            return super()._op_reduce(op, args, env, state)
        (x,) = args
        axis = int(op.attrs["axis"].split(":")[0])
        self.trace[(op.node_id, "textbook_sequential_sum")] += 1
        vals = np.moveaxis(np.where(x.st == ST_OK, x.lo, 0.0), axis, -1)
        acc = vals[..., 0]
        for i in range(1, vals.shape[-1]):
            acc = self._rn_add(acc, vals[..., i], x.elem)
        bad = np.any(np.moveaxis(x.st != ST_OK, axis, -1), axis=-1) | np.isnan(acc)
        st = np.where(bad, ST_NE, ST_OK).astype(np.int8)
        acc = np.where(bad, 0.0, acc)
        return E._ftv(x.elem, acc, acc.copy(), st, np.any(x.cond, axis=axis), x.reasons)

    def _op_scan(self, op, args, env, state):
        x = args[0] if args else None
        if ("order" not in self.premises or len(args) != 1 or x.kind != "f" or len(x.shape) != 1
                or recognize_combiner(op) != "sum" or op.node_id in self.exact_nodes):
            return super()._op_scan(op, args, env, state)
        self.trace[(op.node_id, "textbook_sequential_scan")] += 1
        ok = x.st == ST_OK
        vals = np.where(ok, x.lo, 0.0)
        out = np.empty_like(vals)
        acc = vals[0]
        out[0] = acc
        for i in range(1, vals.shape[0]):
            acc = self._rn_add(np.asarray(acc), np.asarray(vals[i]), x.elem)
            out[i] = acc
        bad = np.isnan(out) | ~np.logical_and.accumulate(ok)
        st = np.where(bad, ST_NE, ST_OK).astype(np.int8)
        out = np.where(bad, 0.0, out)
        return E._ftv(x.elem, out, out.copy(), st, x.cond, x.reasons)

    # -- tensor core ------------------------------------------------------------------

    def _op_dot(self, op, args, env, state):
        a, b, c = args
        precision = op.attrs.get("inputPrecision", "ieee").strip()
        tc = E.tensor_core.mma_format(a.elem, b.elem, c.elem, precision)
        if "tensor_core" not in self.premises or tc is None or op.node_id in self.exact_nodes:
            return super()._op_dot(op, args, env, state)
        self.trace[(op.node_id, f"textbook_dot_{tc}")] += 1
        x_all = iv.round_nearest_even(a.lo, "tf32")[0] if tc == "tf32" else a.lo
        y_all = iv.round_nearest_even(b.lo, "tf32")[0] if tc == "tf32" else b.lo
        shape = np.broadcast_shapes(c.shape)
        acc = c.lo.copy()
        for k in range(a.shape[-1]):        # products of 11-bit (or narrower) significands are exact in float64
            p = np.broadcast_to(x_all[..., :, k][..., :, None], shape) * np.broadcast_to(y_all[..., k, :][..., None, :], shape)
            acc = self._rn_add(acc, p, "f32")
        bad = (np.isnan(acc) | (c.st != ST_OK) | np.any(a.st != ST_OK, axis=-1)[..., :, None]
               | np.any(b.st != ST_OK, axis=-2)[..., None, :])
        st = np.where(bad, ST_NE, ST_OK).astype(np.int8)
        acc = np.where(bad, 0.0, acc)
        return E._ftv(c.elem, acc, acc.copy(), st, c.cond | np.any(a.cond, axis=-1)[..., :, None], c.reasons)


def textbook_emulate(launch, premises: Iterable[str] = PREMISES, oracle: Optional[E.HardwareOracle] = None,
                     programs: Optional[list] = None):
    """emulate.emulate with the given premises replaced by their textbook counterparts; returns (result, emulator)."""
    from .ttir_parser import parse_ttir

    module = parse_ttir(launch.asm["ttir"])
    module.source_text = launch.asm["ttir"]
    if oracle is None:
        options = {k: launch.metadata[k] for k in ("enable_fp_fusion", "enable_reflect_ftz") if k in launch.metadata}
        oracle = E.HardwareOracle(options={"num_warps": 4, **options})
    emu = TextbookEmulator(module, launch.asm["ttgir"], premises=premises,
                           enable_fp_fusion=bool(launch.metadata.get("enable_fp_fusion", True)), oracle=oracle,
                           ptx=launch.asm.get("ptx"), sass=E.launch_sass(launch),
                           enable_reflect_ftz=bool(launch.metadata.get("enable_reflect_ftz", True)))
    result = emu.evaluate(launch, programs=programs)
    return result, emu
