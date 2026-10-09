"""Producer records for upstream buffers (external audit F04 follow-up).

A Triton kernel of the call may read a float buffer that a torch / ATen op made.  Equal values prove nothing about where
a buffer came from (audit F04), so by default such a buffer is an upstream intermediate and the reference is
kernel-level.  This module supplies the evidence the audit asks for -- the real producer and its element mapping:

* the call is run once more for the same seed, **not measured**, under an ATen dispatch trace;
* every ATen op is classified: a view (same storage, no data change), a copy (each output element is an input element
  by the op's definition: clone, repeat, cat, gather, index, flip, ...), an exact constant (zeros / ones / full / fill
  with a value representable in the dtype), or anything else (computed);
* per storage, in event order, the provenance is "input" (a declared input's own storage, unchanged), "const",
  "copy" (only copies of inputs / constants / copies), or "computed"; a Triton launch that the measured reference
  records as writing a storage makes it "triton";
* the trace run is used only when its Triton launches match the measured run's launch for launch (kernel name, grid,
  argument names and dtypes).  A compiled callable runs as eager ATen ops under a dispatch mode (no Triton launch), so
  for compiled code the sequences differ and no evidence is given: the reference stays kernel-level.

The dispatch mode never wraps the measured launches.
"""
from __future__ import annotations

from typing import Optional

import torch

VIEW_OPS = {"view", "_unsafe_view", "_reshape_alias", "reshape", "t", "transpose", "permute", "slice", "select",
            "expand", "as_strided", "unsqueeze", "squeeze", "alias", "detach", "lift_fresh", "narrow", "unfold",
            "diagonal", "unbind", "split", "split_with_sizes", "chunk", "view_as", "expand_as", "movedim",
            "_conj", "conj", "resolve_conj", "resolve_neg", "unflatten", "flatten", "numpy_T", "mT", "T"}
COPY_OPS = {"clone", "contiguous", "repeat", "cat", "stack", "index_select", "gather", "index", "take", "flip",
            "roll", "expand_copy", "permute_copy", "narrow_copy", "select_copy", "slice_copy", "t_copy",
            "transpose_copy", "view_copy", "unsqueeze_copy", "squeeze_copy", "_to_copy", "copy", "where",
            "repeat_interleave", "tile", "masked_select", "_unsafe_index"}
CONST_OPS = {"zeros", "ones", "full", "zeros_like", "ones_like", "full_like", "new_zeros", "new_ones", "new_full",
             "zero", "fill", "scalar_tensor"}
CLEAN = ("input", "const", "copy")


def _base(func) -> str:
    """aten.clone.default -> clone; aten.fill_.Scalar -> fill (in-place suffix dropped)."""
    name = str(func).split(".")
    base = name[1] if len(name) > 1 else name[0]
    return base[:-1] if base.endswith("_") and not base.startswith("_") else base


def _flat(obj):
    from torch.utils._pytree import tree_flatten
    return tree_flatten(obj)[0]


def _exact_scalar(value, dtype) -> bool:
    try:
        v = float(value)
        return float(torch.tensor(v, dtype=dtype).item()) == v
    except (TypeError, ValueError, RuntimeError):
        return False


class _Trace:
    """ATen dispatch trace: (launches so far, op base name, args, kwargs, outputs, mutated argument tensors)."""

    def __init__(self, recorder):
        from torch.utils._python_dispatch import TorchDispatchMode
        trace = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                mutated = []
                for i, a in enumerate(func._schema.arguments):
                    if a.alias_info is not None and a.alias_info.is_write:
                        v = args[i] if i < len(args) else kwargs.get(a.name)
                        if torch.is_tensor(v):
                            mutated.append(v)
                trace.events.append((len(recorder.launches), _base(func), func, args, kwargs, out, mutated))
                return out

        self.events = []
        self.mode = Mode()


def _clean(prov, tensors) -> bool:
    return all(prov.get(t.untyped_storage().data_ptr(), ("computed",))[0] in CLEAN for t in tensors)


def _argument(func, args, kwargs, name):
    for i, a in enumerate(func._schema.arguments):
        if a.name == name:
            return args[i] if i < len(args) else kwargs.get(name)
    return None


def _apply(prov, event):
    """Update the per-storage provenance with one ATen event."""
    _, base, func, args, kwargs, out, mutated = event
    ins = [t for t in _flat((args, kwargs)) if torch.is_tensor(t)]
    in_ptrs = {t.untyped_storage().data_ptr() for t in ins}
    mut_ptrs = {t.untyped_storage().data_ptr() for t in mutated}
    outs = [t for t in _flat(out) if torch.is_tensor(t)]
    if base in VIEW_OPS and not mutated:
        return
    targets = mutated or outs
    # data sources: float tensors other than the written ones (integer / bool tensors are indices and masks)
    sources = [t for t in ins if t.is_floating_point() and t.untyped_storage().data_ptr() not in mut_ptrs]
    if base in CONST_OPS:
        if base in ("zeros", "zeros_like", "new_zeros", "zero", "ones", "ones_like", "new_ones"):
            exact = True
        else:
            value = _argument(func, args, kwargs, "fill_value")
            value = _argument(func, args, kwargs, "value") if value is None else value
            if torch.is_tensor(value):
                exact = value.numel() == 1 and _clean(prov, [value]) if value.is_floating_point() else True
            else:
                exact = all(not t.is_floating_point() or _exact_scalar(value, t.dtype) for t in targets)
        label = ("const", str(func)) if exact else ("computed", f"{func} (value not exact in the dtype)")
    elif base in COPY_OPS:
        if any(s_.dtype != t.dtype for s_ in sources for t in targets if t.is_floating_point()):
            label = ("computed", f"{func} (dtype change rounds)")
        elif sources and _clean(prov, sources):
            label = ("copy", str(func))
        else:
            label = ("computed", f"{func} of a computed or missing source")
    else:
        label = ("computed", str(func))
    for t in mutated:
        p = t.untyped_storage().data_ptr()
        old = prov.get(p, ("computed",))
        partial = t.numel() * t.element_size() < t.untyped_storage().nbytes()
        if label[0] == "computed" or (partial and old[0] not in CLEAN):
            prov[p] = ("computed", label[1])
        elif partial:
            prov[p] = ("copy", f"{old[1]}; {label[1]} (part of the storage)")
        else:
            prov[p] = label
    for t in outs:
        p = t.untyped_storage().data_ptr()
        if p in in_ptrs or p in mut_ptrs:
            continue                                                     # a view or the mutated argument itself
        prov[p] = label


def _signature(launch):
    return (launch.kernel_name, tuple(launch.grid),
            tuple((a.name, str(a.dtype)) for a in launch.args if a.kind == "tensor"))


def producer_records(case, seed, measured_launches, seq) -> Optional[dict]:
    """{(launch index, argument name): (provenance, producer chain)} for the float tensor arguments of the measured
    launches, from an unmeasured traced run of the call for ``seed``; None when the traced run cannot be aligned with
    the measured launches (different Triton launch sequence, an error in the traced run)."""
    from .reference_eval.capture import TritonLaunchRecorder

    try:
        inp = case.inputs(seed)
        prov = {}
        for t in _flat(inp):
            if torch.is_tensor(t) and t.is_floating_point():
                prov[t.untyped_storage().data_ptr()] = ("input", "declared input")
        rec = TritonLaunchRecorder(copy_tensors=False, keep_storages=True)
        trace = _Trace(rec)
        with rec:
            with trace.mode:
                case.launch(inp)
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 -- no evidence; the reference stays kernel-level
        return None
    if len(rec.launches) != len(measured_launches) or \
            any(_signature(a) != _signature(b) for a, b in zip(rec.launches, measured_launches)):
        return None
    out = {}
    events = iter(trace.events)
    pending = next(events, None)
    for i, (tl_, ml) in enumerate(zip(rec.launches, measured_launches)):
        while pending is not None and pending[0] <= i:
            _apply(prov, pending)
            pending = next(events, None)
        for a in tl_.args:
            if a.kind == "tensor" and str(a.dtype).startswith(("float", "bfloat")):
                out[(i, a.name)] = prov.get(a.storage_ptr, ("computed", "no producer recorded"))
        stored_measured = getattr(seq.launches[i], "stored", set())
        names = {b.name for b in ml.args if b.kind == "tensor" and b.storage_ptr in stored_measured}
        for a in tl_.args:
            if a.kind == "tensor" and a.name in names:
                prov[a.storage_ptr] = ("triton", tl_.kernel_name)
    return out
