"""Producer records for upstream buffers (external audit F04 follow-up).

A Triton kernel of the call may read a float buffer that a torch / ATen op made.  Equal values prove nothing about where
a buffer came from (audit F04), so by default such a buffer is an upstream intermediate and the reference is
kernel-level.  This module supplies the evidence the audit asks for -- the real producer and its element mapping:

* the call is run once more for the same seed, **not measured**, under an ATen dispatch trace;
* every ATen op is classified: a view (same storage, no data change), a copy (each output element is an input element
  by the op's definition: clone, repeat, cat, gather, index, flip, ...), an exact constant (zeros / ones / full / fill
  with a value representable in the dtype), or anything else (computed);
* per storage and per byte (``storage_effects.StorageMap``, rules M1, M2, M5), in event order, the provenance is
  "input" (the bytes of a declared input tensor, unchanged), "const", "copy" (only copies of inputs / constants /
  copies), or "computed"; a Triton launch that the measured reference records as writing a storage makes it
  "triton".  A write through a view changes exactly the bytes of its elements (offset, strides, overlap, stride 0):
  a fill through an overlapping or stride-0 view of an uninitialised buffer leaves the other bytes uninitialised,
  not constant.  Storages with a clean byte are held alive, so an address names one storage while its label lives;
* the label of a launch argument is the join over the bytes the measured reference read from its initial values
  (``KernelReference.loaded_elements``), not over the whole storage;
* the trace run is used only when its Triton launches match the measured run's launch for launch (kernel name, grid,
  argument names and dtypes).  A compiled callable runs as eager ATen ops under a dispatch mode (no Triton launch), so
  for compiled code the sequences differ and no evidence is given: the reference stays kernel-level.

The dispatch mode never wraps the measured launches, and the traced run must not change them either: a
``torch.compile``d function called once under a dispatch mode runs eagerly at every later call (no Triton launch any
more; found in batch 1, finding D12).  The traced run therefore runs under ``torch.compiler.set_stance("force_eager")``,
which leaves the compiled code and its caches untouched; without that API no traced run is made (no evidence).
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
from .reference_eval.storage_effects import CLEAN, StorageMap, byte_runs, hull_run


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


def _runs(t):
    """(byte runs, exact) of the view ``t`` (storage_effects M2)."""
    runs = byte_runs(t)
    return (runs, True) if runs is not None else (hull_run(t), False)


def _clean(prov, tensors) -> bool:
    """Every byte each tensor view reads carries a clean label."""
    return all(prov.read(t.untyped_storage().data_ptr(), _runs(t)[0])[0] in CLEAN for t in tensors)


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
        # exactly the bytes of the written view's elements (M2); a hull (coverage not enumerable) joins (M5)
        runs, exact = _runs(t)
        prov.write(t.untyped_storage(), runs, label, exact=exact)
    born = set()
    for t in outs:
        p = t.untyped_storage().data_ptr()
        if p in in_ptrs or p in mut_ptrs:
            continue                                                     # a view or the mutated argument itself
        runs, exact = _runs(t)
        if p not in born:                                                # a storage born at this event (M1)
            born.add(p)
            prov.bind(t.untyped_storage(), runs if exact else [], label)
            if not exact:
                prov.write(t.untyped_storage(), runs, label, exact=False)
        else:
            prov.write(t.untyped_storage(), runs, label, exact=exact)


def _signature(launch):
    return (launch.kernel_name, tuple(launch.grid),
            tuple((a.name, str(a.dtype)) for a in launch.args if a.kind == "tensor"))


def producer_records(case, seed, measured_launches, seq) -> Optional[dict]:
    """{(launch index, argument name): (provenance, producer chain)} for the float tensor arguments of the measured
    launches, from an unmeasured traced run of the call for ``seed``; None when the traced run cannot be aligned with
    the measured launches (different Triton launch sequence, an error in the traced run)."""
    from .reference_eval.capture import TritonLaunchRecorder

    stance = getattr(getattr(torch, "compiler", None), "set_stance", None)
    if stance is None:
        return None
    try:
        inp = case.inputs(seed)
        prov = StorageMap()
        for t in _flat(inp):
            if torch.is_tensor(t) and t.is_floating_point():
                runs, exact = _runs(t)
                if not exact:
                    continue                      # no claim about bytes whose coverage is not enumerable
                if t.untyped_storage().data_ptr() in prov:
                    prov.write(t.untyped_storage(), runs, ("input", "declared input"))
                else:
                    prov.bind(t.untyped_storage(), runs, ("input", "declared input"))
        rec = TritonLaunchRecorder(copy_tensors=False, keep_storages=True)
        trace = _Trace(rec)
        with rec:
            with stance("force_eager"):
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
    storages = {}
    for launch in rec.launches:            # the recorder held every launch storage alive (keep_storages)
        for a in launch.args:
            if a.kind == "tensor":
                storages.setdefault(a.storage_ptr, a)
    for i, (tl_, ml) in enumerate(zip(rec.launches, measured_launches)):
        while pending is not None and pending[0] <= i:
            _apply(prov, pending)
            pending = next(events, None)
        measured = {b.name: b for b in ml.args if b.kind == "tensor"}
        for a in tl_.args:
            if a.kind == "tensor" and str(a.dtype).startswith(("float", "bfloat")):
                out[(i, a.name)] = prov.read(a.storage_ptr, _read_runs(seq.launches[i], measured.get(a.name), a))
        stored_measured = getattr(seq.launches[i], "stored", set())
        names = {b.name for b in ml.args if b.kind == "tensor" and b.storage_ptr in stored_measured}
        for a in tl_.args:
            if a.kind == "tensor" and a.name in names:
                prov.write(_StorageRef(a), None, ("triton", tl_.kernel_name))
    return out


class _StorageRef:
    """The storage of a recorded trace-run argument, by address (the recorder keeps it alive)."""

    def __init__(self, arg):
        self._ptr, self._nbytes = arg.storage_ptr, arg.storage_nbytes

    def data_ptr(self):
        return self._ptr

    def nbytes(self):
        return self._nbytes


def _read_runs(ref, measured_arg, trace_arg):
    """The bytes the measured reference read from ``measured_arg``'s initial values, as runs of the trace run's
    storage: only when the two arguments sit at the same offset in storages of the same size (else every byte)."""
    if measured_arg is None:
        return None
    if (measured_arg.data_ptr - measured_arg.storage_ptr, measured_arg.storage_nbytes) != \
            (trace_arg.data_ptr - trace_arg.storage_ptr, trace_arg.storage_nbytes):
        return None
    got = (getattr(ref, "loaded_elements", None) or {}).get(measured_arg.storage_ptr)
    if got is None:
        return None
    from .reference_eval.storage_effects import element_runs
    elems, size = got
    return element_runs(elems, size) if size else None
