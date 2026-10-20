#!/usr/bin/env python3
"""Path A recorder (docs/OPERATOR_ENTRY_PROTOCOL.md): ATen calls at the dispatch boundary and the Triton launches inside
each call, for a workload run with FlagGems registered on the CUDA dispatch key.

The recorder is a TorchDispatchMode, i.e. it sits above the CUDA key, so the recorded call still dispatches to the
registered implementation. Whether recording changes the executed object is checked separately (check_launches.py):
the same workload is run with and without the recorder and the launch sequences are compared.

Per call: overload, a structural key, the Triton launches (kernel name, cache hash) issued while the call ran, and
for the first call of each key a pre-state copy of every tensor argument that preserves size, stride and the values
(the storage span the view reads is copied, so non-contiguous and overlapping views keep their layout).
Launches issued by the recorder's own copies are excluded.
"""
from __future__ import annotations

import contextlib
import json
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_LAUNCHES = None          # list receiving (kernel name, hash) while a call is open; None = not recording
_SUPPRESS = [False]       # True while the recorder copies tensors (those launches are not the candidate's)
_ALL = []                 # every launch of the run (name, hash, attributed call index or None)
_HOOKED = [False]


def install_launch_hook():
    """Hook CompiledKernel.run once, before the workload's first launch (launchers are cached at first use)."""
    if _HOOKED[0]:
        return
    from triton.compiler.compiler import CompiledKernel
    original = CompiledKernel.run.fget

    def patched(kernel):
        real = original(kernel)

        def launcher(*args):
            if not _SUPPRESS[0]:
                item = (kernel.name, getattr(kernel, "hash", None))
                _ALL.append(item + (Recorder.current,))
                if _LAUNCHES is not None:
                    _LAUNCHES.append(item)
            return real(*args)
        return launcher

    CompiledKernel.run = property(patched)
    _HOOKED[0] = True


def all_launches():
    return list(_ALL)


def reset_launches():
    _ALL.clear()


def snapshot(t: torch.Tensor) -> torch.Tensor:
    """A copy with the same size, stride and logical values (storage offset 0)."""
    if t.numel() == 0:
        return torch.empty_strided(t.size(), t.stride(), dtype=t.dtype, device=t.device)
    span = 1 + sum((s - 1) * st for s, st in zip(t.size(), t.stride()))
    flat = torch.as_strided(t, (span,), (1,), t.storage_offset()).clone()
    return flat.as_strided(t.size(), t.stride(), 0)


def _walk(x, fn):
    if isinstance(x, torch.Tensor):
        return fn(x)
    if isinstance(x, (list, tuple)):
        return type(x)(_walk(v, fn) for v in x)
    if isinstance(x, dict):
        return {k: _walk(v, fn) for k, v in x.items()}
    return x


def structural(x):
    """Hashable description of an argument: tensors by dtype/size/stride, everything else by value."""
    if isinstance(x, torch.Tensor):
        return ("T", str(x.dtype), tuple(x.size()), tuple(x.stride()), x.device.type)
    if isinstance(x, (list, tuple)):
        return (type(x).__name__,) + tuple(structural(v) for v in x)
    if isinstance(x, dict):
        return ("dict",) + tuple(sorted((k, structural(v)) for k, v in x.items()))
    if isinstance(x, float):
        return ("f", "nan" if math.isnan(x) else x)
    return (type(x).__name__, repr(x))


class Recorder(TorchDispatchMode):
    """Records every ATen call that reaches Python dispatch; keeps tensor copies for the first call of each key."""
    current = None

    def __init__(self, keep_values=True, max_calls=None):
        super().__init__()
        self.calls = []            # one entry per call
        self.values = {}           # key index -> (args, kwargs) with tensor copies
        self.keys = {}             # structural key -> key index
        self.keep_values = keep_values
        self.max_calls = max_calls

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        global _LAUNCHES
        kwargs = kwargs or {}
        key = (str(func), structural(args), structural(kwargs))
        new = key not in self.keys
        if new:
            self.keys[key] = len(self.keys)
            if self.keep_values:
                _SUPPRESS[0] = True
                try:
                    self.values[self.keys[key]] = (_walk(args, snapshot), _walk(kwargs, snapshot))
                finally:
                    _SUPPRESS[0] = False
        index = len(self.calls)
        outer, Recorder.current = Recorder.current, index
        _LAUNCHES, saved = [], _LAUNCHES
        try:
            out = func(*args, **kwargs)
        finally:
            launches, _LAUNCHES = _LAUNCHES, saved
            Recorder.current = outer
        self.calls.append({"index": index, "op": str(func), "key": self.keys[key], "first": new,
                           "launches": [n for n, _ in launches], "hashes": [h for _, h in launches]})
        return out


def describe(x):
    """JSON description of a recorded argument (tensors by metadata)."""
    if isinstance(x, torch.Tensor):
        return {"tensor": {"dtype": str(x.dtype), "size": list(x.size()), "stride": list(x.stride())}}
    if isinstance(x, (list, tuple)):
        return [describe(v) for v in x]
    if isinstance(x, dict):
        return {k: describe(v) for k, v in x.items()}
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    return {"repr": repr(x)}


def save(recorder: Recorder, out_dir, extra=None):
    """calls.jsonl (every call), keys.jsonl (one line per key with argument description), values/<k>.pt."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "values").mkdir()
    with open(out / "calls.jsonl", "w") as fh:
        for c in recorder.calls:
            fh.write(json.dumps(c) + "\n")
    unserializable = []
    by_key = {}
    for c in recorder.calls:
        by_key.setdefault(c["key"], []).append(c)
    with open(out / "keys.jsonl", "w") as fh:
        for k, calls in sorted(by_key.items()):
            args, kwargs = recorder.values.get(k, (None, None))
            first = calls[0]
            fh.write(json.dumps({"key": k, "op": first["op"], "calls": len(calls),
                                 "launches": first["launches"], "hashes": first["hashes"],
                                 "launch_sets_agree": all(c["launches"] == first["launches"] for c in calls),
                                 "args": describe(args), "kwargs": describe(kwargs)}) + "\n")
            if args is not None:
                try:
                    torch.save({"args": args, "kwargs": kwargs}, out / "values" / f"{k}.pt")
                except Exception as err:  # noqa: BLE001 -- deviation D-A2: the key stays in the account
                    (out / "values" / f"{k}.pt").unlink(missing_ok=True)
                    unserializable.append({"key": k, "error": f"{type(err).__name__}: {err}"[:200]})
    if extra is not None:
        extra = dict(extra, keys_values_unserializable=unserializable)
        (out / "run.json").write_text(json.dumps(extra, indent=1))


@contextlib.contextmanager
def recording(keep_values=True):
    install_launch_hook()
    rec = Recorder(keep_values=keep_values)
    with rec:
        yield rec
