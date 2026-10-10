"""Memory-effect composition rules (DSL v2 rc3 04 W4): storage identity, lifetime, exact coverage and producer
evidence.  One implementation for every place that decides where a buffer's bytes came from: the declared-input
test and the cross-launch carry-over of ``check.torch_intermediates``, the carry-over of
``ttir_eval.evaluate_sequence``, and the per-byte producer map of ``provenance.producer_records``.

Obligations (each one is a function or method below; the tests are in ``tests/test_storage_effects.py`` and the
end-to-end ones in ``tests/test_batch1_guarantees.py``):

M1  identity and lifetime -- an address names one storage instance only while that instance lives.  Every decision
    is made on storages the analysis holds alive (recorder, ``InputSnapshot``, ``StorageMap``), and a captured
    argument is matched by address *and* StorageImpl identity.
M2  exact coverage -- a write through a view covers exactly the bytes of its elements (offset, strides, overlap,
    stride 0), ``byte_runs``; a read covers the bytes of the elements the reference loaded from initial values.  A
    write that covers part of a storage changes only those bytes.  When the coverage cannot be enumerated, the hull is
    used and the write *joins* with what was there (it never replaces bytes it may not have written).
M3  content is not provenance -- equal bytes (to an input, to zero) prove nothing.  A read is a declared-input read
    only when the bytes read lie inside the declared input tensors on that storage, equal *that storage's own* bytes
    before the call (``InputSnapshot.declared_read``), and there is no write evidence: no ATen-visible version
    increment that was not announced as a raw-pointer write (``StorageWatch``), and no ATen write of those bytes in
    the aligned producer trace.
M4  carry-over -- the reference memory of a recorded launch carries over to a later launch only when the bytes are
    those after the earlier launch and there is no write evidence in between (``carry_over``).
M5  producer lattice -- labels ``input`` < ``const`` < ``copy`` are clean (the call's own values, exactly);
    ``computed`` and ``triton`` are not; an unlabeled byte is ``computed``.  A read joins the labels of the bytes it
    reads (all clean: the largest; otherwise not clean).  An exact write replaces the labels of the bytes it covers; a
    hull write joins: clean with clean gives ``copy`` (an exact mixture; the same label stays), anything with a
    non-clean label gives ``computed``.

Not observed (stated limit): a writer outside ATen and Triton that leaves every byte unchanged and goes through
neither a version counter nor the dispatcher (a raw-pointer write from foreign code).  ``tensor.data`` writes do not
bump the version counter; the producer trace sees them as ATen writes when it can be aligned with the measured run.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

CLEAN = ("input", "const", "copy")
_RANK = {"input": 0, "const": 1, "copy": 2}
ENUMERATION_CAP = 1 << 22          # elements enumerated for a non-dense view before falling back to its hull


# ------------------------------------------------------------------------------------------------ M2: coverage

def merge_runs(runs: Iterable) -> list:
    """Sorted, disjoint, non-adjacent [start, end) runs."""
    out = []
    for s, e in sorted((int(s), int(e)) for s, e in runs if e > s):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(r) for r in out]


def element_runs(elements, element_size: int) -> list:
    """Byte runs of storage-relative element indices (any order, repeats allowed)."""
    idx = np.unique(np.asarray(elements, dtype=np.int64))
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]]) + 1
    return [(int(s) * element_size, int(e) * element_size) for s, e in zip(starts, ends)]


def view_elements(offset: int, sizes, strides, cap: int = ENUMERATION_CAP) -> Optional[np.ndarray]:
    """Storage-relative element indices of a strided view (with repeats when the view overlaps itself); None when
    more than ``cap`` elements would have to be enumerated."""
    n = int(np.prod(sizes, dtype=np.int64)) if len(sizes) else 1
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n > cap:
        return None
    idx = np.array([offset], dtype=np.int64)
    for sz, st in zip(sizes, strides):
        if sz != 1:
            idx = (idx[:, None] + np.arange(sz, dtype=np.int64)[None, :] * int(st)).reshape(-1)
    return idx


def _dense(sizes, strides) -> bool:
    """Non-overlapping and dense: the dimensions of size > 1, sorted by stride, have strides 1, s0, s0 s1, ..."""
    expect = 1
    for sz, st in sorted(((sz, st) for sz, st in zip(sizes, strides) if sz != 1), key=lambda d: d[1]):
        if st != expect:
            return False
        expect *= sz
    return True


def byte_runs(t, cap: int = ENUMERATION_CAP) -> Optional[list]:
    """Exact byte coverage of the tensor view ``t`` relative to its storage start (M2), or None when the view is not
    dense and has more than ``cap`` elements (callers then use ``hull_run`` and join)."""
    if t.numel() == 0:
        return []
    es = t.element_size()
    off = t.storage_offset()
    if _dense(tuple(t.shape), tuple(t.stride())):   # numel distinct elements filling [off, off + numel)
        return [(off * es, (off + t.numel()) * es)]
    idx = view_elements(off, tuple(t.shape), tuple(t.stride()), cap)
    return None if idx is None else element_runs(idx, es)


def hull_run(t) -> list:
    """The smallest byte range containing every element of ``t``."""
    if t.numel() == 0:
        return []
    es = t.element_size()
    off = t.storage_offset()
    last = off + sum((sz - 1) * st for sz, st in zip(t.shape, t.stride()))
    return [(off * es, (last + 1) * es)]


def runs_within(inner: list, outer: list) -> bool:
    """Every byte of ``inner`` lies in ``outer`` (both merged runs)."""
    j = 0
    for s, e in inner:
        while j < len(outer) and outer[j][1] <= s:
            j += 1
        if j == len(outer) or not (outer[j][0] <= s and e <= outer[j][1]):
            return False
    return True


def gather(raw: np.ndarray, runs: list) -> np.ndarray:
    """The bytes of ``raw`` (uint8, storage-relative) at ``runs``."""
    if not runs:
        return np.zeros(0, dtype=np.uint8)
    return np.concatenate([raw[s:e] for s, e in runs])


# ------------------------------------------------------------------------------------------------ M5: label lattice

def join(a: tuple, b: tuple) -> tuple:
    """Join of two (kind, chain) labels for bytes that may hold either value (M5)."""
    if a[0] not in CLEAN:
        return a
    if b[0] not in CLEAN:
        return b
    if a[0] == b[0]:
        return a if a[1] == b[1] else (a[0], f"{a[1]}; {b[1]}"[:400])
    return ("copy", f"{a[1]}; {b[1]}"[:400])


def read_join(labels: list) -> tuple:
    """The label of a read over bytes carrying ``labels`` (M5): the first non-clean label, else the largest clean."""
    if not labels:
        return ("computed", "no bytes read")
    for lab in labels:
        if lab[0] not in CLEAN:
            return lab
    top = max(labels, key=lambda lab: _RANK[lab[0]])
    chains = []
    for lab in labels:
        if lab[1] not in chains:
            chains.append(lab[1])
    return (top[0], "; ".join(chains)[:400])


class StorageMap:
    """Per-storage byte-interval producer labels (M1, M2, M5).  Storages with a clean byte are held alive, so their
    address cannot be handed to another storage while the label exists; a storage whose bytes are all non-clean is
    released (a stale non-clean label only makes a later read more conservative)."""

    UNLABELED = ("computed", "no producer recorded")

    def __init__(self):
        self._m = {}      # ptr -> {"storage": obj or None, "nbytes": int, "segs": [[s, e, label]]}

    def __contains__(self, ptr):
        return ptr in self._m

    def _entry(self, storage):
        ptr = storage.data_ptr()
        ent = self._m.get(ptr)
        if ent is None:
            ent = self._m[ptr] = {"storage": None, "nbytes": storage.nbytes(), "segs": []}
        return ent

    def _hold(self, ent, storage):
        if not any(lab[0] in CLEAN for _, _, lab in ent["segs"]):
            ent["storage"] = None
        elif hasattr(storage, "_cdata"):          # a real storage object (not an address-only reference)
            ent["storage"] = storage

    def bind(self, storage, runs: Optional[list], label: tuple):
        """A storage born at this event (or a declared input): only ``runs`` carry ``label``; every other byte is
        unlabeled.  ``runs`` None: the whole storage."""
        ptr = storage.data_ptr()
        self._m.pop(ptr, None)
        ent = self._entry(storage)
        runs = [(0, ent["nbytes"])] if runs is None else merge_runs(runs)
        ent["segs"] = [[s, e, label] for s, e in runs]
        self._hold(ent, storage)

    def write(self, storage, runs: Optional[list], label: tuple, exact: bool = True):
        """A write of ``label`` over ``runs`` (exact coverage: replace; ``exact`` False: the runs are a hull that may
        contain unwritten bytes, join).  ``runs`` None: the whole storage."""
        ent = self._entry(storage)
        runs = [(0, ent["nbytes"])] if runs is None else merge_runs(runs)
        for s, e in runs:
            ent["segs"] = self._overlay(ent["segs"], s, e, label, exact)
        self._hold(ent, storage)

    @classmethod
    def _overlay(cls, segs, s, e, label, exact):
        out = []
        covered = []
        for a, b, lab in segs:
            if b <= s or a >= e:
                out.append([a, b, lab])
                continue
            if a < s:
                out.append([a, s, lab])
            if b > e:
                out.append([e, b, lab])
            covered.append([max(a, s), min(b, e), lab])
        if exact:
            out.append([s, e, label])
        else:
            # the hull: bytes that had a label join with the write, bytes without one are unlabeled (computed)
            pos = s
            for a, b, lab in sorted(covered):
                if a > pos:
                    out.append([pos, a, join(cls.UNLABELED, label)])
                out.append([a, b, join(lab, label)])
                pos = b
            if pos < e:
                out.append([pos, e, join(cls.UNLABELED, label)])
        out.sort(key=lambda seg: seg[0])
        return out

    def read(self, ptr: int, runs: Optional[list] = None) -> tuple:
        """The joined label of the bytes ``runs`` (None: the whole storage) of the storage at ``ptr``."""
        ent = self._m.get(ptr)
        if ent is None:
            return self.UNLABELED
        runs = [(0, ent["nbytes"])] if runs is None else merge_runs(runs)
        labels = []
        for s, e in runs:
            pos = s
            for a, b, lab in ent["segs"]:
                if b <= s or a >= e:
                    continue
                if a > pos:
                    labels.append(self.UNLABELED)
                labels.append(lab)
                pos = max(pos, b)
            if pos < e:
                labels.append(self.UNLABELED)
        return read_join(labels)


# ------------------------------------------------------------------------------------------------ M3 / M4: write evidence

class StorageWatch:
    """ATen-visible write evidence per storage from version counters (M3, M4).  Every tensor the analysis sees on a
    storage is held (so the storage lives, M1) with its version when first seen; ``count`` is the number of version
    increments since then that were not announced through ``torch.autograd.graph.increment_version`` -- the call
    AOTAutograd makes, before a compiled function runs, for inputs the compiled code writes through raw pointers
    (those writes are recorded Triton launches, or show as changed bytes).  Views share their base's counter, so an
    in-place op on any view of a held tensor counts.  None: no evidence (a tensor without a version counter, e.g. an
    inference-mode tensor)."""

    def __init__(self):
        self._held = {}        # ptr -> list of [tensor, version when first held]
        self._ids = {}         # ptr -> {id(tensor)}
        self._announced = {}   # id(tensor) -> announced increments
        self._original = None

    @staticmethod
    def _version(t) -> Optional[int]:
        try:
            return int(t._version)
        except (RuntimeError, AttributeError):
            return None

    def hold(self, t):
        ptr = t.untyped_storage().data_ptr()
        ids = self._ids.setdefault(ptr, set())
        if id(t) in ids:
            return
        ids.add(id(t))
        self._held.setdefault(ptr, []).append([t, self._version(t)])

    def count(self, ptr: int) -> Optional[int]:
        total = 0
        for t, v0 in self._held.get(ptr, ()):
            v = self._version(t)
            if v is None or v0 is None:
                return None
            total += v - v0 - self._announced.get(id(t), 0)
        return total

    def observe(self, t) -> Optional[int]:
        """The count of ``t``'s storage before ``t`` itself is held (the evidence at this launch)."""
        ptr = t.untyped_storage().data_ptr()
        c = self.count(ptr)
        self.hold(t)
        return c

    # announced raw-pointer writes ---------------------------------------------------------------
    def _announce(self, tensors):
        import torch
        ts = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
        ptrs = {t.untyped_storage().data_ptr() for t in ts if isinstance(t, torch.Tensor)}
        before = {id(h): self._version(h) for p in ptrs for h, _ in self._held.get(p, ())}
        self._original(ts)
        for p in ptrs:
            for h, _ in self._held.get(p, ()):
                b, a = before.get(id(h)), self._version(h)
                if b is not None and a is not None and a > b:
                    self._announced[id(h)] = self._announced.get(id(h), 0) + (a - b)

    def install(self):
        import torch.autograd.graph as graph
        if self._original is None:
            self._original = graph.increment_version
            graph.increment_version = self._announce

    def uninstall(self):
        import torch.autograd.graph as graph
        if self._original is not None:
            graph.increment_version = self._original
            self._original = None


def carry_over(bytes_equal: bool, mutations_now: Optional[int], mutations_then: Optional[int]) -> tuple:
    """M4: (carries over, evidence).  Bytes must equal those after the earlier launch; with version evidence on both
    sides the counts must agree too.  Without version evidence the decision rests on bytes alone and says so."""
    if not bytes_equal:
        return False, "bytes changed"
    if mutations_now is None or mutations_then is None:
        return True, "bytes only (no version evidence)"
    if mutations_now != mutations_then:
        return False, f"ATen-visible write in between ({mutations_now - mutations_then} version increments)"
    return True, "bytes and version counters"


# ------------------------------------------------------------------------------------------------ declared inputs

def _tensors(obj, path=""):
    import torch
    if torch.is_tensor(obj):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _tensors(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _tensors(v, f"{path}[{i}]")


def _host_bytes(storage):
    """A host copy of ``storage`` as a uint8 tensor, copied device -> host directly: no device allocation, so the
    snapshot leaves the caching allocator of the measured run as it was."""
    import torch
    view = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage, 0, (storage.nbytes(),))
    return view.to("cpu", copy=True)      # a copy also for host storages (.cpu() would alias them)


def storage_bytes(storage, device=None) -> np.ndarray:
    return _host_bytes(storage).numpy()


def _host_view(t, host):
    """The tensor ``t`` rebuilt on the host from its storage's host copy ``host`` (same offset, sizes, strides)."""
    import torch
    h = torch.empty(0, dtype=t.dtype).set_(host.untyped_storage(), t.storage_offset(), tuple(t.shape),
                                           tuple(t.stride()))
    return h


class InputSnapshot:
    """The declared inputs before the call (M1, M3): per floating input storage the storage itself (held), its
    StorageImpl identity, its bytes, and the byte runs of the declared input tensors on it; per input path the digest
    of the tensor's bytes (an ordered map: two inputs that swap contents are different inputs)."""

    @staticmethod
    def digests_of(inp, hosts=None) -> dict:
        """Input path -> SHA-1 of the tensor's bytes (dtype and shape included), in declaration order.  Computed from
        host copies of the storages (no device allocation)."""
        import hashlib
        import torch
        hosts = {} if hosts is None else hosts
        out = {}
        for path, t in _tensors(inp):
            st = t.untyped_storage()
            host = hosts.get(st.data_ptr())
            if host is None:
                host = hosts[st.data_ptr()] = _host_bytes(st)
            h = hashlib.sha1(f"{t.dtype}{tuple(t.shape)}".encode())
            h.update(_host_view(t.detach(), host).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            out[path] = h.hexdigest()
        return out

    def __init__(self, inp):
        self.tensors = []
        self.storages = {}     # ptr -> storage
        self.cdata = {}        # ptr -> StorageImpl identity
        self.raw = {}          # ptr -> uint8 bytes before the call
        self.cover = {}        # ptr -> merged byte runs of the declared float input tensors
        self.exact = {}        # ptr -> False when some input's coverage had to be approximated (no claim then)
        hosts = {}
        self.digest_map = self.digests_of(inp, hosts)
        for path, t in _tensors(inp):
            if not t.is_floating_point():
                continue
            self.tensors.append(t)
            s = t.untyped_storage()
            p = s.data_ptr()
            if p not in self.storages:
                self.storages[p] = s
                self.cdata[p] = s._cdata
                self.raw[p] = hosts[p].numpy()
                self.cover[p] = []
                self.exact[p] = True
            runs = byte_runs(t)
            if runs is None:
                self.exact[p] = False
            else:
                self.cover[p] = merge_runs(self.cover[p] + runs)

    def ptrs(self) -> set:
        return set(self.storages)

    def declared_read(self, arg, read_runs: Optional[list]) -> tuple:
        """(declared input, reason) for a captured argument whose reference read ``read_runs`` (storage-relative
        bytes; None: every captured byte) from its initial values, on content and identity only (M1, M3); the write
        evidence is checked by the caller."""
        p = arg.storage_ptr
        if p not in self.storages:
            return False, "not a declared input's storage"
        if getattr(arg, "storage_id", None) not in (None, self.cdata[p]):
            return False, "another storage instance at a declared input's address"
        if not self.exact[p]:
            return False, "declared input coverage not enumerable"
        raw = self.raw[p]
        before = np.asarray(arg.before.numpy() if hasattr(arg.before, "numpy") else arg.before, dtype=np.uint8)
        window = getattr(arg, "window", None)
        if window is not None:
            es = int(arg.element_size)
            elems = np.asarray(window, dtype=np.int64)
            runs = element_runs(elems, es) if read_runs is None else read_runs
            if not runs_within(runs, self.cover[p]):
                return False, "bytes read outside the declared input tensors"
            # the window copy holds the elements in storage order: compare element by element
            pos = {int(e): k for k, e in enumerate(elems)}
            want = []
            for s, e in runs:
                for el in range(s // es, e // es):
                    k = pos.get(el)
                    if k is None:
                        return False, "bytes read outside the captured window"
                    want.append((el, k))
            for el, k in want:
                if not np.array_equal(raw[el * es:(el + 1) * es], before[k * es:(k + 1) * es]):
                    return False, "bytes differ from the input's own bytes before the call"
            return True, "declared input"
        runs = [(0, min(raw.size, before.size))] if read_runs is None else read_runs
        if not runs_within(runs, self.cover[p]):
            return False, "bytes read outside the declared input tensors"
        if before.size != raw.size or not np.array_equal(gather(raw, runs), gather(before, runs)):
            return False, "bytes differ from the input's own bytes before the call"
        return True, "declared input"


def read_runs_of(ref, arg) -> Optional[list]:
    """Byte runs of the elements the launch reference ``ref`` read from the captured initial values of ``arg``'s
    storage (None: unknown, every captured byte counts)."""
    got = (getattr(ref, "loaded_elements", None) or {}).get(arg.storage_ptr)
    if got is None:
        return None
    elems, size = got
    return element_runs(elems, int(size)) if size else None
