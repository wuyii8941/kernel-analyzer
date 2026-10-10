"""Memory-effect composition rules (``reference_eval.storage_effects``, M1-M5) against independent answers: the bytes
a real torch write through a view changes (coverage), a per-byte model of the label lattice (StorageMap), and the
version counters of real tensors (StorageWatch)."""
from __future__ import annotations

import itertools
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from kernel_analyzer.reference_eval import storage_effects as S


# ------------------------------------------------------------------------------------------------ M2: coverage

def _written_bytes(n, offset, sizes, strides):
    """Independent answer: write -1 (all bytes 0xFF) through the view on a zeroed int32 storage, read which bytes
    changed."""
    base = torch.zeros(n, dtype=torch.int32)
    v = base.as_strided(sizes, strides, offset)
    v.fill_(-1)
    raw = base.view(torch.uint8).numpy()
    return np.flatnonzero(raw == 0xFF)


def _bytes_of(runs):
    return np.array(sorted(b for s, e in runs for b in range(s, e)), dtype=np.int64)


def test_byte_runs_equal_the_bytes_a_real_write_changes():
    rng = random.Random(7)
    cases = [((8,), (0,), 0), ((4, 2), (1, 1), 0), ((2, 8), (0, 1), 3), ((3, 3), (3, 1), 1), ((4,), (2,), 1),
             ((2, 3, 2), (6, 1, 3), 0), ((5,), (1,), 4), ((1, 4), (9, 1), 2)]
    for _ in range(200):
        nd = rng.randint(1, 3)
        sizes = tuple(rng.randint(1, 4) for _ in range(nd))
        strides = tuple(rng.randint(0, 5) for _ in range(nd))
        cases.append((sizes, strides, rng.randint(0, 5)))
    for sizes, strides, offset in cases:
        n = offset + sum((s - 1) * st for s, st in zip(sizes, strides)) + 1
        v = torch.zeros(n, dtype=torch.int32).as_strided(sizes, strides, offset)
        got = S.byte_runs(v)
        assert np.array_equal(_bytes_of(got), _written_bytes(n, offset, sizes, strides)), (sizes, strides, offset)
        # the hull contains every written byte
        assert S.runs_within(got, S.hull_run(v))


def test_overlapping_full_size_view_is_not_full_coverage():
    """audit counterexample class: numel * element_size equals the storage size, yet a stride-0 or overlapping view
    covers part of it"""
    buf = torch.zeros(8, dtype=torch.float32)
    assert S.byte_runs(buf.as_strided((8,), (0,))) == [(0, 4)]
    assert S.byte_runs(buf.as_strided((4, 2), (1, 1))) == [(0, 20)]
    assert S.byte_runs(buf.expand(2, 8)) == [(0, 32)]
    assert S.byte_runs(buf[2:6]) == [(8, 24)]
    assert S.byte_runs(buf.view(2, 4).t()) == [(0, 32)]


def test_enumeration_cap_falls_back_to_none():
    v = torch.zeros(10, dtype=torch.float32).as_strided((6,), (0,))
    assert S.byte_runs(v, cap=4) is None
    assert S.hull_run(v) == [(0, 4)]


# ------------------------------------------------------------------------------------------------ M5: lattice

LABELS = [("input", "i"), ("const", "c"), ("copy", "p"), ("computed", "x"), ("triton", "t")]


def test_join_is_commutative_associative_idempotent_on_kinds():
    k = lambda lab: lab[0]  # noqa: E731
    for a, b, c in itertools.product(LABELS, repeat=3):
        assert k(S.join(a, b)) == k(S.join(b, a)) or {k(S.join(a, b)), k(S.join(b, a))} <= {"computed", "triton"}
        assert k(S.join(S.join(a, b), c)) in (k(S.join(a, S.join(b, c))),) or \
            {k(S.join(S.join(a, b), c)), k(S.join(a, S.join(b, c)))} <= {"computed", "triton"}
        assert k(S.join(a, a)) == k(a)
        clean = a[0] in S.CLEAN and b[0] in S.CLEAN
        assert (k(S.join(a, b)) in S.CLEAN) == clean


class _Model:
    """Per-byte model of StorageMap (independent implementation of M2 / M5)."""

    def __init__(self, n):
        self.lab = [None] * n

    def bind(self, runs, label):
        self.lab = [None] * len(self.lab)
        self.write(runs, label, True)

    def write(self, runs, label, exact):
        for s, e in runs:
            for b in range(s, e):
                if exact:
                    self.lab[b] = label
                else:
                    self.lab[b] = S.join(self.lab[b] or S.StorageMap.UNLABELED, label)

    def read(self, runs):
        labs = [self.lab[b] or S.StorageMap.UNLABELED for s, e in runs for b in range(s, e)]
        return S.read_join(labs)[0] if labs else "computed"


class _Store:
    def __init__(self, ptr, n):
        self._p, self._n, self._cdata = ptr, n, ptr

    def data_ptr(self):
        return self._p

    def nbytes(self):
        return self._n


def test_storage_map_matches_a_per_byte_model():
    rng = random.Random(11)
    for trial in range(300):
        n = rng.randint(4, 40)
        st = _Store(1000 + trial, n)
        m, model = S.StorageMap(), _Model(n)

        def runs():
            out = []
            for _ in range(rng.randint(0, 3)):
                s = rng.randint(0, n - 1)
                out.append((s, rng.randint(s + 1, n)))
            return S.merge_runs(out)

        r0, l0 = runs(), rng.choice(LABELS)
        m.bind(st, r0, l0)
        model.bind(r0, l0)
        for _ in range(rng.randint(0, 6)):
            r, lab, exact = runs(), rng.choice(LABELS), rng.random() < 0.6
            m.write(st, r, lab, exact=exact)
            model.write(r, lab, exact)
        for _ in range(4):
            r = runs() or [(0, n)]
            got = m.read(st.data_ptr(), r)[0]
            want = model.read(r)
            assert got == want or {got, want} <= {"computed", "triton"}, (trial, r, got, want)


def test_partial_constant_write_over_uninitialised_bytes_stays_computed():
    """the audit's overlapping-view class through StorageMap: an empty buffer filled through a stride-0 view is
    constant in one element only"""
    buf = torch.empty(8, dtype=torch.float32)
    m = S.StorageMap()
    m.bind(buf.untyped_storage(), S.byte_runs(buf), ("computed", "aten.empty"))
    m.write(buf.untyped_storage(), S.byte_runs(buf.as_strided((8,), (0,))), ("const", "fill"))
    assert m.read(buf.untyped_storage().data_ptr())[0] == "computed"
    assert m.read(buf.untyped_storage().data_ptr(), [(0, 4)])[0] == "const"
    # an exact write that does cover everything replaces
    m.write(buf.untyped_storage(), S.byte_runs(buf), ("const", "zero"))
    assert m.read(buf.untyped_storage().data_ptr())[0] == "const"


def test_clean_storages_are_held_alive_and_released_when_not_clean():
    m = S.StorageMap()
    t = torch.zeros(4)
    st = t.untyped_storage()
    m.bind(st, None, ("const", "zeros"))
    assert m._m[st.data_ptr()]["storage"] is st
    m.write(st, None, ("computed", "add"))
    assert m._m[st.data_ptr()]["storage"] is None


# ------------------------------------------------------------------------------------------------ M3 / M4: write evidence

def test_storage_watch_counts_aten_writes_through_views_and_detach():
    w = S.StorageWatch()
    x = torch.rand(8)
    w.hold(x)
    p = x.untyped_storage().data_ptr()
    assert w.count(p) == 0
    x[:2].add_(0.0)                 # an in-place op that changes no byte still counts
    assert w.count(p) == 1
    x.detach().mul_(1.0)
    assert w.count(p) == 2
    v = x.view(2, 4)
    assert w.observe(v) == 2        # the evidence before the view itself is held
    x.add_(0.0)
    assert w.count(p) == 2 + 2      # x and its view share the counter: both see the increment


def test_storage_watch_excludes_announced_raw_pointer_writes():
    import torch.autograd.graph as graph
    w = S.StorageWatch()
    x = torch.rand(8)
    w.hold(x)
    w.install()
    try:
        graph.increment_version(x)                  # what AOTAutograd does before a compiled in-place kernel
        graph.increment_version(t for t in [x])     # it passes a generator
    finally:
        w.uninstall()
    assert graph.increment_version is not w._announce
    assert w.count(x.untyped_storage().data_ptr()) == 0
    x.add_(0.0)
    assert w.count(x.untyped_storage().data_ptr()) == 1


def test_storage_watch_states_its_limits():
    w = S.StorageWatch()
    x = torch.rand(8)
    w.hold(x)
    x.data.add_(0.0)                # .data has its own counter: not observed here (the producer trace sees it)
    assert w.count(x.untyped_storage().data_ptr()) == 0
    with torch.inference_mode():
        y = torch.rand(4)
    w.hold(y)
    assert w.count(y.untyped_storage().data_ptr()) is None   # no evidence, not "no write"


def test_carry_over_needs_bytes_and_agreeing_counts():
    assert S.carry_over(True, 3, 3) == (True, "bytes and version counters")
    assert S.carry_over(True, 4, 3)[0] is False
    assert S.carry_over(False, 3, 3)[0] is False
    assert S.carry_over(True, None, 3) == (True, "bytes only (no version evidence)")


# ------------------------------------------------------------------------------------------------ M1 / M3: declared inputs

def _arg(t, before=None, storage_id=None):
    st = t.untyped_storage()
    raw = S.storage_bytes(st, t.device) if before is None else before
    return NS(storage_ptr=st.data_ptr(), storage_id=st._cdata if storage_id is None else storage_id,
              before=raw, window=None, element_size=t.element_size())


def test_declared_read_uses_the_storages_own_bytes():
    """cross-input class: bytes equal to another input's bytes are not this input"""
    x = torch.rand(16) + 1
    y = torch.rand(16) + 1
    snap = S.InputSnapshot({"x": x, "y": y})
    assert snap.declared_read(_arg(x), None) == (True, "declared input")
    x.data.copy_(y * (1 + 2.0 ** -30))       # rounds to y's bytes
    assert np.array_equal(S.storage_bytes(x.untyped_storage(), "cpu"), S.storage_bytes(y.untyped_storage(), "cpu"))
    ok, why = snap.declared_read(_arg(x), None)
    assert not ok and "own bytes" in why


def test_declared_read_identity_and_coverage():
    big = torch.rand(32)
    x = big[8:24]                              # the declared input is a view of a larger storage
    snap = S.InputSnapshot({"x": x})
    a = _arg(x)
    assert snap.declared_read(a, [(32, 96)])[0] is True                       # inside the view
    ok, why = snap.declared_read(a, [(0, 40)])
    assert not ok and "outside the declared input" in why                     # bytes of the storage outside it
    ok, why = snap.declared_read(_arg(x, storage_id=12345), [(32, 96)])
    assert not ok and "another storage instance" in why


def test_input_digest_map_is_ordered_by_path():
    x, y = torch.rand(4), torch.rand(4)
    assert S.InputSnapshot.digests_of({"x": x, "y": y}) != S.InputSnapshot.digests_of({"x": y, "y": x})


def test_element_runs_and_read_runs():
    assert S.element_runs([3, 1, 2, 7, 7], 4) == [(4, 16), (28, 32)]
    ref = NS(loaded_elements={5: (np.array([0, 1, 4]), 2)})
    assert S.read_runs_of(ref, NS(storage_ptr=5, element_size=2)) == [(0, 4), (8, 10)]
    assert S.read_runs_of(ref, NS(storage_ptr=6, element_size=2)) is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
