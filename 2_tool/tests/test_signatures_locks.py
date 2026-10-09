"""CAS spin locks (DSL v2 increment 7, rc3 02 6.3 / 6.9): one serialization of the contended CAS plus the reverse
order as evidence, happens-before through release / acquire with transitivity.  Compiled-only kernels on synthetic
captures (CPU only); expected values are exact."""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402
from test_signatures_structural import _run  # noqa: E402

P = 16  # programs


def _serialized_add(sem):
    s = "" if sem is None else f", sem=\"{sem}\""
    # official test_atomic_cas serialized_add, 8 elements
    return ("    i = tl.arange(0, N)\n"
            f"    while tl.atomic_cas(lock, 0, 1{s}) == 1:\n        pass\n"
            "    tl.store(data + i, tl.load(data + i) + 1.0)\n"
            "    tl.debug_barrier()\n"
            "    tl.atomic_xchg(lock, 0)\n")


@pytest.mark.parametrize("sem", [None, "acquire", "acq_rel"])
def test_lock_protected_accumulation_is_established_under_the_order_premise(sem):
    lo, hi, st = _run("lock_add", _serialized_add(sem), {"data": ("fp32", np.zeros(8)), "lock": ("int32", [0])},
                      out_name="data", grid=(P, 1, 1))
    assert (st == H.ST_OK).all() and (lo == P).all() and (hi == P).all()
    lo, hi, st = _run("lock_add", _serialized_add(sem), {"data": ("fp32", np.zeros(8)), "lock": ("int32", [0])},
                      out_name="lock", grid=(P, 1, 1))
    assert st[0] == H.ST_OK and int(lo[0]) == 0


@pytest.mark.parametrize("sem", ["relaxed", "release"])
def test_lock_without_acquire_leaves_the_critical_section_racing(sem):
    lo, hi, st = _run("lock_add", _serialized_add(sem), {"data": ("fp32", np.zeros(8)), "lock": ("int32", [0])},
                      out_name="data", grid=(P, 1, 1))
    assert (st != H.ST_OK).all()


def test_first_writer_inside_a_lock_depends_on_the_order():
    body = ("    pid = tl.program_id(0)\n"
            "    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    f = tl.load(flag)\n"
            "    if f == 0:\n        tl.store(out, pid)\n        tl.store(flag, 1)\n"
            "    tl.debug_barrier()\n"
            "    tl.atomic_xchg(lock, 0)\n")
    lo, hi, st = _run("lock_first", body, {"out": ("int32", [-1]), "flag": ("int32", [0]), "lock": ("int32", [0])},
                      grid=(P, 1, 1))
    assert st[0] != H.ST_OK   # program 0 in program order, program P - 1 in reverse order
    lo, hi, st = _run("lock_first", body, {"out": ("int32", [-1]), "flag": ("int32", [0]), "lock": ("int32", [0])},
                      out_name="flag", grid=(P, 1, 1))
    assert st[0] == H.ST_OK and int(lo[0]) == 1   # the flag is 1 in every order


def test_layer_norm_style_first_store_then_accumulate():
    # tutorial 05 _layer_norm_bwd_dx_fused: the first holder stores its partial, later holders add theirs
    body = ("    pid = tl.program_id(0)\n    i = tl.arange(0, N)\n"
            "    part = tl.load(x + pid * N + i)\n"
            "    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    count = tl.load(cnt)\n"
            "    if count == 0:\n        tl.atomic_xchg(cnt, 1)\n"
            "    else:\n        part += tl.load(dw + i)\n"
            "    tl.store(dw + i, part)\n"
            "    tl.debug_barrier()\n"
            "    tl.atomic_xchg(lock, 0)\n")
    rng = np.random.default_rng(4)
    x = rng.standard_normal((P, 8)).astype(np.float32)
    bufs = {"x": ("fp32", x.reshape(-1)), "dw": ("fp32", np.zeros(8)), "cnt": ("int32", [0]), "lock": ("int32", [0])}
    lo, hi, st = _run("lock_ln", body, bufs, out_name="dw", grid=(P, 1, 1))
    from fractions import Fraction as Fr
    assert (st == H.ST_OK).all()
    for j in range(8):
        exact = sum(Fr(float(v)) for v in x[:, j])
        assert Fr(float(lo[j])) <= exact <= Fr(float(hi[j]))
    lo, hi, st = _run("lock_ln", body, bufs, out_name="cnt", grid=(P, 1, 1))
    assert st[0] == H.ST_OK and int(lo[0]) == 1


def test_a_lock_never_released_is_not_established_and_does_not_hang():
    body = ("    pid = tl.program_id(0)\n"
            "    while tl.atomic_cas(lock, 0, 1) == 1:\n        pass\n"
            "    tl.store(out + pid, pid + 1)\n")              # no release: every later program spins forever
    lo, hi, st = _run("lock_held", body, {"out": ("int32", np.zeros(4)), "lock": ("int32", [0])}, grid=(4, 1, 1))
    assert (st[1:] != H.ST_OK).all()
