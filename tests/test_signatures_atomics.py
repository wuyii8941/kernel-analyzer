"""Atomic return values, bit-view atomics and order laws (DSL v2 increment 4, rc3 02 5.3, 6.3, 6.9, 10), on compiled-only
kernels evaluated on synthetic captures (CPU only).  Every assertion is against an independent enumeration of the
interleavings or an exact value."""
from __future__ import annotations

import itertools
from fractions import Fraction as Fr

import mpmath as mp
import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402
from test_signatures_structural import C, _check, _run  # noqa: E402

INF = float("inf")


def _orders_old(init, contribs, op):
    """Every value each update can return over all interleavings (by enumerating the permutations)."""
    seen = [set() for _ in contribs]
    for perm in itertools.permutations(range(len(contribs))):
        acc = init
        for k in perm:
            seen[k].add(acc)
            acc = op(acc, contribs[k])
    return seen


def test_returned_value_without_contention_is_the_value_before_the_launch():
    x = np.arange(1, C + 1, dtype=np.float32)
    init = np.asarray([10, 20, 30, 40, 50, 60, 70, 80], np.float32)
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    old = tl.atomic_add(acc + i, x)\n"
            "    tl.store(out + i, old)\n")
    bufs = {"x_ptr": ("fp32", x), "acc": ("fp32", init), "out": ("fp32", np.zeros(C))}
    lo, hi, st = _run("ret_free", body, bufs)
    _check("positive", lo, hi, st, [mp.mpf(float(v)) for v in init])
    assert (lo == hi).all()                                      # a point, not a set
    lo, hi, st = _run("ret_free", body, bufs, out_name="acc")
    _check("positive", lo, hi, st, [mp.mpf(float(a)) + mp.mpf(float(b)) for a, b in zip(init, x)])
    # integers, one program per address across a grid
    body = ("    pid = tl.program_id(0)\n    v = tl.load(x_ptr + pid)\n    old = tl.atomic_add(acc + pid, v)\n"
            "    tl.store(out + pid, old)\n")
    lo, hi, st = _run("ret_free_int", body, {"x_ptr": ("int32", np.arange(4)), "acc": ("int32", [5, 6, 7, 8]),
                                             "out": ("int32", np.zeros(4))}, grid=(4, 1, 1))
    _check("positive", lo, hi, st, [5, 6, 7, 8], is_int=True)


@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_official_pattern_contended_float_add_returns_a_set(dtype):
    # test_atomic_rmw (official): 5 programs update one address and store the old value
    x = np.asarray([2.0 ** i for i in range(5)], np.float32)
    body = ("    pid = tl.program_id(0)\n    v = tl.load(x_ptr + pid)\n    old = tl.atomic_add(z, v)\n"
            "    tl.store(out + pid, old)\n")
    bufs = {"x_ptr": (dtype, x), "z": (dtype, np.zeros(1)), "out": (dtype, np.zeros(5))}
    lo, hi, st = _run("ret_add_" + dtype, body, bufs, grid=(5, 1, 1))
    seen = _orders_old(Fr(0), [Fr(float(v)) for v in x], lambda a, b: a + b)
    for i in range(5):
        assert st[i] == H.ST_OK
        assert all(Fr(float(lo[i])) <= v <= Fr(float(hi[i])) for v in seen[i]), (i, lo[i], hi[i])
    lo, hi, st = _run("ret_add_" + dtype, body, bufs, out_name="z", grid=(5, 1, 1))
    _check("positive", lo, hi, st, [mp.mpf(31)])


def test_contended_integer_return_is_not_a_point_unless_every_order_agrees():
    body = ("    pid = tl.program_id(0)\n    v = tl.load(x_ptr + pid)\n    old = tl.atomic_add(z, v)\n"
            "    tl.store(out + pid, old)\n")
    lo, hi, st = _run("ret_int", body, {"x_ptr": ("int32", [1, 2, 4, 8]), "z": ("int32", [0]),
                                        "out": ("int32", np.zeros(4))}, grid=(4, 1, 1))
    assert (st != H.ST_OK).all()
    # adding zeros: every interleaving returns the initial value
    lo, hi, st = _run("ret_int", body, {"x_ptr": ("int32", [0, 0, 0, 0]), "z": ("int32", [9]),
                                        "out": ("int32", np.zeros(4))}, grid=(4, 1, 1))
    _check("positive", lo, hi, st, [9, 9, 9, 9], is_int=True)
    # max below the initial value: every interleaving returns it
    body_max = body.replace("atomic_add", "atomic_max")
    lo, hi, st = _run("ret_imax", body_max, {"x_ptr": ("int32", [1, 2, 3, 4]), "z": ("int32", [7]),
                                             "out": ("int32", np.zeros(4))}, grid=(4, 1, 1))
    _check("positive", lo, hi, st, [7, 7, 7, 7], is_int=True)


@pytest.mark.parametrize("init", [-INF, -100.0, 0.5])
def test_frontend_float_max_on_bit_patterns(init):
    # tl.atomic_max on fp32 lowers to max (non-negative values) and umin (negative values) on the i32 view
    x = np.asarray([-3.0, 2.5, -0.25, 7.0, -8.0, 1.0, 0.0, -1.0], np.float32)
    body = "    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    tl.atomic_max(out + i * 0, x)\n"
    lo, hi, st = _run("bv_max", body, {"x_ptr": ("fp32", x), "out": ("fp32", np.asarray([init], np.float32))})
    _check("positive", lo, hi, st, [mp.mpf(max([float(v) for v in x] + [init]))])
    body = "    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    tl.atomic_min(out + i * 0, x)\n"
    lo, hi, st = _run("bv_min", body, {"x_ptr": ("fp32", x), "out": ("fp32", np.asarray([-init], np.float32))})
    _check("positive", lo, hi, st, [mp.mpf(min([float(v) for v in x] + [-init]))])


def test_bit_view_needs_definite_bits():
    # the stored value is a non-point real (x / 3 enclosed): its bits are not definite, so the max is not established
    x = np.asarray([1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0], np.float32)
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    tl.store(out + i, x / 3.0)\n"
            "    tl.debug_barrier()\n    tl.atomic_max(out + i * 0 + 1, x)\n")
    lo, hi, st = _run("bv_indef", body, {"x_ptr": ("fp32", x), "out": ("fp32", np.zeros(C))})
    assert st[1] != H.ST_OK


def test_mixed_kinds_without_a_joint_law_are_not_established():
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i)\n    tl.atomic_add(out + i * 0, x)\n"
            "    tl.atomic_max(out + i * 0, x)\n")
    lo, hi, st = _run("mixed", body, {"x_ptr": ("int32", np.arange(C)), "out": ("int32", np.zeros(1))})
    assert st[0] != H.ST_OK


def test_undefined_integer_contribution_leaves_the_fold_not_established():
    # defect fixed in increment 4: an undefined (masked, no other) integer contribution was folded as established
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i, mask=i < 4)\n    tl.atomic_add(out + i * 0, x)\n")
    lo, hi, st = _run("undef_add", body, {"x_ptr": ("int32", np.arange(C)), "out": ("int32", np.zeros(1))})
    assert st[0] != H.ST_OK
    body = ("    i = tl.arange(0, N)\n    x = tl.load(x_ptr + i, mask=i < 4, other=0)\n"
            "    tl.atomic_add(out + i * 0, x)\n")
    lo, hi, st = _run("undef_add_other", body, {"x_ptr": ("int32", np.arange(C)), "out": ("int32", np.zeros(1))})
    _check("positive", lo, hi, st, [0 + 1 + 2 + 3], is_int=True)


def test_exchange_final_value_and_returns():
    x = np.asarray([1.5, -2.0, 3.25, 0.5], np.float32)
    body = ("    pid = tl.program_id(0)\n    v = tl.load(x_ptr + pid)\n    old = tl.atomic_xchg(z, v)\n"
            "    tl.store(out + pid, old)\n")
    bufs = {"x_ptr": ("fp32", x), "z": ("fp32", [9.0]), "out": ("fp32", np.zeros(4))}
    lo, hi, st = _run("xchg", body, bufs, out_name="z", grid=(4, 1, 1))
    assert st[0] == H.ST_OK and all(lo[0] <= v <= hi[0] for v in x)       # the last exchange: any contribution
    lo, hi, st = _run("xchg", body, bufs, grid=(4, 1, 1))
    seen = _orders_old(Fr(9), [Fr(float(v)) for v in x], lambda a, b: b)
    for i in range(4):
        assert st[i] == H.ST_OK and all(Fr(float(lo[i])) <= v <= Fr(float(hi[i])) for v in seen[i])
    # one program: a point
    lo, hi, st = _run("xchg", body, bufs, out_name="z", grid=(1, 1, 1))
    _check("positive", lo, hi, st, [mp.mpf(1.5)])


def test_returned_value_used_as_an_address_does_not_crash_and_is_not_established():
    body = ("    pid = tl.program_id(0)\n    t = tl.atomic_add(cnt, 1)\n    tl.store(out + t, pid)\n")
    lo, hi, st = _run("ticket", body, {"cnt": ("int32", [0]), "out": ("int32", np.zeros(4))}, grid=(4, 1, 1))
    assert (st != H.ST_OK).all()
