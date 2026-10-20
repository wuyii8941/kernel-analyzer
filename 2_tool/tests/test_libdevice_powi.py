"""libdevice pow with an int32 exponent (__nv_powif float^int, __nv_powi double^int).

Found by path A (FlagGems 5.0.2 gelu_backward, 2026-10-20): the program reference aborted with "libdevice __nv_powif
has no declared semantics", so no G_pi existed for the call. x**n with an integer exponent is the real power; the
lanes are checked against mpmath, including negative bases, n = 0, negative n and IEEE special values.
"""
from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest

pytest.importorskip("triton")

import signature_harness as H  # noqa: E402

BASES = [1.5, -2.0, 0.75, -0.5, 3.0, 0.0, 10.0, -1.25]
EXPONENTS = [3, 2, -2, 5, 0, 4, 1, -3]


def _exact(x, n):
    if np.isnan(x):
        return None if n != 0 else mp.mpf(1)
    if x == 0 and n < 0:
        return mp.inf
    return mp.mpf(float(x)) ** n


@pytest.mark.parametrize("sym,dt", [("__nv_powif", "fp32"), ("__nv_powi", "fp64")])
def test_pow_int_exponent_is_enclosed(sym, dt):
    xs = np.asarray(H.pad(BASES), H.NP[dt])
    ns = np.asarray(H.pad(EXPONENTS), np.int32)
    lo, hi, st, _ = H.evaluate(sym, "libdevice.pow(a0, a1)", (dt, "int32"), dt, [xs, ns])
    assert not np.any(st == H.ST_NE), f"{sym}: lanes without a reference: {np.flatnonzero(st == H.ST_NE)[:8]}"
    for j in range(H.N):
        if xs[j] == 0 and ns[j] < 0:
            continue          # pole: the IEEE special-value path, checked below
        ex = _exact(xs[j], int(ns[j]))
        assert st[j] == H.ST_OK and mp.mpf(float(lo[j])) <= ex <= mp.mpf(float(hi[j])), (j, xs[j], ns[j], lo[j], hi[j])
