"""Integer tt.dot (DSL v2 increment 4): exact matmul(a, b) + c on signed int8 operands into an int32 accumulator, compared
with Python integers; a result outside the accumulator range is not established (saturation or wrap is lowering
dependent).  Found by the official-main capture (test_dot int8 crashed the evaluator)."""
from __future__ import annotations

import numpy as np
import pytest

triton = pytest.importorskip("triton")

import signature_harness as H  # noqa: E402
from test_signatures_structural import _check, _run  # noqa: E402

M, K = 16, 32  # int8 dot needs K >= 32


def _body():
    return ("    r = tl.arange(0, 16)\n    k = tl.arange(0, 32)\n    a = tl.load(a_ptr + r[:, None] * 32 + k[None, :])\n"
            "    b = tl.load(b_ptr + k[:, None] * 16 + r[None, :])\n    c = tl.load(c_ptr + r[:, None] * 16 + r[None, :])\n"
            "    d = tl.dot(a, b, c, out_dtype=tl.int32)\n    tl.store(out + r[:, None] * 16 + r[None, :], d)\n")


@pytest.mark.parametrize("category", ["positive", "boundary", "premise_violation"])
def test_int8_dot_signature(category):
    rng = np.random.default_rng(3)
    a = rng.integers(-128, 128, (M, K)).astype(np.int8)
    b = rng.integers(-128, 128, (K, M)).astype(np.int8)
    c = rng.integers(-1000, 1000, (M, M)).astype(np.int32)
    if category == "boundary":
        a[0, :], b[:, 0] = -128, -128          # 32 * 16384: far inside int32
        dot = int(np.dot(a[1].astype(np.int64), b[:, 1].astype(np.int64)))
        c[1, 1] = (2 ** 31 - 1 - dot) if dot >= 0 else (-2 ** 31 - dot)  # the result is exactly INT32_MAX / MIN
    if category == "premise_violation":
        c[0, 0] = 2 ** 31 - 1                  # + a positive dot product: leaves the accumulator range
        a[0, :], b[:, 0] = 127, 127
    lo, hi, st = _run("idot", _body(), {"a_ptr": ("int8", a), "b_ptr": ("int8", b), "c_ptr": ("int32", c),
                                        "out": ("int32", np.zeros((M, M)))})
    exact = (a.astype(object) @ b.astype(object) + c.astype(object)).reshape(-1)
    expect = [int(v) if -2 ** 31 <= v < 2 ** 31 else None for v in exact]
    _check(category, lo, lo, st, expect, is_int=True)
    if category == "premise_violation":
        assert st[0] != H.ST_OK
