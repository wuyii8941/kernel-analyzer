"""SumK / DotK enclosures against exact rational sums (tool 3.x).  The counterexample row came from the audit of
2026-10-08: with a pairwise final pass over all n terms the dominant term was rounded several times and the enclosure
missed the exact sum; the final pass now adds p_n last."""
from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path

import numpy as np

from kernel_analyzer.reference_eval import intervals as iv

DATA = Path(__file__).resolve().parent / "data"


def _exact(row):
    return sum(F(float(v)) for v in row)


def test_audit_counterexample_is_enclosed():
    x = np.load(DATA / "sumk_counterexample.npy")[None, :]
    lo, hi = iv.isum(x, x, 1)
    res, err, ok = iv.sum_k(x, 1)
    s = _exact(x[0])
    assert F(float(lo[0])) <= s <= F(float(hi[0]))
    assert abs(F(float(res[0])) - s) <= F(float(err[0]))


def test_ill_conditioned_rows_are_enclosed():
    rng = np.random.default_rng(7)
    for n in (2, 8, 33, 256):
        for _ in range(150):
            m = rng.normal(0, 1, n) * np.exp2(rng.integers(0, 80, n))
            row = np.concatenate([m[: n // 2], -m[: n // 2] * (1 + 2.0 ** -52), m[n // 2:]])[:n]
            rng.shuffle(row)
            X = row[None, :]
            lo, hi = iv.isum(X, X, 1)
            res, err, _ = iv.sum_k(X, 1)
            s = _exact(row)
            assert F(float(lo[0])) <= s <= F(float(hi[0]))
            assert abs(F(float(res[0])) - s) <= F(float(err[0]))


def test_dot_k_encloses_exact_products():
    rng = np.random.default_rng(3)
    for _ in range(30):
        a = rng.normal(0, 1, (5, 17)) * np.exp2(rng.integers(-20, 20, (5, 17)))
        b = rng.normal(0, 1, (17, 4))
        lo, hi = iv.idot(a, a, b, b)
        for i in range(5):
            for j in range(4):
                e = sum(F(float(a[i, k])) * F(float(b[k, j])) for k in range(17))
                assert F(float(lo[i, j])) <= e <= F(float(hi[i, j]))
