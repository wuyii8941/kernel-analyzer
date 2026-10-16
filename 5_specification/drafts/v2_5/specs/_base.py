"""Shared base for the v2.5 specs: exact rationals (fractions.Fraction) plus the existing rigorous interval module
(specification_v2/algorithm_specs/phase2/rigorous_mp.py, mpmath.iv at 200 bits). No new interval arithmetic is added."""
import os, sys, itertools
from fractions import Fraction as F
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', '..', 'specification_v2', 'algorithm_specs', 'phase2'))
from rigorous_mp import I, iv, isum, tanh, sigmoid, softmax_row, contains, overlaps, SpecNotEstablished, SpecInputError  # noqa: F401
import mpmath as mp

def Q(x):
    """exact rational from int / str / Fraction / float (floats are exact binary values)."""
    return x if isinstance(x, F) else F(x) if not isinstance(x, str) else F(x)

def is_iv(x): return isinstance(x, type(iv.mpf(0)))

def floor_log2(a):
    """exact floor(log2(a)) for a positive rational."""
    a = Q(a)
    if a <= 0: raise SpecInputError('floor_log2 of non-positive value')
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if F(2) ** e > a: e -= 1
    if F(2) ** (e + 1) <= a: e += 1
    return e

def ceil_log2(a):
    e = floor_log2(a)
    return e if F(2) ** e == a else e + 1

def gt(a, b):
    """strict order that is certain: True / False / None (undecided when intervals overlap)."""
    if not is_iv(a) and not is_iv(b): return a > b
    a, b = I(a), I(b)
    if a.a > b.b: return True
    if a.b <= b.a: return False
    return None

def topk_sets(vals, k):
    """all index sets of size k that are admissible top-k selections: no excluded value is certainly larger
    than an included one. Ties (exact equality or overlapping intervals) give several admissible sets."""
    n = len(vals); k = min(k, n); out = set()
    for S in itertools.combinations(range(n), k):
        rest = [j for j in range(n) if j not in S]
        if all(gt(vals[j], vals[i]) is not True for i in S for j in rest):
            out.add(frozenset(S))
    return out

def matvec_T(S, q):
    """S is dk x dv (list of rows), q has length dk: returns S^T q (length dv)."""
    dk, dv = len(S), len(S[0])
    return [sum((S[i][j] * q[i] for i in range(dk)), F(0) if not any(is_iv(S[i][j]) or is_iv(q[i]) for i in range(dk)) else I(0)) for j in range(dv)]
