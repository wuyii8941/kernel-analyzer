import sys, mpmath as mp
from fractions import Fraction as F
import os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'specification_v2')  # draft/ moved one level down
sys.path[:0] = [R, R + '/algorithm_specs/phase1', R + '/algorithm_specs/phase2', R + '/algorithm_specs/phase3']
mp.mp.dps = 60
UNDEF = 'undefined'
def q(v): return F(str(v)) if isinstance(v, str) else F(v)
def mpq(v): v = q(v); return mp.mpf(v.numerator) / v.denominator
def bounds(x):
    """reference output -> (lo, hi) as mpf; accepts Fraction, Decimal Interval (lo/hi), mpmath iv (a/b), rigorous.Interval"""
    if isinstance(x, F): return mpq(x), mpq(x)
    for a, b in (('lo', 'hi'), ('a', 'b')):
        if hasattr(x, a):
            lo, hi = getattr(x, a), getattr(x, b)
            conv = lambda t: mp.mpf(str(t).strip('[]').split(',')[0]) if not isinstance(t, F) else mpq(t)
            return conv(lo), conv(hi)
    return mp.mpf(str(x)), mp.mpf(str(x))
