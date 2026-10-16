from _c import *
import spec_quant as S
def reference(call, a): return S.round_fmt(Q(a['x']), a['fmt'], a.get('mode', 'rne'), a.get('saturate', False), u=a.get('u'))
def reading(name, call, a): return S.round_fmt(Q(a['x']), a['fmt'], a.get('mode', 'rne'), a.get('saturate', False), overflow_test='before_rounding')
def _m(mode=None, fp4_nan=False, le=False):
    def f(call, a):
        x, fmt = Q(a['x']), a['fmt']
        if fp4_nan and fmt in ('e2m1', 'e2m3', 'e3m2') and abs(x) > S.FMT[fmt]['maxv']: return 'nan'
        if le and a.get('mode') == 'stochastic':
            e = max(S.floor_log2(abs(x)), S.FMT[fmt]['emin']); q = F(2) ** (e - S.FMT[fmt]['m']); n = abs(x) / q
            lo = n.numerator // n.denominator; fr = n - lo
            return (1 if x > 0 else -1) * (lo + (1 if fr > 0 and Q(a['u']) <= fr else 0)) * q
        return S.round_fmt(x, fmt, mode or a.get('mode', 'rne'), a.get('saturate', False), u=a.get('u'))
    return f
MUTANTS = dict(M1=_m(mode='rna'), M2=_m(mode='rtz'), M3=_m(fp4_nan=True), M4=_m(le=True))
