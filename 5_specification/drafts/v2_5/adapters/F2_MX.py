from _c import *
import spec_quant as S
def _out(X, el): return ('none' if X is None else X, tuple(el))
def reference(call, a): return _out(*S.mx_quantize(QL(a['block']), a['fmt']))
def reading(name, call, a): return _out(*S.mx_quantize(QL(a['block']), a['fmt'], scale_rule='ceil_pow2'))
def M1(call, a):
    b = QL(a['block']); amax = max(abs(v) for v in b); X = F(2) ** (S.ceil_log2(amax) - S.emax_elem(a['fmt']))
    return _out(X, [S.round_fmt(v / X, a['fmt'], saturate=True) for v in b])
def M2(call, a):
    b = QL(a['block']); amax = max(abs(v) for v in b); X = F(2) ** (S.floor_log2(amax) - S.emax_elem(a['fmt']))
    return _out(X, [S.round_fmt(v / X, a['fmt'], saturate=False) for v in b])
def M3(call, a):
    b = QL(a['block']); X = max(abs(v) for v in b) / S.FMT[a['fmt']]['maxv']
    return _out(X, [S.round_fmt(v / X, a['fmt'], saturate=True) for v in b])
MUTANTS = dict(M1=M1, M2=M2, M3=M3)
