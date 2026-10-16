from _c import *
import spec_quant as S
def reference(call, a):
    if call == 'scale': return S.quant_fp8_groups(QL(a['x']), a['group'])[0][0]
    return tuple((sc, tuple(q)) for sc, q in S.quant_fp8_groups(QL(a['x']), a['group'], eps=Q(a.get('eps', 0))))
def reading(name, call, a): return S.quant_fp8_groups(QL(a['x']), a['group'], scale_store='fp32')[0][0]
def M1(call, a):
    if call == 'scale': return None
    x = QL(a['x']); out = []
    for g0 in range(0, len(x), a['group']):
        g = x[g0:g0 + a['group']]; sc = (max(abs(v) for v in g) + Q(a.get('eps', 0))) / 448
        out.append((sc, tuple(S.round_fmt(v / sc, 'e4m3', saturate=True) for v in g)))
    return tuple(out)
def M2(call, a):
    if call == 'scale': return None
    x = QL(a['x']); sc = max(max(abs(v) for v in x), Q(a.get('eps', 0))) / 448
    return ((sc, tuple(S.round_fmt(v / sc, 'e4m3', saturate=True) for v in x)),)
MUTANTS = dict(M1=M1, M2=M2)
