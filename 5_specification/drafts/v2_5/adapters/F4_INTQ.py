from _c import *
import spec_quant as S
def reference(call, a):
    if call == 'sym':
        (sc, q), = S.quant_int_sym(QL(a['x']), bits=a.get('bits', 8)); return (sc, tuple(q))
    (sc, z, q), = S.quant_int_asym(QL(a['x']), bits=a['bits']); return (sc, z, tuple(q))
def M1(call, a):        # round half away from zero (LMDeploy per_token_quant_int8 kernel, survey)
    if call != 'sym': return None
    (sc, q), = S.quant_int_sym(QL(a['x']), bits=a.get('bits', 8), mode='rna'); return (sc, tuple(q))
def M2(call, a):        # divide by 2^(b-1) instead of 2^(b-1) - 1
    if call != 'sym': return None
    x = QL(a['x']); sc = max(abs(v) for v in x) / 2 ** (a.get('bits', 8) - 1)
    return (sc, tuple(max(-127, min(127, S._round_int(v / sc, 'rne'))) for v in x))
def M3(call, a):        # asymmetric range without forcing 0 inside [min, max]
    if call != 'asym': return None
    x = QL(a['x']); top = 2 ** a['bits'] - 1; lo, hi = min(x), max(x); sc = (hi - lo) / top
    z = max(0, min(top, S._round_int(-lo / sc, 'rne'))); return (sc, z, tuple(max(0, min(top, S._round_int(v / sc, 'rne') + z)) for v in x))
def M4(call, a):
    return (F(0), tuple(0 for _ in a['x'])) if call == 'sym' and all(Q(v) == 0 for v in a['x']) else None
MUTANTS = dict(M1=M1, M2=M2, M3=M3, M4=M4)
