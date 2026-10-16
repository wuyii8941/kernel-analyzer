from _c import *
import spec_quant as S
def reference(call, a):
    return S.pack_int4(a['vals'], a['order']) if call == 'pack' else tuple(S.unpack_int4(int(a['word'], 0), a['order']))
def M1(call, a):        # AWQ applied with the packing map [0,2,4,6,1,3,5,7] as positions (inverse confusion)
    if a['order'] != 'awq': return None
    pos = [0, 2, 4, 6, 1, 3, 5, 7]
    if call == 'pack': return sum(v << (4 * pos[i]) for i, v in enumerate(a['vals']))
    w = int(a['word'], 0); return tuple((w >> (4 * pos[i])) & 0xF for i in range(8))
def M2(call, a):        # high nibble first
    if call != 'pack': return None
    return sum(v << (4 * (7 - i)) for i, v in enumerate(a['vals']))
MUTANTS = dict(M1=M1, M2=M2)
