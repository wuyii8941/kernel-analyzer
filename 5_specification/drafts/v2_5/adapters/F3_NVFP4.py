from _c import *
import spec_quant as S
def _out(sg, blocks): return ('none' if sg is None else sg, tuple((sb, tuple(el)) for sb, el in blocks))
def reference(call, a): return _out(*S.nvfp4_quantize(QL(a['blocks']), a.get('global_amax')))
def M1(call, a):        # no second-level (global) scale
    out = []
    for b in QL(a['blocks']):
        sb = S.round_fmt(max(abs(v) for v in b) / 6, 'e4m3', saturate=True)
        out.append((sb, [S.round_fmt(v / sb, 'e2m1', saturate=True) for v in b]))
    return _out(F(1), out)
def M2(call, a):        # MX-style power-of-two block scale instead of E4M3
    sg, _ = S.nvfp4_quantize(QL(a['blocks']), a.get('global_amax')); out = []
    for b in QL(a['blocks']):
        sb = F(2) ** S.ceil_log2((max(abs(v) for v in b) / 6) / sg)
        out.append((sb, [S.round_fmt(v / (sb * sg), 'e2m1', saturate=True) for v in b]))
    return _out(sg, out)
def M3(call, a):
    sg, blocks = S.nvfp4_quantize(QL(a['blocks']), a.get('global_amax'))
    return _out(sg, [(sb, ['nan'] * len(el)) if sb == 0 else (sb, el) for sb, el in blocks])
MUTANTS = dict(M1=M1, M2=M2, M3=M3)
