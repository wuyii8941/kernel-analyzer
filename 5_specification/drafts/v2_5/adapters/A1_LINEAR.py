from _c import *
import spec_linear_attention as L
def _run(a, **over):
    v = a['variant']; kw = dict(a=QL(a['a']) if 'a' in a else None, b=QL(a['b']) if 'b' in a else None, scale=F(1))
    if 'cu' in a:
        o, _ = L.varlen(QL(a['q']), QL(a['k']), QL(a['v']), a['cu'], v, **kw); return tuple(x[0] for x in o)
    o, S = L.recurrent(QL(a['q']), QL(a['k']), QL(a['v']), v, **kw); return tuple(x[0] for x in o) + (S[0][0],)
def reference(call, a):
    if call == 'default_scale':
        o, _ = L.recurrent(QL(a['q']), QL(a['k']), QL(a['v']), 'linear'); return o[0][0]
    if call == 'gate': return L.gate_from_log(Q(a['g']) if 'ln_of' not in a else -iv.log(iv.mpf(a['ln_of'])), a['base'])
    if call == 'chunked':
        o, S = L.chunked(QL(a['q']), QL(a['k']), QL(a['v']), a['C'], a['variant'], a=QL(a['a'])); return tuple(x[0] for x in o) + (S[0][0],)
    if call == 'grad':
        g = L.backward(QL(a['q']), QL(a['k']), QL(a['v']), QL(a['dO']), QL(a['dST']), a['variant'], a=QL(a['a']), S0=QL(a['S0']))
        return g['dS0'][0][0]
    return _run(a)
def M1(call, a):        # gated delta with the gate applied after the write
    if call != 'run' or a['variant'] != 'gated_delta': return None
    S = F(0); outs = []
    for t in range(len(a['q'])):
        k, v, al, be, q = Q(a['k'][t][0]), Q(a['v'][t][0]), Q(a['a'][t]), Q(a['b'][t]), Q(a['q'][t][0])
        S = al * ((1 - be * k * k) * S + be * k * v); outs.append(S * q)
    return tuple(outs) + (S,)
def M2(call, a):        # state carried across sequence boundaries
    if 'cu' not in a: return None
    o, _ = L.recurrent(QL(a['q']), QL(a['k']), QL(a['v']), a['variant'], scale=F(1)); return tuple(x[0] for x in o)
def M3(call, a):        # final-state gradient dropped (fla HGRN / ABC chunked backward, survey)
    if call != 'grad': return None
    g = L.backward(QL(a['q']), QL(a['k']), QL(a['v']), QL(a['dO']), None, a['variant'], a=QL(a['a']), S0=QL(a['S0']))
    return g['dS0'][0][0]
def M4(call, a):        # decay applied after adding k v^T
    if call != 'run' or a['variant'] != 'decay': return None
    S = F(0); outs = []
    for t in range(len(a['q'])):
        S = Q(a['a'][t]) * (S + Q(a['k'][t][0]) * Q(a['v'][t][0])); outs.append(S * Q(a['q'][t][0]))
    return tuple(outs) + (S,)
def M5(call, a):        # natural-log gate used where the kernel declares log2
    if call != 'gate' or a['base'] != 'log2': return None
    return L.gate_from_log(Q(a['g']), 'ln')
def M6(call, a):        # chunked: carried state not decayed between chunks
    if call != 'chunked': return None
    S = F(0); outs = []; al = QL(a['a'])
    for t in range(len(a['q'])):
        k, v, q = Q(a['k'][t][0]), Q(a['v'][t][0]), Q(a['q'][t][0])
        outs.append(q * (al[t] * S + k * v)); S = S + k * v
    return tuple(outs) + (S,)
MUTANTS = dict(M1=M1, M2=M2, M3=M3, M4=M4, M5=M5, M6=M6)
