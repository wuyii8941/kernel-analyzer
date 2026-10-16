from _c import *
import spec_paged_attention as P
def _o(res, i=0): return res[i][0][0]
def reference(call, a):
    if call == 'slots': return tuple(P.slots_block_table(a['block_table'], a['seq_len'], a['page_size']))
    if call == 'kv_head': return P.kv_head(a['h'], a['hq'], a['hkv'])
    kw = {k: (Q(v) if k in ('scale', 'softcap', 'sink') else v) for k, v in a.get('kw', {}).items()}
    if 'key_set' in kw: kw['key_set'] = set(kw['key_set'])
    return _o(P.attention_rows(QL(a['q']), QL(a['k']), QL(a['v']), **kw), a.get('row', 0))
def reading(name, call, a):
    if call == 'kv_head': return P.kv_head(a['h'], a['hq'], a['hkv'], mapping='modulo')
    kw = {k: (Q(v) if k in ('scale', 'softcap', 'sink') else v) for k, v in a.get('kw', {}).items()}; kw['window_rule'] = 'inclusive'
    return _o(P.attention_rows(QL(a['q']), QL(a['k']), QL(a['v']), **kw), a.get('row', 0))
def _attn(a, top_left=False, no_sink=False, cap_first=False, mask_first=False):
    q, K, V = QL(a['q']), QL(a['k']), QL(a['v']); kw = a.get('kw', {}); i = a.get('row', 0)
    Lq, Lk = len(q), len(K); p = i if (top_left or kw.get('alignment') == 'top_left') else Lk - Lq + i
    sc = I(Q(kw['scale'])) if 'scale' in kw else 1 / iv.sqrt(iv.mpf(len(q[0])))
    cap = Q(kw['softcap']) if 'softcap' in kw else None; logits, vals = [], []
    for j in range(Lk):
        visible = (not kw.get('causal', True) or j <= p) and ('window' not in kw or p - j < kw['window'])
        if not visible and not mask_first: continue
        s = isum(I(q[i][t]) * I(K[j][t]) for t in range(len(q[0])))
        s = (I(cap) * tanh(s / I(cap)) * sc) if (cap is not None and cap_first) else sc * s
        if cap is not None and not cap_first: s = I(cap) * tanh(s / I(cap))
        if not visible: s = -I(cap) if cap is not None else None
        if s is None: continue
        logits.append(s); vals.append(V[j][0])
    if 'sink' in kw and not no_sink: logits.append(I(Q(kw['sink']))); vals.append(None)
    pr = softmax_row(logits)[0]
    return isum(pr[n] * I(v) for n, v in enumerate(vals) if v is not None)
M5 = lambda c, a: 'nan' if c == 'attn' and a.get('kw', {}).get('key_set') == [] else None
MUTANTS = dict(M5=M5, M1=lambda c, a: _attn(a, top_left=True) if c == 'attn' else None,
               M2=lambda c, a: _attn(a, no_sink=True) if c == 'attn' else None,
               M3=lambda c, a: _attn(a, cap_first=True) if c == 'attn' else None,
               M4=lambda c, a: _attn(a, mask_first=True) if c == 'attn' else None)
