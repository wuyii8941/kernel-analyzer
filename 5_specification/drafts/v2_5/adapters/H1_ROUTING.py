from _c import *
import spec_moe_routing as M
def _route(a, **o):
    kw = dict(bias=QL(a['bias']) if 'bias' in a else None, n_group=a.get('n_group', 1), topk_group=a.get('topk_group', 1),
              routed_scaling_factor=Q(a.get('rsf', 1))); kw.update(o)
    return M.route(QL(a['logits']), a['k'], a['method'], **kw)
def reference(call, a):
    res = _route(a)
    if call == 'ids': return frozenset(r[0] for r in res)
    e = a['expert']; ws = set()
    for ids, w in res:
        if e in ids: ws.add(w[ids.index(e)])
    (lo, hi), = ws; return iv.mpf([lo, hi])
def M1(call, a):        # selection bias also used in the weights
    if call != 'weight' or a['method'] != 'deepseek_v3': return None
    s = [sigmoid(x) + I(b) for x, b in zip(QL(a['logits']), QL(a['bias']))]
    (ids, _), = [r for r in _route(a)][:1]; tot = isum(s[i] for i in ids); return s[a['expert']] / tot * I(Q(a.get('rsf', 1)))
def M2(call, a):        # group score = max instead of sum of top-2, ties broken by lowest index
    if call != 'ids' or a['method'] != 'deepseek_v3': return None
    lg = QL(a['logits']); E = len(lg); g = a['n_group']; gs = E // g; s = [sigmoid(x) for x in lg]
    gmax = [max(s[e].b for e in range(i * gs, (i + 1) * gs)) for i in range(g)]
    groups = sorted(range(g), key=lambda i: (-gmax[i], i))[:a['topk_group']]
    cand = sorted(e for i in groups for e in range(i * gs, (i + 1) * gs))
    return tuple(sorted(sorted(cand, key=lambda e: (-s[e].b, e))[:a['k']]))
def M3(call, a):        # no renormalization over the selected experts
    if call != 'weight' or a['method'] != 'softmax_topk_renorm': return None
    p = softmax_row(QL(a['logits']))[0]; return p[a['expert']]
MUTANTS = dict(M1=M1, M2=M2, M3=M3)
