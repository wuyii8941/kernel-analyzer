from _c import *
import spec_sparse_attention as C
def reference(call, a):
    if call == 'score': return C.indexer_scores(QL(a['q']), QL(a['w']), QL(a['k']), a['t'])[a['s']]
    if call == 'topk':
        sc = QL(a['scores']) if 'scores' in a else C.indexer_scores(QL(a['q']), QL(a['w']), QL(a['k']), a['t'])
        return frozenset(C.select_topk(sc, a['k_sel']))
    if call == 'blocks': return frozenset(C.select_blocks(C.block_scores(QL(a['tok']), a['block'], 'sum'), a['n'], tuple(a['forced'])))
    return tuple(C.key_range(a['ks'], a['ke']))
def reading(name, call, a):
    if call == 'blocks': return frozenset(C.select_blocks(C.block_scores(QL(a['tok']), a['block'], 'max'), a['n'], tuple(a['forced'])))
    return tuple(C.key_range(a['ks'], a['ke'], 'closed'))
def M1(call, a):        # ReLU applied after the head weighting
    if call != 'score': return None
    q, w, k, s = QL(a['q']), QL(a['w']), QL(a['k']), a['s']
    return max(F(0), sum(w[j] * sum(x * y for x, y in zip(q[j], k[s])) for j in range(len(q))))
def M2(call, a):        # only strictly earlier tokens (s < t)
    if call != 'topk' or 'scores' in a: return None
    sc = C.indexer_scores(QL(a['q']), QL(a['w']), QL(a['k']), a['t'])[:-1]
    order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:a['k_sel']]
    return tuple(sorted(order) + [-1] * (a['k_sel'] - len(order)))
def M3(call, a):        # pad with 0 instead of -1
    if call != 'topk': return None
    sc = QL(a['scores']) if 'scores' in a else C.indexer_scores(QL(a['q']), QL(a['w']), QL(a['k']), a['t'])
    ids = sorted(sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:a['k_sel']]); return tuple(ids + [0] * (a['k_sel'] - len(ids)))
MUTANTS = dict(M1=M1, M2=M2, M3=M3)
