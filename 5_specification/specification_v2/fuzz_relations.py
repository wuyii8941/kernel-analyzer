"""Random-input validation of the derived relations (the generator the spec layer needs: inputs sampled inside each
role's accepted domain, not hand-picked examples). Every trial records holds / fails / declined (the spec itself
refused the input as outside its domain). Failures are printed with their witness. Seeds are fixed per relation."""
import sys, os, json, random, hashlib, traceback
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'algorithm_specs', 'phase1'), os.path.join(HERE, 'algorithm_specs', 'phase2'), os.path.join(HERE, 'algorithm_specs', 'phase3'), HERE]
from derive import enclosure_zero, _frac
import spec_cross_entropy as ce, spec_index_scatter as ix, spec_accumulation as acc, spec_pooling as pool
import spec_base_ops as bo, spec_attention as at, spec_normalization as sn, spec_optimizers as so, spec_moe as moe
from rigorous_mp import iv
import spec_ema as ema, spec_kvcache as kvc, spec_kvcomp as kcp, spec_sparse as sps

def q(rng, lo=-4, hi=4, den=7): return F(rng.randint(lo * den, hi * den), den)
def vec(rng, n, **kw): return [q(rng, **kw) for _ in range(n)]
def isum(vals):
    s = None
    for v in vals: s = v if s is None else s + v
    return s
def encl_all(pairs): return all(enclosure_zero(a - b) for a, b in pairs)

RELS = {}
def rel(name):
    def deco(f): RELS[name] = f; return f
    return deco

@rel('SOFTMAX: softmax(x + c) = softmax(x)')
def _(rng):
    n = rng.randint(2, 6); x = vec(rng, n); c = q(rng)
    return encl_all(zip(bo.softmax(x), bo.softmax([v + c for v in x])))
@rel('SOFTMAX: Σ softmax = 1')
def _(rng):
    x = vec(rng, rng.randint(2, 6)); return enclosure_zero(isum(bo.softmax(x)) - 1)
@rel('SOFTMAX: online logsumexp = whole')
def _(rng):
    n = rng.randint(2, 8); x = vec(rng, n, lo=-20, hi=20); k = rng.randint(1, n - 1)
    la, lb = bo.logsumexp(x[:k]), bo.logsumexp(x[k:]); m = la if la.b >= lb.b else lb
    return enclosure_zero(bo.logsumexp(x) - (m + iv.log(iv.exp(la - m) + iv.exp(lb - m))))
@rel('CE: row shift invariance')
def _(rng):
    N, K = rng.randint(1, 4), rng.randint(2, 5); L = [vec(rng, K) for _ in range(N)]; T = [rng.randrange(K) for _ in range(N)]
    cs = vec(rng, N); L2 = [[v + cs[n] for v in row] for n, row in enumerate(L)]
    return enclosure_zero(ce.cross_entropy(L, T)['loss'] - ce.cross_entropy(L2, T)['loss'])
@rel('CE: ignored rows irrelevant')
def _(rng):
    N, K = rng.randint(2, 5), rng.randint(2, 5); L = [vec(rng, K) for _ in range(N)]; T = [rng.choice([rng.randrange(K), -100]) for _ in range(N)]
    if all(t == -100 for t in T): T[0] = 0
    L2 = [[v + q(rng) if T[n] == -100 else v for v in row] for n, row in enumerate(L)]
    r1, r2 = ce.cross_entropy(L, T), ce.cross_entropy(L2, T); return r1['loss'].lo == r2['loss'].lo and r1['loss'].hi == r2['loss'].hi
@rel('CE: Σ_k grad = 0 per row')
def _(rng):
    N, K = rng.randint(1, 4), rng.randint(2, 5); L = [vec(rng, K) for _ in range(N)]; T = [rng.randrange(K) for _ in range(N)]
    return all(s.contains(0) for s in ce.prop_grad_row_sums(ce.cross_entropy(L, T)['grad']))
@rel('CE: reduction=sum splits over batches')
def _(rng):
    N, K = rng.randint(2, 6), rng.randint(2, 5); L = [vec(rng, K) for _ in range(N)]; T = [rng.randrange(K) for _ in range(N)]; h = rng.randint(1, N - 1)
    w = ce.cross_entropy(L, T, reduction='sum'); a = ce.cross_entropy(L[:h], T[:h], reduction='sum'); b = ce.cross_entropy(L[h:], T[h:], reduction='sum')
    return enclosure_zero(w['loss'] - (a['loss'] + b['loss']))
@rel('INDEXREDUCE: Σ index_add = Σ self + α Σ src')
def _(rng):
    n, m = rng.randint(1, 6), rng.randint(1, 6); s = vec(rng, n); idx = [rng.randrange(n) for _ in range(m)]; src = vec(rng, m); a = q(rng)
    return sum(ix.index_add(s, 0, idx, src, alpha=a), F(0)) == sum(s, F(0)) + a * sum(src, F(0))
@rel('INDEXREDUCE: include_self=False ignores self at written rows')
def _(rng):
    n, m = rng.randint(1, 6), rng.randint(1, 6); s = vec(rng, n); idx = [rng.randrange(n) for _ in range(m)]; src = vec(rng, m); op = rng.choice(['amax', 'amin', 'prod'])
    alt = [v + q(rng) if i in set(idx) else v for i, v in enumerate(s)]
    return ix.index_reduce(s, 0, idx, src, op, include_self=False) == ix.index_reduce(alt, 0, idx, src, op, include_self=False)
@rel('ACCUM: window loss = Σ / count and regroup invariance')
def _(rng):
    B = rng.randint(1, 4); per = [vec(rng, rng.randint(1, 5)) for _ in range(B)]; flat = [v for b in per for v in b]
    ok1 = acc.window_loss(per) == sum(flat, F(0)) / len(flat)
    k = rng.randint(1, len(flat)); a, b = acc.prop_split_invariance(per, lambda fl: [fl[:k], fl[k:]] if k < len(fl) else [fl]); return ok1 and a == b
@rel('MATMUL: (sA)B = s(AB)')
def _(rng):
    n, k, m = (rng.randint(1, 4) for _ in range(3)); A = [vec(rng, k) for _ in range(n)]; B = [vec(rng, m) for _ in range(k)]; s = q(rng)
    return bo.matmul2([[s * v for v in r] for r in A], B) == [[s * v for v in r] for r in bo.matmul2(A, B)]
@rel('NORM_LN: shift invariance, any eps')
def _(rng):
    x = vec(rng, rng.randint(2, 6)); b = q(rng); eps = F(rng.randint(0, 3), 100)
    if len(set(x)) == 1 and eps == 0: return 'declined'
    y1, y2 = sn.prop_shift_scale_invariance(sn.layer_norm, x, F(1), b, gamma=None, beta=None, eps=eps); return encl_all(zip(y1, y2))
@rel('NORM_RMS: positive scale invariance, eps=0')
def _(rng):
    x = vec(rng, rng.randint(2, 6)); a = q(rng, lo=1, hi=5)
    if all(v == 0 for v in x): return 'declined'
    y1, y2 = sn.prop_shift_scale_invariance(sn.rms_norm, x, a, F(0), gamma=None, eps=0); return encl_all(zip(y1, y2))
@rel('ADAM: zero effective gradient -> geometric decay of m, v (decoupled)')
def _(rng):
    n = rng.randint(1, 4); th = vec(rng, n); g = vec(rng, n); st = so.AdamState(n)
    so.adam_step(th, g, st, F(1, 1000)); m0, v0 = list(st.m), list(st.v); so.adam_step(th, [F(0)] * n, st, F(1, 1000))
    return st.m == [F(9, 10) * m for m in m0] and st.v == [F(999, 1000) * v for v in v0]
@rel('AVGPOOL: linearity and adjoint')
def _(rng):
    L = rng.randint(2, 9); x = [vec(rng, L)]; k = rng.randint(1, min(3, L)); s = rng.randint(1, k); a = q(rng)
    y = pool.avg_pool(x, k, s); ok1 = pool.avg_pool([[a * v for v in x[0]]], k, s) == [[a * v for v in r] for r in y]
    g = [vec(rng, len(y[0]))]; gx = pool.avg_pool_backward(g, [1, L], k, s); lhs, rhs = pool.prop_adjoint(y, g, x, gx); return ok1 and lhs == rhs
@rel('ROPE: relative position and norm preservation')
def _(rng):
    d = rng.choice([2, 4, 6]); qv, kv = vec(rng, d), vec(rng, d); m, n = rng.randint(0, 20), rng.randint(0, 20)
    a, b = at.prop_rope_relative(qv, kv, m, n, d); y = at.rope_apply(qv, m, d); nb, na = at.prop_rope_norm_preserving(qv, y)
    return enclosure_zero(a - b) and enclosure_zero(na - nb)
@rel('ATTN: query-row permutation equivariance; masked K/V irrelevant')
def _(rng):
    Lq, Lk, E = rng.randint(1, 3), rng.randint(2, 4), rng.randint(1, 3)
    Q = [vec(rng, E) for _ in range(Lq)]; K = [vec(rng, E) for _ in range(Lk)]; V = [vec(rng, 2) for _ in range(Lk)]
    mask = [[rng.random() < 0.7 for _ in range(Lk)] for _ in range(Lq)]
    for row in mask:
        if not any(row): row[rng.randrange(Lk)] = True
    perm = list(range(Lq)); rng.shuffle(perm)
    o1, _ = at.attention(Q, K, V, attn_mask=mask); o2, _ = at.attention([Q[i] for i in perm], K, V, attn_mask=[mask[i] for i in perm])
    ok1 = all(encl_all(zip(o1[perm[r]], o2[r])) for r in range(Lq))
    dead = [j for j in range(Lk) if not any(mask[i][j] for i in range(Lq))]
    K2 = [[v + q(rng) for v in K[j]] if j in dead else K[j] for j in range(Lk)]; V2 = [[v + q(rng) for v in V[j]] if j in dead else V[j] for j in range(Lk)]
    o3, _ = at.attention(Q, K2, V2, attn_mask=mask); return ok1 and all(encl_all(zip(o1[r], o3[r])) for r in range(Lq))
@rel('MOE: renormalised top-k weights sum to 1')
def _(rng):
    E = rng.randint(2, 5); k = rng.randint(1, E); p = moe.router_probs(vec(rng, E)); S = sorted(moe.topk_sets(p, k)[0])
    return enclosure_zero(isum(moe.combine_weights(p, S, True).values()) - 1)

@rel('EMA: SWA = mean, EMA closed form')
def _(rng):
    n, m = rng.randint(1, 6), rng.randint(1, 3); hist = [vec(rng, m) for _ in range(n)]; d = F(rng.randint(0, 10), 10)
    a, b = ema.prop_swa_is_mean(hist); c, e = ema.prop_ema_closed_form(hist, d); return a == b and c == e
@rel('KVCACHE: decode (contiguous / paged, with and without RoPE) = causal prefill')
def _(rng):
    L, d = rng.randint(1, 4), rng.choice([2, 4]); Q = [vec(rng, d) for _ in range(L)]; K = [vec(rng, d) for _ in range(L)]; V = [vec(rng, 2) for _ in range(L)]
    rope = rng.random() < 0.5; cache = kvc.ContiguousCache() if rng.random() < 0.5 else kvc.PagedCache(rng.randint(1, 3), list(range(8))[::-1])
    pre = kvc.prefill(Q, K, V, rope=rope); dec = kvc.decode(Q, K, V, cache, rope=rope); return all(encl_all(zip(a, b)) for a, b in zip(pre, dec))
@rel('KVCOMP: shift by the stride is equivariant (uniform and gated, shared or per-offset)')
def _(rng):
    S = W = rng.randint(1, 3); d = rng.randint(1, 2); L = rng.randint(W, 3 * W + 2); H = [vec(rng, d) for _ in range(L)]; per = rng.random() < 0.5; mode = rng.choice(['uniform', 'gated'])
    C = [[vec(rng, d) for _ in range(d)] for _ in range(W)] if per else [vec(rng, d) for _ in range(d)]
    Z = [vec(rng, d) for _ in range(W)] if per else vec(rng, d); b = vec(rng, W) if per else q(rng)
    a = kcp.compress(H, W, S, C, mode, Z, b, per); s = kcp.compress([vec(rng, d) for _ in range(S)] + H, W, S, C, mode, Z, b, per)[1:]
    return all(((x == y) if mode == 'uniform' else enclosure_zero(x - y)) for ra, rb in zip(a, s) for x, y in zip(ra, rb))
@rel('SPARSE: spmm = dense matmul; coalesce and entry order do not change the dense value')
def _(rng):
    r, c, m, n = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 3), rng.randint(0, 8)
    idx = [[rng.randrange(r), rng.randrange(c)] for _ in range(n)]; val = vec(rng, n); B = [vec(rng, m) for _ in range(c)]; D = sps.to_dense(idx, val, (r, c))
    ok = sps.spmm(idx, val, (r, c), B) == [[sum((D[i][k] * B[k][j] for k in range(c)), F(0)) for j in range(m)] for i in range(r)]
    ci, cv = sps.coalesce(idx, val, (r, c)); return ok and sps.to_dense(ci, cv, (r, c)) == D

# ---- negative controls: relations known to be false; the generator must find failures, otherwise it is not testing anything
@rel('NEGATIVE CONTROL NORM_RMS: shift invariance (false)')
def _(rng):
    x = vec(rng, rng.randint(2, 6)); b = q(rng)
    if all(v == 0 for v in x) or b == 0: return 'declined'
    y1, y2 = sn.prop_shift_scale_invariance(sn.rms_norm, x, F(1), b, gamma=None, eps=0); return encl_all(zip(y1, y2))
@rel('NEGATIVE CONTROL CE: reduction=mean splits additively (false)')
def _(rng):
    N, K = rng.randint(3, 6), rng.randint(2, 5); L = [vec(rng, K) for _ in range(N)]; T = [rng.randrange(K) for _ in range(N)]; h = rng.randint(1, N - 1)
    w = ce.cross_entropy(L, T); a = ce.cross_entropy(L[:h], T[:h]); b = ce.cross_entropy(L[h:], T[h:])
    return enclosure_zero(w['loss'] - (a['loss'] + b['loss']))
@rel('NEGATIVE CONTROL KVCOMP: shift by one position is equivariant (false: phase dependence)')
def _(rng):
    S = W = rng.randint(2, 3); L = rng.randint(2 * W, 3 * W); H = [vec(rng, 1, lo=1, hi=9) for _ in range(L)]
    a = kcp.compress(H, W, S, [[F(1)]]); s = kcp.compress([[F(0)]] + H, W, S, [[F(1)]])
    if len(set(tuple(h) for h in H)) == 1: return 'declined'
    return any(r in s for r in a) and len(a) == len([r for r in a if r in s])
@rel('NEGATIVE CONTROL ADAM: zero raw gradient decays m with coupled wd (false)')
def _(rng):
    n = rng.randint(1, 3); th = vec(rng, n, lo=1, hi=4); st = so.AdamState(n)
    so.adam_step(th, [F(0)] * n, st, F(1, 1000), weight_decay=F(1, 10), decoupled=False); return st.m == [F(0)] * n

from rigorous import SpecInputError as SIE1
from rigorous_mp import SpecInputError as SIE2, SpecNotEstablished
DOMAIN_EXC = (SIE1, SIE2, SpecNotEstablished)

class Recorder(random.Random):                       # records every draw so a failing trial can be replayed exactly
    def __init__(self, seed): super().__init__(seed); self.draws = []
    def random(self): v = super().random(); self.draws.append(v); return v

def seed_of(name): return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'big')   # stable across processes

def main(trials=200):
    report = {}; errors = 0
    for name, f in RELS.items():
        rng = Recorder(seed_of(name)); h = fl = dc = er = 0; witnesses = []
        for t in range(trials):
            start = len(rng.draws); state = rng.getstate()
            try:
                r = f(rng)
            except DOMAIN_EXC as e:
                dc += 1; continue                                 # the spec declined the input: outside its declared domain
            except Exception as e:                                # anything else is an implementation error, never 'declined'
                er += 1; errors += 1
                if len(witnesses) < 3: witnesses.append(dict(trial=t, kind='error', exc=type(e).__name__, msg=str(e)[:200], rng_state=str(state)[:120]))
                continue
            if r == 'declined': dc += 1
            elif r: h += 1
            else:
                fl += 1
                if len(witnesses) < 3: witnesses.append(dict(trial=t, kind='fail', draws=rng.draws[start:], rng_state=repr(state)[:120]))
        report[name] = dict(holds=h, fails=fl, declined=dc, errors=er, witnesses=witnesses, seed=seed_of(name))
    neg = {k: v for k, v in report.items() if k.startswith('NEGATIVE')}; pos = {k: v for k, v in report.items() if not k.startswith('NEGATIVE')}
    verdict = dict(positive_relations_all_hold=all(v['fails'] == 0 and v['errors'] == 0 for v in pos.values()),
                   negative_controls_each_fail_at_least_once=all(v['fails'] >= 1 for v in neg.values()), implementation_errors=errors)
    json.dump(dict(trials_per_relation=trials, seed_scheme='sha256(name)[:8]', report=report, verdict=verdict,
                   totals=dict(relations=len(RELS), holds=sum(v['holds'] for v in report.values()), fails=sum(v['fails'] for v in report.values()),
                               declined=sum(v['declined'] for v in report.values()), errors=errors)), open('fuzz_log.json', 'w'), indent=1, ensure_ascii=False)
    for k, v in report.items(): print(f"{v['holds']:4d} holds {v['fails']:3d} fails {v['declined']:3d} declined {v['errors']:2d} errors | {k}")
    print('verdict:', verdict)
    ok = verdict['positive_relations_all_hold'] and verdict['negative_controls_each_fail_at_least_once'] and errors == 0
    sys.exit(0 if ok else 1)
if __name__ == '__main__': main()
