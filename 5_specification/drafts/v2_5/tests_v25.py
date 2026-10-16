"""Property checks for the v2.5 specs against independent computations (numpy IEEE casts, the v2 rounding oracle,
exact algebra). Each check prints its count; any failure raises."""
import os, sys, random, struct
from fractions import Fraction as F
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'specs')]
sys.path.append(os.path.join(HERE, '..', 'specification_v2'))
import spec_quant as SQ, spec_linear_attention as LA, spec_collectives as K, spec_paged_attention as PA, spec_moe_routing as MR
import oracle_rounding as ORC
import mpmath as mp
rng = random.Random(20261016); n_ok = {}
def ok(name): n_ok[name] = n_ok.get(name, 0) + 1

# 1. element rounding vs the v2 rounding oracle (FP8, RNE, with and without saturation)
for _ in range(3000):
    x = F(rng.randint(-10**6, 10**6), rng.choice([1, 3, 7, 64, 1000, 4096])) * F(2) ** rng.randint(-12, 3)
    for fmt, ofmt in (('e4m3', 'f8e4m3fn'), ('e5m2', 'f8e5m2')):
        for sat in (False, True):
            a = SQ.round_fmt(x, fmt, saturate=sat); b = ORC.round_to(x, ofmt, 'rne', saturate=sat)
            assert (str(a) == str(b)) if isinstance(a, str) or isinstance(b, str) else a == b, (x, fmt, sat, a, b); ok('FP8 rounding == v2 oracle')
# 2. fp16 / fp32 rounding vs numpy IEEE casts
for _ in range(3000):
    v = rng.uniform(-70000, 70000) * 10.0 ** rng.randint(-8, 0)
    a = SQ.round_fmt(F(v), 'fp16'); b = float(np.float16(v))
    assert (a == 'inf' and b == np.inf) or (a == '-inf' and b == -np.inf) or a == F(b), (v, a, b); ok('fp16 rounding == numpy')
    a = SQ.round_fmt(F(v), 'fp32'); b = struct.unpack('f', struct.pack('f', v))[0]
    assert a == F(b), (v, a, b); ok('fp32 rounding == IEEE cast')
# 3. chunked algorithm == recurrence (exact), several chunk sizes
R = lambda: F(rng.randint(-6, 6), rng.choice([1, 2, 3]))
for _ in range(60):
    T, dk, dv = rng.randint(1, 9), rng.randint(1, 3), rng.randint(1, 3)
    q = [[R() for _ in range(dk)] for _ in range(T)]; k = [[R() for _ in range(dk)] for _ in range(T)]; v = [[R() for _ in range(dv)] for _ in range(T)]
    S0 = [[R() for _ in range(dv)] for _ in range(dk)]
    for var, a in (('linear', None), ('decay', [F(rng.randint(1, 9), 10) for _ in range(T)]), ('gla', [[F(rng.randint(1, 9), 10) for _ in range(dk)] for _ in range(T)])):
        ref = LA.recurrent(q, k, v, var, a=a, S0=S0, scale=F(1))
        for C in (1, 2, 3, 8):
            assert LA.chunked(q, k, v, C, var, a=a, S0=S0) == ref; ok('chunked == recurrent')
# 4. exact adjoint (S0, q, v enter linearly: exact differences equal the gradient)
for _ in range(40):
    T, dk, dv = rng.randint(1, 5), rng.randint(1, 3), rng.randint(1, 3); var = rng.choice(['linear', 'decay', 'gla'])
    q = [[R() for _ in range(dk)] for _ in range(T)]; k = [[R() for _ in range(dk)] for _ in range(T)]; v = [[R() for _ in range(dv)] for _ in range(T)]
    a = None if var == 'linear' else ([F(rng.randint(1, 9), 10) for _ in range(T)] if var == 'decay' else [[F(rng.randint(1, 9), 10) for _ in range(dk)] for _ in range(T)])
    S0 = [[R() for _ in range(dv)] for _ in range(dk)]; dO = [[R() for _ in range(dv)] for _ in range(T)]; dST = [[R() for _ in range(dv)] for _ in range(dk)]
    def Lf(q_, v_, S0_):
        o, S = LA.recurrent(q_, k, v_, var, a=a, S0=S0_, scale=F(1))
        return sum(dO[t][j] * o[t][j] for t in range(T) for j in range(dv)) + sum(dST[i][j] * S[i][j] for i in range(dk) for j in range(dv))
    g = LA.backward(q, k, v, dO, dST, var, a=a, S0=S0); base = Lf(q, v, S0)
    i, j = rng.randrange(dk), rng.randrange(dv); S1 = [r[:] for r in S0]; S1[i][j] += 1
    assert Lf(q, v, S1) - base == g['dS0'][i][j]; ok('dS0 exact')
    t = rng.randrange(T); q1 = [r[:] for r in q]; q1[t][i] += 1; assert Lf(q1, v, S0) - base == g['dq'][t][i]; ok('dq exact')
    v1 = [r[:] for r in v]; v1[t][j] += 1; assert Lf(q, v1, S0) - base == g['dv'][t][j]; ok('dv exact')
# 5. split-KV merge identity on random splits
for _ in range(40):
    Lk, d = rng.randint(2, 7), rng.randint(1, 3)
    q = [[R() for _ in range(d)]]; Kr = [[R() for _ in range(d)] for _ in range(Lk)]; V = [[R() for _ in range(2)] for _ in range(Lk)]
    cut = rng.randint(1, Lk - 1)
    full = PA.attention_rows(q, Kr, V, causal=False, scale=F(1, 2))[0]
    p1 = PA.attention_rows(q, Kr[:cut], V[:cut], causal=False, scale=F(1, 2))[0]; p2 = PA.attention_rows(q, Kr[cut:], V[cut:], causal=False, scale=F(1, 2))[0]
    o, lse = K.merge_partials([(p1[1], 1, p1[0]), (p2[1], 1, p2[0])])
    assert all(abs(mp.mpf(x.mid) - mp.mpf(y.mid)) < mp.mpf(10) ** -45 for x, y in zip(o, full[0])); ok('split-KV merge == full attention')
# 6. packing round trips
for _ in range(500):
    vals = [rng.randrange(16) for _ in range(8)]
    for order in ('gptq', 'awq'): assert SQ.unpack_int4(SQ.pack_int4(vals, order), order) == vals; ok('pack/unpack round trip')
# 7. MX: dequantized error bounded by half a quantum of the element grid (in range), saturation only above max * X
for _ in range(500):
    blk = [F(rng.randint(-10**4, 10**4), rng.choice([1, 7, 64])) for _ in range(rng.randint(1, 32))]
    X, el = SQ.mx_quantize(blk, 'e4m3')
    if X is None: continue
    for v, e in zip(blk, el):
        if abs(v / X) <= 448:
            ex = max(SQ.floor_log2(abs(v / X)), -6) if v != 0 else -6
            assert abs(v - e * X) <= X * F(2) ** (ex - 3) / 2; ok('MX in-range error <= half quantum')
        else: assert abs(e) == 448; ok('MX saturation')
# 8. alignment totals and routing set sizes
for _ in range(200):
    E, B, N, k = rng.randint(2, 6), rng.choice([2, 4, 8]), rng.randint(1, 6), 2
    ids = [rng.sample(range(E), k) for _ in range(N)]
    r = MR.moe_align(ids, E, B); cnt = {e: sum(row.count(e) for row in ids) for e in range(E)}
    assert r['total'] == sum(-(-c // B) * B for c in cnt.values()) and len(r['expert_of_block']) * B == r['total']; ok('moe_align totals')
for name, c in n_ok.items(): print(f'{c:6d}  {name}')
print('all property checks passed')
