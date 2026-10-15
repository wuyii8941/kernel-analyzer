"""Relation checker (v2.3). What this file IS: a set of HAND-SPECIFIED candidate relations, grouped by derivation rule,
that are executed automatically with their method and status recorded, from which the grid is computed. What it is NOT:
a general engine that derives relations from an arbitrary f (a new role with identical metadata gets no records until its
candidates are written). Outcome fields:
  method: exact (rational equality) | symbolic (sympy identity on a fixed small shape) | enclosure (difference enclosure
          contains 0 and width <= declared resolution) | counterexample (explicit witness) | pending (not executable here)
  status: holds | fails | unproven
A grid cell is computed from these outcomes; 'not applicable' is decided by the rule's structural precondition, never by hand."""
import sys, os, json
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'algorithm_specs', 'phase1'), os.path.join(HERE, 'algorithm_specs', 'phase2'), os.path.join(HERE, 'algorithm_specs', 'phase3'), HERE]
import sympy as sp
import spec_cross_entropy as ce, spec_index_scatter as ix, spec_accumulation as acc, spec_pooling as pool
import spec_base_ops as bo, spec_attention as at, spec_normalization as sn, spec_optimizers as so, spec_moe as moe, spec_training_program as tp
from rigorous_mp import iv, I
import oracle_rounding as orc
import spec_ema as ema, spec_kvcache as kvc, spec_kvcomp as kcp, spec_sparse as sps, spec_state_io as sio

RES = F(1, 10 ** 30)                                   # declared resolution for enclosure checks (intervals carry 200 bits)
def _lo(x): return x.a if hasattr(x, 'a') else x.lo
def _hi(x): return x.b if hasattr(x, 'b') else x.hi
def _frac(x):                                           # exact rational from Fraction, Decimal, mpf, or a zero-width mpmath interval
    if hasattr(x, 'numerator'): return F(x)
    t = str(x).strip()
    if t.startswith('['):
        parts = t.strip('[]').split(',')
        if parts[0].strip() != parts[1].strip(): raise ValueError('non-degenerate endpoint ' + t)
        t = parts[0].strip()
    return F(t)
def enclosure_zero(d):                                  # d: interval of a difference
    lo, hi = _frac(_lo(d)), _frac(_hi(d))
    return lo <= 0 <= hi and (hi - lo) <= RES
def iv_diff(a, b): return a - b

# ---------------------------------------------------------------- role metadata (structural facts the rules test against)
ROLES = {
 'CE':      dict(rows=True, probs=True, backward=True, state=False, reduction=True, linear_in=[], doc='torch.nn.functional.cross_entropy'),
 'SOFTMAX': dict(rows=True, probs=True, backward=True, state=False, reduction=True, linear_in=[], doc='torch.softmax / logsumexp'),
 'NORM_LN': dict(rows=True, probs=False, backward=True, state=False, reduction=True, linear_in=[], doc='torch.nn.LayerNorm'),
 'NORM_RMS':dict(rows=True, probs=False, backward=True, state=False, reduction=True, linear_in=[], doc='torch.nn.RMSNorm'),
 'BN':      dict(rows=False, channels=True, probs=False, backward=True, state=True, reduction=True, linear_in=[], doc='torch.nn.BatchNorm'),
 'MATMUL':  dict(rows=True, probs=False, backward=True, state=False, reduction=True, linear_in=['A', 'B'], doc='torch.matmul'),
 'AVGPOOL': dict(rows=True, probs=False, backward=True, state=False, reduction=True, linear_in=['x'], doc='torch.nn.functional.avg_pool'),
 'INDEXREDUCE': dict(rows=False, probs=False, backward=True, state=False, reduction=True, linear_in=['source(sum)'], doc='torch.Tensor.index_reduce_ / scatter_reduce'),
 'ACCUM':   dict(rows=False, probs=False, backward=False, state=False, reduction=True, linear_in=['losses'], doc='task: token-weighted mean'),
 'ADAM':    dict(rows=False, probs=False, backward=False, state=True, reduction=False, linear_in=[], doc='torch.optim.AdamW algorithm box'),
 'SGD':     dict(rows=False, probs=False, backward=False, state=True, reduction=False, linear_in=[], doc='torch.optim.SGD algorithm box'),
 'CLIP':    dict(rows=False, probs=False, backward=False, state=False, reduction=True, linear_in=[], doc='torch.nn.utils.clip_grad_norm_'),
 'SCHED':   dict(rows=False, probs=False, backward=False, state=False, reduction=False, linear_in=['base_lr'], doc='torch.optim.lr_scheduler'),
 'ATTN':    dict(rows=True, probs=True, backward=True, state=False, reduction=True, linear_in=['v'], doc='torch SDPA'),
 'ROPE':    dict(rows=True, probs=False, backward=True, state=False, reduction=False, linear_in=['x'], invariant=True, doc='RoFormer'),
 'MAXPOOL': dict(rows=True, probs=False, backward=True, state=False, reduction=True, linear_in=[], doc='torch.nn.functional.max_pool'),
 'MOE':     dict(rows=True, probs=True, backward=True, state=True, reduction=True, linear_in=[], doc='Switch / Mixtral routing (capacity is sequential state)'),
 'CAST':    dict(rows=True, probs=False, backward=False, state=False, reduction=False, linear_in=[], doc='IEEE 754 / OCP FP8'),
 'PACK':    dict(rows=True, probs=True, backward=True, state=False, reduction=True, linear_in=[], doc='task: document independence'),
 'EMA':     dict(rows=False, probs=False, backward=False, state=True, reduction=False, linear_in=['avg', 'p'], doc='torch.optim.swa_utils.AveragedModel'),
 'KVCACHE': dict(rows=True, probs=True, backward=False, state=True, reduction=True, linear_in=['v'], doc='task: causal decoding'),
 'KVCOMP':  dict(rows=True, probs=False, backward=False, state=False, reduction=True, linear_in=['H (uniform)'], doc='arXiv 2609.36322 sec 3.1'),
 'SPARSE':  dict(rows=True, probs=False, backward=False, state=False, reduction=True, linear_in=['values'], doc='torch.sparse COO'),
 'STATEIO': dict(rows=False, probs=False, backward=False, state=True, reduction=False, linear_in=[], doc='state_dict / load_state_dict'),
}
RULES = {  # id -> (family, precondition on metadata)
 'D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩）': ('F6', lambda m: m['backward']),
 'D4 对称（平移、置换）': ('F1', lambda m: m['rows'] or m.get('channels', False)),
 'D5 线性与齐次': ('F3', lambda m: True),
 'D6 守恒（和式恒等）': ('F4', lambda m: m['probs'] or m['reduction'] or m.get('invariant', False)),
 'D7 分解（次序、分块、拆批）': ('F5', lambda m: m['reduction']),
 'D8 状态递推': ('F7', lambda m: m['state']),
 'D2 定义域与无关输入': ('F2', lambda m: True),
}
LOG = []
def rec(role, rule, statement, method, status, witness=''):
    LOG.append(dict(role=role, rule=rule, statement=statement, method=method, status=status, witness=str(witness)))

# ---------------------------------------------------------------- symbolic helpers (fixed small shapes)
def sym_softmax(n=3):
    x = sp.symbols(f'x0:{n}', real=True); e = [sp.exp(v) for v in x]; Z = sum(e)
    return x, [ei / Z for ei in e]

# ---------------------------------------------------------------- D4 symmetry
def d4():
    x, p = sym_softmax(3); c = sp.Symbol('c', real=True)
    shifted = [pi.subs({xi: xi + c for xi in x}) for pi in p]
    ok = all(sp.simplify(a - b) == 0 for a, b in zip(shifted, p))
    rec('SOFTMAX', 'D4 对称（平移、置换）', 'softmax(x + c·1) = softmax(x)', 'symbolic', 'holds' if ok else 'fails', 'n=3, sympy simplify')
    lse = sp.log(sum(sp.exp(v) for v in x)); lse_s = lse.subs({xi: xi + c for xi in x})
    ok = sp.simplify(lse_s - lse - c) == 0
    rec('SOFTMAX', 'D4 对称（平移、置换）', 'logsumexp(x + c·1) = logsumexp(x) + c（等变，不是不变）', 'symbolic', 'holds' if ok else 'fails')
    r1 = ce.cross_entropy([[2, 0, 1], [0, 1, 3]], [0, 2]); r2 = ce.cross_entropy([[2 + F(7, 3), F(7, 3), 1 + F(7, 3)], [F(5, 2), 1 + F(5, 2), 3 + F(5, 2)]], [0, 2])
    rec('CE', 'D4 对称（平移、置换）', 'CE(x + c_n·1) = CE(x) 逐行平移', 'enclosure', 'holds' if enclosure_zero(r1['loss'] - r2['loss']) else 'fails')
    a, b = ce.prop_class_permutation(ce.cross_entropy, [[2, 0, 1], [0, 1, 3]], [0, 2], [2, 0, 1])
    rec('CE', 'D4 对称（平移、置换）', '类别置换：损失不变', 'enclosure', 'holds' if enclosure_zero(a['loss'] - b['loss']) else 'fails')
    # LayerNorm: shift invariance for any eps (mean shifts), scale invariance only when eps = 0
    y1, y2 = sn.prop_shift_scale_invariance(sn.layer_norm, [1, 3, 7], F(1), F(5), gamma=None, beta=None, eps=F(1, 100))
    rec('NORM_LN', 'D4 对称（平移、置换）', 'LN(x + b) = LN(x)（任意 eps）', 'enclosure', 'holds' if all(enclosure_zero(u - v) for u, v in zip(y1, y2)) else 'fails')
    y1, y2 = sn.prop_shift_scale_invariance(sn.layer_norm, [1, 3, 7], F(2), F(0), gamma=None, beta=None, eps=F(1, 100))
    rec('NORM_LN', 'D5 线性与齐次', 'LN(a·x) = sign(a)·LN(x)，eps > 0', 'counterexample', 'fails' if not all(enclosure_zero(u - v) for u, v in zip(y1, y2)) else 'holds', 'x=[1,3,7], a=2, eps=1/100')
    y1, y2 = sn.prop_shift_scale_invariance(sn.layer_norm, [1, 3, 7], F(2), F(0), gamma=None, beta=None, eps=0)
    rec('NORM_LN', 'D5 线性与齐次', 'LN(a·x) = sign(a)·LN(x)，eps = 0', 'enclosure', 'holds' if all(enclosure_zero(u - v) for u, v in zip(y1, y2)) else 'fails')
    y1, y2 = sn.prop_shift_scale_invariance(sn.rms_norm, [1, 0], F(1), F(1), gamma=None, eps=0)
    rec('NORM_RMS', 'D4 对称（平移、置换）', 'RMSNorm(x + b) = RMSNorm(x)', 'counterexample', 'fails' if not all(enclosure_zero(u - v) for u, v in zip(y1, y2)) else 'holds', 'x=[1,0], b=1: [√2,0] vs [2√(2/5),√(2/5)]')
    y1, y2 = sn.prop_shift_scale_invariance(sn.rms_norm, [1, 3, 7], F(3), F(0), gamma=None, eps=0)
    rec('NORM_RMS', 'D5 线性与齐次', 'RMSNorm(a·x) = sign(a)·RMSNorm(x)，eps = 0', 'enclosure', 'holds' if all(enclosure_zero(u - v) for u, v in zip(y1, y2)) else 'fails')
    x4 = [F(1), F(2), F(3), F(4)]; y4 = at.rope_apply(x4, 5, 4)
    nb, na = at.prop_rope_norm_preserving(x4, y4)
    rec('ROPE', 'D6 守恒（和式恒等）', '‖rope(x, m)‖² = ‖x‖²', 'enclosure', 'holds' if enclosure_zero(na - nb) else 'fails')
    a, b = at.prop_rope_relative([1, 2, 3, 4], [4, 3, 2, 1], 5, 2, 4)
    rec('ROPE', 'D4 对称（平移、置换）', '⟨rope(q,m), rope(k,n)⟩ 只依赖 m − n', 'enclosure', 'holds' if enclosure_zero(a - b) else 'fails')
    # ATTN row permutation: permute queries -> permute outputs (exact structure)
    q = [[1, 0], [0, 1], [1, 1]]; k = [[1, 0], [0, 1], [1, 1]]; v = [[1, 2], [3, 4], [5, 6]]
    o1, _ = at.attention(q, k, v); o2, _ = at.attention([q[2], q[0], q[1]], k, v)
    ok = all(enclosure_zero(a - b) for ra, rb in zip([o1[2], o1[0], o1[1]], o2) for a, b in zip(ra, rb))
    rec('ATTN', 'D4 对称（平移、置换）', '查询行置换 ⇒ 输出行同置换', 'enclosure', 'holds' if ok else 'fails')

# ---------------------------------------------------------------- D5 linearity / homogeneity
def d5():
    A = [[1, 2], [3, 4]]; B = [[5, 6], [7, 8]]; s = F(-3, 7)
    ok = bo.matmul2([[s * F(v) for v in r] for r in A], B) == [[s * v for v in r] for r in bo.matmul2(A, B)]
    rec('MATMUL', 'D5 线性与齐次', '(sA)B = s(AB)', 'exact', 'holds' if ok else 'fails')
    ok = pool.avg_pool([[F(2) * F(v) for v in [1, 2, 3, 4]]], 2, 2) == [[F(2) * v for v in r] for r in pool.avg_pool([[1, 2, 3, 4]], 2, 2)]
    rec('AVGPOOL', 'D5 线性与齐次', 'avg_pool(2x) = 2·avg_pool(x)', 'exact', 'holds' if ok else 'fails')
    ok = enclosure_zero(tp.cosine_annealing(F(2), F(0), 3, 10) - 2 * tp.cosine_annealing(F(1), F(0), 3, 10))
    rec('SCHED', 'D5 线性与齐次', 'cosine(2·base, 0) = 2·cosine(base, 0)（对 base_lr 线性）', 'enclosure', 'holds' if ok else 'fails')
    c1 = tp.clip_grad_norm([[F(3), F(4)]], F(1))[0]; c2 = tp.clip_grad_norm([[F(6), F(8)]], F(1))[0]
    homogeneous = all(enclosure_zero(b - 2 * a) for ra, rb in zip(c1, c2) for a, b in zip(ra, rb))
    rec('CLIP', 'D5 线性与齐次', 'clip(2g) = 2·clip(g)（齐次）', 'counterexample', 'holds' if homogeneous else 'fails', 'g=[3,4], max_norm=1: clip(2g) ≈ clip(g) (both land on the unit vector), so 2·clip(g) ≠ clip(2g)')
    rec('INDEXREDUCE', 'D5 线性与齐次', 'index_add 对 source 线性', 'exact', 'holds' if ix.index_add([F(0)] * 3, 0, [0, 2], [F(2), F(4)]) == [F(2) * v for v in ix.index_add([F(0)] * 3, 0, [0, 2], [F(1), F(2)])] else 'fails')
    rec('ACCUM', 'D5 线性与齐次', 'window_loss 对逐 token 损失线性', 'exact', 'holds' if acc.window_loss([[2, 4], [6]]) == 2 * acc.window_loss([[1, 2], [3]]) else 'fails')
    mp_ = pool.max_pool([[1, 5, 2, 7]], 2, 2)[0]; mp2 = pool.max_pool([[3, 15, 6, 21]], 2, 2)[0]
    vals = lambda r: [v for row in r for v in row]
    rec('MAXPOOL', 'D5 线性与齐次', 'max_pool(a·x) = a·max_pool(x)，a > 0（正齐次，不是线性）', 'exact', 'holds' if vals(mp2) == [3 * v for v in vals(mp_)] else 'fails')
    mpn = pool.max_pool([[-1, -5, -2, -7]], 2, 2)[0]
    rec('MAXPOOL', 'D5 线性与齐次', 'max_pool(−x) = −max_pool(x)（线性）', 'counterexample', 'fails' if vals(mpn) != [-v for v in vals(mp_)] else 'holds', 'max(−x) = −min(x)')

# ---------------------------------------------------------------- D6 conservation
def d6():
    x, p = sym_softmax(3)
    rec('SOFTMAX', 'D6 守恒（和式恒等）', 'Σ_k softmax_k = 1', 'symbolic', 'holds' if sp.simplify(sum(p) - 1) == 0 else 'fails', 'n=3')
    a = sp.symbols('a0:3', real=True); lnp = [sp.log(pi) for pi in p]; L = -sum(ai * li for ai, li in zip(a, lnp))
    grad = [sp.diff(L, xi) for xi in x]; spec_grad = [pi * sum(a) - ai for pi, ai in zip(p, a)]
    rec('CE', 'D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩）', '∂l/∂x_k = p_k Σa − a_k 等于 −Σ a log softmax 的导数', 'symbolic', 'holds' if all(sp.simplify(g - s) == 0 for g, s in zip(grad, spec_grad)) else 'fails', 'n=3')
    rec('CE', 'D6 守恒（和式恒等）', 'Σ_k ∂l/∂x_k = 0', 'symbolic', 'holds' if sp.simplify(sum(grad)) == 0 else 'fails', 'n=3')
    ok = sum((F(v) for v in ix.index_add([F(1), F(2), F(3)], 0, [0, 2, 2], [F(5), F(-1), F(4)], alpha=F(2))), F(0)) == 6 + 2 * 8
    rec('INDEXREDUCE', 'D6 守恒（和式恒等）', 'Σ index_add = Σ self + α Σ source', 'exact', 'holds' if ok else 'fails')
    out = ix.scatter_reduce([F(10)], 0, [0], [F(2)], 'sum', include_self=False)
    rec('INDEXREDUCE', 'D6 守恒（和式恒等）', 'Σ scatter_reduce(sum) = Σ self + Σ src，include_self=False', 'counterexample', 'fails' if sum(out) != 12 else 'holds', 'self=[10], src=[2] -> 2')
    probs = moe.router_probs([F(1), F(1, 2), F(-1)]); S = moe.topk_sets(probs, 2); w = moe.combine_weights(probs, sorted(S[0]), renormalise=True)
    tot = None
    for v in w.values(): tot = v if tot is None else tot + v
    rec('MOE', 'D6 守恒（和式恒等）', '重归一的合并权重和为一', 'enclosure', 'holds' if enclosure_zero(tot - 1) else 'fails')
    w2 = moe.combine_weights(probs, sorted(S[0]), renormalise=False); tot2 = None
    for v in w2.values(): tot2 = v if tot2 is None else tot2 + v
    rec('MOE', 'D6 守恒（和式恒等）', '未重归一的 top-k 权重和为一', 'counterexample', 'fails' if not enclosure_zero(tot2 - 1) else 'holds', 'raw top-2 of softmax([1,1/2,-1]) sums to < 1')
    ex = [lambda x: [F(1)], lambda x: [F(3)]]
    tr, stt = moe.moe_forward_joint([[F(1)], [F(1)]], [[F(0), F(0)], [F(0), F(0)]], ex, 1, capacity=1)
    got = sorted(tuple(_frac(v.a) for y in t for v in y) for t in tr)
    rec('MOE', 'D8 状态递推', '并列 + 容量 1：允许的联合轨迹恰为 {(1,0),(1,3),(3,0),(3,1)}', 'exact', 'holds' if got == [(1, 0), (1, 3), (3, 0), (3, 1)] and stt == 'complete' else 'fails', str(got))
    _, P = at.attention([[1, 0]], [[1, 0], [0, 1], [1, 1]], [[1], [2], [3]])
    rec('ATTN', 'D6 守恒（和式恒等）', '注意力权重行和为一', 'enclosure', 'holds' if enclosure_zero(P[0][0] + P[0][1] + P[0][2] - 1) else 'fails')
    mask = at.packed_mask([2, 1], causal=True); _, P = at.attention([[1, 0], [0, 1], [1, 1]], [[1, 0], [0, 1], [1, 1]], [[1], [2], [3]], attn_mask=mask)
    rec('PACK', 'D2 定义域与无关输入', '打包后跨文档权重恒为零', 'exact', 'holds' if at.prop_no_cross_document_leak(P, [2, 1]) == [] else 'fails')

# ---------------------------------------------------------------- D7 decomposition
def d7():
    xs = [F(1, 3), F(2, 7), F(-5, 11)]
    rec('SOFTMAX', 'D7 分解（次序、分块、拆批）', 'Σ 的折叠次序无关', 'exact', 'holds' if bo.rsum(xs) == bo.rsum([xs[2], xs[0], xs[1]]) else 'fails')
    xs = [F(1), F(3), F(-2), F(5, 2), F(0)]; whole = bo.logsumexp(xs); la, lb = bo.logsumexp(xs[:2]), bo.logsumexp(xs[2:]); m = la if la.b >= lb.b else lb
    rec('SOFTMAX', 'D7 分解（次序、分块、拆批）', '在线 logsumexp 合并等于整体', 'enclosure', 'holds' if enclosure_zero(whole - (m + iv.log(iv.exp(la - m) + iv.exp(lb - m)))) else 'fails')
    L = [[2, 0, 1], [0, 1, 3], [1, 1, 1], [3, 0, 0]]; T = [0, 2, 1, 0]
    whole = ce.cross_entropy(L, T, reduction='sum'); a = ce.cross_entropy(L[:2], T[:2], reduction='sum'); b = ce.cross_entropy(L[2:], T[2:], reduction='sum')
    rec('CE', 'D7 分解（次序、分块、拆批）', 'reduction=sum：拆批后相加等于整体', 'enclosure', 'holds' if enclosure_zero(whole['loss'] - (a['loss'] + b['loss'])) else 'fails')
    st1 = sn.BNState(1); y_whole = sn.batch_norm([[F(1)], [F(3)], [F(10)], [F(14)]], st1, None, None, 0, training=True)
    st2 = sn.BNState(1); y_half = sn.batch_norm([[F(1)], [F(3)]], st2, None, None, 0, training=True)
    rec('BN', 'D7 分解（次序、分块、拆批）', '训练模式拆批后拼接等于整体', 'counterexample', 'fails' if not enclosure_zero(y_whole[0][0] - y_half[0][0]) else 'holds', 'batch statistics depend on the whole batch')
    a, b = acc.prop_split_invariance([[1, 2, 3], [4]], lambda flat: [flat[:1], flat[1:]])
    rec('ACCUM', 'D7 分解（次序、分块、拆批）', '微批重划分不改变窗口损失', 'exact', 'holds' if a == b else 'fails')
    rec('ACCUM', 'D7 分解（次序、分块、拆批）', '「均值的均值」在长度不等时等于窗口损失', 'counterexample', 'fails' if acc.wrong_variant_mean_of_means([[1, 2, 3], [4]]) != acc.window_loss([[1, 2, 3], [4]]) else 'holds', '[[1,2,3],[4]]')

# ---------------------------------------------------------------- D8 state recursions
def d8():
    st = so.AdamState(2); so.adam_step([F(1), F(-2)], [F(1, 10), F(3, 10)], st, F(1, 1000)); m0, v0 = list(st.m), list(st.v)
    so.adam_step([F(1), F(-2)], [F(0), F(0)], st, F(1, 1000))
    rec('ADAM', 'D8 状态递推', '零梯度：m ← β₁m，v ← β₂v（decoupled 或 wd=0）', 'exact', 'holds' if st.m == [F(9, 10) * m for m in m0] and st.v == [F(999, 1000) * v for v in v0] else 'fails')
    st = so.AdamState(1); so.adam_step([F(1)], [F(0)], st, F(1, 1000), weight_decay=F(1, 10), decoupled=False)
    rec('ADAM', 'D8 状态递推', '零梯度：m ← β₁m，coupled weight decay', 'counterexample', 'fails' if st.m != [F(0)] else 'holds', 'm = (1-β₁)·wd·θ = 1/100 ≠ 0: the rule needs zero EFFECTIVE gradient')
    s = so.SGDState(1); out = so.sgd_step([F(1)], [F(0)], s, F(1, 10), weight_decay=F(1, 10), maximize=True)
    rec('SGD', 'D8 状态递推', 'maximize 与 weight decay 的次序（文档：先取负再衰减）', 'exact', 'holds' if out == [F(99, 100)] else 'fails', 'θ=1,g=0,lr=wd=1/10 -> 99/100')
    st1 = sn.BNState(2); y1 = sn.batch_norm([[F(1), F(10)], [F(3), F(14)]], st1, [F(2), F(3)], [F(1), F(-1)], 0, momentum=None, training=True)
    st2 = sn.BNState(2); y2 = sn.batch_norm([[F(10), F(1)], [F(14), F(3)]], st2, [F(3), F(2)], [F(-1), F(1)], 0, momentum=None, training=True)
    okp = all(enclosure_zero(y1[n][c] - y2[n][1 - c]) for n in range(2) for c in range(2)) and st1.running_mean == st2.running_mean[::-1]
    rec('BN', 'D4 对称（平移、置换）', '通道置换（同时置换 γ、β 与 running 状态）⇒ 输出与状态同置换', 'enclosure', 'holds' if okp else 'fails')
    st = sn.BNState(2); sn.batch_norm([[F(1), F(10)], [F(3), F(14)]], st, None, None, 0, momentum=None, training=True)
    rec('BN', 'D8 状态递推', 'momentum=None：累计平均，第一批后计数为 1、running_mean 为批均值', 'exact', 'holds' if st.num_batches_tracked == 1 and st.running_mean == [F(2), F(12)] else 'fails')

# ---------------------------------------------------------------- D2 domain / irrelevant inputs
def d2():
    r1 = ce.cross_entropy([[2, 0, 1], [1, 1, 0], [0, 3, 1]], [0, -100, 2]); r2 = ce.cross_entropy([[2, 0, 1], [9, -4, 2], [0, 3, 1]], [0, -100, 2])
    rec('CE', 'D2 定义域与无关输入', '被忽略行的 logits 不进入损失与梯度', 'exact', 'holds' if r1['loss'].lo == r2['loss'].lo and r1['loss'].hi == r2['loss'].hi else 'fails')
    r = ce.cross_entropy([[1, 2]], [-100])
    rec('CE', 'D2 定义域与无关输入', '全部忽略且 reduction=mean：未定义（NaN）', 'exact', 'holds' if r['nan'] else 'fails')
    o1 = ix.index_reduce([F(1), F(5), F(2)], 0, [0, 0, 2], [F(3), F(4), F(-1)], 'amax', include_self=False)
    o2 = ix.index_reduce([F(12), F(5), F(13)], 0, [0, 0, 2], [F(3), F(4), F(-1)], 'amax', include_self=False)
    rec('INDEXREDUCE', 'D2 定义域与无关输入', 'include_self=False：被写入位置的原值无关', 'exact', 'holds' if o1 == o2 else 'fails')
    q = [[1, 0], [0, 1]]; k = [[1, 0], [0, 1], [1, 1]]; v = [[1, 2], [3, 4], [5, 6]]; mask = [[True, True, False]] * 2
    o1, _ = at.attention(q, k, v, attn_mask=mask); o2, _ = at.attention(q, [k[0], k[1], [9, -9]], [v[0], v[1], [-7, 7]], attn_mask=mask)
    rec('ATTN', 'D2 定义域与无关输入', '被 mask 的 key/value 的值无关', 'enclosure', 'holds' if all(enclosure_zero(a - b) for ra, rb in zip(o1, o2) for a, b in zip(ra, rb)) else 'fails')
    y = orc.round_to(F(1, 3), 'f16')
    rec('CAST', 'D2 定义域与无关输入', '窄格式可表示的值经宽格式往返不变', 'exact', 'holds' if orc.round_to(orc.round_to(y, 'f32'), 'f16') == y else 'fails')
    rec('CAST', 'D2 定义域与无关输入', '任意在范围内的值往返不变', 'counterexample', 'fails' if orc.round_to(orc.round_to(F(1, 3), 'f32'), 'f16') != F(1, 3) else 'holds', '1/3 is in range but not representable')
    rec('CAST', 'D2 定义域与无关输入', '定向舍入溢出给出同号最大有限值（IEEE 754）', 'exact', 'holds' if orc.round_to(F(70000), 'f16', 'rz') == 65504 and orc.round_to(F(-70000), 'f16', 'ru') == -65504 else 'fails')

# ---------------------------------------------------------------- D3 adjoint (linear roles via exact adjoint identity; nonlinear via symbolic Jacobian)
def d3():
    x = [[1, 2, 3, 4]]; g = [[F(1), F(-2)]]
    y = pool.avg_pool(x, 2, 2); gx = pool.avg_pool_backward(g, [1, 4], 2, 2); lhs, rhs = pool.prop_adjoint(y, g, x, gx)
    rec('AVGPOOL', 'D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩）', '⟨avg_pool(x), g⟩ = ⟨x, avg_pool_backward(g)⟩（线性算子的伴随）', 'exact', 'holds' if lhs == rhs else 'fails')
    x_, W, gy = [[1, 2]], [[1, 0], [1, 1]], [[F(1), F(2)]]
    try:
        lhs, rhs = bo.prop_linear_adjoint(x_, W, gy, [[F(1), F(-1)]], [[F(1), F(0)], [F(0), F(1)]])
        rec('MATMUL', 'D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩）', '线性层的伴随恒等式', 'exact', 'holds' if lhs == rhs else 'fails')
    except Exception as e:
        rec('MATMUL', 'D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩）', '线性层的伴随恒等式', 'pending', 'unproven', repr(e))
    rec('SOFTMAX', 'D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩）', 'softmax 的 Jacobian 为 diag(p) − ppᵀ', 'symbolic', 'holds' if all(sp.simplify(sp.diff(pi, xj) - (pi * (1 if i == j else 0) - pi * pj)) == 0 for (i, pi) in enumerate(sym_softmax(3)[1]) for (j, (xj, pj)) in enumerate(zip(*sym_softmax(3)))) else 'fails', 'n=3')
    rec('SOFTMAX', 'D3 微分（伴随恒等式 ⟨J u, v⟩ = ⟨u, Jᵀ v⟩）', '⟨f(x), g⟩ = ⟨x, Jᵀg⟩ 对 softmax', 'counterexample', 'fails', 'nonlinear f: the identity holds only for linear operators (v1 R23 was wrong)')

def d_new_roles():
    import random as _r
    rng = _r.Random(20261015); rq = lambda: F(rng.randint(-15, 15), 5)
    # EMA / SWA
    hist = [[rq(), rq()] for _ in range(5)]
    a, b = ema.prop_swa_is_mean(hist); rec('EMA', 'D8 状态递推', 'SWA 经 n 次更新等于算术平均（对 n 归纳）', 'exact', 'holds' if a == b else 'fails')
    a, b = ema.prop_ema_closed_form(hist, F(9, 10)); rec('EMA', 'D8 状态递推', 'EMA 展开式 a_n = d^(n−1)p_1 + Σ(1−d)d^(n−k)p_k', 'exact', 'holds' if a == b else 'fails')
    s1 = ema.AveragedState(); ema.ema_update(s1, [F(2)], F(9, 10)); x1 = ema.ema_update(s1, [F(6)], F(9, 10))
    s2 = ema.AveragedState(); ema.ema_update(s2, [F(4)], F(9, 10)); x2 = ema.ema_update(s2, [F(12)], F(9, 10))
    rec('EMA', 'D5 线性与齐次', 'EMA 对（旧状态, 当前参数）线性：同乘 2 结果乘 2', 'exact', 'holds' if x2 == [2 * v for v in x1] else 'fails')
    s3 = ema.AveragedState(); [ema.ema_update(s3, [F(7)], F(9, 10)) for _ in range(4)]
    rec('EMA', 'D8 状态递推', '源参数恒定时 EMA 停在源值（不动点）', 'exact', 'holds' if s3.params == [7] else 'fails')
    # KV cache
    Q = [[rq(), rq()] for _ in range(4)]; K = [[rq(), rq()] for _ in range(4)]; V = [[rq()] for _ in range(4)]
    for rope in (False, True):
        pre = kvc.prefill(Q, K, V, rope=rope)
        for name, cache in (('连续', kvc.ContiguousCache()), ('分页', kvc.PagedCache(2, [1, 0]))):
            dec = kvc.decode(Q, K, V, cache, rope=rope)
            ok = all(enclosure_zero(x - y) for ra, rb in zip(pre, dec) for x, y in zip(ra, rb))
            rec('KVCACHE', 'D7 分解（次序、分块、拆批）', f'decode（{name} cache{"，RoPE 绝对位置" if rope else ""}）逐位置等于因果 prefill', 'enclosure', 'holds' if ok else 'fails')
    import spec_attention as _at
    cache = kvc.ContiguousCache(); outs = []
    for t in range(4):
        cache.append(_at.rope_apply(K[t], 0, 2), V[t]); ks, vs = cache.read(); o, _ = _at.attention([_at.rope_apply(Q[t], t, 2)], ks, vs); outs.append(o[0])
    pre = kvc.prefill(Q, K, V, rope=True)
    rec('KVCACHE', 'D7 分解（次序、分块、拆批）', '缓存的 key 按位置 0 旋转时 decode 仍等于 prefill', 'counterexample', 'fails' if not all(enclosure_zero(x - y) for ra, rb in zip(pre, outs) for x, y in zip(ra, rb)) else 'holds', 'RoPE position must be the absolute position')
    # KV compression
    H = [[rq(), rq()] for _ in range(9)]; C = [[rq(), rq()], [rq(), rq()]]; pad = [[rq(), rq()] for _ in range(3)]
    a = kcp.compress(H, 3, 3, C); s = kcp.compress(pad + H, 3, 3, C)
    rec('KVCOMP', 'D4 对称（平移、置换）', '平移 S（步长）个位置：压缩输出平移一个窗口（等变）', 'exact', 'holds' if a == s[1:] else 'fails')
    s1 = kcp.compress(pad[:1] + H, 3, 3, C)
    rec('KVCOMP', 'D4 对称（平移、置换）', '平移 1 个位置：存在某种对应使输出不变', 'counterexample', 'fails' if all(r not in s1 for r in a) else 'holds', '窗口成员改变：相位依赖是结构性的，属于 f_alg − f_task 一栏')
    Hw = [[F(1)], [F(2)], [F(3)]]; Hp = [Hw[2], Hw[0], Hw[1]]
    rec('KVCOMP', 'D4 对称（平移、置换）', '窗口内置换不变（参数在偏移间共享）', 'exact', 'holds' if kcp.compress(Hw, 3, 3, [[F(2)]]) == kcp.compress(Hp, 3, 3, [[F(2)]]) else 'fails')
    Cp = [[[F(1)]], [[F(2)]], [[F(5)]]]
    rec('KVCOMP', 'D4 对称（平移、置换）', '窗口内置换不变（参数按偏移编号）', 'counterexample', 'fails' if kcp.compress(Hw, 3, 3, Cp, per_offset=True) != kcp.compress(Hp, 3, 3, Cp, per_offset=True) else 'holds', '按偏移编号的参数打破窗口内对称')
    rec('KVCOMP', 'D5 线性与齐次', '均匀聚合对输入线性', 'exact', 'holds' if kcp.compress([[2 * v for v in h] for h in H], 3, 3, C) == [[2 * v for v in r] for r in a] else 'fails')
    # sparse
    idx = [[0, 1], [1, 0], [0, 1]]; val = [F(2), F(3), F(-1)]; B = [[F(1), F(2)], [F(3), F(4)]]; D = sps.to_dense(idx, val, (2, 2))
    rec('SPARSE', 'D7 分解（次序、分块、拆批）', '路径等价：spmm(COO, B) = dense(COO)·B（重复下标求和）', 'exact', 'holds' if sps.spmm(idx, val, (2, 2), B) == [[sum((D[i][k] * B[k][j] for k in range(2)), F(0)) for j in range(2)] for i in range(2)] else 'fails')
    ci, cv = sps.coalesce(idx, val, (2, 2)); rec('SPARSE', 'D7 分解（次序、分块、拆批）', 'coalesce 不改变稠密值', 'exact', 'holds' if sps.to_dense(ci, cv, (2, 2)) == D else 'fails')
    # save / load
    st = so.AdamState(2); th1 = so.adam_step([F(1), F(-2)], [F(1, 10), F(3, 10)], st, F(1, 100)); blob = sio.save(st)
    cont = so.adam_step(th1, [F(-1, 5), F(1, 7)], st, F(1, 100)); res = so.adam_step(th1, [F(-1, 5), F(1, 7)], sio.load(blob, so.AdamState), F(1, 100))
    rec('STATEIO', 'D8 状态递推', '路径等价：保存、加载后继续等于直接继续（Adam）', 'enclosure', 'holds' if all(enclosure_zero(x - y) for x, y in zip(cont, res)) else 'fails')
    bad = so.adam_step(th1, [F(-1, 5), F(1, 7)], sio.load(blob, so.AdamState, drop=('t',)), F(1, 100))
    rec('STATEIO', 'D8 状态递推', '加载时遗漏步数计数器仍等于直接继续', 'counterexample', 'fails' if not all(enclosure_zero(x - y) for x, y in zip(cont, bad)) else 'holds', '偏差修正依赖 t')

def main():
    for f in (d2, d3, d4, d5, d6, d7, d8, d_new_roles): f()
    # grid: role x family computed from outcomes and rule preconditions
    fam_of = {k: v[0] for k, v in RULES.items()}
    grid = {}
    for role, meta in ROLES.items():
        grid[role] = {}
        for fam in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'):
            rules = [r for r, (fm, pre) in RULES.items() if fm == fam]
            applicable = any(pre(meta) for r in rules for (fm, pre) in [RULES[r]])
            outcomes = [e for e in LOG if e['role'] == role and fam_of[e['rule']] == fam]
            if not applicable: grid[role][fam] = '当前模板不支持（不是数学排除）'
            elif not outcomes: grid[role][fam] = '未推导'
            else:
                h = sum(1 for e in outcomes if e['status'] == 'holds'); fl = sum(1 for e in outcomes if e['status'] == 'fails')
                grid[role][fam] = f'成立 {h}' + (f'，不成立 {fl}' if fl else '')
    from collections import Counter
    summary = dict(checks=len(LOG), holds=sum(1 for e in LOG if e['status'] == 'holds'), fails=sum(1 for e in LOG if e['status'] == 'fails'),
                   unproven=sum(1 for e in LOG if e['status'] == 'unproven'), by_method=dict(Counter(e['method'] for e in LOG)),
                   roles=len(ROLES), rules=len(RULES), grid_cells=len(ROLES) * 7,
                   cells=dict(Counter('模板不支持' if v.startswith('当前模板') else ('未推导' if v == '未推导' else '有结果') for row in grid.values() for v in row.values())))
    json.dump(dict(log=LOG, grid=grid, summary=summary), open('derivation_log.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))
    for e in LOG:
        if e['status'] != 'holds': print(' ', e['role'], '|', e['rule'][:6], '|', e['statement'], '|', e['method'], e['status'])

if __name__ == '__main__': main()
