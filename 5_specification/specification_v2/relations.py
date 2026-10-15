"""Executable relation instances (v2). Every instance names its derivation rule, the exact proposition it tests, the
comparison method, and (where cheap) a mutation that must FAIL. Enclosure checks require the difference enclosure to
contain zero with width below the declared resolution; they are evidence, not proof (proof lives in derive.py's symbolic
records). Interval helpers are shared with derive.py."""
import sys, os
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, 'algorithm_specs', 'phase1'), os.path.join(HERE, 'algorithm_specs', 'phase2'), HERE]
from derive import enclosure_zero
import spec_cross_entropy as ce, spec_index_scatter as ix, spec_accumulation as acc, spec_pooling as pool
import spec_base_ops as bo, spec_attention as at, spec_normalization as sn, spec_optimizers as so, spec_moe as moe
from rigorous_mp import iv

def isum(vals):
    s = None
    for v in vals: s = v if s is None else s + v
    return s
# ---- D6 conservation, with the T09 defect fixed: the proposition is 'the sum encloses 1 tightly', and doubling must fail
def softmax_row_sum_is_one(xs):           return enclosure_zero(isum(bo.softmax(xs)) - 1)
def softmax_row_sum_mutation(xs):         return not enclosure_zero(isum([2 * p for p in bo.softmax(xs)]) - 1)
def moe_renormalised_weights_sum_one(logits_row, k):
    p = moe.router_probs(logits_row); S = sorted(moe.topk_sets(p, k)[0]); return enclosure_zero(isum(moe.combine_weights(p, S, True).values()) - 1)
def moe_raw_weights_mutation(logits_row, k):
    p = moe.router_probs(logits_row); S = sorted(moe.topk_sets(p, k)[0]); return not enclosure_zero(isum(moe.combine_weights(p, S, False).values()) - 1)
# ---- D2 irrelevance
def ignore_index_irrelevant(logits, target, ignore_index=-100):
    r1 = ce.cross_entropy(logits, target, ignore_index=ignore_index)
    changed = [[F(v) + F(7, 3) if target[n] == ignore_index else F(v) for v in row] for n, row in enumerate(logits)]
    r2 = ce.cross_entropy(changed, target, ignore_index=ignore_index)
    return r1['loss'].lo == r2['loss'].lo and r1['loss'].hi == r2['loss'].hi
def include_self_false_irrelevant(self_t, index, source, op='amax'):
    alt = [F(v) + 11 if i in set(index) else F(v) for i, v in enumerate(self_t)]
    return ix.index_reduce(self_t, 0, index, source, op, include_self=False) == ix.index_reduce(alt, 0, index, source, op, include_self=False)
def include_self_true_mutation(self_t, index, source, op='amax'):
    alt = [F(v) + 11 if i in set(index) else F(v) for i, v in enumerate(self_t)]
    return ix.index_reduce(self_t, 0, index, source, op, include_self=True) != ix.index_reduce(alt, 0, index, source, op, include_self=True)
# ---- D5 / D7
def token_weighted(per_token):            return acc.window_loss(per_token) == sum((F(v) for b in per_token for v in b), F(0)) / sum(len(b) for b in per_token)
def mean_of_means_mutation(per_token):    return acc.wrong_variant_mean_of_means(per_token) != acc.window_loss(per_token)
def online_logsumexp(xs, split):
    whole = bo.logsumexp(xs); la, lb = bo.logsumexp(xs[:split]), bo.logsumexp(xs[split:]); m = la if la.b >= lb.b else lb
    return enclosure_zero(whole - (m + iv.log(iv.exp(la - m) + iv.exp(lb - m))))
def ce_sum_split(logits, target):
    h = len(logits) // 2; whole = ce.cross_entropy(logits, target, reduction='sum')
    a = ce.cross_entropy(logits[:h], target[:h], reduction='sum'); b = ce.cross_entropy(logits[h:], target[h:], reduction='sum')
    return enclosure_zero(whole['loss'] - (a['loss'] + b['loss']))
# ---- D3 adjoint for a linear operator; D8 state
def avgpool_adjoint(x, g, **kw):
    y = pool.avg_pool(x, **kw); gx = pool.avg_pool_backward(g, [len(x), len(x[0])], **kw); lhs, rhs = pool.prop_adjoint(y, g, x, gx); return lhs == rhs
def adam_zero_effective_grad_decay(theta, grad, lr=F(1, 1000), b1=F(9, 10), b2=F(999, 1000)):
    st = so.AdamState(len(theta)); so.adam_step(theta, grad, st, lr, beta1=b1, beta2=b2); m0, v0 = list(st.m), list(st.v)
    so.adam_step(theta, [F(0)] * len(theta), st, lr, beta1=b1, beta2=b2); return st.m == [b1 * m for m in m0] and st.v == [b2 * v for v in v0]
