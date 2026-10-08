"""Hand-computed checks for the phase-2 draft specs. Run: python3 test_specs_phase2.py (exit 1 on failure)."""
import sys
from fractions import Fraction as F
from decimal import Decimal, getcontext
getcontext().prec = 90

import spec_optimizers as so
import spec_normalization as sn
import spec_embedding as se
from rigorous import Interval
from rigorous import SpecNotEstablished as _SNE1, SpecInputError as _SIE1
from rigorous_mp import SpecNotEstablished as _SNE2, SpecInputError as _SIE2
SpecNotEstablished = (_SNE1, _SNE2); SpecInputError = (_SIE1, _SIE2)

results = []


def check(name, cond):
    results.append((name, bool(cond)))


def val(x):
    return x if isinstance(x, Interval) else Interval.exact(x)


# ---------------------------------------------------------------- optimizers
# AdamW, theta=1, g=1, lr=1/10, eps=0, wd=0: m=1/10, v=1/1000, mhat=1, vhat=1 -> 1 - 0.1 = 0.9
st = so.AdamState(1)
out = so.adam_step([1], [1], st, F(1, 10), eps=0)
check("adamw/step1", val(out[0]).contains(Decimal("0.9")))
# AdamW decoupled decay: theta=1, g=0 (v=0 -> v_hat=0 -> division by eps): lr=1/10, wd=1/2, eps=1: theta_pre = 1 - 1/20 = 19/20,
# m=0 -> update 0 -> 19/20
st = so.AdamState(1)
out = so.adam_step([1], [0], st, F(1, 10), eps=1, weight_decay=F(1, 2))
check("adamw/decoupled-decay", val(out[0]).contains(Decimal(19) / 20))
# Adam (L2): g <- g + wd*theta = 0 + 1/2 ; m=1/20, v=(1/2)^2/1000; mhat=1/2, vhat=1/4 -> sqrt=1/2; eps=0 -> theta - lr*(1/2)/(1/2) = 1 - 1/10
st = so.AdamState(1)
out = so.adam_step([1], [0], st, F(1, 10), eps=0, weight_decay=F(1, 2), decoupled=False)
check("adam/l2-decay", val(out[0]).contains(Decimal("0.9")))
# maximize mirror (wd=0)
a, b = so.prop_maximize_mirror(so.adam_step, [1, -2], [F(1, 3), 5], so.AdamState, lr=F(1, 10), eps=F(1, 7))
check("adam/prop/maximize-mirror", all(val(x).overlaps(val(y)) for x, y in zip(a, b)))
# grad None skips everything
check("adam/prop/none-skip", so.prop_none_is_skip(so.adam_step, [1, 2], so.AdamState, lr=F(1, 10)))
# amsgrad readings differ from t=2: g1=1, g2=0 with beta2=1/2: v1=1/2, v2=1/4; main: vmax=1/2, vhat=(1/2)/(1-1/4)=2/3
# R_old: vhat1 = (1/2)/(1/2)=1, vhat2=(1/4)/(3/4)=1/3 -> max=1
for rd, exp in (("main", F(2, 3)), ("R_old", F(1))):
    st = so.AdamState(1); so.adam_step([0], [1], st, F(1), beta2=F(1, 2), eps=0, amsgrad=True, amsgrad_reading=rd)
    so.adam_step([0], [0], st, F(1), beta2=F(1, 2), eps=0, amsgrad=True, amsgrad_reading=rd)
    vhat = (st.vmax[0] / (1 - F(1, 2) ** 2)) if rd == "main" else st.vmax[0]
    check(f"adam/amsgrad/{rd}", vhat == exp)
# SGD momentum: first step b=g (no dampening even with tau=1/2): theta=1,g=1,lr=1/10,mu=1/2,tau=1/2 -> 0.9; second g=1: b=1/2*1+1/2*1=1 -> 0.8
st = so.SGDState(1)
o1 = so.sgd_step([1], [1], st, F(1, 10), momentum=F(1, 2), dampening=F(1, 2))
o2 = so.sgd_step(o1, [1], st, F(1, 10), momentum=F(1, 2), dampening=F(1, 2))
check("sgd/first-step-no-dampening", o1 == [F(9, 10)] and o2 == [F(8, 10)])
# nesterov: g <- g + mu*b: step1 b=1 -> g=1+1/2 -> theta = 1 - 0.15 = 0.85
st = so.SGDState(1)
check("sgd/nesterov", so.sgd_step([1], [1], st, F(1, 10), momentum=F(1, 2), nesterov=True) == [F(17, 20)])
try:
    so.sgd_step([1], [1], so.SGDState(1), F(1, 10), momentum=F(1, 2), dampening=F(1, 10), nesterov=True)
    check("sgd/nesterov-precondition", False)
except SpecInputError:
    check("sgd/nesterov-precondition", True)
# RMSprop: alpha=1/2, g=2, v=(1/2)*0+(1/2)*4=2, denom=sqrt(2)+0 -> theta - lr*2/sqrt2 = 1 - (1/10)*sqrt(2)
st = so.RMSpropState(1)
out = so.rmsprop_step([1], [2], st, F(1, 10), alpha=F(1, 2), eps=0)
check("rmsprop/step", val(out[0]).contains(1 - Decimal(2).sqrt() / 10))
# centered: gave=(1/2)*2=1, vt=2-1=1 -> theta - lr*2/1 = 0.8
st = so.RMSpropState(1)
out = so.rmsprop_step([1], [2], st, F(1, 10), alpha=F(1, 2), eps=0, centered=True)
check("rmsprop/centered", val(out[0]).contains(Decimal("0.8")))
# sequence with None in the middle: state untouched on the skipped step
st = so.AdamState(1)
traj = so.run_sequence(so.adam_step, [1], [[1], None, [1]], st, lr=F(1, 10), eps=0)
check("adam/sequence/none-skips", st.t == 2 and val(traj[1][0]).overlaps(val(traj[0][0])))

# ---------------------------------------------------------------- normalization
# LayerNorm of (1, 3), eps=0: mean 2, var 1 -> (-1, 1)
y = sn.layer_norm([1, 3], None, None, 0)
check("ln/basic", y[0].contains(-1) and y[1].contains(1))
# gamma/beta: (2,3)*(-1,1)+(5,7) -> (3, 10)
y = sn.layer_norm([1, 3], [2, 3], [5, 7], 0)
check("ln/affine", y[0].contains(3) and y[1].contains(10))
# RMSNorm of (3, 4), eps=0: ms = 25/2 -> rms = 5/sqrt2 -> (3 sqrt2/5, 4 sqrt2/5)
y = sn.rms_norm([3, 4], None, 0)
check("rms/basic", y[0].contains(3 * Decimal(2).sqrt() / 5) and y[1].contains(4 * Decimal(2).sqrt() / 5))
# GroupNorm: 2 channels, 1 group, spatial 2: x = [[1,3],[5,7]] -> mean 4, var 5 -> (x-4)/sqrt5
y = sn.group_norm([[1, 3], [5, 7]], 1, None, None, 0)
check("gn/basic", y[0][0].contains(-3 / Decimal(5).sqrt()) and y[1][1].contains(3 / Decimal(5).sqrt()))
# BatchNorm training: N=2, C=1, spatial 1: x = [[[1]], [[3]]]: biased var 1, unbiased 2; momentum 1/10:
# running_mean = 0.9*0 + 0.1*2 = 1/5 ; running_var = 0.9*1 + 0.1*2 = 11/10 ; output (-1, 1)
st = sn.BNState(1)
y = sn.batch_norm([[[1]], [[3]]], st, None, None, 0, momentum=F(1, 10), training=True)
check("bn/train-output", y[0][0][0].contains(-1) and y[1][0][0].contains(1))
check("bn/running-stats-unbiased", st.running_mean[0] == F(1, 5) and st.running_var[0] == F(11, 10) and st.num_batches_tracked == 1)
# momentum None -> cumulative average: first batch factor 1 -> running = batch stats
st = sn.BNState(1)
sn.batch_norm([[[1]], [[3]]], st, None, None, 0, momentum=None, training=True)
check("bn/momentum-none", st.running_mean[0] == 2 and st.running_var[0] == 2)
# eval uses running stats: with running_mean=2, running_var=2, eps=0: x=3 -> (3-2)/sqrt2
y = sn.batch_norm([[[3]]], st, None, None, 0, training=False)
check("bn/eval", y[0][0][0].contains(1 / Decimal(2).sqrt()))
# unbiased variance with a single element is undefined -> declines
try:
    sn.batch_norm([[[1]]], sn.BNState(1), None, None, 0, training=True); check("bn/n=1-declines", False)
except SpecNotEstablished:
    check("bn/n=1-declines", True)

# ---------------------------------------------------------------- embedding
W = [[1, 2], [3, 4], [5, 6]]
check("emb/forward", se.embedding_forward(W, [2, 0]) == [[5, 6], [1, 2]])
g = se.embedding_grad_weight((3, 2), [2, 0, 2], [[1, 1], [1, 1], [1, 1]])
check("emb/grad-accumulates", g[2] == [2, 2] and g[0] == [1, 1] and g[1] == [0, 0])
g = se.embedding_grad_weight((3, 2), [2, 0, 2], [[1, 1], [1, 1], [1, 1]], scale_grad_by_freq=True)
check("emb/scale-by-freq", g[2] == [1, 1])
g = se.embedding_grad_weight((3, 2), [2, 0], [[1, 1], [1, 1]], padding_idx=0)
check("emb/padding-zero-grad", se.prop_padding_row_zero_grad(g, 0) and g[2] == [1, 1])
r = se.embedding_renorm([[3, 4]], [0], 1)                 # norm 5 -> scaled to 1: (3/5, 4/5)
check("emb/renorm", r[0][0].contains(Decimal(3) / 5) and r[0][1].contains(Decimal(4) / 5))
# embedding_bag: bags [0:2) and [2:3): sum -> (1+3, 2+4) and (5,6); mean of first -> (2,3)
check("bag/sum", se.embedding_bag(W, [0, 1, 2], [0, 2], "sum") == [[4, 6], [5, 6]])
check("bag/mean", se.embedding_bag(W, [0, 1, 2], [0, 2], "mean")[0] == [2, 3])
check("bag/max", se.embedding_bag(W, [0, 1, 2], [0, 2], "max")[0] == [3, 4])
check("bag/empty", se.embedding_bag(W, [0, 1], [0, 2, 2], "sum")[1] == [0, 0])
check("bag/padding-excluded", se.embedding_bag(W, [0, 1], [0], "mean", padding_idx=0) == [[3, 4]])
try:
    se.embedding_bag(W, [0, 0], [0], "mean", padding_idx=0); check("bag/all-padding-mean-declines", False)
except SpecNotEstablished:
    check("bag/all-padding-mean-declines", True)
a, b = se.prop_bag_sum_linearity(W, [0, 1, 2], [0, 2], [1, 2, 3], [4, 5, 6])
check("bag/prop/linearity", a == b)

# ---------------------------------------------------------------- base ops
import spec_base_ops as B
import rigorous_mp as R
import mpmath as mp
check("base/matmul", B.matmul2([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]])
y = B.linear([[1, 2]], [[1, 0], [0, 1], [1, 1]], [1, 1, 1])
check("base/linear", y == [[2, 3, 4]])
dX, dW, db = B.linear_backward([[1, 2]], [[1, 0], [0, 1], [1, 1]], [[1, 1, 1]])
check("base/linear-grads", dX == [[2, 2]] and dW == [[1, 2], [1, 2], [1, 2]] and db == [1, 1, 1])
lhs, rhs = B.prop_linear_adjoint([[1, 2], [3, 5]], [[2, 1], [0, 1], [1, 1]], [[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1]], [[1, 1], [2, 0], [0, 3]])
check("base/prop/linear-adjoint", lhs == rhs)
check("base/var", B.rvar([1, 2, 3, 4], 1) == F(5, 3) and B.rvar([1, 2, 3, 4], 0) == F(5, 4))
try:
    B.rvar([1], 1); check("base/var-dof0-declines", False)
except SpecNotEstablished:
    check("base/var-dof0-declines", True)
p = B.softmax([0, float("-inf"), 0])
check("base/softmax/-inf-entry", p[1].a == 0 == p[1].b and R.contains(p[0], mp.mpf(1) / 2))
try:
    B.softmax([float("-inf"), float("-inf")]); check("base/softmax/all-inf-declines", False)
except SpecNotEstablished:
    check("base/softmax/all-inf-declines", True)
check("base/logsumexp", R.contains(B.logsumexp([0, 0]), mp.log(2)) and B.logsumexp([float("-inf")]) == float("-inf"))
check("base/cumsum", B.cumsum([1, 2, 3]) == [1, 3, 6])
check("base/relu-grad-set", B.relu_grad(0, 1) == {0, 1} and B.relu_grad(2, 1) == {1})
check("base/silu", R.contains(B.silu(0), 0) and R.contains(B.silu(1), 1 / (1 + mp.e ** -1)))
# gelu tanh at x=1: 0.5*(1+tanh(sqrt(2/pi)*(1.044715)))
exp_gelu = mp.mpf(1) / 2 * (1 + mp.tanh(mp.sqrt(2 / mp.pi) * (1 + mp.mpf("0.044715"))))
check("base/gelu-tanh", R.contains(B.gelu_tanh(1), exp_gelu))
check("base/gelu-erf(approx)", R.contains(B.gelu_erf(1), (1 + mp.erf(1 / mp.sqrt(2))) / 2))
check("base/gather", B.gather([[1, 2], [3, 4]], 0, [[1, 0]]) == [[3, 2]] and B.gather([[1, 2], [3, 4]], 1, [[1, 1], [0, 0]]) == [[2, 2], [3, 3]])
check("base/index_select", B.index_select([[1, 2], [3, 4]], 1, [1]) == [[2], [4]])
lhs, rhs = B.prop_gather_adjoint([[1, 2], [3, 4]], 0, [[1, 0], [1, 1]], [[1, 2], [3, 4]], [[5, 6], [7, 8]])
check("base/prop/gather-adjoint", lhs == rhs)
try:
    B.gather([[1, 2]], 0, [[1, 0]]); check("base/gather-oob-refused", False)
except SpecInputError:
    check("base/gather-oob-refused", True)

# ---------------------------------------------------------------- attention / packing / rope
import spec_attention as A
out, P = A.attention([[1, 0]], [[1, 0], [1, 0]], [[1, 0], [0, 1]])
check("attn/equal-keys", R.contains(out[0][0], mp.mpf(1) / 2) and R.contains(out[0][1], mp.mpf(1) / 2))
# scale: default 1/sqrt(E): q=(2,0), k1=(1,0), k2=(0,0): scores 2/sqrt2 = sqrt2 and 0
out, P = A.attention([[2, 0]], [[1, 0], [0, 0]], [[1], [0]])
check("attn/default-scale", R.contains(P[0][0], mp.e ** mp.sqrt(2) / (mp.e ** mp.sqrt(2) + 1)))
# bool mask False excludes a key exactly
out, P = A.attention([[1, 0]], [[1, 0], [1, 0]], [[1], [0]], attn_mask=[[True, False]])
check("attn/bool-mask", P[0][1].a == 0 == P[0][1].b and R.contains(out[0][0], 1))
# causal alignment: Lq=2, Lk=3, UL: row0 sees {0}, row1 sees {0,1}; BR: row0 sees {0,1}, row1 sees {0,1,2}
check("attn/causal-UL", [A.causal_allowed(1, j, 2, 3, "upper_left") for j in range(3)] == [True, True, False])
check("attn/causal-BR", [A.causal_allowed(0, j, 2, 3, "bottom_right") for j in range(3)] == [True, True, False])
try:
    A.attention([[1, 0]], [[1, 0]], [[1]], attn_mask=[[False]]); check("attn/empty-row-declines", False)
except SpecNotEstablished:
    check("attn/empty-row-declines", True)
try:
    A.attention([[1, 0]], [[1, 0]], [[1]], attn_mask=[[True]], is_causal=True); check("attn/mask+causal-refused", False)
except SpecInputError:
    check("attn/mask+causal-refused", True)
# softcap: score s -> c*tanh(s/c); with c=1 and scores (sqrt2, 0) P0 = e^tanh(sqrt2)/(e^tanh(sqrt2)+1)
out, P = A.attention([[2, 0]], [[1, 0], [0, 0]], [[1], [0]], softcap=1)
check("attn/softcap", R.contains(P[0][0], mp.e ** mp.tanh(mp.sqrt(2)) / (mp.e ** mp.tanh(mp.sqrt(2)) + 1)))
check("attn/gqa-head-map", A.gqa_kv_head(5, 8, 2) == 1)
m = A.packed_mask([2, 1]); check("pack/mask", m == [[True, False, False], [True, True, False], [False, False, True]])
check("pack/positions", A.packed_position_ids([2, 1]) == [0, 1, 0])
out, P = A.attention([[1, 0], [1, 0], [1, 0]], [[1, 0], [1, 0], [1, 0]], [[1], [2], [3]], attn_mask=m)
check("pack/prop/no-leak", A.prop_no_cross_document_leak(P, [2, 1]) == [])
y = A.rope_apply([1, 0], 1, 2)
check("rope/pos1-d2", R.contains(y[0], mp.cos(1)) and R.contains(y[1], mp.sin(1)))
nx, ny = A.prop_rope_norm_preserving([3, 4, 1, 2], A.rope_apply([3, 4, 1, 2], 9, 4))
check("rope/prop/norm", R.overlaps(nx, ny))
a, b = A.prop_rope_relative([1, 2, 3, 4], [4, 3, 2, 1], 2, 5, 4)
check("rope/prop/relative", R.overlaps(a, b))
yi = A.rope_apply([1, 0, 0, 0], 1, 4, layout="interleaved")
check("rope/interleaved-pairs", R.contains(yi[1], mp.sin(1)) and yi[2].a == 0 == yi[2].b)

# ---------------------------------------------------------------- training program layer
import spec_training_program as TP
check("sch/cosine-ends", R.contains(TP.cosine_annealing(1, 0, 0, 10), 1) and R.contains(TP.cosine_annealing(1, 0, 10, 10), 0) and R.contains(TP.cosine_annealing(1, 0, 5, 10), mp.mpf(1) / 2))
check("sch/linear", TP.linear_lr(1, F(1, 2), 1, 4, 2) == F(3, 4) and TP.linear_lr(1, F(1, 2), 1, 4, 9) == 1)
check("sch/hf-warmup", R.contains(TP.hf_cosine_with_warmup(1, 1, 4, 12), F(1, 4)) and R.contains(TP.hf_cosine_with_warmup(1, 12, 4, 12), 0))
g, tn = TP.clip_grad_norm([[3, 4]], 1)                                # norm 5 -> coef 1/(5+1e-6)
check("clip/norm", R.contains(tn, 5))
check("clip/coef", R.contains(g[0][0], mp.mpf(3) / (5 + mp.mpf(10) ** -6)))
g, tn = TP.clip_grad_norm([[3, 4]], 10)
check("clip/no-clip", R.contains(g[0][1], 4) and R.contains(tn, 5))
check("clip/inf-norm", R.contains(TP.total_norm([[3, -7]], float("inf")), 7))
st = TP.ScalerState(init_scale=4, growth_interval=2)
u, stepped = TP.scaler_step(st, [[8]], False); check("amp/unscale", u == [[2]] and stepped and st.scale == 4 and st.tracker == 1)
u, stepped = TP.scaler_step(st, [[8]], True); check("amp/skip-backoff", u is None and not stepped and st.scale == 2 and st.tracker == 0)
TP.scaler_step(st, [[2]], False); TP.scaler_step(st, [[2]], False)
check("amp/growth", st.scale == 4 and st.tracker == 0)
check("prog/shift", TP.shift_labels([5, 6, 7]) == [6, 7, -100])
check("prog/count", TP.count_items([[6, 7, -100], [-100, -100, 1]]) == 3)
check("prog/cross-rank", TP.cross_rank_mean([F(-1, 2), F(3, 2)], [1, 3]) == F(1, 4) and TP.wrong_variant_mean_of_rank_means([F(-1, 2), F(3, 2)], [1, 3]) == 0)

# ---------------------------------------------------------------- MoE
import spec_moe as M
probs = M.router_probs([0, 0, 1])
check("moe/topk-unique", M.topk_sets(probs, 1) == [frozenset({2})])
check("moe/topk-ties", set(M.topk_sets(M.router_probs([1, 1, 0]), 1)) == {frozenset({0}), frozenset({1})})
w = M.combine_weights(probs, [1, 2], renormalise=True)
check("moe/renorm-sum-1", R.contains(M.prop_weights_sum_to_one(w), 1))
experts = [lambda x: [F(2) * F(v) for v in x], lambda x: [F(3) * F(v) for v in x]]
outs, rec = M.moe_forward([[1]], [[0, 0]], experts, k=2, renormalise=True)
check("moe/forward-equal-weights", R.contains(outs[0][0], mp.mpf(5) / 2) and set(rec[0]["sets"]) == {frozenset({0, 1})})
outs, rec = M.moe_forward([[1], [1]], [[0, 10], [0, 10]], experts, k=1, capacity=1)
check("moe/capacity-drop", rec[1]["used"] == [] and outs[1][0].a == 0 == outs[1][0].b)
aux = M.load_balancing_loss([[0, 0], [0, 0]], [[0], [1]], alpha=1)   # f=(1/2,1/2), P=(1/2,1/2): 1*2*(1/4+1/4)=1
check("moe/aux-loss", R.contains(aux, 1))
check("moe/prop/permute", M.prop_permute_unpermute_identity(["a", "b", "c"], [2, 0, 1]))

failed = [n for n, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
for n in failed:
    print("FAILED:", n)
sys.exit(1 if failed else 0)
