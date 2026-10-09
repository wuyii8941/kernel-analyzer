"""Hand-computed checks (v0.2). Every expected value is derived by hand in the adjacent comment or copied from the
official documentation example (marked DOC). Audit counterexamples from 2026-10-07 are marked AUDIT.
Run: python3 test_specs.py   (exit code 1 on any failure)"""
from decimal import Decimal, localcontext, getcontext
getcontext().prec = 90          # expected values in this file are formed at 90 digits (the spec sets its own contexts)
from fractions import Fraction as F
import math
import sys

import spec_cross_entropy as ce
import spec_pooling as pool
import spec_index_scatter as ix
import spec_accumulation as acc

results = []


def check(name, cond):
    results.append((name, bool(cond)))


def ln(v):
    with localcontext() as c:
        c.prec = 90
        return Decimal(v).ln()


def same(a, b):
    """two rigorous intervals of the same real value must overlap."""
    return a.lo <= b.hi and b.lo <= a.hi


# ------------------------------------------------------------------ cross-entropy
r = ce.cross_entropy([[0, 0]], [0])                      # p=(1/2,1/2): loss = ln2, grad = (-1/2, 1/2)
check("ce/uniform/loss-contains-ln2", r["loss"].contains(ln(2)))
check("ce/uniform/width<1e-55", r["loss"].width() < Decimal(10) ** -55)
check("ce/uniform/grad", r["grad"][0][0].contains(Decimal("-0.5")) and r["grad"][0][1].contains(Decimal("0.5")))

with localcontext() as c:
    c.prec = 90
    l3 = Decimal(3).ln()                                  # logits (ln3, 0): p=(3/4,1/4), loss = ln(4/3), grad=(-1/4,1/4)
r = ce.cross_entropy([[l3, Decimal(0)]], [0])
check("ce/3:1/loss", r["loss"].contains(ln(Decimal(4) / 3)))
check("ce/3:1/grad", r["grad"][0][0].contains(Decimal(-1) / 4))

r = ce.cross_entropy([[0, 0]], [1], label_smoothing=F(1, 5))   # uniform logits: smoothing changes nothing -> ln2
check("ce/smooth-uniform", r["loss"].contains(ln(2)))

r = ce.cross_entropy([[0, 0], [0, 0]], [0, 1], weights=[1, 3])  # (1*ln2 + 3*ln2)/(1+3) = ln2
check("ce/weighted-mean", r["loss"].contains(ln(2)))
r = ce.cross_entropy([[0, 0], [0, 0]], [0, 1], weights=[1, 3], reduction="sum")
check("ce/weighted-sum=4ln2", r["loss"].contains(4 * ln(2)))

r = ce.cross_entropy([[l3, Decimal(0)], [5, -5]], [0, -100])    # second row ignored
check("ce/ignore/loss", r["loss"].contains(ln(Decimal(4) / 3)))
check("ce/ignore/zero-grad-row", all(v.contains(0) for v in r["grad"][1]))
check("ce/ignore/prop-no-offenders", ce.prop_ignored_rows_zero_grad(r, [0, -100]) == [])

r = ce.cross_entropy([[0, 0]], [-100])                   # CE-A3: all ignored, 'mean' -> undefined
check("ce/all-ignored/nan", r["nan"] and not r["grad_defined"])
check("ce/all-ignored/prop-skipped (AUDIT)", ce.prop_ignored_rows_zero_grad(r, [-100]) == [])
r = ce.cross_entropy([[0, 0]], [-100], reduction="sum")  # 'sum' with all ignored is defined: 0 loss, 0 grad
check("ce/all-ignored/sum=0", r["loss"].contains(0) and all(v.contains(0) for v in r["grad"][0]))

# CE-A1/A2 reading split: uniform logits, target 0, w=(1,3), eps=1/2 -> q=(3/4,1/4), log p = -ln2
# R_A: (1*3/4 + 3*1/4) ln2 / w_y=1 = (3/2) ln2 ; R_B: 1*(3/4+1/4) ln2 / 1 = ln2 ; R_C: (3/2) ln2 / (3/2) = ln2
for rd, exp in (("R_A", Decimal(3) / 2 * ln(2)), ("R_B", ln(2)), ("R_C", ln(2))):
    r = ce.cross_entropy([[0, 0]], [0], weights=[1, 3], label_smoothing=F(1, 2), reading=rd)
    check(f"ce/A1/{rd}", r["loss"].contains(exp))

for rd in ("R_A", "R_B", "R_C"):                         # gradient rows sum to zero for every reading
    r = ce.cross_entropy([[1, -2, F(1, 3)], [0, 4, -1]], [2, 0], weights=[2, 1, 5], label_smoothing=F(1, 10), reading=rd)
    check(f"ce/prop/grad-row-sum/{rd}", all(s.contains(0) for s in ce.prop_grad_row_sums(r["grad"])))

a, b = ce.prop_shift_invariance(ce.cross_entropy, [[1, -2, F(1, 3)]], [1], 7)
check("ce/prop/shift", same(a["loss"], b["loss"]))

# class permutation with ignore_index inside the class range (AUDIT counterexample): must be invariant
a, b = ce.prop_class_permutation(ce.cross_entropy, [[2, 0], [0, 0]], [0, 1], [1, 0], ignore_index=0)
check("ce/prop/perm/ignore-in-range (AUDIT)", same(a["loss"], b["loss"]))
# probability targets permute along the class axis (AUDIT: used to raise TypeError)
a, b = ce.prop_class_permutation(ce.cross_entropy, [[2, 0, 1]], [[F(1, 2), F(1, 4), F(1, 4)]], [2, 0, 1], prob_target=True)
check("ce/prop/perm/prob-target (AUDIT)", same(a["loss"], b["loss"]))

r = ce.cross_entropy([[0, 0], [0, 0]], [[1, 0], [0, 1]], weights=[2, 2], prob_target=True)  # D4: mean by N -> 2 ln2
check("ce/prob-target/mean-by-N", r["loss"].contains(2 * ln(2)))

# AUDIT: huge logits; the enclosure must contain the true value A + ln(1 + e^-A) (> A, within 1e-60 relative of A)
A = Decimal(1e100)
r = ce.cross_entropy([[-1e100, 0]], [0])
check("ce/huge-logit/encloses-A (AUDIT)", r["loss"].lo <= A <= r["loss"].hi)
check("ce/huge-logit/width-is-ulp-scale", r["loss"].width() <= Decimal(10) ** 46)

# ------------------------------------------------------------------ pooling (1-D, channel dim first)
x = [[1, 2, 3, 4, 5]]
check("pool/avg/floor", pool.avg_pool(x, 2, 2) == [[F(3, 2), F(7, 2)]])
check("pool/avg/ceil/R1", pool.avg_pool(x, 2, 2, ceil_mode=True, reading="R1") == [[F(3, 2), F(7, 2), F(5, 2)]])
check("pool/avg/ceil/R2", pool.avg_pool(x, 2, 2, ceil_mode=True, reading="R2") == [[F(3, 2), F(7, 2), F(5)]])
check("pool/avg/ceil/exclude-pad", pool.avg_pool(x, 2, 2, ceil_mode=True, count_include_pad=False) == [[F(3, 2), F(7, 2), F(5)]])
x4 = [[1, 2, 3, 4]]
check("pool/avg/pad/include", pool.avg_pool(x4, 3, 2, 1) == [[F(1), F(3)]])
check("pool/avg/pad/exclude", pool.avg_pool(x4, 3, 2, 1, count_include_pad=False) == [[F(3, 2), F(3)]])
check("pool/avg/pad/ceil/R1", pool.avg_pool(x4, 3, 2, 1, ceil_mode=True, reading="R1")[0][2] == F(4, 3))
check("pool/avg/pad/ceil/default-is-R2", pool.avg_pool(x4, 3, 2, 1, ceil_mode=True)[0][2] == F(2))
check("pool/avg/pad/ceil/R2", pool.avg_pool(x4, 3, 2, 1, ceil_mode=True, reading="R2")[0][2] == F(2))
check("pool/avg/pad/ceil/exclude", pool.avg_pool(x4, 3, 2, 1, ceil_mode=True, count_include_pad=False)[0][2] == F(4))
check("pool/avg/divisor_override", pool.avg_pool(x, 2, 2, divisor_override=5) == [[F(3, 5), F(7, 5)]])
for rd in ("R1", "R2"):
    kw = dict(kernel=3, stride=2, padding=1, ceil_mode=True, reading=rd)
    y = pool.avg_pool(x4, **kw); g = [[F(1), F(-2), F(5)]]
    gx = pool.avg_pool_backward(g, [1, 4], **kw)
    lhs, rhs = pool.prop_adjoint(y, g, x4, gx)
    check(f"pool/prop/adjoint/{rd}", lhs == rhs)
check("pool/avg2d", pool.avg_pool([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], 2, 1) == [[[F(3), F(4)], [F(6), F(7)]]])

vals, sets = pool.max_pool([[3, 1, 3, 0]], 2, 1, 0, dilation=2)   # windows {0,2},{1,3}
check("pool/max/dilation-values", vals == [[F(3), F(1)]])
check("pool/max/tie-set", sets[0][0] == {(0,), (2,)})
vals, _ = pool.max_pool([[1.0, float("nan"), 2.0]], 3, 1, nan_reading="R_prop")
check("pool/max/nan-prop", math.isnan(vals[0][0]))
vals, _ = pool.max_pool([[1.0, float("nan"), 2.0]], 3, 1, nan_reading="R_ignore")
check("pool/max/nan-ignore", vals[0][0] == 2)
a, b = pool.prop_max_shift(pool.max_pool, [[3, 1, 3, 0]], 10, kernel=2, stride=1)
check("pool/prop/max-shift", all(u == v + 10 for u, v in zip(a[0], b[0])))
# AUDIT: legal window sampling only padding: input [7], k=2, s=1, p=1, d=2 -> positions {-1, 1} -> -inf, empty set
vals, sets = pool.max_pool([[7]], 2, 1, 1, dilation=2)
check("pool/max/empty-window=-inf (AUDIT)", vals == [[float("-inf")]] and sets == [[set()]])
vals, sets = pool.max_pool([[float("-inf"), 1.0, float("inf")]], 3, 1)
check("pool/max/inf-inputs (AUDIT)", vals[0][0] == float("inf") and sets[0][0] == {(2,)})
try:
    pool.avg_pool([[1.0, float("inf")]], 2, 1); check("pool/avg/non-finite-raises", False)
except ValueError:
    check("pool/avg/non-finite-raises", True)

# ------------------------------------------------------------------ index / scatter
X = [[2, 2, 2] for _ in range(5)]
T = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
I = [0, 4, 2, 0]
r = ix.index_reduce(X, 0, I, T, "prod")
check("idx/DOC/prod/include_self", r[0] == [20, 44, 72])
check("idx/prod/rows", r[4] == [8, 10, 12] and r[2] == [14, 16, 18] and r[1] == [2, 2, 2])
r = ix.index_reduce(X, 0, I, T, "prod", include_self=False)
check("idx/DOC/prod/exclude_self", r[0] == [10, 22, 36])
check("idx/prod/exclude/untouched", r[1] == [2, 2, 2] and r[3] == [2, 2, 2])
check("idx/mean/include_self", ix.index_reduce(X, 0, I, T, "mean")[0] == [F(13, 3), F(15, 3), F(17, 3)])
check("idx/mean/exclude_self", ix.index_reduce(X, 0, I, T, "mean", include_self=False)[0] == [F(11, 2), F(13, 2), F(15, 2)])
check("idx/amax/exclude_self", ix.index_reduce(X, 0, I, T, "amax", include_self=False)[0] == [10, 11, 12])
# AUDIT: the only valid mean reading preserves constants; the error variant does not
check("idx/mean/constant-preserved (AUDIT)", ix.index_reduce([1], 0, [0], [1], "mean") == [1])
check("idx/mean/error-variant-labelled (AUDIT)", ix.index_reduce([1], 0, [0], [1], "mean", error_variant="mean_divides_by_contrib_count") == [2])

src = [1, 2, 3, 4, 5, 6]; idx = [0, 1, 0, 1, 2, 1]; inp = [1, 2, 3, 4]
check("sc/DOC/sum", ix.scatter_reduce(inp, 0, idx, src, "sum") == [5, 14, 8, 4])
check("sc/DOC/sum/exclude", ix.scatter_reduce(inp, 0, idx, src, "sum", include_self=False) == [4, 12, 5, 4])
inp2 = [5, 4, 3, 2]
check("sc/DOC/amax", ix.scatter_reduce(inp2, 0, idx, src, "amax") == [5, 6, 5, 2])
check("sc/DOC/amax/exclude", ix.scatter_reduce(inp2, 0, idx, src, "amax", include_self=False) == [3, 6, 5, 2])
check("sc/mean/include_self", ix.scatter_reduce(inp, 0, idx, src, "mean") == [F(5, 3), F(7, 2), F(4), F(4)])
check("sc/amax/inf-input", ix.scatter_reduce([0.0], 0, [0, 0], [float("-inf"), 2.0], "amax") == [2])
check("idx/add", ix.index_add([0, 0, 0], 0, [1, 1, 2], [1, 2, 3], alpha=2) == [0, 6, 6])
check("idx/prop/counting", ix.index_add([0], 0, [0] * 5, [1] * 5) == [ix.prop_counting(5)])
check("idx/prod-grad-with-zero", ix.prod_grad_factors([2, 0, 3]) == [0, 6, 0])
a, b = ix.prop_order_invariance(ix.index_reduce, X, 0, I, T, [3, 2, 1, 0], "mean")
check("idx/prop/order", a == b)
after = ix.scatter_reduce(inp, 0, idx, src, "sum", include_self=False)
check("sc/prop/untouched", ix.prop_untouched_rows(inp, after, {(0,), (1,), (2,)}) == [])

# ------------------------------------------------------------------ accumulation window (AUDIT construction)
# two micro-batches with 1 and 3 tokens; gradient sums -1/2 and 3/2.
# wrong (mean of means): (1/2)(-1/2 + (3/2)/3) = 0 ; correct: (-1/2 + 3/2)/(1+3) = 1/4
losses = [[F(-1, 2)], [F(1, 2), F(1, 2), F(1, 2)]]
check("acc/correct=1/4", acc.window_loss(losses) == F(1, 4))
check("acc/wrong-variant=0", acc.wrong_variant_mean_of_means(losses) == 0)
check("acc/grad", acc.window_grad(losses) == F(1, 4))
a, b = acc.prop_split_invariance(losses, lambda flat: [flat[:2], flat[2:]])
check("acc/prop/split", a == b)

# DOC: AvgPool1d(3, stride=2) on [1..7] -> [2, 4, 6]
check("pool/DOC/avg1d", pool.avg_pool([[1, 2, 3, 4, 5, 6, 7]], 3, 2) == [[F(2), F(4), F(6)]])

# ------------------------------------------------------------------ v0.3 regressions (second audit)
# subnormal gradient: must be an enclosing interval, not a non-moving point
r = ce.cross_entropy([[0, -2302718]], [0])
g1 = r["grad"][0][1]
check("ce/subnormal-grad/established (AUDIT2)", r["established"])
check("ce/subnormal-grad/encloses (AUDIT2)", g1.lo <= Decimal("1.90204465038E-1000058") <= g1.hi)
# huge equal logits: loss is exactly ln 2, no overflow
r = ce.cross_entropy([[1e100, 1e100]], [0])
check("ce/huge-equal/ln2 (AUDIT2)", r["established"] and r["loss"].contains(ln(2)))
# illegal labels refused, not answered
try:
    ce.cross_entropy([[0, 0]], [-1]); check("ce/illegal-label-refused (AUDIT2)", False)
except ce.SpecInputError:
    check("ce/illegal-label-refused (AUDIT2)", True)
try:
    ce.cross_entropy([[0, 0]], [0], label_smoothing=2); check("ce/illegal-eps-refused", False)
except ce.SpecInputError:
    check("ce/illegal-eps-refused", True)
# ignore_index inside the class range is legal (D5)
r = ce.cross_entropy([[0, 0], [0, 0]], [0, 1], ignore_index=0)
check("ce/ignore-in-range-legal", r["established"] and r["loss"].contains(ln(2)))
# index out of range refused
try:
    ix.index_add([0, 0], 0, [-1], [7]); check("idx/illegal-index-refused (AUDIT2)", False)
except ix.SpecInputError:
    check("idx/illegal-index-refused (AUDIT2)", True)
try:
    ix.index_reduce([[1, 2], [3, 4]], 2, [0], [[1, 2]], "mean"); check("idx/unsupported-dim-refused", False)
except ix.SpecInputError:
    check("idx/unsupported-dim-refused", True)
# dim=1 order invariance permutes columns (AUDIT2): self 2x2, index [0,1] with source [[1,2],[3,4]]
a, b = ix.prop_order_invariance(ix.index_reduce, [[10, 20], [30, 40]], 1, [0, 1], [[1, 2], [3, 4]], [1, 0], "amax")
check("idx/prop/order/dim1 (AUDIT2)", a == b == [[10, 20], [30, 40]])
a, b = ix.prop_order_invariance(ix.index_add, [[0, 0], [0, 0]], 1, [1, 1], [[1, 2], [3, 4]], [1, 0])
check("idx/prop/order/dim1/add (AUDIT2)", a == b == [[0, 3], [0, 7]])
# pooling preconditions: padding above half the kernel refused (P-D8)
try:
    pool.avg_pool([[1, 2, 3]], 2, 1, padding=2); check("pool/pad-too-large-refused", False)
except pool.SpecInputError:
    check("pool/pad-too-large-refused", True)
# adjoint check refuses mismatched shapes (AUDIT2)
try:
    pool.prop_adjoint([[F(1), F(2)]], [[F(1)]], [[1, 2, 3]], [[F(0), F(0), F(0)]]); check("pool/adjoint-shape-check (AUDIT2)", False)
except pool.SpecInputError:
    check("pool/adjoint-shape-check (AUDIT2)", True)
# gradient mass with an empty window: input [7], k=2, s=1, p=1, d=2 -> one empty window; routed mass must be 0
vals, sets = pool.max_pool([[7]], 2, 1, 1, dilation=2)
routed, input_mass, n_empty = pool.prop_max_grad_mass([[F(3)]], [[F(0)]], sets)
check("pool/grad-mass/empty-window (AUDIT2)", routed == 0 and input_mass == 0 and n_empty == 1)
vals, sets = pool.max_pool([[3, 1, 3, 0]], 2, 1)
routed, input_mass, n_empty = pool.prop_max_grad_mass([[F(1), F(1), F(1)]], [[F(1), F(0), F(2), F(0)]], sets)
check("pool/grad-mass/non-empty", routed == input_mass == 3 and n_empty == 0)

# ------------------------------------------------------------------ v0.4: adopted decisions (2026-10-07 review)
# reviewer's distinguishing example: [1,2,3,4], k=3, s=2, p=0, ceil -> second window covers 2,3,4; index 4 is overhang
check("pool/A1/review-example/R2", pool.avg_pool([[1, 2, 3, 4]], 3, 2, 0, ceil_mode=True, reading="R2") == [[F(2), F(7, 2)]])
check("pool/A1/review-example/R1", pool.avg_pool([[1, 2, 3, 4]], 3, 2, 0, ceil_mode=True, reading="R1") == [[F(2), F(7, 3)]])
# all-NaN window under R_ignore is 'not established', distinct from the geometric empty window
try:
    pool.max_pool([[float("nan"), float("nan")]], 2, 1, nan_reading="R_ignore"); check("pool/A3/all-nan-ignore-declines", False)
except pool.SpecNotEstablished:
    check("pool/A3/all-nan-ignore-declines", True)
try:
    ix.scatter_reduce([0.0], 0, [0], [float("nan")], "amax", include_self=False, nan_reading="R_ignore")
    check("idx/A3/all-nan-ignore-declines (REVIEW)", False)
except ix.SpecNotEstablished:
    check("idx/A3/all-nan-ignore-declines (REVIEW)", True)
# include_self=False: old self NaN does not participate (reviewer example -> 2)
check("idx/A3/excluded-self-nan", ix.scatter_reduce([float("nan")], 0, [0], [2.0], "amax", include_self=False) == [2])
# max indices must lie in the argmax set
vals, sets = pool.max_pool([[3, 1, 3, 0]], 2, 1)
check("pool/A2/indices-in-set", pool.check_max_indices([[(0,), (2,), (2,)]], sets) == [])
check("pool/A2/index-outside-set", len(pool.check_max_indices([[(1,), (2,), (2,)]], sets)) == 1)
# subgradient set check on non-overlapping windows: x=[1,1 | 5,2], k=2,s=2, upstream (1, 1)
vals, sets = pool.max_pool([[1, 1, 5, 2]], 2, 2)
check("pool/A2/subgrad/single-ok", pool.check_max_subgradient([[F(1), F(1)]], [[F(1), F(0), F(1), F(0)]], sets, 2, 2) == [])
check("pool/A2/subgrad/split-ok", pool.check_max_subgradient([[F(1), F(1)]], [[F(1, 2), F(1, 2), F(1), F(0)]], sets, 2, 2) == [])
check("pool/A2/subgrad/(2,-1)-rejected", len(pool.check_max_subgradient([[F(1), F(1)]], [[F(2), F(-1), F(1), F(0)]], sets, 2, 2)) == 1)
check("pool/A2/subgrad/outside-set-rejected", len(pool.check_max_subgradient([[F(1), F(1)]], [[F(1), F(0), F(0), F(1)]], sets, 2, 2)) == 1)
try:
    pool.check_max_subgradient([[F(1), F(1), F(1)]], [[F(1), F(0), F(1), F(0)]], pool.max_pool([[1, 1, 5, 2]], 2, 1)[1], 2, 1)
    check("pool/A2/subgrad/overlap-refused", False)
except pool.SpecInputError:
    check("pool/A2/subgrad/overlap-refused", True)
# IDX-A4 set check: values (1,1), upstream 1 -> (1,0) ok, (1/2,1/2) ok, (2,-1) rejected
check("idx/A4/subgrad/single", ix.check_extremum_subgradient([1, 1], [1, 0], 1) is None)
check("idx/A4/subgrad/split", ix.check_extremum_subgradient([1, 1], [F(1, 2), F(1, 2)], 1) is None)
check("idx/A4/subgrad/(2,-1)", ix.check_extremum_subgradient([1, 1], [2, -1], 1) is not None)
check("idx/A4/subgrad/unique", ix.check_extremum_subgradient([1, 3], [0, 1], 1) is None and ix.check_extremum_subgradient([1, 3], [1, 0], 1) is not None)
# CE-A3 wording: positive numerator over zero denominator (weights (0,1), target 0, eps=1/2) is undefined, not 0/0
r = ce.cross_entropy([[0, 0]], [0], weights=[0, 1], label_smoothing=F(1, 2))
check("ce/A3/positive-over-zero-undefined (REVIEW)", r["nan"] and not r["grad_defined"])
r = ce.cross_entropy([[0, 0]], [0], weights=[0, 1], label_smoothing=F(1, 2), reduction="none")
check("ce/A3/unreduced-defined=ln2/4", r["loss"][0].contains(ln(2) / 4))

# ------------------------------------------------------------------ report
failed = [n for n, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
for n in failed:
    print("FAILED:", n)
sys.exit(1 if failed else 0)
