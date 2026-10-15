"""Cross-check: our CE spec (spec_cross_entropy.py, written from PyTorch docs) against the declared semantics of the
Liger-Kernel cross-entropy kernel (formulas transcribed from liger_cross_entropy_kernel: online-softmax lse, label
smoothing via eps = ls/V, z-loss, softcap, mean = divide by n_non_ignore). Liger side evaluated in 80-digit mpmath."""
import sys, os, random
from fractions import Fraction as F
from decimal import Decimal
import mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'specification_v2', 'algorithm_specs', 'phase1'))
import spec_cross_entropy as ce
mp.mp.dps = 80

def liger_rows(X, y, ignore_index, ls, s, softcap):
    """per-row loss and gradient following the kernel; returns list of (loss_row or None, grad_row)"""
    out = []
    for xr, yr in zip(X, y):
        V = len(xr)
        if yr == ignore_index:
            out.append((None, [mp.mpf(0)] * V)); continue
        x = [mp.mpf(v.numerator) / v.denominator for v in xr]
        inter = None
        if softcap is not None:
            inter = [mp.tanh(v / softcap) for v in x]
            x = [softcap * t for t in inter]
        m = max(x); d = sum(mp.e ** (v - m) for v in x); lse = m + mp.log(d)
        eps = mp.mpf(ls) / V
        loss = lse - x[yr]
        if ls > 0:
            smooth = sum(-eps * v for v in x) + ls * lse
            loss = loss * (1 - ls) + smooth
        loss += s * lse * lse
        g = []
        for i, v in enumerate(x):
            p = mp.e ** (v - m) / d
            gi = p + 2 * s * lse * p - eps
            if i == yr: gi -= (1 - ls)
            if softcap is not None: gi = gi * (1 - inter[i] * inter[i])
            g.append(gi)
        out.append((loss, g))
    return out

def liger_total(X, y, ignore_index, ls, s, softcap, reduction):
    rows = liger_rows(X, y, ignore_index, ls, s, softcap)
    n_non = sum(1 for yr in y if yr != ignore_index)
    losses = [r[0] for r in rows if r[0] is not None]
    grads = [r[1] for r in rows]
    if reduction == 'mean':
        # kernel divides each non-ignored row by n_non_ignore (0 rows -> loss stays 0); wrapper sums
        return (sum(l / n_non for l in losses) if n_non else mp.mpf(0)), [[gi / n_non if n_non else gi for gi in g] for g in grads]
    return sum(losses, mp.mpf(0)), grads

def inside(val, iv, tol=mp.mpf('1e-45')):
    lo, hi = mp.mpf(str(iv.lo)), mp.mpf(str(iv.hi))
    return lo - tol <= val <= hi + tol

rng = random.Random(20261015)
stats = dict(cases=0, loss_ok=0, grad_ok=0, all_ignored=0, declined=0, mismatches=[])
for case in range(400):
    N, C = rng.randint(1, 5), rng.randint(2, 7)
    X = [[F(rng.randint(-40, 40), rng.choice([1, 2, 4, 5, 8])) for _ in range(C)] for _ in range(N)]
    ignore_index = -100
    y = [rng.randint(0, C - 1) if rng.random() > 0.25 else ignore_index for _ in range(N)]
    if case % 50 == 0: y = [ignore_index] * N                      # force the all-ignored edge case
    ls = rng.choice([F(0), F(1, 10), F(3, 10)])
    reduction = rng.choice(['mean', 'sum'])
    softcap = rng.choice([None, 2, 5])
    # softcap is not in our CE spec: feed the softcapped logits to the spec (composition check)
    Xs = X if softcap is None else [[F(mp.nstr(softcap * mp.tanh(mp.mpf(v.numerator) / v.denominator / softcap), 60)) for v in row] for row in X]
    res = ce.cross_entropy(Xs, y, weights=None, ignore_index=ignore_index, reduction=reduction, label_smoothing=ls)
    stats['cases'] += 1
    if not res.get('established', True):
        stats['declined'] += 1; continue
    L, G = liger_total(X, y, ignore_index, mp.mpf(ls.numerator) / ls.denominator, mp.mpf(0), softcap, reduction)
    if res['nan']:
        stats['all_ignored'] += 1
        stats['mismatches'].append((case, 'spec: NaN (undefined, CE-A3); Liger kernel: returns 0'))
        continue
    lok = inside(L, res['loss'], tol=mp.mpf('1e-40') if softcap is None else mp.mpf('1e-30'))
    gok = True
    for n in range(N):
        for k in range(C):
            if softcap is not None:   # spec grad is w.r.t. the capped logits; apply the chain rule tanh' = 1 - tanh^2
                t = mp.tanh(mp.mpf(X[n][k].numerator) / X[n][k].denominator / softcap)
                lo, hi = mp.mpf(str(res['grad'][n][k].lo)) * (1 - t * t), mp.mpf(str(res['grad'][n][k].hi)) * (1 - t * t)
                lo, hi = min(lo, hi), max(lo, hi)
                if not (lo - mp.mpf('1e-30') <= G[n][k] <= hi + mp.mpf('1e-30')): gok = False
            elif not inside(G[n][k], res['grad'][n][k], tol=mp.mpf('1e-40')): gok = False
    stats['loss_ok'] += lok; stats['grad_ok'] += gok
    if not (lok and gok): stats['mismatches'].append((case, dict(N=N, C=C, ls=str(ls), red=reduction, softcap=softcap, loss=lok, grad=gok)))

print({k: v for k, v in stats.items() if k != 'mismatches'})
for m in stats['mismatches'][:12]: print(' ', m)
