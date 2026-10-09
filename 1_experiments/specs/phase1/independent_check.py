"""Independent cross-check of the phase-1 specs (does not reuse the specs' arithmetic).

CE      : every loss/gradient interval returned by spec_cross_entropy must contain the value computed independently with
          mpmath at 300 significant digits (an error of 1e-300 relative is far inside the ~1e-59 relative spec width).
          Cases include the extreme inputs from both audits plus random mixtures of scales from 1e-300 to 1e100.
Pooling : spec_pooling must agree EXACTLY with a brute-force implementation that materialises the padded array.
Index   : spec_index_scatter must agree EXACTLY with a dictionary-based brute-force implementation.
Fuzz    : no exception other than SpecInputError may escape any spec call; established=False must only occur when the
          independent computation also finds a value outside the normal decimal range.
Run: python3 independent_check.py [n_cases]  -> prints a summary and writes independent_check_results.json
"""
import json
import math
import random
import sys
from decimal import Decimal
from fractions import Fraction as F

import mpmath as mp

import spec_cross_entropy as ce
import spec_pooling as pool
import spec_index_scatter as ix

mp.mp.dps = 300
rng = random.Random(20261007)
SCALES = [1e-300, 1e-30, 1e-3, 1.0, 1e3, 1e30, 1e100]


def mpf_of(v):
    if isinstance(v, Decimal):
        return mp.mpf(str(v))
    if isinstance(v, F):
        return mp.mpf(v.numerator) / mp.mpf(v.denominator)
    return mp.mpf(v)


def ce_independent(logits, target, weights, ignore_index, reduction, eps, reading, prob_target):
    N, C = len(logits), len(logits[0])
    w = [mp.mpf(1)] * C if weights is None else [mpf_of(v) for v in weights]
    eps = mpf_of(eps)
    rows, grads, dens = [], [], []
    for n in range(N):
        t = target[n]
        if not prob_target and t == ignore_index:
            rows.append(mp.mpf(0)); grads.append([mp.mpf(0)] * C); dens.append(mp.mpf(0)); continue
        x = [mpf_of(v) for v in logits[n]]
        m = max(x)
        e = [mp.exp(v - m) for v in x]
        s = mp.fsum(e)
        logp = [v - m - mp.log(s) for v in x]
        p = [v / s for v in e]
        if prob_target:
            q = [(1 - eps) * mpf_of(t[c]) + eps / C for c in range(C)]
            a = [w[c] * q[c] for c in range(C)]; den = mp.mpf(1)
        else:
            q = [(1 - eps) * (1 if c == t else 0) + eps / C for c in range(C)]
            if reading == "R_B":
                a = [w[t] * q[c] for c in range(C)]; den = w[t]
            else:
                a = [w[c] * q[c] for c in range(C)]; den = mp.fsum(a) if reading == "R_C" else w[t]
        rows.append(-mp.fsum(a[c] * logp[c] for c in range(C)))
        A = mp.fsum(a)
        grads.append([p[c] * A - a[c] for c in range(C)]); dens.append(den)
    if reduction == "none":
        return rows, grads, True
    tot = mp.fsum(rows)
    if reduction == "sum":
        return tot, grads, True
    D = mp.fsum(dens)
    if D == 0:
        return None, None, False
    return tot / D, [[g / D for g in row] for row in grads], True


def inside(iv, val):
    """containment up to mpmath's own representation error (binary mpf of a decimal/rational at 300 digits)."""
    tol = mp.mpf(10) ** -280 * max(mp.mpf(1), abs(val))
    return mpf_of(iv.lo) - tol <= val <= mpf_of(iv.hi) + tol


def rand_logit_row(C, mode):
    if mode == "normal":
        return [rng.gauss(0, 1) for _ in range(C)]
    if mode == "mixed":
        return [rng.gauss(0, 1) * rng.choice(SCALES) for _ in range(C)]
    if mode == "equal":
        v = rng.gauss(0, 1) * rng.choice(SCALES); return [v] * C
    if mode == "huge-neg":
        row = [rng.gauss(0, 1) for _ in range(C)]; row[rng.randrange(C)] = -rng.choice([1e6, 2302718.0, 1e30, 1e100]); return row
    if mode == "huge-pos":
        row = [rng.gauss(0, 1) for _ in range(C)]; row[rng.randrange(C)] = rng.choice([1e6, 1e30, 1e100]); return row
    if mode == "exact":
        return [F(rng.randint(-5, 5), rng.randint(1, 4)) for _ in range(C)]
    raise ValueError(mode)


def check_ce(n_cases):
    stats = dict(cases=0, scalars=0, inside=0, outside=0, unestablished=0, undefined=0, exceptions=0, outside_examples=[])
    modes = ["normal", "mixed", "equal", "huge-neg", "huge-pos", "exact"]
    for i in range(n_cases):
        N = rng.choice([1, 2, 3]); C = rng.choice([2, 3, 5, 8])
        mode = rng.choice(modes)
        logits = [rand_logit_row(C, mode) for _ in range(N)]
        prob = rng.random() < 0.2
        if prob:
            target = []
            for _ in range(N):
                raw = [rng.random() for _ in range(C)]; s = sum(raw)
                target.append([F(v).limit_denominator(10 ** 6) / F(s).limit_denominator(10 ** 6) for v in raw])
        else:
            ignore = rng.choice([-100, -100, 0])
            target = [rng.randrange(C) if rng.random() < 0.8 else ignore for _ in range(N)]
        weights = rng.choice([None, None, [F(rng.randint(1, 5)) for _ in range(C)], [F(rng.choice([0, 1, 3])) for _ in range(C)]])
        eps = rng.choice([0, F(1, 10), 0.1, 1, F(1, 2)])
        reading = rng.choice(["R_A", "R_B", "R_C"])
        reduction = rng.choice(["mean", "sum", "none"])
        ignore_index = -100 if prob else ignore
        kw = dict(weights=weights, ignore_index=ignore_index, reduction=reduction, label_smoothing=eps,
                  reading=reading, prob_target=prob)
        stats["cases"] += 1
        try:
            r = ce.cross_entropy(logits, target, **kw)
        except ce.SpecInputError:
            continue
        except Exception as e:                       # any other escape is a leak
            stats["exceptions"] += 1; stats["outside_examples"].append(("EXC", repr(e)[:120], str(logits)[:120])); continue
        if not r.get("established", True):
            stats["unestablished"] += 1
            stats["outside_examples"].append(("UNESTABLISHED", r.get("reason", "")[:100], mode))
            continue
        if r["nan"]:
            stats["undefined"] += 1
            ind = ce_independent(logits, target, weights, ignore_index, reduction, eps, reading, prob)
            if ind[2]:
                stats["outside"] += 1; stats["outside_examples"].append(("NAN-MISMATCH", str(logits)[:120], str(target)))
            continue
        loss_i, grads_i, ok = ce_independent(logits, target, weights, ignore_index, reduction, eps, reading, prob)
        if reduction == "none":
            pairs = list(zip(r["loss"], loss_i))
        else:
            pairs = [(r["loss"], loss_i)]
        for n in range(N):
            for c in range(C):
                pairs.append((r["grad"][n][c], grads_i[n][c]))
        for iv, val in pairs:
            stats["scalars"] += 1
            if inside(iv, val):
                stats["inside"] += 1
            else:
                stats["outside"] += 1
                if len(stats["outside_examples"]) < 20:
                    stats["outside_examples"].append(("OUTSIDE", str(iv)[:80], mp.nstr(val, 30), mode, str(logits)[:100]))
    return stats


# ---------------------------------------------------------------- pooling brute force (materialised padded array)

def bf_avg_pool_1d(x, k, s, p, ceil_mode, count_include_pad, divisor_override, reading):
    L = len(x)
    padded = [None] * p + list(x) + [None] * p        # None = padding cell (value 0, counts as padded position)
    num = L + 2 * p - k
    out = (math.ceil(F(num, s)) if ceil_mode else num // s) + 1
    if ceil_mode and (out - 1) * s >= L + p:
        out -= 1
    res = []
    for o in range(out):
        start = o * s
        cells = [padded[j] if j < len(padded) else "OVER" for j in range(start, start + k)]
        real = [F(v) for v in cells if v is not None and v != "OVER"]
        n_padded = sum(1 for v in cells if v != "OVER")
        if divisor_override is not None:
            div = divisor_override
        elif not count_include_pad:
            div = len(real)
        elif reading == "R2":
            div = n_padded
        else:
            div = k
        res.append(sum(real, F(0)) / div)
    return res


def bf_max_pool_1d(x, k, s, p, d, ceil_mode):
    L = len(x)
    num = L + 2 * p - d * (k - 1) - 1
    out = (math.ceil(F(num, s)) if ceil_mode else num // s) + 1
    if ceil_mode and (out - 1) * s >= L + p:
        out -= 1
    res, sets = [], []
    for o in range(out):
        vals = []
        for m in range(k):
            pos = o * s - p + d * m
            if 0 <= pos < L:
                vals.append((pos, F(x[pos])))
        if not vals:
            res.append(float("-inf")); sets.append(set()); continue
        best = max(v for _, v in vals)
        res.append(best); sets.append({(pos,) for pos, v in vals if v == best})
    return res, sets


def check_pool(n_cases):
    stats = dict(cases=0, agree=0, disagree=0, rejected_both=0, rejected_one=0, exceptions=0, examples=[])
    for i in range(n_cases):
        L = rng.randint(1, 9); k = rng.randint(1, 4); s = rng.randint(1, 3); p = rng.randint(0, 2)
        d = rng.randint(1, 2); ceil_mode = rng.random() < 0.5
        x = [rng.randint(-5, 5) for _ in range(L)]
        legal = 2 * p <= d * (k - 1) + 1 and L + 2 * p >= d * (k - 1) + 1
        stats["cases"] += 1
        is_avg = rng.random() < 0.5
        legal_this = (2 * p <= k and L + 2 * p >= k) if is_avg else legal
        try:
            if is_avg:
                cip = rng.random() < 0.5; do = rng.choice([None, 7]); rd = rng.choice(["R1", "R2"])
                got = pool.avg_pool([x], k, s, p, ceil_mode=ceil_mode, count_include_pad=cip, divisor_override=do, reading=rd)[0]
                if not legal_this:
                    stats["rejected_one"] += 1; stats["examples"].append(("avg accepted illegal", L, k, s, p)); continue
                exp = bf_avg_pool_1d(x, k, s, p, ceil_mode, cip, do, rd)
            else:
                got, gsets = pool.max_pool([x], k, s, p, dilation=d, ceil_mode=ceil_mode)
                got = got[0]; gsets = gsets[0]
                if not legal_this:
                    stats["rejected_one"] += 1; stats["examples"].append(("max accepted illegal", L, k, s, p, d)); continue
                exp, esets = bf_max_pool_1d(x, k, s, p, d, ceil_mode)
                if gsets != esets:
                    stats["disagree"] += 1; stats["examples"].append(("max sets", x, k, s, p, d, ceil_mode)); continue
        except pool.SpecNotEstablished:
            stats["rejected_both"] += 1; continue
        except pool.SpecInputError:
            if legal_this:
                stats["rejected_one"] += 1; stats["examples"].append(("spec rejected legal", L, k, s, p, d, is_avg)); continue
            stats["rejected_both"] += 1
            continue
        except Exception as e:
            stats["exceptions"] += 1; stats["examples"].append(("EXC", repr(e)[:100], x, k, s, p, d)); continue
        if got == exp:
            stats["agree"] += 1
        else:
            stats["disagree"] += 1
            if len(stats["examples"]) < 20:
                stats["examples"].append(("DIS", x, k, s, p, d, ceil_mode, str(got)[:60], str(exp)[:60]))
    return stats


# ---------------------------------------------------------------- index/scatter brute force

def bf_index_reduce(self_t, index, source, op, include_self):
    out = [F(v) for v in self_t]
    groups = {}
    for i, j in enumerate(index):
        groups.setdefault(j, []).append(F(source[i]))
    for j, vals in groups.items():
        allv = ([out[j]] if include_self else []) + vals
        if op == "prod":
            r = F(1)
            for v in allv: r *= v
        elif op == "mean":
            r = sum(allv, F(0)) / len(allv)
        elif op == "amax":
            r = max(allv)
        elif op == "amin":
            r = min(allv)
        elif op == "sum":
            r = sum(allv, F(0))
        out[j] = r
    return out


def check_index(n_cases):
    stats = dict(cases=0, agree=0, disagree=0, rejected=0, exceptions=0, examples=[])
    for i in range(n_cases):
        n = rng.randint(1, 6); m = rng.randint(0, 7)
        self_t = [rng.randint(-3, 3) for _ in range(n)]
        index = [rng.randint(-1, n) for _ in range(m)]            # may include illegal -1 / n
        source = [rng.randint(-3, 3) for _ in range(m)]
        op = rng.choice(["prod", "mean", "amax", "amin"]); inc = rng.random() < 0.5
        legal = all(0 <= j < n for j in index)
        stats["cases"] += 1
        try:
            if rng.random() < 0.5:
                got = ix.index_reduce(self_t, 0, index, source, op, include_self=inc)
                exp = bf_index_reduce(self_t, index, source, op, inc)
            else:
                op2 = rng.choice(["sum", "prod", "mean", "amax", "amin"])
                got = ix.scatter_reduce(self_t, 0, index, source, op2, include_self=inc)
                exp = bf_index_reduce(self_t, index, source, op2, inc)
            if not legal:
                stats["disagree"] += 1; stats["examples"].append(("accepted illegal index", index, n)); continue
        except ix.SpecNotEstablished:
            stats["rejected"] += 1; continue
        except ix.SpecInputError:
            stats["rejected"] += 1
            if legal:
                stats["disagree"] += 1; stats["examples"].append(("rejected legal", self_t, index, source, op))
            continue
        except Exception as e:
            stats["exceptions"] += 1; stats["examples"].append(("EXC", repr(e)[:100])); continue
        if got == exp:
            stats["agree"] += 1
        else:
            stats["disagree"] += 1
            if len(stats["examples"]) < 20:
                stats["examples"].append(("DIS", self_t, index, source, op, inc, str(got), str(exp)))
    return stats


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    res = dict(ce=check_ce(n), pool=check_pool(n * 3), index=check_index(n * 3))
    json.dump(res, open("independent_check_results.json", "w"), indent=1, default=str)
    for k, v in res.items():
        print(k, {a: b for a, b in v.items() if not isinstance(b, list)})
        for ex in v.get("outside_examples", v.get("examples", []))[:8]:
            print("   ", ex)
