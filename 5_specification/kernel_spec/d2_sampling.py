"""Top-k / top-p filtering and exponential-race sampling (contract SAMPLING), exact semantics with allowed sets.

Per row with logits l (rationals or -inf; F = the finite positions):
- top-k (Holtzman et al. 3.2: a set of size k maximising the mass): if k < |F|, every k-subset made of all tokens above
  the k-th largest value plus any of the copies of that value; otherwise F.
- top-p on the renormalised distribution of each top-k set S (Holtzman et al. 3.1: the smallest set with mass >= p): walk
  the distinct values in decreasing order; inside the group of value v (n copies of probability P_v) the j-th copy is kept
  iff mass_above + j P_v < p; when a group is cut, any m of its n copies may be the kept ones.
The result is the set of all kept index sets (an allowed set). Probabilities exp(l_i - max) / Z are R values: exact when
every kept logit is equal, rigorous enclosures otherwise (then a comparison with the rational p is strict, so an undecided
enclosure raises NotEstablished instead of guessing).

sample(l, q): argmax over the kept set of P_i / q_i (torch.argmax: first maximal index), for every allowed kept set.
"""
from __future__ import annotations

import itertools
from math import comb

from . import real as rl
from .num import NotEstablished
from .real import NEG, R

MAX_SETS = 20000


def groups(values, idx):
    """[(value, [indices])] for the given indices in decreasing value order (exact values)."""
    out = {}
    for i in idx:
        out.setdefault(values[i].v, []).append(i)
    return [(R(v), sorted(ix)) for v, ix in sorted(out.items(), key=lambda kv: kv[0], reverse=True)]


def _choose(base, pool, m):
    if comb(len(pool), m) > MAX_SETS:
        raise NotEstablished("too many tied selections to enumerate")
    return [frozenset(base) | frozenset(c) for c in itertools.combinations(pool, m)]


def topk_sets(l, finite, k):
    if k is None or k >= len(finite):
        return [frozenset(finite)]
    base, out = [], None
    for v, ix in groups(l, finite):
        if len(base) + len(ix) <= k:
            base += ix
            if len(base) == k:
                return [frozenset(base)]
            continue
        return _choose(base, ix, k - len(base))
    return [frozenset(base)]


def probs(l, S):
    m = max((l[i] for i in S), key=lambda r: r.v)
    e = {i: rl.exp(l[i] - m) for i in S}
    z = R(0)
    for i in S:
        z = z + e[i]
    return {i: e[i] / z for i in S}


def topp_sets(l, S, p):
    if p is None or p == 1:
        return [frozenset(S)]
    P = probs(l, S)
    base, above = [], R(0)
    for v, ix in groups(l, S):
        pv = P[ix[0]]
        m = 0
        for j in range(len(ix)):
            if above + R(j) * pv < R(p):
                m += 1
            else:
                break
        if m < len(ix):
            return _choose(base, ix, m)
        base += ix
        above = above + R(len(ix)) * pv
    return [frozenset(base)]


def kept_sets(l, finite, k, p):
    out = set()
    for S in topk_sets(l, finite, k):
        out.update(topp_sets(l, S, p))
    return out


def masked_row(l, keep):
    return tuple(l[i].v if i in keep else NEG for i in range(len(l)))


def argmax_ratio(l, keep, q):
    """First index maximising P_i / q_i over keep, i.e. exp(l_i) / q_i (the common factor 1/Z cancels)."""
    best, bi = None, None
    for i in sorted(keep):
        r = rl.exp(l[i]) / q[i]
        if best is None:
            best, bi = r, i
            continue
        if l[i].v == l[bi].v and q[i].v == q[bi].v:
            continue                                    # exactly equal ratio: the first index stays
        if r > best:
            best, bi = r, i
    return bi
