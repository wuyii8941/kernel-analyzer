"""Adapter for contracts/SAMPLING.yaml: vLLM top-k / top-p filtering and sampling given the kernel's exponential noise.

Input {op, logits [B][V], k [B] | null, p [B] | null, q [B][V] (op sample), keep [B][V] booleans (op filtered_probs)}.
Output `out`, one entry per row:
- op mask: the allowed set of masked rows (kept logits unchanged, the rest -inf);
- op sample: the allowed set of sampled indices argmax_{i kept} P_i / q_i over all allowed kept sets;
- op filtered_probs: the renormalised distribution on the kernel's own kept set (which must be an allowed one).
The semantics are in kernel_spec.d2_sampling; MUTANTS are independent implementations below.
"""
from __future__ import annotations

import itertools

from kernel_spec import d2_common as U
from kernel_spec import d2_sampling as S
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Undefined, Unsupported, integer, rational, vector
from kernel_spec.real import NEG, R

KEYS = {"op", "logits", "k", "p", "q", "keep"}
OPS = ("mask", "sample", "filtered_probs")


def parse(profile, inp):
    if profile != "nucleus":
        raise Unsupported(f"profile {profile!r}")
    U.check_keys(inp, KEYS, required={"op", "logits"})
    op = inp.get("op")
    if op not in OPS:
        raise Invalid(f"op must be one of {OPS}")
    l, (B, V) = U.tensor(inp["logits"], 2, "logits", allow_inf=True)
    for row in l:
        for x in row:
            if x.v == "+inf":
                raise Invalid("+inf logit")
    k = inp.get("k")
    if k is not None:
        k = vector(k, "k", n=B, item=integer)
        if any(x < 1 for x in k):
            raise Invalid("k: one integer >= 1 per row")
    p = inp.get("p")
    if p is not None:
        p = vector(p, "p", n=B, item=rational)
        if any(x < 0 or x > 1 for x in p):
            raise Invalid("p: one value in [0, 1] per row")
    q = keep = None
    if op == "sample":
        q, _ = U.tensor(inp.get("q"), 2, "q", shape=(B, V))
        if any(x.sign() <= 0 for row in q for x in row):
            raise Invalid("exponential noise must be > 0 (support of exponential_)")
    if op == "filtered_probs":
        keep = inp.get("keep")
        if U.shape_of(keep, "keep") != (B, V) or any(not isinstance(x, bool) for row in keep for x in row):
            raise Invalid("keep: [B][V] booleans")
    return dict(op=op, l=l, B=B, V=V, k=k, p=p, q=q, keep=keep)


def _row_sets(a, b, kept_sets=S.kept_sets):
    l = a["l"][b]
    finite = [i for i, x in enumerate(l) if x.v != NEG]
    if not finite:
        raise Undefined("row without a finite logit: softmax undefined")
    p = a["p"][b] if a["p"] is not None else None
    if p is not None and p == 0:
        raise Undefined("top-p with p = 0: the smallest set of mass >= 0 is empty")
    k = a["k"][b] if a["k"] is not None else None
    return l, kept_sets(l, finite, k, p)


def _evaluate(profile, inp, kept_sets=S.kept_sets, argmax=S.argmax_ratio):
    a = parse(profile, inp)
    out = []
    for b in range(a["B"]):
        l, sets = _row_sets(a, b, kept_sets)
        if a["op"] == "mask":
            out.append(frozenset(S.masked_row(l, kp) for kp in sets))
        elif a["op"] == "sample":
            out.append(frozenset(argmax(l, kp, a["q"][b]) for kp in sets))
        else:
            kp = frozenset(i for i, x in enumerate(a["keep"][b]) if x)
            if kp not in sets:
                raise Invalid(f"row {b}: the conditioning keep set is not an allowed top-k / top-p set")
            P = S.probs(l, kp)
            out.append([P[i].v if i in kp else 0 for i in range(a["V"])])
    return Result("value", {"out": out})


def reference(profile, inp):
    return _evaluate(profile, inp)


# ------------------------------------------------------------------ reading: vLLM PyTorch fallback top-k (ties kept)
def _kept_torch_topk(l, finite, k, p):
    if k is not None and k < len(finite):
        kth = sorted((l[i].v for i in finite), reverse=True)[k - 1]
        finite = [i for i in finite if l[i].v >= kth]
    out = set()
    out.update(S.topp_sets(l, finite, p))
    return out


READINGS = {"vllm_torch_topk": lambda profile, inp: _evaluate(profile, inp, kept_sets=_kept_torch_topk)}


# ------------------------------------------------------------------ mutants (independent code)
def _softmax(l, idx):
    m = max(l[i].v for i in idx)
    e = {i: rl.exp(l[i] - R(m)) for i in idx}
    z = U.rsum(e.values())
    return {i: e[i] / z for i in idx}


def _order_sets(l, idx):
    """All descending orders of idx (ties permuted) - small rows only."""
    vals = sorted({l[i].v for i in idx}, reverse=True)
    blocks = [[i for i in idx if l[i].v == v] for v in vals]
    for perm in itertools.product(*[list(itertools.permutations(bk)) for bk in blocks]):
        yield [i for bk in perm for i in bk]


def _topp_by_order(l, idx, p, strict=True, inclusive_ties=False):
    if p is None or p == 1:
        return {frozenset(idx)}
    P = _softmax(l, idx)
    res = set()
    for order in _order_sets(l, idx):
        kept, above = [], R(0)
        for i in order:
            if inclusive_ties:
                higher = U.rsum(P[j] for j in idx if l[j].v > l[i].v)
                ok = higher < R(p)
            else:
                ok = above < R(p) if strict else above <= R(p)
            if ok:
                kept.append(i)
            above = above + P[i]
        res.add(frozenset(kept))
    return res


def _topk_by_order(l, idx, k, extra=0):
    if k is None or k >= len(idx):
        return {frozenset(idx)}
    return {frozenset(order[:k + extra]) for order in _order_sets(l, idx)}


def _mut_kept(kind):
    def kept(l, finite, k, p):
        if kind == "topp_before_topk":
            out = set()
            for s1 in _topp_by_order(l, finite, p):
                out |= _topk_by_order(l, sorted(s1), k)
            return out
        if kind == "topp_full_distribution":
            P = _softmax(l, finite)
            out = set()
            for s1 in _topk_by_order(l, finite, k):
                for order in _order_sets(l, sorted(s1)):
                    kp, above = [], R(0)
                    for i in order:
                        if p is None or p == 1 or above < R(p):
                            kp.append(i)
                        above = above + P[i]
                    out.add(frozenset(kp))
            return out
        out = set()
        for s1 in _topk_by_order(l, finite, k, extra=1 if kind == "topk_off_by_one" else 0):
            out |= _topp_by_order(l, sorted(s1), p, strict=kind != "topp_non_strict",
                                  inclusive_ties=kind == "topp_tie_inclusive")
        return out
    return kept


def _argmax_mul(l, keep, q):
    return max(sorted(keep), key=lambda i: (rl.exp(l[i]) * q[i]).bounds()[0])


def _argmax_unfiltered(l, keep, q):
    idx = [i for i, x in enumerate(l) if x.v != NEG]
    return max(idx, key=lambda i: (rl.exp(l[i]) / q[i]).bounds()[0])


def _mask_zero(profile, inp):
    r = _evaluate(profile, inp, kept_sets=_mut_kept("plain"))
    if inp.get("op") == "mask":
        r.fields["out"] = [frozenset(tuple(0 if x == NEG else x for x in row) for row in s) for s in r.fields["out"]]
    return r


def _mk(kind):
    return lambda profile, inp: _evaluate(profile, inp, kept_sets=_mut_kept(kind))


MUTANTS = {
    "SP.M1": _mk("topp_tie_inclusive"),
    "SP.M2": _mk("topp_non_strict"),
    "SP.M3": _mk("topp_full_distribution"),
    "SP.M4": _mk("topp_before_topk"),
    "SP.M5": _mk("topk_off_by_one"),
    "SP.M6": lambda profile, inp: _evaluate(profile, inp, kept_sets=_mut_kept("plain"), argmax=_argmax_mul),
    "SP.M7": lambda profile, inp: _evaluate(profile, inp, kept_sets=_mut_kept("plain"), argmax=_argmax_unfiltered),
    "SP.M8": _mask_zero,
}
