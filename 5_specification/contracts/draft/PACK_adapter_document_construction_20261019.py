"""Adapter for contracts/PACK.yaml: packed-sequence (document) attention mask, position ids and the attention output
of the packed sequence.

The reference builds the document mask and positions from the document lengths and evaluates attention with that
boolean mask through kernel_spec.d1_attention (SDPA semantics). The mutants are independent code (own mask builders,
kernel_spec.d1_indep for the attention).
"""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import d1_attention as A
from kernel_spec import d1_indep as D
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational, require_fields
from kernel_spec.real import R

PROFILES = {"packed_causal": True, "packed_bidirectional": False}
KEYS = {"doc_lengths", "q", "k", "v", "scale"}


def parse(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    require_fields(inp, {"doc_lengths", "q", "k", "v"}, "PACK")
    extra = set(inp) - KEYS
    if extra:
        raise Invalid(f"unknown input fields {sorted(extra)}")
    lens = inp["doc_lengths"]
    if not isinstance(lens, list) or not lens:
        raise Invalid("doc_lengths must be a non-empty list")
    lens = [integer(n, "doc_lengths", lo=1) for n in lens]
    q, qs = A.tensor(inp["q"], 3, "q")
    k, ks = A.tensor(inp["k"], 3, "k")
    v, vs = A.tensor(inp["v"], 3, "v")
    H, L, E = qs
    if ks != (H, L, E) or vs[:2] != (H, L):
        raise Invalid("q, k, v must share heads and the packed length")
    if sum(lens) != L:
        raise Invalid(f"document lengths sum to {sum(lens)}, the packed sequence has {L} tokens")
    scale = A.default_scale(E) if inp.get("scale") is None else R(rational(inp["scale"], "scale"))
    return lens, q, k, v, scale, PROFILES[profile]


def doc_ids(lens):
    out = []
    for d, n in enumerate(lens):
        out += [d] * n
    return out


def reference(profile, inp):
    lens, q, k, v, scale, causal = parse(profile, inp)
    doc = doc_ids(lens)
    L = len(doc)
    mask = [[doc[i] == doc[j] and (j <= i or not causal) for j in range(L)] for i in range(L)]
    pos = [t for n in lens for t in range(n)]
    res = A.attention([q], [k], [v], allowed=lambda b, h, i, j: mask[i][j], bias=lambda b, h, i, j: Fraction(0),
                      scale=scale, empty="undefined")
    return Result("value", {"mask": mask, "position_ids": pos, "out": A.values(res["out"])[0]})


# ------------------------------------------------------------------ mutants (independent code)
def _indep(profile, inp, m):
    lens, q, k, v, scale, causal = parse(profile, inp)
    L = sum(lens)
    ends, acc = [], 0
    for n in lens:
        acc += n
        ends.append(acc)

    def doc_of(t):
        for d, e in enumerate(ends):
            if (t <= e - 1) if m != "boundary_off_by_one" else (t <= e):
                return d
        return len(ends) - 1
    doc = [doc_of(t) for t in range(L)]
    cz = causal and m != "bidirectional_within_document"

    def ok(i, j):
        same = True if m == "no_document_boundaries" else doc[i] == doc[j]
        return same and (j <= i or not cz)
    mask = [[ok(i, j) for j in range(L)] for i in range(L)]
    if m == "positions_global":
        pos = list(range(L))
    elif m == "positions_from_one":
        pos = [t + 1 for n in lens for t in range(n)]
    else:
        pos = []
        for d in range(len(lens)):
            pos += [t - (ends[d] - lens[d]) for t in range(ends[d] - lens[d], ends[d])]
    res = D.iv_attention([q], [k], [v], allowed=lambda b, h, i, j: mask[i][j], bias=lambda b, h, i, j: Fraction(0),
                         scale=D.I(scale.v), empty="undefined")
    return Result("value", {"mask": mask, "position_ids": pos, "out": res["out"][0]})


def _m(kind):
    return lambda profile, inp: _indep(profile, inp, kind)


MUTANTS = {
    "PK.M1": _m("no_document_boundaries"),
    "PK.M2": _m("positions_global"),
    "PK.M3": _m("positions_from_one"),
    "PK.M4": _m("bidirectional_within_document"),
    "PK.M5": _m("boundary_off_by_one"),
}
READINGS = {}
