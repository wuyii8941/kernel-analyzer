"""Adapter for contracts/PAGEDATTN.yaml: vLLM unified attention over a paged KV cache.

f: gather the keys/values of each sequence through its block table, place the query tokens at the last query_len
positions of the sequence, and evaluate softmax attention (kernel_spec.d1_attention) with the declared causal flag,
window (FlashAttention convention) and softcap. The mutants are independent interval code (kernel_spec.d1_indep, or the
loop below for the V-mask misreading of bug B013).
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import d1_attention as A
from kernel_spec import d1_indep as D
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational, require_fields, vector
from kernel_spec.real import R

KEYS = {"query", "cu_seqlens_q", "seq_lens", "key_cache", "value_cache", "block_table", "scale", "causal", "window",
        "softcap"}


def _cache(x, what):
    """[num_blocks][block_size][H][D] with rationals, or null for a slot that is never read."""
    if not isinstance(x, list) or not x:
        raise Invalid(f"{what}: expected [num_blocks][block_size][H][D]")
    blocks = []
    dims = None
    for b, blk in enumerate(x):
        if not isinstance(blk, list):
            raise Invalid(f"{what}[{b}]: not a list")
        slots = []
        for s, slot in enumerate(blk):
            if slot is None:
                slots.append(None)
                continue
            vals, sh = A.tensor(slot, 2, f"{what}[{b}][{s}]")
            if dims is None:
                dims = sh
            if sh != dims:
                raise Invalid(f"{what}: inconsistent head / dimension sizes")
            slots.append(vals)
        blocks.append(slots)
    bs = {len(b) for b in blocks}
    if len(bs) != 1:
        raise Invalid(f"{what}: blocks of different sizes")
    return blocks, bs.pop(), dims


def parse(profile, inp):
    if profile != "vllm_unified":
        raise Unsupported(f"profile {profile!r}")
    require_fields(inp, KEYS - {"softcap"}, "PAGEDATTN")
    extra = set(inp) - KEYS
    if extra:
        raise Invalid(f"unknown input fields {sorted(extra)}")
    q, qs = A.tensor(inp["query"], 3, "query")
    T, Hq, Dh = qs
    cu = vector(inp["cu_seqlens_q"], "cu_seqlens_q", item=integer)
    sl = vector(inp["seq_lens"], "seq_lens", item=lambda x, name: integer(x, name, lo=0))
    if not cu or cu[0] != 0 or any(b < a for a, b in zip(cu, cu[1:])) or cu[-1] != T:
        raise Invalid("cu_seqlens_q must start at 0, be non-decreasing and end at the number of query tokens")
    if len(sl) != len(cu) - 1:
        raise Invalid("seq_lens must have one entry per sequence")
    kc, bs, kd = _cache(inp["key_cache"], "key_cache")
    vc, bs2, vd = _cache(inp["value_cache"], "value_cache")
    if bs != bs2 or len(kc) != len(vc):
        raise Invalid("key and value caches differ in block count or block size")
    if kd is None or vd is None:
        raise Invalid("caches hold no data")
    Hkv, Dk = kd
    if vd[0] != Hkv or Dk != Dh:
        raise Invalid("cache heads / head size do not match the query")
    if Hkv == 0 or Hq % Hkv:
        raise Invalid("query heads must be a multiple of key/value heads")
    bt = inp["block_table"]
    if not isinstance(bt, list) or len(bt) != len(sl):
        raise Invalid("block_table needs one row per sequence")
    scale = R(rational(inp["scale"], "scale"))
    causal = inp["causal"]
    if not isinstance(causal, bool):
        raise Invalid("causal must be a boolean (per-sequence causal flags are outside this profile)")
    wl, wr = A.window_pair(inp["window"])
    if not causal and wl != wr:
        raise Unsupported("asymmetric bidirectional window: FlashAttention convention and vLLM kernel disagree (PA.WINDOW)")
    cap = inp.get("softcap")
    cap = rational(0 if cap is None else cap, "softcap")
    if cap < 0:
        raise Invalid("softcap must be >= 0")
    seqs = []
    for s in range(len(sl)):
        ql = cu[s + 1] - cu[s]
        if ql > sl[s]:
            raise Invalid(f"sequence {s}: query length {ql} exceeds its kv length {sl[s]}")
        row = vector(bt[s], f"block_table[{s}]", item=integer)
        K, V = [], []
        for j in range(sl[s]):
            lb = j // bs
            if lb >= len(row) or not 0 <= row[lb] < len(kc):
                raise Invalid(f"sequence {s}: block table entry for logical block {lb} missing or out of range")
            ks, vs = kc[row[lb]][j % bs], vc[row[lb]][j % bs]
            if ks is None or vs is None:
                raise Invalid(f"sequence {s}: position {j} reads an empty cache slot")
            K.append(ks)
            V.append(vs)
        seqs.append(dict(ql=ql, sl=sl[s], start=cu[s], K=K, V=V, row=row))
    return dict(q=q, T=T, Hq=Hq, Hkv=Hkv, Dv=vd[1], seqs=seqs, scale=scale, causal=causal, wl=wl, wr=wr,
                cap=None if cap == 0 else R(cap), kc=kc, vc=vc, bs=bs)


def _mask(causal, wl, wr):
    def allowed(pos, j):
        if causal and j > pos:
            return False
        if wl >= 0 and pos - j > wl:
            return False
        if not causal and wr >= 0 and j - pos > wr:
            return False
        return True
    return allowed


def _heads(seq, a):
    """q [1][Hq][ql][D], k [1][Hkv][S][D], v [1][Hkv][S][Dv] for one sequence."""
    q = [[[a["q"][seq["start"] + i][h] for i in range(seq["ql"])] for h in range(a["Hq"])]]
    k = [[[seq["K"][j][g] for j in range(seq["sl"])] for g in range(a["Hkv"])]]
    v = [[[seq["V"][j][g] for j in range(seq["sl"])] for g in range(a["Hkv"])]]
    return q, k, v


def reference(profile, inp):
    a = parse(profile, inp)
    allowed = _mask(a["causal"], a["wl"], a["wr"])
    out = [[None] * a["Hq"] for _ in range(a["T"])]
    for seq in a["seqs"]:
        if seq["ql"] == 0:
            continue
        q, k, v = _heads(seq, a)
        ctx = seq["sl"] - seq["ql"]
        res = A.attention(q, k, v, allowed=lambda b, h, i, j, c=ctx: allowed(c + i, j),
                          bias=lambda b, h, i, j: Fraction(0), scale=a["scale"], softcap=a["cap"], empty="undefined")
        for i in range(seq["ql"]):
            for h in range(a["Hq"]):
                out[seq["start"] + i][h] = A.values(res["out"][0][h][i])
    return Result("value", {"out": out})


# ------------------------------------------------------------------ mutants (independent interval code)
def _indep(profile, inp, m=None):
    a = parse(profile, inp)
    causal, wl, wr = a["causal"], a["wl"], a["wr"]
    if m == "window_no_right_edge":
        allowed = _mask(causal, wl, -1)
    elif m == "window_strict":
        base = _mask(causal, wl, wr)
        allowed = lambda pos, j: base(pos, j) and (wl < 0 or pos - j < wl) and (causal or wr < 0 or j - pos < wr)  # noqa: E731
    else:
        allowed = _mask(causal, wl, wr)
    seqs = a["seqs"]
    if m == "block_table_ignored":                      # logical block b read from physical block b
        seqs = []
        for s in a["seqs"]:
            K = [a["kc"][j // a["bs"]][j % a["bs"]] for j in range(s["sl"])]
            V = [a["vc"][j // a["bs"]][j % a["bs"]] for j in range(s["sl"])]
            if any(x is None for x in K + V):
                raise Invalid("mutant read an empty slot")
            seqs.append(dict(s, K=K, V=V))
    out = [[None] * a["Hq"] for _ in range(a["T"])]
    for seq in seqs:
        if seq["ql"] == 0:
            continue
        q, k, v = _heads(seq, a)
        ctx = 0 if m == "positions_top_left" else seq["sl"] - seq["ql"]
        if m == "v_mask_block_first_query":
            rows = _b013(q, k, v, a, ctx, allowed, block_q=2)
        else:
            res = D.iv_attention(q, k, v, allowed=lambda b, h, i, j, c=ctx: allowed(c + i, j),
                                 bias=lambda b, h, i, j: Fraction(0), scale=D.I(a["scale"].v),
                                 softcap=None if (a["cap"] is None or m == "softcap_ignored") else a["cap"].v,
                                 empty="undefined")
            rows = [[res["out"][0][h][i] for h in range(a["Hq"])] for i in range(seq["ql"])]
        for i in range(seq["ql"]):
            out[seq["start"] + i] = rows[i]
    return Result("value", {"out": out})


def _b013(q, k, v, a, ctx, allowed, block_q):
    """Bug B013: the scores are masked per query row, but V rows are zeroed by a window computed from the first query
    of each block of block_q queries, so keys right of that window drop out of the numerator only."""
    Hq, grp = a["Hq"], a["Hq"] // a["Hkv"]
    ql, S, Dv = len(q[0][0]), len(k[0][0]), a["Dv"]
    sw = a["wl"] + 1
    rows = []
    for i in range(ql):
        row = []
        lo = ctx + (i // block_q) * block_q
        for h in range(Hq):
            g = h // grp
            js = [j for j in range(S) if allowed(ctx + i, j)]
            z = [sum((D.I(x) * D.I(y) for x, y in zip(q[0][h][i], k[0][g][j])), D.ZERO) * D.I(a["scale"].v) for j in js]
            w = [iv.exp(t) for t in z]
            tot = sum(w[1:], w[0])
            keepv = [(lo - j < sw) and (a["causal"] or lo - j > -sw) for j in js]
            row.append([sum((wj / tot * D.I(v[0][g][j][d]) for wj, j, kv in zip(w, js, keepv) if kv), D.ZERO)
                        for d in range(Dv)])
        rows.append(row)
    return rows


def _m(kind):
    return lambda profile, inp: _indep(profile, inp, kind)


MUTANTS = {
    "PA.M1": _m("v_mask_block_first_query"),
    "PA.M2": _m("block_table_ignored"),
    "PA.M3": _m("positions_top_left"),
    "PA.M4": _m("window_no_right_edge"),
    "PA.M5": _m("window_strict"),
    "PA.M6": _m("softcap_ignored"),
}
READINGS = {}
