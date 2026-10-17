"""Adapter for contracts/ATTN.yaml: scaled dot-product attention (SDPA, flex attention with the tested score/mask
modifications, FlashAttention conventions, the Triton tutorial's M) with outputs out, lse and the VJP (dq, dk, dv).

The reference is kernel_spec.d1_attention (exact rationals, rigorous enclosures, +-inf). The mutants call the
independent interval implementation kernel_spec.d1_indep with the misreading as a parameter.
"""
from __future__ import annotations

from fractions import Fraction

from mpmath import iv

from kernel_spec import d1_attention as A
from kernel_spec import d1_indep as D
from kernel_spec import real as rl
from kernel_spec.num import Invalid, Result, Unsupported, rational
from kernel_spec.real import NEG, R

# profile -> (causal / window alignment, rows without an allowed key, base of lse)
PROFILES = {"sdpa": ("top_left", "undefined", "e"), "sdpa_zero_rows": ("top_left", "zero", "e"),
            "lower_right": ("bottom_right", "undefined", "e"),
            "lower_right_zero_rows": ("bottom_right", "zero", "e"),
            "sdpa_lse_base2": ("top_left", "undefined", "2")}
KEYS = {"q", "k", "v", "dout", "attn_mask", "is_causal", "scale", "softcap", "window", "enable_gqa", "dropout_p"}


def parse(profile, inp):
    if profile not in PROFILES:
        raise Unsupported(f"profile {profile!r}")
    extra = set(inp) - KEYS
    if extra:
        raise Invalid(f"unknown input fields {sorted(extra)}")
    align, empty, base = PROFILES[profile]
    rank = len(A.shape_of(inp["q"], "q"))
    if rank not in (3, 4):
        raise Invalid("q must be [H][L][E] or [B][H][L][E]")
    q, qs = A.tensor(inp["q"], rank, "q")
    k, ks = A.tensor(inp["k"], rank, "k")
    v, vs = A.tensor(inp["v"], rank, "v")
    if rank == 3:
        q, k, v, qs, ks, vs = [q], [k], [v], (1,) + qs, (1,) + ks, (1,) + vs
    B, Hq, L, E = qs
    if ks[0] != B or vs[0] != B:
        raise Invalid("batch sizes of q, k, v differ")
    Hkv, S = ks[1], ks[2]
    if vs[1] != Hkv or vs[2] != S:
        raise Invalid("key and value must have the same heads and length")
    if ks[3] != E:
        raise Invalid("query and key head dimensions differ")
    Ev = vs[3]
    gqa = inp.get("enable_gqa", False)
    if not isinstance(gqa, bool):
        raise Invalid("enable_gqa must be a boolean")
    if Hkv == 0 or (Hkv != Hq and (Hq % Hkv if gqa else Hkv != 1)):
        raise Invalid(f"{Hq} query heads and {Hkv} key/value heads" + ("" if gqa else " without enable_gqa"))
    dout, ds = A.tensor(inp["dout"], rank, "dout")
    if rank == 3:
        dout, ds = [dout], (1,) + ds
    if ds != (B, Hq, L, Ev):
        raise Invalid(f"dout shape {ds[1:] if rank == 3 else ds} does not match out")
    p = rational(inp.get("dropout_p", 0), "dropout_p")
    if not 0 <= p <= 1:
        raise Invalid("dropout_p outside [0, 1]")
    if p != 0:
        raise Unsupported("dropout_p > 0: random (outside these profiles)")
    causal = inp.get("is_causal", False)
    if not isinstance(causal, bool):
        raise Invalid("is_causal must be a boolean")
    mask = inp.get("attn_mask")
    if causal and mask is not None:
        raise Invalid("attn_mask and is_causal are both set (documented error)")
    win = inp.get("window")
    if win is not None:
        if align == "top_left":
            raise Unsupported("window_size is the FlashAttention convention (bottom-right profiles only)")
        win = A.window_pair(win)
    cap = inp.get("softcap")
    cap = None if cap is None else rational(cap, "softcap")
    if cap is not None and cap < 0:
        raise Invalid("softcap must be >= 0 (0 = off)")
    cap = None if not cap else R(cap)
    scale = A.default_scale(E) if inp.get("scale") is None else R(rational(inp["scale"], "scale"))
    target = (B, Hq, L, S) if rank == 4 else (Hq, L, S)
    getm, is_bool = None, False
    if mask is not None:
        mv, msh, is_bool = A.mask_tensor(mask, "attn_mask")
        getm = A.broadcast_get(mv, msh, target)
    off = S - L if align == "bottom_right" else 0

    def idx(b, h, i, j):
        return (b, h, i, j) if rank == 4 else (h, i, j)

    def allowed(b, h, i, j):
        if causal and j > i + off:
            return False
        if win is not None and not A.in_window(i + off, j, *win):
            return False
        if getm is not None and is_bool and not getm(idx(b, h, i, j)):
            return False
        return True

    def bias(b, h, i, j):
        return getm(idx(b, h, i, j)) if getm is not None and not is_bool else Fraction(0)

    return dict(q=q, k=k, v=v, dout=dout, rank=rank, allowed=allowed, bias=bias, scale=scale, softcap=cap,
                empty=empty, base=base, causal=causal, off=off, win=win, getm=getm, is_bool=is_bool, idx=idx,
                E=E, Hq=Hq, Hkv=Hkv, L=L, S=S)


def _strip(res, rank):
    return {k: (A.values(v) if rank == 4 else A.values(v)[0]) for k, v in res.items()}


def reference(profile, inp):
    a = parse(profile, inp)
    res = A.attention(a["q"], a["k"], a["v"], allowed=a["allowed"], bias=a["bias"], scale=a["scale"],
                      softcap=a["softcap"], empty=a["empty"], dout=a["dout"])
    if a["base"] == "2":
        ln2 = rl.log(R(2))
        res["lse"] = [[[x / ln2 for x in row] for row in hb] for hb in res["lse"]]
    return Result("value", _strip(res, a["rank"]))


# ------------------------------------------------------------------ mutants: independent implementation
def _iv(x):
    if isinstance(x, R):
        x = x.v
    return D.I(x) if isinstance(x, (Fraction, int)) else x


def _indep(profile, inp, m=None):
    a = parse(profile, inp)
    allowed, bias, empty = a["allowed"], a["bias"], a["empty"]
    scale = _iv(a["scale"])
    if m == "scale_one_over_E":
        scale = D.I(Fraction(1, a["E"]))
    if m == "explicit_scale_ignored":
        scale = _iv(A.default_scale(a["E"]))
    getm, idx, off, causal, win = a["getm"], a["idx"], a["off"], a["causal"], a["win"]
    if m == "bool_mask_inverted" and getm is not None and a["is_bool"]:
        allowed = lambda b, h, i, j: not getm(idx(b, h, i, j))           # noqa: E731
    if m == "float_mask_keep_eq_1" and getm is not None and not a["is_bool"]:
        allowed = lambda b, h, i, j: getm(idx(b, h, i, j)) == 1          # noqa: E731
        bias = lambda b, h, i, j: Fraction(0)                             # noqa: E731
    if m in ("causal_bottom_right", "causal_top_left", "causal_no_diagonal") and causal:
        sh = {"causal_bottom_right": a["S"] - a["L"], "causal_top_left": 0, "causal_no_diagonal": off}[m]
        strict = m == "causal_no_diagonal"
        allowed = lambda b, h, i, j: (j < i + sh) if strict else (j <= i + sh)   # noqa: E731
    if m == "window_query_local" and win is not None:
        allowed = lambda b, h, i, j: A.in_window(i, j, *win)              # noqa: E731
    kv_head = None
    if m == "gqa_tiled":
        kv_head = lambda h: h % a["Hkv"]                                   # noqa: E731
    if m == "empty_row_uniform":
        empty = "uniform"
    bwd_allowed = None
    if m == "backward_causal" and not causal:
        bwd_allowed = lambda b, h, i, j: allowed(b, h, i, j) and j <= i + off   # noqa: E731
    res = D.iv_attention(a["q"], a["k"], a["v"], allowed=allowed, bias=bias, scale=scale,
                         softcap=None if a["softcap"] is None else a["softcap"].v,
                         softcap_on_raw=(m == "softcap_on_raw"), empty=empty, dout=a["dout"], kv_head=kv_head,
                         bwd_allowed=bwd_allowed, group_sum=(m != "gqa_no_group_sum"))
    if a["base"] == "2" and m != "lse_natural_for_base2":
        ln2 = iv.log(2)
        res["lse"] = [[[x if isinstance(x, R) else x / ln2 for x in row] for row in hb] for hb in res["lse"]]
    return Result("value", {k: (v if a["rank"] == 4 else v[0]) for k, v in res.items()})


def _causal_and_mask(profile, inp):
    """Misreading: is_causal together with attn_mask is accepted and both are applied."""
    if inp.get("is_causal") and inp.get("attn_mask") is not None:
        inp = dict(inp, is_causal=False)
        a = parse(profile, inp)
        off, al = a["S"] - a["L"] if PROFILES[profile][0] == "bottom_right" else 0, a["allowed"]
        res = D.iv_attention(a["q"], a["k"], a["v"], allowed=lambda b, h, i, j: al(b, h, i, j) and j <= i + off,
                             bias=a["bias"], scale=_iv(a["scale"]), empty=a["empty"], dout=a["dout"])
        return Result("value", {k: (v if a["rank"] == 4 else v[0]) for k, v in res.items()})
    return _indep(profile, inp)


def _m(kind):
    return lambda profile, inp: _indep(profile, inp, kind)


MUTANTS = {
    "AT.M1": _m("scale_one_over_E"),
    "AT.M2": _m("explicit_scale_ignored"),
    "AT.M3": _m("bool_mask_inverted"),
    "AT.M4": _m("float_mask_keep_eq_1"),
    "AT.M5": _m("causal_bottom_right"),
    "AT.M6": _m("causal_top_left"),
    "AT.M7": _m("window_query_local"),
    "AT.M8": _m("gqa_tiled"),
    "AT.M9": _m("softcap_on_raw"),
    "AT.M10": _m("empty_row_uniform"),
    "AT.M11": _m("causal_no_diagonal"),
    "AT.M12": _m("backward_causal"),
    "AT.M13": _m("gqa_no_group_sum"),
    "AT.M14": _m("lse_natural_for_base2"),
    "AT.M15": _causal_and_mask,
}
READINGS = {}
