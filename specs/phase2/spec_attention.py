"""Independent specification f for attention, sequence packing and RoPE.  (v0.1)

Documentation (fetched 2026-10-08):
  SDPA : https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html
         "scale_factor = 1 / sqrt(query.size(-1)) if scale is None else scale"; boolean attn_mask: "a value of True
         indicates that the element should take part in attention"; float attn_mask is added to the scores;
         is_causal: "the attention masking is a lower triangular matrix when the mask is a square matrix. The attention
         masking has the form of the upper left causal bias due to the alignment ... when the mask is a non-square
         matrix"; "An error is thrown if both attn_mask and is_causal are set"; enable_gqa repeats key/value heads.
  FlashAttention's documented causal convention for q_len != k_len is bottom-right alignment (FA2 README) - a DIFFERENT
  declared convention, recorded as ATT-C1; a library is compared against the convention it declares.
  softcap: s' = softcap * tanh(s / softcap) applied to the scaled scores before the mask (declared by the libraries that
  offer it: FlashAttention, Liger); not part of SDPA.
  RoPE   : Su et al. 2021 (RoFormer); two layouts in common use: "rotate-half" (pairs (i, i + d/2), HF) and
           "interleaved" (pairs (2i, 2i+1), GPT-J/original). theta_i = base^(-2i/d).

Ambiguities / conventions (ambiguities_phase2.md):
  ATT-A1 a query row with every key masked out: softmax over an empty set is undefined; implementations return NaN or
         zeros - recorded as a convention, not judged.
  ATT-C1 causal alignment for q_len != k_len: SDPA documents upper-left; FlashAttention documents bottom-right. Each
         candidate is compared against its own declared convention; mixing conventions inside one model is the error.
  ATT-C2 GQA: query head h attends key/value head h // (H_q / H_kv) (SDPA enable_gqa; also the HF convention).
  ATT-A2 sliding window: whether the window includes exactly `window` past tokens or `window - 1` is a per-library
         declaration; the spec takes the window as "keys j with i - j < window" and records the declared variant.
  ROPE-C1 rotate-half vs interleaved are declared conventions; the position offset (0- or 1-based, start position) is
         part of the declaration.
Numerics: scores and the weighted sum are exact rationals; softmax, tanh, cos and sin are rigorous intervals.
"""
from fractions import Fraction as F
import rigorous_mp as R
from rigorous_mp import I, iv, SpecInputError, SpecNotEstablished


def causal_allowed(i, j, Lq, Lk, alignment="upper_left"):
    """whether query i may attend key j."""
    if alignment == "upper_left":
        return j <= i
    if alignment == "bottom_right":
        return j <= i + (Lk - Lq)
    raise SpecInputError("alignment must be upper_left or bottom_right")


def attention(q, k, v, attn_mask=None, is_causal=False, scale=None, alignment="upper_left", softcap=None,
              window=None, allow_empty_row="decline"):
    """Single head. q: [Lq][E], k,v: [Lk][E] (v may have a different last dim). attn_mask: None, or [Lq][Lk] of
    bools (True = attend) or numbers (added to the scaled scores). Returns (out as intervals [Lq][Ev], P as intervals)."""
    Lq, Lk = len(q), len(k)
    if is_causal and attn_mask is not None:
        raise SpecInputError("attn_mask and is_causal cannot both be set (documented)")
    E = len(q[0])
    sc = I(scale) if scale is not None else 1 / iv.sqrt(iv.mpf(E))
    out, P = [], []
    for i in range(Lq):
        scores, allowed = [], []
        for j in range(Lk):
            ok = True
            if is_causal:
                ok = causal_allowed(i, j, Lq, Lk, alignment)
            if window is not None and not (0 <= i - j < window):
                ok = False
            bias = F(0)
            if attn_mask is not None:
                m = attn_mask[i][j]
                if isinstance(m, bool):
                    ok = ok and m
                else:
                    if m == float("-inf"):
                        ok = False
                    else:
                        bias = F(m)
            if not ok:
                continue
            s = I(sum((F(q[i][e]) * F(k[j][e]) for e in range(E)), F(0))) * sc
            if softcap is not None:
                s = I(softcap) * R.tanh(s / I(softcap))
            scores.append(s + I(bias)); allowed.append(j)
        if not allowed:
            if allow_empty_row == "decline":
                raise SpecNotEstablished(f"query row {i}: every key masked (ATT-A1)")
            out.append([iv.mpf(0)] * len(v[0])); P.append([iv.mpf(0)] * Lk); continue
        p, _ = R.softmax_row(scores)
        row_p = [iv.mpf(0)] * Lk
        for pj, j in zip(p, allowed):
            row_p[j] = pj
        P.append(row_p)
        out.append([R.isum(row_p[j] * I(v[j][d]) for j in allowed) for d in range(len(v[0]))])
    return out, P


def gqa_kv_head(h_q, H_q, H_kv):
    if H_q % H_kv != 0:
        raise SpecInputError("H_q must be a multiple of H_kv")
    return h_q // (H_q // H_kv)


# ------------------------------------------------------------------ sequence packing
def packed_mask(doc_lengths, causal=True):
    """block-diagonal (and causal within each document) boolean mask for a packed sequence."""
    L = sum(doc_lengths)
    doc = []
    for d, n in enumerate(doc_lengths):
        doc += [d] * n
    return [[(doc[i] == doc[j]) and (j <= i if causal else True) for j in range(L)] for i in range(L)]


def packed_position_ids(doc_lengths, start=0):
    """positions restart at `start` at every document boundary."""
    out = []
    for n in doc_lengths:
        out += list(range(start, start + n))
    return out


def prop_no_cross_document_leak(P, doc_lengths):
    """attention weights between different documents must be exactly zero."""
    doc = []
    for d, n in enumerate(doc_lengths):
        doc += [d] * n
    return [(i, j) for i in range(len(P)) for j in range(len(P[0])) if doc[i] != doc[j] and not (P[i][j].a == 0 == P[i][j].b)]


# ------------------------------------------------------------------ RoPE
def rope_inv_freq(d, base=10000):
    """theta_i = base^(-2i/d), evaluated rigorously as exp(-(2i/d) * ln(base))."""
    lb = iv.log(I(base))
    return [iv.exp(-I(F(2 * i, d)) * lb) for i in range(d // 2)]


def rope_apply(x, pos, d, base=10000, layout="rotate_half"):
    """x: vector of length d (even); pos: integer position. Returns list of intervals."""
    if d % 2:
        raise SpecInputError("head dim must be even")
    invf = rope_inv_freq(d, base)
    x = [I(v) for v in x]
    out = [None] * d
    for i in range(d // 2):
        ang = invf[i] * pos
        c, s = iv.cos(ang), iv.sin(ang)
        if layout == "rotate_half":
            a, b = i, i + d // 2
        elif layout == "interleaved":
            a, b = 2 * i, 2 * i + 1
        else:
            raise SpecInputError("layout must be rotate_half or interleaved")
        out[a] = x[a] * c - x[b] * s
        out[b] = x[b] * c + x[a] * s
    return out


def prop_rope_norm_preserving(x, y):
    """|rope(x)| == |x| (rotation), as overlapping intervals of the squared norms."""
    nx = R.isum(I(v) * I(v) for v in x)
    ny = R.isum(v * v for v in y)
    return nx, ny


def prop_rope_relative(q, k, m, n, d, base=10000, layout="rotate_half"):
    """<rope(q, m), rope(k, n)> depends only on m - n: compare with positions shifted by the same offset."""
    a = R.isum(u * w for u, w in zip(rope_apply(q, m, d, base, layout), rope_apply(k, n, d, base, layout)))
    b = R.isum(u * w for u, w in zip(rope_apply(q, m + 7, d, base, layout), rope_apply(k, n + 7, d, base, layout)))
    return a, b
