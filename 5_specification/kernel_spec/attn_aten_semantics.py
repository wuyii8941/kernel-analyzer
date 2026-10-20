"""ATen-call semantics for the fused PyTorch 2.10 SDPA overloads in the ATTN contract (draft path A A2).

Covered overloads (schemas from the retained PyTorch 2.10 native_functions.yaml, TS_NATIVE_FUNCTIONS.SDPA_FUSED_SCHEMAS):

    aten._scaled_dot_product_flash_attention
    aten._scaled_dot_product_flash_attention_backward
    aten._scaled_dot_product_efficient_attention_backward

Only dropout_p == 0 is defined (decision AT.ATEN_RNG). The reference values are exact rationals or rigorous
kernel_spec.real.R enclosures. The backward overloads use the supplied `out` and `logsumexp` (decision
AT.ATEN_SAVED): P_ij = exp(scale*q_i.k_j - L_i), D_i = dO_i.O_i, dS_ij = P_ij (dO_i.v_j - D_i),
dq_i = scale sum_j dS_ij k_j, dk_j = scale sum_i dS_ij q_i, dv_j = sum_i P_ij dO_i, with key/value gradients summed
over every query head of the contiguous group that uses that key/value head (AT.ATEN_HEADS).

Input tensors are path-A tensor records {"t": dtype, "v": nested values, "shape": [...]} (shape optional in
hand-written examples). Floats are taken at their exact binary value.
"""
from __future__ import annotations

import math
from fractions import Fraction

from . import d1_attention as DA
from . import ieee
from . import real as rl
from .aten import dt
from .num import Invalid, Unsupported, rational
from .real import POS, R

FLASH = "aten._scaled_dot_product_flash_attention"
FLASH_BWD = "aten._scaled_dot_product_flash_attention_backward"
EFF_BWD = "aten._scaled_dot_product_efficient_attention_backward"
COVERS = (FLASH, FLASH_BWD, EFF_BWD)

FLASH_DTYPES = {"f16", "bf16"}
EFF_DTYPES = {"f16", "bf16", "f32"}
LSE_ALIGN = 32          # TS_META_SDPA.EFF_FWD_LSE / TS_MEMEFF_FWD_H.DROPOUT_RNG (kAlignLSE)

NAN = "nan"


# ------------------------------------------------------------------ tensor records

def _leaf(x, name):
    """Exact value of one element: Fraction, POS / NEG markers via strings, or NAN."""
    if isinstance(x, bool):
        raise Invalid(f"{name}: boolean element in a floating tensor")
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return NAN
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return Fraction(x)
    if isinstance(x, str):
        s = x.strip()
        if s in ("inf", "+inf"):
            return "inf"
        if s == "-inf":
            return "-inf"
        if s.lower() == "nan":
            return NAN
        return rational(s, name)
    raise Invalid(f"{name}: element {x!r} is not a number")


def _shape(v):
    if not isinstance(v, (list, tuple)):
        return ()
    if not v:
        return (0,)
    subs = [_shape(x) for x in v]
    if any(s != subs[0] for s in subs):
        raise Invalid("ragged tensor values")
    return (len(v),) + subs[0]


def _map(v, fn):
    if isinstance(v, (list, tuple)):
        return [_map(x, fn) for x in v]
    return fn(v)


def is_tensor_record(x):
    return isinstance(x, dict) and "t" in x and "v" in x


def tensor(rec, name, *, dtypes=None, rank=None):
    """(dtype, shape, nested exact values) of a tensor record; no element finiteness is required here."""
    if not is_tensor_record(rec):
        raise Invalid(f"{name} must be a tensor record")
    if set(rec) - {"t", "v", "shape"}:
        raise Unsupported(f"{name}: tensor record contains layout or metadata fields")
    dtype = dt(rec["t"])
    if dtypes is not None and dtype not in dtypes:
        raise Unsupported(f"{name} dtype {dtype} is outside {sorted(dtypes)} for this overload")
    values = rec["v"]
    shape = _shape(values)
    if "shape" in rec:
        declared = rec["shape"]
        if not isinstance(declared, (list, tuple)) or any(type(n) is not int or n < 0 for n in declared):
            raise Invalid(f"{name}: invalid shape {declared!r}")
        declared = tuple(declared)
        if declared != shape and not (math.prod(declared) == 0 and values == []):
            raise Invalid(f"{name}: values have shape {shape}, record declares {declared}")
        shape = declared
    if rank is not None and len(shape) != rank:
        raise Invalid(f"{name} must have rank {rank} (got shape {shape})")
    return dtype, shape, _map(values, lambda x: _leaf(x, name))


def _walk(v):
    if isinstance(v, list):
        for x in v:
            yield from _walk(x)
    else:
        yield v


def _finite_exact(values, dtype, name):
    for x in _walk(values):
        if not isinstance(x, Fraction):
            raise Unsupported(f"{name} contains a non-finite element; the fused SDPA profile is finite")
        r = ieee.round_to(R(x), dtype)
        if not r.exact or r.v != x:
            raise Invalid(f"{name} element {x} is not exactly representable as {dtype}")


def _bool(x, name):
    if type(x) is not bool:
        raise Invalid(f"{name} must be a boolean")
    return x


def _int(x, name):
    if type(x) is not int:
        raise Invalid(f"{name} must be an integer")
    return x


def _ignored_rng(x, name):
    """philox seed / offset inputs: ignored when dropout_p == 0, but must be a tensor record or None."""
    if x is not None and not is_tensor_record(x):
        raise Invalid(f"{name} must be a tensor or None")


# ------------------------------------------------------------------ common parse

def parse_qkv(op, query, key, value, dropout_p, is_causal, scale, *, dtypes, gqa, allow_dropout=False,
              allow_nonsquare_causal=False):
    qd, qs, q = tensor(query, "query", dtypes=dtypes, rank=4)
    kd, ks, k = tensor(key, "key", dtypes=dtypes, rank=4)
    vd, vs, v = tensor(value, "value", dtypes=dtypes, rank=4)
    if not qd == kd == vd:
        raise Unsupported("query, key and value dtypes differ; the fused overloads take one dtype")
    if any(n <= 0 for n in (*qs, *ks, *vs)):
        raise Unsupported("empty dimensions are outside the fused SDPA profile")
    for name, vals in (("query", q), ("key", k), ("value", v)):
        _finite_exact(vals, qd, name)
    B, Hq, L, E = qs
    if ks[0] != B or vs[0] != B:
        raise Invalid("batch sizes of query, key and value differ")
    Hkv, S = ks[1], ks[2]
    if ks[3] != E:
        raise Invalid("query and key head dimensions differ")
    if vs[1] != Hkv or vs[2] != S:
        raise Invalid("key and value head counts and sequence lengths must match")
    if gqa:
        if Hq % Hkv:
            raise Invalid("the query head count must be divisible by the key/value head count")
    elif Hq != Hkv:
        raise Unsupported("the efficient overload profile requires equal query and key/value head counts")
    p = rational(dropout_p, "dropout_p")
    if not 0 <= p <= 1:
        raise Invalid("dropout_p must lie in [0, 1]")
    if p != 0 and not allow_dropout:
        raise Unsupported("dropout_p > 0: the result depends on an RNG stream that no documented source specifies "
                          "(decision AT.ATEN_RNG)")
    causal = _bool(is_causal, "is_causal")
    if causal and L != S and not allow_nonsquare_causal:
        raise Unsupported("is_causal with seqlen_q != seqlen_k: the fused kernels' alignment is not the documented "
                          "public one (decision AT.ATEN_CAUSAL)")
    if scale is None:
        sc = DA.default_scale(E)
    else:
        if isinstance(scale, bool):
            raise Invalid("scale must be a real number or None")
        sc = R(rational(scale, "scale"))
    return {"op": op, "dtype": qd, "q": q, "k": k, "v": v, "B": B, "Hq": Hq, "Hkv": Hkv, "L": L, "S": S, "E": E,
            "Ev": vs[3], "qshape": qs, "kshape": ks, "vshape": vs, "causal": causal, "scale": sc,
            "scale_given": scale is not None, "dropout_p": p}


def allowed_fn(parsed):
    causal = parsed["causal"]
    return lambda b, h, i, j: (j <= i) if causal else True


def _zero_bias(b, h, i, j):
    return Fraction(0)


# ------------------------------------------------------------------ per-overload parse

def _flash_value_dim(parsed):
    """The flash overloads return attention = empty_like(query) (TS_META_SDPA.FLASH_FWD): value must have the
    query/key head dimension (AT.ATEN_DOMAIN). The efficient backward keeps Ev != E."""
    if parsed["Ev"] != parsed["E"]:
        raise Invalid(f"flash overloads require the value head dimension ({parsed['Ev']}) to equal the query/key "
                      f"head dimension ({parsed['E']}): the output is shaped like query")


def parse_flash(query, key, value, dropout_p=0.0, is_causal=False, return_debug_mask=False, *, scale=None,
                allow_value_dim=False, **kw):
    parsed = parse_qkv(FLASH, query, key, value, dropout_p, is_causal, scale, dtypes=FLASH_DTYPES, gqa=True, **kw)
    if not allow_value_dim:
        _flash_value_dim(parsed)
    if _bool(return_debug_mask, "return_debug_mask"):
        raise Unsupported("return_debug_mask=True: the debug mask content is backend-specific and PyTorch 2.10 "
                          "rejects it without dropout")
    return parsed


def _saved(parsed, grad_out, out, logsumexp, *, padded_lse):
    B, Hq, L, Ev = parsed["B"], parsed["Hq"], parsed["L"], parsed["Ev"]
    dtype = parsed["dtype"]
    gd, gs, g = tensor(grad_out, "grad_out", rank=4)
    od, os_, o = tensor(out, "out", rank=4)
    if gd != dtype or od != dtype:
        raise Unsupported("grad_out and out must have the dtype of query")
    if gs != (B, Hq, L, Ev):
        raise Invalid(f"grad_out shape {gs} differs from the forward output shape {(B, Hq, L, Ev)}")
    if os_ != (B, Hq, L, Ev):
        raise Invalid(f"out shape {os_} differs from the forward output shape {(B, Hq, L, Ev)}")
    _finite_exact(g, dtype, "grad_out")
    _finite_exact(o, dtype, "out")
    ld, ls, lse = tensor(logsumexp, "logsumexp", rank=3)
    if ld != "f32":
        raise Unsupported("logsumexp must be float32")
    widths = {L, -(-L // LSE_ALIGN) * LSE_ALIGN} if padded_lse else {L}
    if ls[:2] != (B, Hq) or ls[2] not in widths:
        raise Unsupported(f"logsumexp shape {ls} is not (B, H, L){' or the 32-aligned padded form' if padded_lse else ''}")
    rows = []
    for b in range(B):
        rows.append([])
        for h in range(Hq):
            row = lse[b][h]
            head = row[:L]
            _finite_exact(head, "f32", "logsumexp")
            if any(x != "inf" for x in row[L:]):
                raise Unsupported("logsumexp padding entries other than +inf are outside the profile")
            rows[-1].append(head)
    parsed.update(dout=g, out=o, lse=rows)
    return parsed


def parse_flash_bwd(grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p,
                    is_causal, philox_seed, philox_offset, *, scale=None, allow_value_dim=False, **kw):
    parsed = parse_qkv(FLASH_BWD, query, key, value, dropout_p, is_causal, scale, dtypes=FLASH_DTYPES, gqa=True, **kw)
    if not allow_value_dim:
        _flash_value_dim(parsed)
    if cum_seq_q is not None or cum_seq_k is not None:
        raise Unsupported("cumulative sequence tensors (packed variable-length inputs) are outside the profile")
    if _int(max_q, "max_q") != parsed["L"] or _int(max_k, "max_k") != parsed["S"]:
        raise Unsupported("max_q / max_k differ from the dense sequence lengths returned by the forward")
    _ignored_rng(philox_seed, "philox_seed")
    _ignored_rng(philox_offset, "philox_offset")
    return _saved(parsed, grad_out, out, logsumexp, padded_lse=False)


def parse_eff_bwd(grad_out_, query, key, value, attn_bias, out, logsumexp, philox_seed, philox_offset, dropout_p,
                  grad_input_mask, is_causal=False, *, scale=None, **kw):
    parsed = parse_qkv(EFF_BWD, query, key, value, dropout_p, is_causal, scale, dtypes=EFF_DTYPES, gqa=False, **kw)
    if attn_bias is not None:
        raise Unsupported("an attention bias is outside the profile (recorded calls pass None)")
    if not isinstance(grad_input_mask, (list, tuple)) or len(grad_input_mask) != 4 or \
            any(type(x) is not bool for x in grad_input_mask):
        raise Invalid("grad_input_mask must be four booleans")
    if not all(grad_input_mask[:3]):
        raise Unsupported("grad_input_mask with an unrequested query/key/value gradient is outside the profile")
    if grad_input_mask[3]:
        raise Unsupported("a bias gradient is requested although attn_bias is None")
    _ignored_rng(philox_seed, "philox_seed")
    _ignored_rng(philox_offset, "philox_offset")
    return _saved(parsed, grad_out_, out, logsumexp, padded_lse=True)


# ------------------------------------------------------------------ reference arithmetic

def _dot(a, b):
    s = Fraction(0)
    for x, y in zip(a, b):
        s += x * y
    return s


def forward(parsed):
    """(out, lse) as nested lists of R."""
    res = DA.attention(parsed["q"], parsed["k"], parsed["v"], allowed=allowed_fn(parsed), bias=_zero_bias,
                       scale=parsed["scale"], empty="undefined")
    return res["out"], res["lse"]


def saved_backward(parsed):
    """(dq, dk, dv) from the supplied out and logsumexp (AT.ATEN_SAVED), nested lists of R."""
    q, k, v, dout, o, lse = (parsed[n] for n in ("q", "k", "v", "dout", "out", "lse"))
    B, Hq, Hkv, L, S, E, Ev = (parsed[n] for n in ("B", "Hq", "Hkv", "L", "S", "E", "Ev"))
    scale, allowed = parsed["scale"], allowed_fn(parsed)
    group = Hq // Hkv
    zero = R(0)
    dq = [[[[zero] * E for _ in range(L)] for _ in range(Hq)] for _ in range(B)]
    dk = [[[[zero] * E for _ in range(S)] for _ in range(Hkv)] for _ in range(B)]
    dv = [[[[zero] * Ev for _ in range(S)] for _ in range(Hkv)] for _ in range(B)]
    for b in range(B):
        for h in range(Hq):
            g = h // group
            for i in range(L):
                do = dout[b][h][i]
                D = _dot(do, o[b][h][i])
                Li = R(lse[b][h][i])
                for j in range(S):
                    if not allowed(b, h, i, j):
                        continue
                    p = rl.exp(scale * R(_dot(q[b][h][i], k[b][g][j])) - Li)
                    ds = p * R(_dot(do, v[b][g][j]) - D) * scale
                    for d in range(Ev):
                        dv[b][g][j][d] = dv[b][g][j][d] + p * R(do[d])
                    for e in range(E):
                        dq[b][h][i][e] = dq[b][h][i][e] + ds * R(k[b][g][j][e])
                        dk[b][g][j][e] = dk[b][g][j][e] + ds * R(q[b][h][i][e])
    return dq, dk, dv


def recomputed_backward(parsed):
    """Diagnostic reading: the exact VJP of the forward at (q, k, v), ignoring the supplied out and logsumexp."""
    res = DA.attention(parsed["q"], parsed["k"], parsed["v"], allowed=allowed_fn(parsed), bias=_zero_bias,
                       scale=parsed["scale"], empty="undefined", dout=parsed["dout"])
    return res["dq"], res["dk"], res["dv"]


def values(x):
    return DA.values(x)


# ------------------------------------------------------------------ entry points (ATen schema names)

def flash(query, key, value, dropout_p=0.0, is_causal=False, return_debug_mask=False, *, scale=None):
    parsed = parse_flash(query, key, value, dropout_p, is_causal, return_debug_mask, scale=scale)
    out, lse = forward(parsed)
    # positions 2, 3, 6, 7, 8 are not compared (AT.ATEN_AUX); 4 and 5 are the dense lengths
    return [values(out), values(lse), None, None, parsed["L"], parsed["S"], None, None, None]


def flash_backward(grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p,
                   is_causal, philox_seed, philox_offset, *, scale=None):
    parsed = parse_flash_bwd(grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
                             dropout_p, is_causal, philox_seed, philox_offset, scale=scale)
    return [values(x) for x in saved_backward(parsed)]


def efficient_backward(grad_out_, query, key, value, attn_bias, out, logsumexp, philox_seed, philox_offset,
                       dropout_p, grad_input_mask, is_causal=False, *, scale=None):
    parsed = parse_eff_bwd(grad_out_, query, key, value, attn_bias, out, logsumexp, philox_seed, philox_offset,
                           dropout_p, grad_input_mask, is_causal, scale=scale)
    return [values(x) for x in saved_backward(parsed)] + [None]


OPS = {FLASH: flash, FLASH_BWD: flash_backward, EFF_BWD: efficient_backward}

# Output dtypes and shapes of each returned position (for complete-output comparison).
def output_layout(op, parsed):
    B, Hq, Hkv, L, S, E, Ev, dtype = (parsed[n] for n in ("B", "Hq", "Hkv", "L", "S", "E", "Ev", "dtype"))
    if op == FLASH:
        return [(dtype, (B, Hq, L, Ev)), ("f32", (B, Hq, L)), None, None, "int", "int", None, None, None]
    grads = [(dtype, (B, Hq, L, E)), (dtype, (B, Hkv, S, E)), (dtype, (B, Hkv, S, Ev))]
    return grads if op == FLASH_BWD else grads + [None]


PARSERS = {FLASH: parse_flash, FLASH_BWD: parse_flash_bwd, EFF_BWD: parse_eff_bwd}
POS_INF = POS
