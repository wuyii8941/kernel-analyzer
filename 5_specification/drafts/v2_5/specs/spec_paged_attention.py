"""Class B (paged / variable-length KV attention) and the attention core reused by class C.

Sources and identity
  torch.nn.functional.scaled_dot_product_attention documentation (scale default 1/sqrt(E), GQA grouping) ... torch_doc
  GPT-OSS reference implementation (sinks: extra per-head logit column, dropped after softmax; window q - k < W) ... model_impl
  engine layout descriptions (block table, req_to_token, kv_indptr + kv_indices) ............................ engine_ref
Dimensions: D2 (scale, softcap, ALiBi slope), D3 (KV layout, GQA mapping), D6 (causal alignment with a cached prefix,
window endpoint, sink, softcap order), D8 (quantized KV: compare with attention on the dequantized cache), D11 (fully masked rows,
seq_len counting)."""
from _base import *

# ---------------------------------------------------------------- KV layouts -> ordered list of cache slots (D3)
def slots_block_table(block_table, seq_len, page_size):
    """token j of the request lives in page block_table[j // page_size], offset j % page_size."""
    return [(block_table[j // page_size], j % page_size) for j in range(seq_len)]

def slots_req_to_token(req_to_token_row, seq_len):
    return [req_to_token_row[j] for j in range(seq_len)]

def slots_indptr(kv_indptr, kv_indices, req):
    return list(kv_indices[kv_indptr[req]:kv_indptr[req + 1]])

def gather(cache, slots):
    """cache: dict slot -> row (exact values). Raises if a slot is missing or is the padding index -1."""
    rows = []
    for s in slots:
        if s == -1 or s not in cache: raise SpecInputError(f'slot {s} is padding or absent (D11)')
        rows.append(cache[s])
    return rows

def kv_head(h, n_q_heads, n_kv_heads, mapping='grouped'):
    """grouped: q head h uses kv head h // (Hq / Hkv) (SDPA enable_gqa); modulo: h % Hkv (another convention, must be declared)."""
    if n_q_heads % n_kv_heads: raise SpecInputError('Hq must be a multiple of Hkv')
    return h // (n_q_heads // n_kv_heads) if mapping == 'grouped' else h % n_kv_heads

# ---------------------------------------------------------------- exact attention for one head
def attention_rows(q_rows, k_rows, v_rows, q_pos=None, alignment='bottom_right', scale=None, causal=True,
                   window=None, window_rule='strict', softcap=None, sink=None, alibi_slope=None, empty_row='zero',
                   key_set=None):
    """rigorous attention. q_pos: absolute positions of the queries; default bottom_right: the last Lq of Lk positions
    (queries continue a cached prefix); top_left: positions 0..Lq-1. A key j is visible iff causal -> j <= p,
    window -> p - j < W (strict, GPT-OSS) or p - j <= W (inclusive), key_set -> j in key_set (sparse selection).
    Score s = scale * q.k, then softcap c*tanh(s/c), then ALiBi -slope*(p - j). Sink: extra logit in the softmax, dropped after.
    Returns per query (output row, logsumexp) as intervals; a row with no visible key gives zeros and lse = +inf
    (empty_row='zero') or raises (empty_row='undefined')."""
    Lq, Lk, d = len(q_rows), len(k_rows), len(q_rows[0])
    if q_pos is None:
        q_pos = [Lk - Lq + i for i in range(Lq)] if alignment == 'bottom_right' else list(range(Lq))
    sc = I(scale) if scale is not None else 1 / iv.sqrt(iv.mpf(d))
    out = []
    for i in range(Lq):
        p = q_pos[i]; vis = []
        for j in range(Lk):
            if causal and j > p: continue
            if window is not None and ((p - j >= window) if window_rule == 'strict' else (p - j > window)): continue
            if key_set is not None and j not in key_set: continue
            vis.append(j)
        if not vis and sink is None:
            if empty_row == 'undefined': raise SpecNotEstablished(f'query {i} sees no key (D11)')
            out.append(([I(0)] * len(v_rows[0]), iv.mpf('inf'))); continue
        logits = []
        for j in vis:
            s = sc * isum(I(q_rows[i][t]) * I(k_rows[j][t]) for t in range(d))
            if softcap is not None: s = I(softcap) * tanh(s / I(softcap))
            if alibi_slope is not None: s = s - I(alibi_slope) * (p - j)
            logits.append(s)
        if sink is not None: logits.append(I(sink))
        probs, logp = softmax_row(logits)
        m = max(v.b for v in logits); lse = m + iv.log(isum(iv.exp(v - m) for v in logits))
        o = [isum(probs[n] * I(v_rows[j][c]) for n, j in enumerate(vis)) for c in range(len(v_rows[0]))]
        out.append((o, lse))
    return out

def paged_attention(q_rows, cache_k, cache_v, slots, **kw):
    """class B reference: gather K and V by the layout's slots, then exact attention (bottom_right alignment by default)."""
    return attention_rows(q_rows, gather(cache_k, slots), gather(cache_v, slots), **kw)
