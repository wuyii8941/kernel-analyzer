"""Class C (sparse attention, indexers, KV compression) - selection part; attention itself reuses spec_paged_attention.

Sources and identity
  DeepSeek-V3.2 (DSA): index score I_{t,s} = sum_j w_{t,j} * ReLU(q_{t,j} . k_s) over preceding tokens, then top-k ... paper
  NSA (native sparse attention): block scores, forced initial / local blocks, top-n blocks ..................... paper
Dimensions: D1 (block aggregation: sum / mean / max / logsumexp; forced blocks), D3 (index -1 = no entry; [ks, ke) vs
[ks, ke]), D4 (ties -> set of admissible selections), D6 (only preceding positions)."""
from _base import *

def indexer_scores(q_heads, w, keys, t):
    """exact for rational inputs: I_{t,s} = sum_j w_j * max(0, q_j . k_s), s = 0..t."""
    return [sum(Q(w[j]) * max(F(0), sum(Q(a) * Q(b) for a, b in zip(q_heads[j], keys[s]))) for j in range(len(q_heads))) for s in range(t + 1)]

def select_topk(scores, k, pad=-1):
    """admissible selections, each as a sorted tuple padded with -1 when fewer than k candidates exist."""
    out = set()
    for S in topk_sets(scores, k):
        ids = sorted(S); out.add(tuple(ids + [pad] * (k - len(ids))))
    return out

def block_scores(token_scores, block, agg='sum'):
    blocks = [token_scores[i:i + block] for i in range(0, len(token_scores), block)]
    if agg == 'sum': return [sum(map(Q, b)) for b in blocks]
    if agg == 'mean': return [sum(map(Q, b)) / len(b) for b in blocks]
    if agg == 'max': return [max(map(Q, b)) for b in blocks]
    if agg == 'lse': return [softmax_row(b)[1] and iv.log(isum(iv.exp(I(x)) for x in b)) for b in blocks]
    raise SpecInputError(agg)

def select_blocks(bscores, n, forced=()):
    """forced blocks are always kept; the remaining n - len(forced) are admissible top selections of the others."""
    rest = [i for i in range(len(bscores)) if i not in forced]
    out = set()
    for S in topk_sets([bscores[i] for i in rest], max(0, n - len(forced))):
        out.add(tuple(sorted(set(forced) | {rest[i] for i in S})))
    return out

def key_range(ks, ke, rule='half_open'):
    return list(range(ks, ke)) if rule == 'half_open' else list(range(ks, ke + 1))
