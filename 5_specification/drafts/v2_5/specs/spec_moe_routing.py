"""Class H (MoE routing, alignment, grouped GEMM) and the routing part of EP dispatch (K14).

Sources and identity
  Mixtral (softmax over experts -> top-k -> optional renormalization) ............ model_impl (HF)
  DeepSeek-V3: sigmoid -> + bias (selection only) -> group score = sum of top-2 per group -> top topk_group groups
    -> top-k experts in those groups -> weights = unbiased sigmoid scores normalized * routed_scaling_factor
    (technical report; FlashInfer documents the same pipeline) ................. paper + engine_ref
  Llama-4 (top-1 then sigmoid), sigmoid with / without renormalization ......... engine_ref (FlashInfer RoutingMethodType)
  moe_align_block_size (pad each expert's token list to a multiple of the block size; pad id = numel) ... engine_ref (vLLM)
Dimensions: D1 (scoring, where bias enters, normalization), D2 (routed_scaling_factor, eps), D3 (alignment layout),
D4 (ties and token order inside an expert: compared as sets), D11 (groups larger than needed, padding)."""
from _base import *

def scores_for(logits, method):
    if method in ('softmax_topk', 'softmax_topk_renorm'): return softmax_row(logits)[0]
    if method in ('deepseek_v3', 'sigmoid', 'sigmoid_renorm', 'llama4'): return [sigmoid(x) for x in logits]
    if method == 'topk_softmax': return [I(x) for x in logits]
    raise SpecInputError(method)

def route(logits, k, method='softmax_topk', bias=None, n_group=1, topk_group=1, routed_scaling_factor=F(1)):
    """returns the set of admissible (expert tuple, weight tuple) results; ties give several admissible results (D4)."""
    E = len(logits); s = scores_for(logits, method)
    sel_val = [s[e] + (I(bias[e]) if bias is not None else 0) for e in range(E)] if method == 'deepseek_v3' else s
    if method == 'llama4': sel_val = [I(x) for x in logits]
    candidates = [frozenset(range(E))]
    if method == 'deepseek_v3' and n_group > 1:
        gsz = E // n_group
        gscore = []
        for g in range(n_group):
            vals = sorted((sel_val[e] for e in range(g * gsz, (g + 1) * gsz)), key=lambda x: -mp.mpf(x.b))
            gscore.append(vals[0] + vals[1])                                   # sum of the top-2 (value is tie-independent)
        candidates = [frozenset(e for g in G for e in range(g * gsz, (g + 1) * gsz)) for G in topk_sets(gscore, topk_group)]
    results = set()
    for cand in candidates:
        cl = sorted(cand)
        for S in topk_sets([sel_val[e] for e in cl], k):
            ids = tuple(sorted(cl[i] for i in S))
            if method in ('softmax_topk',): w = [s[e] for e in ids]
            elif method in ('softmax_topk_renorm', 'sigmoid_renorm', 'deepseek_v3'):
                tot = isum(s[e] for e in ids); w = [s[e] / tot for e in ids]
                if method == 'deepseek_v3': w = [x * I(routed_scaling_factor) for x in w]
            elif method == 'topk_softmax': w = softmax_row([I(logits[e]) for e in ids])[0]
            elif method == 'sigmoid': w = [s[e] for e in ids]
            elif method == 'llama4': w = [sigmoid(logits[e]) for e in ids]
            results.add((ids, tuple(_key(x) for x in w)))
    return results

def _key(x):
    """hashable rounding-free summary of an interval for set comparison: the enclosure itself."""
    x = I(x); return (mp.nstr(mp.mpf(x.a), 40), mp.nstr(mp.mpf(x.b), 40))

def moe_align(topk_ids, num_experts, block):
    """per expert: the set of flat positions routed to it, padded length (multiple of block), one expert id per block,
    pad id = numel. Token order inside an expert is not specified (kernels use atomics), so it is compared as a set."""
    flat = [e for row in topk_ids for e in row]; numel = len(flat)
    per = {e: frozenset(i for i, x in enumerate(flat) if x == e) for e in range(num_experts)}
    padded = {e: -(-len(per[e]) // block) * block for e in per}
    expert_of_block = [e for e in range(num_experts) for _ in range(padded[e] // block)]
    return dict(sets=per, padded=padded, total=sum(padded.values()), expert_of_block=expert_of_block, pad_id=numel)

def fused_moe(x, ids, weights, W_gate, W_up, W_down):
    """y = sum_j w_j * W_down[e_j] (silu(W_gate[e_j] x) * (W_up[e_j] x)); the routing weight multiplies the expert output
    (applying it to the input of the first GEMM is a different function because of the activation)."""
    out = [I(0)] * len(W_down[0])
    for e, w in zip(ids, weights):
        g = [isum(I(W_gate[e][r][c]) * I(x[c]) for c in range(len(x))) for r in range(len(W_gate[e]))]
        u = [isum(I(W_up[e][r][c]) * I(x[c]) for c in range(len(x))) for r in range(len(W_up[e]))]
        h = [gi * sigmoid(gi) * ui for gi, ui in zip(g, u)]
        y = [isum(I(W_down[e][h_i][o]) * h[h_i] for h_i in range(len(h))) for o in range(len(W_down[e][0]))]
        out = [a + I(w) * b for a, b in zip(out, y)]
    return out
