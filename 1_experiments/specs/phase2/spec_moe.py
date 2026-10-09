"""Independent specification f for a small Mixture-of-Experts layer (single device).  (v0.1)

Sources (declared formulas, not a single PyTorch API):
  Switch Transformer (Fedus et al. 2021): router p = softmax(W x); top-1; capacity factor; load-balancing loss
      L_aux = alpha * E * sum_e f_e * P_e, with f_e the fraction of tokens dispatched to expert e and P_e the mean router
      probability of expert e over the batch.
  Mixtral (Jiang et al. 2024): top-k over softmax(router logits), weights renormalised over the selected experts.
  Megatron-Core MoE layer docs (router -> dispatch (permute) -> expert compute -> combine (unpermute)); capacity handling.
  (URLs recorded in ambiguities_phase2.md; this spec encodes the formulas as declared by each reading.)

Readings (every candidate must declare which it implements; mixing is the error):
  MOE-C1 renormalisation: weights are softmax probabilities of the selected experts renormalised to sum to 1 (Mixtral),
         or the raw probabilities without renormalisation (Switch top-1 trivially sums to 1).
  MOE-C2 top-k ties: documented nowhere; the spec returns the set of valid top-k selections and accepts any member.
  MOE-C3 capacity: tokens beyond capacity for an expert are dropped (their contribution is 0) in order of appearance,
         or no dropping (dropless). Capacity = ceil(capacity_factor * tokens / E) when declared.
  MOE-C4 aux loss: Switch form above; alpha and whether f_e counts dropped tokens are part of the declaration.
Numerics: router softmax rigorous intervals; everything else exact.
"""
from fractions import Fraction as F
from itertools import combinations
import math
import rigorous_mp as R
from rigorous_mp import I, iv, SpecInputError, SpecNotEstablished


def router_probs(logits_row, kind="softmax"):
    if kind == "softmax":
        return R.softmax_row(logits_row)[0]
    if kind == "sigmoid":
        return [R.sigmoid(v) for v in logits_row]
    raise SpecInputError("router kind must be softmax or sigmoid")


def topk_sets(probs, k):
    """all index sets of size k that are valid top-k selections (ties admit several). probs are intervals: a set S is
    valid iff for every s in S and t outside S, lower(p_s) >= upper(p_t) is not violated in the strict sense, i.e. we
    cannot prove p_t > p_s."""
    E = len(probs)
    if not (1 <= k <= E):
        raise SpecInputError("k must be in [1, E]")
    valid = []
    for S in combinations(range(E), k):
        ok = True
        for s in S:
            for t in range(E):
                if t in S:
                    continue
                if probs[t].a > probs[s].b:                  # provably larger outside the set
                    ok = False
        if ok:
            valid.append(frozenset(S))
    if not valid:
        raise SpecNotEstablished("no valid top-k set (interval overlap too wide)")
    return valid


def combine_weights(probs, S, renormalise=True):
    if renormalise:
        tot = R.isum(probs[e] for e in S)
        return {e: probs[e] / tot for e in S}
    return {e: probs[e] for e in S}


def moe_forward(x_tokens, router_logits, experts, k, renormalise=True, capacity=None):
    """x_tokens: [T][D]; router_logits: [T][E]; experts: list of functions f_e(x_vector) -> list of Fractions.
    Returns (outputs as intervals [T][D], dispatch record). Capacity dropping per MOE-C3 (order of appearance).
    Top-k ties: the FIRST valid set in lexicographic order is used for the value; all valid sets are returned."""
    T, E = len(router_logits), len(router_logits[0])
    load = [0] * E
    outs, record = [], []
    for t in range(T):
        p = router_probs(router_logits[t])
        sets = topk_sets(p, k)
        S = sorted(sets[0])
        w = combine_weights(p, S, renormalise)
        y = [iv.mpf(0)] * len(x_tokens[t])
        used = []
        for e in S:
            if capacity is not None and load[e] >= capacity:
                continue                                      # dropped
            load[e] += 1
            fe = experts[e](x_tokens[t])
            y = [a + w[e] * I(b) for a, b in zip(y, fe)]
            used.append(e)
        outs.append(y); record.append(dict(sets=sets, used=used, weights=w))
    return outs, record


def load_balancing_loss(router_logits, dispatch_used, alpha=F(1, 100), count_dropped=False):
    """Switch form: alpha * E * sum_e f_e * P_e. f_e = fraction of tokens routed to e (dispatch_used lists the experts
    that actually received each token unless count_dropped); P_e = mean router probability of e."""
    T, E = len(router_logits), len(router_logits[0])
    f = [F(0)] * E
    for used in dispatch_used:
        for e in used:
            f[e] += F(1, T)
    P = [iv.mpf(0)] * E
    for t in range(T):
        p = router_probs(router_logits[t])
        P = [a + b / T for a, b in zip(P, p)]
    return I(alpha) * E * R.isum(I(f[e]) * P[e] for e in range(E))


# ------------------------------------------------------------------ reference-free properties
def prop_weights_sum_to_one(weights):
    """Precondition: renormalise=True and no capacity drop for this token."""
    return R.isum(weights.values())


def prop_permute_unpermute_identity(tokens, perm):
    """dispatch permutation followed by its inverse must return the original token order exactly."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    permuted = [tokens[p] for p in perm]
    return [permuted[inv[i]] for i in range(len(tokens))] == list(tokens)
