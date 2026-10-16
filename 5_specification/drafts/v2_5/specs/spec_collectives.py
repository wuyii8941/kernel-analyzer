"""K14 (distributed and communication kernels: all-reduce, reduce-scatter, all-gather, AG+GEMM, GEMM+RS, EP all-to-all,
split-KV / flash-decode partial merge).

Sources and identity
  collective semantics (participants, reduction function, order) .... standard (StableHLO / NCCL documentation)
  Triton-distributed kernels (AG+GEMM, GEMM+RS, EP dispatch/combine) .. engine_ref
  partial softmax merge: exact identity of the softmax over a union of key sets .. mathematical definition
Precondition for order independence (D4): the same finite set of contributions, combined by exact summation. Weighted
averages, dropped or duplicated contributions, compression and non-linear combination are different functions (D1).
Dimensions: D1 (reduction function and weights), D3 (rank order of shards), D4 (order independence precondition),
D7 (symmetric buffers), D11 (uneven shards, capacity drops)."""
from _base import *

def all_reduce(xs, op='sum', weights=None):
    """xs: per-rank vectors. sum; avg = sum / world; weighted = sum_r w_r x_r / sum_r w_r (e.g. token counts)."""
    n = len(xs[0]); tot = [sum(Q(x[i]) for x in xs) for i in range(n)]
    if op == 'sum': return tot
    if op == 'avg': return [t / len(xs) for t in tot]
    if op == 'weighted':
        W = sum(map(Q, weights));
        if W == 0: raise SpecNotEstablished('zero total weight (D11)')
        return [sum(Q(w) * Q(x[i]) for w, x in zip(weights, xs)) / W for i in range(n)]
    raise SpecInputError(op)

def reduce_scatter(xs):
    """shard r of the exact sum, shards in rank order; length must divide evenly (D11)."""
    R, n = len(xs), len(xs[0])
    if n % R: raise SpecInputError('length not divisible by world size (D11)')
    tot = all_reduce(xs); m = n // R
    return [tot[r * m:(r + 1) * m] for r in range(R)]

def all_gather(shards): return [v for s in shards for v in s]

def matmul(A, B): return [[sum(Q(A[i][t]) * Q(B[t][j]) for t in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def ag_gemm(A_shards, B):
    """AllGather + GEMM: rows of A are sharded across ranks (rank order); result = gather(A) @ B."""
    return matmul([row for s in A_shards for row in s], B)

def gemm_rs(A_parts, B_parts):
    """GEMM + ReduceScatter: rank r holds A_r (M x K_r) and B_r (K_r x N); result shard s = rows of sum_r A_r B_r."""
    parts = [matmul(a, b) for a, b in zip(A_parts, B_parts)]
    R, M = len(parts), len(parts[0])
    if M % R: raise SpecInputError('rows not divisible by world size (D11)')
    tot = [[sum(p[i][j] for p in parts) for j in range(len(parts[0][0]))] for i in range(M)]
    m = M // R
    return [tot[r * m:(r + 1) * m] for r in range(R)]

def merge_partials(parts, base='e'):
    """split-KV merge: part i = (m_i, l_i, o_i) with o_i the normalized partial output and lse_i = m_i + log l_i.
    Exact identity: lse = log sum_i exp(lse_i); o = sum_i exp(lse_i - lse) o_i. base='2' if the kernel keeps log2-domain values."""
    lses = [I(p[0]) + iv.log(I(p[1])) if base == 'e' else (I(p[0]) + iv.log(I(p[1])) / iv.log(iv.mpf(2))) for p in parts]
    if base == '2': lses = [x * iv.log(iv.mpf(2)) for x in lses]
    m = max(x.b for x in lses); lse = m + iv.log(isum(iv.exp(x - m) for x in lses))
    o = [isum(iv.exp(lses[i] - lse) * I(parts[i][2][c]) for i in range(len(parts))) for c in range(len(parts[0][2]))]
    return o, lse

def ep_dispatch_combine(x_tokens, routes, placement, expert_fn, capacity=None):
    """routes[t] = [(expert, weight), ...]; placement[e] = rank. Each (token, expert) pair is sent to placement[e],
    processed, sent back, and combined as sum_j w_j * expert_fn(e_j, x_t). With capacity=None nothing is dropped and the
    result equals the local MoE. With a capacity, which pairs are dropped depends on arrival order (D4/D11): the reference
    then returns, per token, the set of admissible results."""
    if capacity is not None: raise SpecNotEstablished('capacity dropping: arrival order must be declared (D4/D11)')
    out = []
    for t, x in enumerate(x_tokens):
        acc = None
        for e, w in routes[t]:
            y = expert_fn(e, x); acc = [Q(w) * Q(v) for v in y] if acc is None else [a + Q(w) * Q(v) for a, v in zip(acc, y)]
        out.append(acc)
    return out
