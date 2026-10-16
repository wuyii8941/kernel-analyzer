"""Class A (linear attention and state recurrences), state S of shape dk x dv, output o_t = scale * S_t^T q_t.

Sources and identity (paper = task definition; fla naive.py and HF modules = model_impl cross-checks)
  linear attention (Katharopoulos et al. 2020)        S_t = S_{t-1} + k_t v_t^T
  RetNet / Lightning Attention / Mamba-2 (scalar)     S_t = a_t S_{t-1} + k_t v_t^T
  GLA (Yang et al. 2024), RWKV-6 (diagonal)           S_t = Diag(a_t) S_{t-1} + k_t v_t^T
  DeltaNet (Schlag et al. 2021; Yang et al. 2024)     S_t = (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T
  Gated DeltaNet (Yang et al. 2025)                   S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T
  KDA (Kimi Linear)                                   S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T
(the v x k layout used by some papers is the transpose; layout is dimension D3/D7.)
Gate parameterizations (D1): a = exp(g) ('ln') or 2^g ('log2'); Mamba-2: a = exp(dt * A), dt = softplus(dt_raw + dt_bias),
A = -exp(A_log); beta = sigmoid(raw) or given. Scale default dk^-1/2 (D2). Chunked algorithms equal the recurrence in exact
arithmetic (D8; checked here for the linear, scalar-decay and diagonal-decay variants). Backward (D12): exact adjoint for
those variants, including the gradient carried by the final state."""
from _base import *

VARIANTS = ('linear', 'decay', 'gla', 'delta', 'gated_delta', 'kda')

def _zero(dk, dv): return [[F(0)] * dv for _ in range(dk)]

def transition(S, k, v, variant, a=None, b=None):
    dk, dv = len(S), len(S[0])
    if variant == 'linear': A = S
    elif variant == 'decay': A = [[a * S[i][j] for j in range(dv)] for i in range(dk)]
    elif variant in ('gla', 'kda'): A = [[a[i] * S[i][j] for j in range(dv)] for i in range(dk)]
    elif variant in ('delta', 'gated_delta'): A = [[(a if variant == 'gated_delta' else 1) * S[i][j] for j in range(dv)] for i in range(dk)]
    else: raise SpecInputError(variant)
    if variant in ('delta', 'gated_delta', 'kda'):
        kA = [sum(k[i] * A[i][j] for i in range(dk)) for j in range(dv)]               # k^T A
        return [[A[i][j] - b * k[i] * kA[j] + b * k[i] * v[j] for j in range(dv)] for i in range(dk)]
    return [[A[i][j] + k[i] * v[j] for j in range(dv)] for i in range(dk)]

def recurrent(q, k, v, variant='linear', a=None, b=None, S0=None, scale=None):
    """q, k: T x dk; v: T x dv; a: per step scalar (decay, gated_delta) or vector (gla, kda); b: per step scalar.
    Returns (outputs T x dv, final state). Exact when every input is rational and scale is rational."""
    T, dk, dv = len(q), len(q[0]), len(v[0])
    S = [row[:] for row in S0] if S0 is not None else _zero(dk, dv)
    sc = Q(scale) if scale is not None else None
    outs = []
    for t in range(T):
        S = transition(S, k[t], v[t], variant, None if a is None else a[t], None if b is None else b[t])
        o = matvec_T(S, q[t])
        outs.append([x * sc for x in o] if sc is not None else [I(x) / iv.sqrt(iv.mpf(dk)) for x in o])
    return outs, S

def varlen(q, k, v, cu_seqlens, variant='linear', a=None, b=None, S0_list=None, scale=None):
    """independent sequences [cu[i], cu[i+1]); each starts from its own initial state (state pool slot i)."""
    outs, finals = [], []
    for i in range(len(cu_seqlens) - 1):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        o, S = recurrent(q[s:e], k[s:e], v[s:e], variant, None if a is None else a[s:e], None if b is None else b[s:e],
                         None if S0_list is None else S0_list[i], scale)
        outs += o; finals.append(S)
    return outs, finals

def chunked(q, k, v, C, variant='linear', a=None, S0=None, scale=F(1)):
    """chunked algorithm (intra-chunk masked products with cumulative decay + inter-chunk state); for linear / decay / gla."""
    if variant not in ('linear', 'decay', 'gla'): raise SpecInputError('chunked form here covers linear, decay, gla')
    T, dk, dv = len(q), len(q[0]), len(v[0]); S = [row[:] for row in S0] if S0 is not None else _zero(dk, dv)
    vec = lambda t: [F(1)] * dk if variant == 'linear' else ([a[t]] * dk if variant == 'decay' else list(a[t]))
    outs = []
    for c0 in range(0, T, C):
        idx = range(c0, min(c0 + C, T)); cum = {}; run = [F(1)] * dk
        for t in idx: run = [run[i] * vec(t)[i] for i in range(dk)]; cum[t] = run[:]
        for t in idx:
            o = [sum(q[t][i] * cum[t][i] * S[i][j] for i in range(dk)) for j in range(dv)]
            for s in idx:
                if s > t: break
                w = sum(q[t][i] * cum[t][i] / cum[s][i] * k[s][i] for i in range(dk))
                o = [o[j] + w * v[s][j] for j in range(dv)]
            outs.append([x * scale for x in o])
        last = max(idx)
        S = [[cum[last][i] * S[i][j] + sum(cum[last][i] / cum[s][i] * k[s][i] * v[s][j] for s in idx) for j in range(dv)] for i in range(dk)]
    return outs, S

def backward(q, k, v, dO, dS_T=None, variant='linear', a=None, S0=None, scale=F(1)):
    """exact adjoint for linear / decay / gla: gradients of sum_t <dO_t, o_t> + <dS_T, S_T> with respect to S0, q, k, v.
    The final-state gradient dS_T must flow back into S0, k and v (a kernel that drops it computes a different function)."""
    if variant not in ('linear', 'decay', 'gla'): raise SpecInputError('exact backward here covers linear, decay, gla')
    T, dk, dv = len(q), len(q[0]), len(v[0])
    Ss = [[row[:] for row in S0] if S0 is not None else _zero(dk, dv)]
    for t in range(T): Ss.append(transition(Ss[-1], k[t], v[t], variant, None if a is None else a[t]))
    G = [row[:] for row in dS_T] if dS_T is not None else _zero(dk, dv)
    dq, dk_, dv_ = [None] * T, [None] * T, [None] * T
    for t in reversed(range(T)):
        G = [[G[i][j] + scale * q[t][i] * dO[t][j] for j in range(dv)] for i in range(dk)]        # dL/dS_t
        dq[t] = [scale * sum(Ss[t + 1][i][j] * dO[t][j] for j in range(dv)) for i in range(dk)]
        dk_[t] = [sum(G[i][j] * v[t][j] for j in range(dv)) for i in range(dk)]
        dv_[t] = [sum(G[i][j] * k[t][i] for i in range(dk)) for j in range(dv)]
        if variant == 'decay': G = [[a[t] * G[i][j] for j in range(dv)] for i in range(dk)]
        elif variant == 'gla': G = [[a[t][i] * G[i][j] for j in range(dv)] for i in range(dk)]
    return dict(dS0=G, dq=dq, dk=dk_, dv=dv_)

# ---------------------------------------------------------------- gate parameterizations (D1), rigorous intervals
def gate_from_log(g, base='ln'): return iv.exp(I(g)) if base == 'ln' else iv.exp(I(g) * iv.log(iv.mpf(2)))
def softplus(x): x = I(x); return iv.log(1 + iv.exp(x)) if x.b < 30 else x + iv.log(1 + iv.exp(-x))
def mamba2_decay(dt_raw, dt_bias, A_log):
    dt = softplus(I(dt_raw) + I(dt_bias)); A = -iv.exp(I(A_log)); return iv.exp(dt * A), dt
def l2norm(x, eps=F(0), rule='add_inside'):
    """x / sqrt(sum x^2 + eps) ('add_inside') or x / max(||x||, eps) ('clamp')."""
    ss = isum(I(t) * I(t) for t in x)
    den = iv.sqrt(ss + I(eps)) if rule == 'add_inside' else iv.sqrt(ss)
    if rule == 'clamp' and den.b < I(eps).a: den = I(eps)
    return [I(t) / den for t in x]
