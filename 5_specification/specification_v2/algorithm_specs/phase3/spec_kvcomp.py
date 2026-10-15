"""Chunked KV compression (window W, stride S). Source: the algorithm family described in 'Periodic Weak Spots: Phase
Sensitivity from Chunked KV-Cache Compression' (arXiv 2609.36322, section 3.1): each window j covers positions
[jS, jS + W); every member t with in-window offset r = t - jS contributes a payload C_r h_t; aggregation is either a
uniform average or a softmax over gate logits z_t = <Z_r, h_t> + b_r. Parameters are shared across offsets
(per_offset=False) or indexed by the offset r (per_offset=True). Only complete windows are compressed."""
import os, sys
from fractions import Fraction as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase2'))
import spec_base_ops as bo
from rigorous_mp import I

def windows(L, W, S):
    return [(j, list(range(j * S, j * S + W))) for j in range((L - W) // S + 1)] if L >= W else []

def _mv(M, h): return [sum((F(M[i][c]) * F(h[c]) for c in range(len(h))), F(0)) for i in range(len(M))]

def compress(H, W, S, C, mode='uniform', Z=None, b=None, per_offset=False):
    """H: [L][d]; C: one matrix (shared) or W matrices (per offset); Z, b likewise for the gated mode."""
    out = []
    for j, members in windows(len(H), W, S):
        pay = [_mv(C[t - j * S] if per_offset else C, H[t]) for t in members]
        if mode == 'uniform':
            out.append([sum((p[i] for p in pay), F(0)) / len(pay) for i in range(len(pay[0]))])
        elif mode == 'gated':
            z = [sum((F((Z[t - j * S] if per_offset else Z)[c]) * F(H[t][c]) for c in range(len(H[t]))), F(0)) + F((b[t - j * S] if per_offset else b))
                 for t in members]
            alpha = bo.softmax(z)
            acc = None
            for a, p in zip(alpha, pay):
                term = [a * I(v) for v in p]; acc = term if acc is None else [x + y for x, y in zip(acc, term)]
            out.append(acc)
        else: raise ValueError(mode)
    return out

def visible_windows(tau, W, S, L):
    """Causal availability: a query at position tau may read window j iff the window is complete and ends before tau."""
    return [j for j, m in windows(L, W, S) if m[-1] < tau]

def influence_uniform(W, S, C, t, per_offset=False):
    """d K_hat_j / d h_t for the window containing t at offset r (uniform mode): C_r / W. Depends on r iff per_offset."""
    r = t % S if S == W else None
    M = C[r] if per_offset else C
    return [[F(x) / W for x in row] for row in M]
