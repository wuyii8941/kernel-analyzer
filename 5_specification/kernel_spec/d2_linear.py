"""Gated delta rule (contract GDR): the recurrence and an independent chunkwise (WY / UT-transform) evaluation.

State layout h = S^T with h[i][j], i over the key dimension K and j over the value dimension V (FLA's [K, V]); the
paper's S in R^{dv x dk} is its transpose, so the two layouts give the same outputs.

recurrent(q, k, v, alpha, beta, scale, h0) for one sequence and one value head:
    h <- alpha_t h;  u = beta_t (v_t - h^T k_t);  h <- h + k_t u^T;  o_t = scale * h^T q_t
which is Eq. 10 of the paper, S_t = S_{t-1} alpha_t (I - beta_t k_t k_t^T) + beta_t v_t k_t^T, transposed.

chunked(...) evaluates the same function chunk by chunk without stepping the state token by token: inside a chunk
with cumulative decay gamma_r = alpha_1 ... alpha_r it solves the unit lower-triangular system
(I + strictLower(diag(beta) K K^T)) U = diag(beta) (V / gamma - K H0) by forward substitution and forms
o_r = scale gamma_r (H0^T q_r + sum_{i<=r} u_i (k_i . q_r)) and the chunk's final state gamma_C (H0 + K^T U).
In exact arithmetic the two agree (paper Sections 2.2 / 3.3: the chunkwise form is an expansion of the recurrence).

All values are kernel_spec.real.R (exact rationals or rigorous enclosures).
"""
from __future__ import annotations

from .real import R


def _dot(a, b):
    s = R(0)
    for x, y in zip(a, b):
        s = s + x * y
    return s


def recurrent(q, k, v, alpha, beta, scale, h0):
    """q, k: T x K; v: T x V; alpha, beta: length T; h0: K x V (copied). Returns (o: T x V, h_T: K x V)."""
    K = len(h0)
    V = len(h0[0]) if K else 0
    h = [row[:] for row in h0]
    outs = []
    for t in range(len(q)):
        h = [[alpha[t] * h[i][j] for j in range(V)] for i in range(K)]
        kh = [_dot([k[t][i] for i in range(K)], [h[i][j] for i in range(K)]) for j in range(V)]   # h^T k_t
        u = [beta[t] * (v[t][j] - kh[j]) for j in range(V)]
        h = [[h[i][j] + k[t][i] * u[j] for j in range(V)] for i in range(K)]
        outs.append([scale * _dot(q[t], [h[i][j] for i in range(K)]) for j in range(V)])
    return outs, h


def chunked(q, k, v, alpha, beta, scale, h0, C, carry_decay=True):
    """Chunkwise evaluation with chunk size C (the last chunk may be shorter). carry_decay=False is only used by a
    mutant (the state passed to the next chunk without its decay)."""
    K = len(h0)
    V = len(h0[0]) if K else 0
    H0 = [row[:] for row in h0]
    outs = []
    T = len(q)
    for c0 in range(0, T, C):
        idx = list(range(c0, min(c0 + C, T)))
        gam, run = [], R(1)
        for t in idx:
            run = run * alpha[t]
            gam.append(run)
        n = len(idx)
        U = []
        for r, t in enumerate(idx):
            kH0 = [_dot(k[t], [H0[i][j] for i in range(K)]) for j in range(V)]
            rhs = [beta[t] * (v[t][j] / gam[r] - kH0[j]) for j in range(V)]
            for i in range(r):
                c = beta[t] * _dot(k[t], k[idx[i]])
                rhs = [rhs[j] - c * U[i][j] for j in range(V)]
            U.append(rhs)
        for r, t in enumerate(idx):
            base = [_dot(q[t], [H0[i][j] for i in range(K)]) for j in range(V)]
            for i in range(r + 1):
                w = _dot(k[idx[i]], q[t])
                base = [base[j] + w * U[i][j] for j in range(V)]
            outs.append([scale * gam[r] * base[j] for j in range(V)])
        Ht = [[H0[i][j] + _dot([k[idx[r]][i] for r in range(n)], [U[r][j] for r in range(n)]) for j in range(V)]
              for i in range(K)]
        g = gam[-1] if carry_decay else R(1)
        H0 = [[g * Ht[i][j] for j in range(V)] for i in range(K)]
    return outs, H0
