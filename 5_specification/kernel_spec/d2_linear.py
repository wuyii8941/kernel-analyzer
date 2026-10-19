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


def _alpha_rows(alpha,K):
    # Parsed scalar gates (GDR) or key-wise gates (KDA).
    return alpha if isinstance(alpha,list) else [alpha]*K


def recurrent(q, k, v, alpha, beta, scale, h0):
    """Parsed R inputs: q/k T x K, v T x V, beta T, alpha T or T x K.

    Key-wise alpha applies before delta correction (KDA). Scalar alpha is GDR.
    h0 K x V is copied; returns (o T x V, h_T K x V).
    """
    K = len(h0)
    V = len(h0[0]) if K else 0
    h = [row[:] for row in h0]
    outs = []
    for t in range(len(q)):
        rows=_alpha_rows(alpha[t],K)
        h = [[rows[i] * h[i][j] for j in range(V)] for i in range(K)]
        kh = [_dot([k[t][i] for i in range(K)], [h[i][j] for i in range(K)]) for j in range(V)]   # h^T k_t
        u = [beta[t] * (v[t][j] - kh[j]) for j in range(V)]
        h = [[h[i][j] + k[t][i] * u[j] for j in range(V)] for i in range(K)]
        outs.append([scale * _dot(q[t], [h[i][j] for i in range(K)]) for j in range(V)])
    return outs, h


def backward(q,k,v,alpha,beta,scale,h0,dO,dHT):
    """VJP of outputs and final state for already validated R-valued inputs.

    alpha is scalar per token or a vector per key; dalpha preserves that shape.
    beta is scalar per token. No gate/normalization parameterization is implied.
    """
    K,V=len(h0),len(h0[0]);T=len(q)
    states=[h0];decayed=[];residuals=[]
    for t in range(T):
        rows=_alpha_rows(alpha[t],K)
        A=[[rows[i]*states[-1][i][j] for j in range(V)] for i in range(K)]
        residual=[v[t][j]-_dot(k[t],[A[i][j] for i in range(K)]) for j in range(V)]
        states.append([[A[i][j]+beta[t]*k[t][i]*residual[j] for j in range(V)] for i in range(K)])
        decayed.append(A);residuals.append(residual)
    dq,dk,dv,da,db=([None]*T for _ in range(5));carry=dHT
    for t in reversed(range(T)):
        dq[t]=[scale*_dot(states[t+1][i],dO[t]) for i in range(K)]
        G=[[carry[i][j]+scale*q[t][i]*dO[t][j] for j in range(V)] for i in range(K)]
        # S=A+beta*k*(v-k^T*A)^T, so both occurrences of k contribute.
        kG=[_dot(k[t],[G[i][j] for i in range(K)]) for j in range(V)]
        dr=[beta[t]*x for x in kG]
        dA=[[G[i][j]-k[t][i]*dr[j] for j in range(V)] for i in range(K)]
        dk[t]=[beta[t]*_dot(G[i],residuals[t])-_dot(decayed[t][i],dr) for i in range(K)]
        dv[t]=dr;db[t]=_dot(kG,residuals[t])
        rowgrad=[_dot(dA[i],states[t][i]) for i in range(K)]
        da[t]=rowgrad if isinstance(alpha[t],list) else _dot(rowgrad,[R(1)]*K)
        rows=_alpha_rows(alpha[t],K)
        carry=[[rows[i]*dA[i][j] for j in range(V)] for i in range(K)]
    return {'dq':dq,'dk':dk,'dv':dv,'dalpha':da,'dbeta':db,'dS0':carry}


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
