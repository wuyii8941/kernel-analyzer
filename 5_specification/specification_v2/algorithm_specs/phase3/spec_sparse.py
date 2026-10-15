"""Sparse COO semantics. Source: torch.sparse documentation: a COO tensor is (indices, values, shape); duplicate
indices are allowed in an uncoalesced tensor and are summed by coalesce() and by every operation that densifies;
coalesce() sorts indices lexicographically and keeps explicit zeros."""
from fractions import Fraction as F

def _check(indices, values, shape):
    if len(indices) != len(values): raise ValueError('indices/values length mismatch')
    for idx in indices:
        if len(idx) != len(shape) or any(not (0 <= i < n) for i, n in zip(idx, shape)): raise ValueError(f'index {idx} out of bounds')

def to_dense(indices, values, shape):
    _check(indices, values, shape)
    rows, cols = shape; D = [[F(0)] * cols for _ in range(rows)]
    for (i, j), v in zip(indices, values): D[i][j] += F(v)
    return D

def coalesce(indices, values, shape):
    _check(indices, values, shape); acc = {}
    for idx, v in zip(indices, values): acc[tuple(idx)] = acc.get(tuple(idx), F(0)) + F(v)
    keys = sorted(acc); return [list(k) for k in keys], [acc[k] for k in keys]

def spmm(indices, values, shape, B):
    _check(indices, values, shape)
    if shape[1] != len(B): raise ValueError('inner dimension mismatch')
    out = [[F(0)] * len(B[0]) for _ in range(shape[0])]
    for (i, k), v in zip(indices, values):
        for j in range(len(B[0])): out[i][j] += F(v) * F(B[k][j])
    return out
