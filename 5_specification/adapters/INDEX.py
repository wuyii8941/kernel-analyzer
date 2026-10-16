"""Adapter for contracts/INDEX.yaml. Reference: kernel_spec.aten; the reading pick_one and the mutants replace one
operation by an independent implementation."""
from __future__ import annotations

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec.aten_adapter import make_reference
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "INDEX.yaml").read_text())["covers"]
reference = make_reference(COVERS)


def _gather_other_axis(a, dim, idx):
    a = A.as_ten(a)
    return A.gather(a, (dim + 1) % a.a.ndim, idx)


def _scatter_add_overwrite(a, dim, idx, src):
    a, s = A.as_ten(a), A.as_ten(src)
    out = a.a.copy()
    ix = A._ints(idx)
    for i in np.ndindex(ix.shape):
        j = list(i)
        j[dim] = int(ix[i])
        out[tuple(j)] = s.a[i]                 # last write wins, no addition
    return A.Ten(out, a.dtype)


def _index_put_no_acc(a, indices, values, accumulate=False, pick_one=False):
    out = A.as_ten(a).a.copy()
    vals = A.as_ten(values).a.reshape(-1)
    ix = A._ints(indices[0]).reshape(-1)
    for k, j in enumerate(ix):
        out[int(j)] = vals[k]
    return A.Ten(out, A.as_ten(a).dtype)


def _emb_bwd_keep_padding(grad, indices, num_weights, padding_idx=-1, scale_grad_by_freq=False):
    return A.embedding_dense_backward(grad, indices, num_weights, -1, scale_grad_by_freq)


def _emb_bwd_no_freq(grad, indices, num_weights, padding_idx=-1, scale_grad_by_freq=False):
    return A.embedding_dense_backward(grad, indices, num_weights, padding_idx, False)


def _index_add_no_alpha(a, dim, idx, src, alpha=1):
    return A.index_add(a, dim, idx, src, 1)


def _scatter_last_wins(a, dim, idx, src=None, value=None, reduce=None):
    a = A.as_ten(a)
    out = a.a.copy()
    ix = A._ints(idx)
    sv = A.as_ten(src).a if src is not None else None
    for i in np.ndindex(ix.shape):
        j = list(i)
        j[dim] = int(ix[i])
        out[tuple(j)] = sv[i] if sv is not None else A.R(value)
    return A.Ten(out, a.dtype)


def _pick_one(a, indices, values, accumulate=False, pick_one=False):
    return A.index_put(a, indices, values, accumulate, pick_one=True)


def _m(**ops):
    return make_reference(COVERS, extra=ops)


READINGS = {"pick_one": _m(**{"aten.index_put": _pick_one, "aten.index_put_": _pick_one})}
MUTANTS = {
    "IX.M1": _m(**{"aten.gather": _gather_other_axis}),
    "IX.M2": _m(**{"aten.scatter_add": _scatter_add_overwrite}),
    "IX.M3": _m(**{"aten.index_put": _index_put_no_acc}),
    "IX.M4": _m(**{"aten.embedding_dense_backward": _emb_bwd_keep_padding}),
    "IX.M5": _m(**{"aten.embedding_dense_backward": _emb_bwd_no_freq}),
    "IX.M6": _m(**{"aten.index_add": _index_add_no_alpha}),
    "IX.M7": _m(**{"aten.scatter": _scatter_last_wins}),
}
