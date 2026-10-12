"""Combination order of a 1-D ``tt.scan`` on the locked build (Triton 3.6.0 ScanOpToLLVM), for a blocked layout whose
axis is covered without replication (n a multiple of spt * 32 * wpc).

Checked bit for bit against the device for f32 / f16 / bf16 add and mul scans, n = 128 .. 4096, num_warps 1 .. 8
(1_experiments/composition_study/scan_rule.py, an independent implementation of the same order):

  1. inside a thread, each chunk of ``spt`` contiguous elements is scanned left to right: acc = acc (+) x;
  2. the last element L of each chunk takes a Hillis-Steele scan over the 32 lanes: for d = 1, 2, 4, 8, 16:
     L = shfl_up(L, d) (+) L on lanes >= d;
  3. one warp along the axis: chunk r > 0 starts from the running total T of the previous chunks (L = T (+) L);
     several warps: the warp totals of each chunk are folded left to right through shared memory, starting from the
     total of all previous chunks, and warp w's last elements get the prefix of the warps before it on the left;
  4. the other elements of a chunk get the inclusive value of the previous lane (lane 0: the prefix that lane 0's
     chunk starts from) on the left: x = P (+) x; the very first chunk of lane 0 / warp 0 keeps its values.
``combine(a, b)`` is the region evaluated elementwise with the device's rounding.
"""
from __future__ import annotations

import numpy as np


def scan_1d(x: np.ndarray, spt: int, warps: int, combine, lanes: int = 32) -> np.ndarray:
    """Inclusive scan of x (..., n) in the lowering's order; n must be a multiple of spt * lanes * warps."""
    lead = x.shape[:-1]
    n = x.shape[-1]
    cover = spt * lanes * warps
    if n % cover:
        raise ValueError("the layout replicates data along the axis")
    R = n // cover
    v = np.array(x).reshape(lead + (R, warps, lanes, spt))      # float64, or int64 for integer scans
    for j in range(1, spt):
        v[..., j] = combine(v[..., j - 1], v[..., j])
    L = v[..., spt - 1].copy()                                             # (..., R, W, lanes)
    lane = np.arange(lanes)
    d = 1
    while d <= lanes // 2:
        sh = np.concatenate([np.zeros_like(L[..., :d]), L[..., :-d]], axis=-1)
        L = np.where(lane >= d, combine(sh, L), L)
        d *= 2
    out = v.copy()

    def spread(prefix, r, w, last):
        """Write chunk (r, w): last elements ``last`` (lanes), other elements get the previous lane's inclusive value,
        lane 0 the chunk prefix (None: the first chunk of warp 0 keeps lane 0's values)."""
        prev = np.concatenate([np.zeros_like(last[..., :1]) if prefix is None else prefix[..., None],
                               last[..., :-1]], axis=-1)
        for j in range(spt - 1):
            col = v[..., r, w, :, j]
            upd = combine(prev, col)
            if prefix is None:
                upd = np.concatenate([col[..., :1], upd[..., 1:]], axis=-1)
            out[..., r, w, :, j] = upd
        out[..., r, w, :, spt - 1] = last

    if warps == 1:
        T = None
        for r in range(R):
            last = L[..., r, 0, :]
            if T is not None:
                last = combine(np.broadcast_to(T[..., None], last.shape), last)
            spread(T, r, 0, last)
            T = last[..., -1]
        return out.reshape(lead + (n,))
    acc = None
    for r in range(R):
        totals = L[..., r, :, lanes - 1]                                   # (..., W)
        masked = [None] * warps
        if acc is None:
            a = totals[..., 0]
            masked = [a] * warps
            for i in range(1, warps):
                a = combine(a, totals[..., i])
                for w in range(i + 1, warps):
                    masked[w] = a
        else:
            a = acc
            masked = [a] * warps
            for i in range(warps):
                a = combine(a, totals[..., i])
                for w in range(i + 1, warps):
                    masked[w] = a
        for w in range(warps):
            last = L[..., r, w, :]
            first = acc is None and w == 0
            if not first:
                last = combine(np.broadcast_to(masked[w][..., None], last.shape), last)
            spread(None if first else masked[w], r, w, last)
        acc = a
    return out.reshape(lead + (n,))
