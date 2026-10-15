"""Independent specification f for torch.nn.functional.embedding and embedding_bag.  (v0.1)

Documentation (fetched 2026-10-08):
  Embedding    : https://docs.pytorch.org/docs/2.10/generated/torch.nn.Embedding.html
  EmbeddingBag : https://docs.pytorch.org/docs/2.10/generated/torch.nn.EmbeddingBag.html

Documented semantics used:
  E-D1 embedding: out[i] = weight[idx[i]]. padding_idx: "the entries at padding_idx do not contribute to the gradient;
       therefore, the embedding vector at padding_idx is not updated during training" -> d loss / d weight[padding_idx] = 0;
       the forward still returns the stored row (whatever it holds).
  E-D2 max_norm: "each embedding vector with p-norm larger than max_norm is renormalized to have norm max_norm" (p =
       norm_type), as an in-place modification of weight for the looked-up rows.
  E-D3 scale_grad_by_freq: "scale gradients by the inverse of frequency of the words in the mini-batch" -> the gradient
       contribution of index j is divided by the number of occurrences of j in the current input.
  E-D4 embedding_bag: bags given by offsets (or one bag per row for 2-D input); mode in {sum, mean, max};
       per_sample_weights only with mode='sum' (weights multiply the looked-up rows before summing);
       padding_idx: "entries at padding_idx ... are excluded from the reduction" (and do not count towards a mean);
       empty bags give zeros; include_last_offset: the last offset is the end of the last bag.
Ambiguities (ambiguities_phase2.md):
  EMB-A1 max_norm renormalisation: the spec rescales exactly to max_norm. An implementation that divides by (norm + c)
         with a small constant c is a numerical/interface difference under the declared interpretation, not a violation,
         unless c changes the result by more than its own rounding scale.
  EMB-A2 embedding_bag mode='mean' when a bag contains only padding_idx entries: count is 0 -> undefined; spec declines.
  EMB-A3 embedding_bag mode='max' of an empty bag: zeros (documented for empty bags); of a bag with only padding -> as A2.
Numerics: exact rationals; the p-norm uses a rigorous Interval (sqrt) for p = 2 and exact rationals for p = 1 / inf.
"""
from fractions import Fraction as F
from rigorous import Interval, SpecInputError, SpecNotEstablished


def _row(weight, j):
    if not (0 <= j < len(weight)):
        raise SpecInputError(f"index {j} outside [0, {len(weight)})")
    return [F(v) for v in weight[j]]


def embedding_forward(weight, indices, padding_idx=None):
    return [_row(weight, j) for j in indices]


def embedding_grad_weight(weight_shape, indices, grad_out, padding_idx=None, scale_grad_by_freq=False):
    """d loss / d weight for upstream grad_out[i] (one vector per looked-up index). Exact rationals."""
    V, D = weight_shape
    g = [[F(0)] * D for _ in range(V)]
    counts = {}
    for j in indices:
        counts[j] = counts.get(j, 0) + 1
    for i, j in enumerate(indices):
        if not (0 <= j < V):
            raise SpecInputError(f"index {j} outside [0, {V})")
        if padding_idx is not None and j == padding_idx:
            continue                                        # E-D1
        scale = F(1, counts[j]) if scale_grad_by_freq else F(1)
        for d in range(D):
            g[j][d] += scale * F(grad_out[i][d])
    return g


def embedding_renorm(weight, indices, max_norm, norm_type=2):
    """E-D2: returns the renormalised rows for the looked-up indices as a dict j -> list of Interval/Fraction."""
    out = {}
    for j in set(indices):
        row = _row(weight, j)
        if norm_type == 2:
            nrm = Interval.exact(sum((v * v for v in row), F(0))).sqrt()
            if nrm.lo > F(max_norm):
                out[j] = [Interval.exact(v) * F(max_norm) / nrm for v in row]
            elif nrm.hi <= F(max_norm):
                out[j] = row
            else:
                raise SpecNotEstablished("norm interval straddles max_norm; cannot decide renormalisation")
        elif norm_type == 1:
            nrm = sum((abs(v) for v in row), F(0))
            out[j] = [v * F(max_norm) / nrm for v in row] if nrm > F(max_norm) else row
        elif norm_type == float("inf"):
            nrm = max(abs(v) for v in row)
            out[j] = [v * F(max_norm) / nrm for v in row] if nrm > F(max_norm) else row
        else:
            raise SpecInputError("norm_type must be 1, 2 or inf in this spec")
    return out


def embedding_bag(weight, indices, offsets, mode="sum", per_sample_weights=None, padding_idx=None,
                  include_last_offset=False):
    """indices: flat list; offsets: bag starts. Returns list of bags, each a list of Fractions (exact)."""
    if mode not in ("sum", "mean", "max"):
        raise SpecInputError("mode must be sum, mean or max")
    if per_sample_weights is not None and mode != "sum":
        raise SpecInputError("per_sample_weights only supported with mode='sum' (documented)")
    D = len(weight[0])
    bounds = list(offsets) + ([] if include_last_offset else [len(indices)])
    if include_last_offset and (not offsets or offsets[-1] != len(indices) and offsets[-1] > len(indices)):
        raise SpecInputError("with include_last_offset the last offset must be the end of the last bag")
    out = []
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        if not (0 <= lo <= hi <= len(indices)):
            raise SpecInputError("offsets must be non-decreasing and within the indices")
        rows = []
        for i in range(lo, hi):
            j = indices[i]
            if padding_idx is not None and j == padding_idx:
                continue                                    # E-D4: excluded from the reduction
            r = _row(weight, j)
            if per_sample_weights is not None:
                r = [F(per_sample_weights[i]) * v for v in r]
            rows.append(r)
        if not rows:
            if hi > lo and mode == "mean":                  # EMB-A2: only padding entries, count 0
                raise SpecNotEstablished("mean over a bag whose entries are all padding_idx is undefined")
            out.append([F(0)] * D)                          # empty bag -> zeros (documented)
            continue
        if mode == "sum":
            out.append([sum((r[d] for r in rows), F(0)) for d in range(D)])
        elif mode == "mean":
            out.append([sum((r[d] for r in rows), F(0)) / len(rows) for d in range(D)])
        else:
            out.append([max(r[d] for r in rows) for d in range(D)])
    return out


# ---------------------------------------------------------------- reference-free properties
def prop_padding_row_zero_grad(grad_weight, padding_idx):
    """Precondition: padding_idx given. d loss / d weight[padding_idx] must be exactly zero (E-D1)."""
    return all(v == 0 for v in grad_weight[padding_idx])


def prop_bag_sum_linearity(weight, indices, offsets, w1, w2):
    """Precondition: mode='sum'. Bags with per_sample_weights w1 + w2 equal the sum of bags with w1 and with w2."""
    a = embedding_bag(weight, indices, offsets, "sum", [F(x) + F(y) for x, y in zip(w1, w2)])
    b = embedding_bag(weight, indices, offsets, "sum", w1)
    c = embedding_bag(weight, indices, offsets, "sum", w2)
    return a, [[u + v for u, v in zip(rb, rc)] for rb, rc in zip(b, c)]
