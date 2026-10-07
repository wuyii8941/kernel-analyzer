"""Independent specification f for Tensor.index_add_, Tensor.index_reduce_, Tensor.scatter_reduce_ (out-of-place forms).  (v0.4)

Written ONLY from the official documentation (no PyTorch source code consulted), fetched 2026-10-07:
  https://docs.pytorch.org/docs/stable/generated/torch.Tensor.index_reduce_.html
  https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html
  index_add_: self[index[i]] += alpha * source[i] along dim (standard documented form; duplicates accumulate).

Documented semantics used:
  I-D1  index_reduce_: "Accumulate the elements of source into the self tensor by accumulating to the indices in the
        order given in index using the reduction given by the reduce argument" ("prod", "mean", "amax", "amin").
  I-D2  include_self=True: "the values in the self tensor are included in the reduction"; include_self=False: "rows in
        the self tensor that are accumulated to are treated as if they were filled with the reduction identities".
  I-D3  documentation example (dim 0, x filled with 2, index [0,4,2,0], prod): include_self=True row 0 = [20,44,72];
        include_self=False row 0 = [10,22,36]  (used as unit tests).
  S-D1  scatter_reduce_: each src value "is reduced to an index in self which is specified by its index in src for
        dimension != dim and by the corresponding value in index for dimension = dim"; reductions sum/prod/mean/amax/amin.
  S-D2  documentation example: src=[1..6], index=[0,1,0,1,2,1], input=[1,2,3,4]: sum -> [5,14,8,4];
        include_self=False -> [4,12,5,4]; input2=[5,4,3,2]: amax -> [5,6,5,2]; include_self=False -> [3,6,5,2].
        (Positions that receive nothing keep their value; used as unit tests.)

Ambiguities (every reading evaluated):
  IDX-A1  reduce='mean' with include_self=False: mean has no identity element. Reading R_contrib: mean over the
          contributions only (count = number of contributions). (Only reading implemented; a candidate that differs is
          recorded under IDX-A1.)
  IDX-A2  reduce='mean' with include_self=True: the only reading consistent with "the values in the self tensor are
          included in the reduction" is (self + sum)/(1 + k). The variant (self + sum)/k is NOT a reading: it is not a
          mean (constant inputs are not preserved). It is kept only as a labelled ERROR VARIANT for classification.
  IDX-A3  NaN in amax/amin: R_prop (NaN wins; only NaNs that actually participate - with include_self=False the old
          self value does not) is the project's diagnostic profile; R_ignore is auxiliary. If filtering leaves no
          value, the spec declines (SpecNotEstablished) - adopted 2026-10-07.
  IDX-A4  backward of amax/amin with ties: not documented. The strict exact-derivative scoring is limited to unique
          extrema; at ties the set check `check_extremum_subgradient` accepts any coefficients that are zero outside the
          extremum set, non-negative on it and sum to the upstream gradient. Single routing and even splitting are not
          judged against each other.
  IDX-A5  integer dtypes for 'mean' (floor vs trunc): out of scope in phase 1 (floating inputs only).
Gradients: exact. For 'prod' the derivative w.r.t. each factor is the product of the OTHER factors (well defined with
zeros, which is where implementations that divide by the factor fail).
Representation: tensors as nested lists; 1-D and 2-D supported; dim in {0, 1}. Anything else, and any index outside
[0, size(dim)), is REFUSED with SpecInputError (v0.3): the spec answers only legal inputs in its supported domain.
"""
from fractions import Fraction
import math


class SpecInputError(ValueError):
    """illegal or unsupported input: refused, not answered."""


class SpecNotEstablished(Exception):
    """the spec declines for this condition (e.g. amax over values that are all NaN under R_ignore)."""


def _isnan(v):
    return isinstance(v, float) and math.isnan(v)


_REDUCES_INDEX = ("prod", "mean", "amax", "amin")
_REDUCES_SCATTER = ("sum", "prod", "mean", "amax", "amin")


def _ndim(t):
    return 2 if isinstance(t[0], list) else 1


def _validate_index_family(self_t, dim, index, source, reduce, allowed):
    nd = _ndim(self_t)
    if nd == 2 and any(len(r) != len(self_t[0]) for r in self_t):
        raise SpecInputError("ragged self tensor")
    if dim not in (0, 1) or dim >= nd:
        raise SpecInputError(f"dim {dim} unsupported for a {nd}-D tensor (phase 1 supports dim in {{0,1}})")
    if reduce is not None and reduce not in allowed:
        raise SpecInputError(f"reduce {reduce!r} not in {allowed}")
    size = len(self_t) if dim == 0 else len(self_t[0])
    for j in index:
        if int(j) != j or not (0 <= j < size):
            raise SpecInputError(f"index {j} outside [0, {size})")
    if nd == 1:
        if len(source) != len(index):
            raise SpecInputError("source must have one entry per index")
    else:
        if dim == 0 and (len(source) != len(index) or any(len(r) != len(self_t[0]) for r in source)):
            raise SpecInputError("source shape must be [len(index)][C] for dim 0")
        if dim == 1 and (len(source) != len(self_t) or any(len(r) != len(index) for r in source)):
            raise SpecInputError("source shape must be [R][len(index)] for dim 1")


def _validate_scatter(self_t, dim, index, src, reduce):
    nd = _ndim(self_t)
    if dim not in (0, 1) or dim >= nd:
        raise SpecInputError(f"dim {dim} unsupported for a {nd}-D tensor")
    if reduce not in _REDUCES_SCATTER:
        raise SpecInputError(f"reduce {reduce!r} not in {_REDUCES_SCATTER}")
    size = len(self_t) if dim == 0 else len(self_t[0])
    flat = index if nd == 1 else [v for row in index for v in row]
    for j in flat:
        if int(j) != j or not (0 <= j < size):
            raise SpecInputError(f"index {j} outside [0, {size})")
    if nd == 1:
        if len(src) != len(index):
            raise SpecInputError("src and index must have equal length")
    else:
        if len(src) != len(index) or any(len(a) != len(b) for a, b in zip(src, index)):
            raise SpecInputError("src and index must have the same shape")
        if dim == 0 and any(len(r) > len(self_t[0]) for r in index) or dim == 1 and len(index) > len(self_t):
            raise SpecInputError("index must not exceed self along the non-dim axis")


def _reduce(vals, op, nan_reading="R_prop"):
    if op in ("amax", "amin") and any(_isnan(v) for v in vals):
        if nan_reading == "R_prop":
            return float("nan")
        vals = [v for v in vals if not _isnan(v)]
        if not vals:
            raise SpecNotEstablished("all participating values are NaN under R_ignore: no numeric value")
    vals = [v if (isinstance(v, float) and math.isinf(v)) else Fraction(v) for v in vals]
    if op in ("sum", "prod") and any(isinstance(v, float) for v in vals):
        raise ValueError("index/scatter spec: non-finite inputs are out of scope for sum/prod")
    if op == "sum":
        return sum(vals, Fraction(0))
    if op == "prod":
        return math.prod(vals, start=Fraction(1))
    if op == "amax":
        return max(vals)
    if op == "amin":
        return min(vals)
    raise ValueError(op)


def _groups_index(self_t, dim, index, source):
    """index_* family: returns {target position: [contributions]} for 1-D/2-D."""
    groups = {}
    if not isinstance(self_t[0], list):          # 1-D
        for i, j in enumerate(index):
            groups.setdefault((j,), []).append(source[i])
        return groups
    R, Cn = len(self_t), len(self_t[0])
    for i, j in enumerate(index):
        if dim == 0:
            for c in range(Cn):
                groups.setdefault((j, c), []).append(source[i][c])
        else:
            for r in range(R):
                groups.setdefault((r, j), []).append(source[r][i])
    return groups


def _groups_scatter(self_t, dim, index, src):
    groups = {}
    if not isinstance(self_t[0], list):
        for i, j in enumerate(index):
            groups.setdefault((j,), []).append(src[i])
        return groups
    for r in range(len(index)):
        for c in range(len(index[0])):
            tgt = (index[r][c], c) if dim == 0 else (r, index[r][c])
            groups.setdefault(tgt, []).append(src[r][c])
    return groups


def _apply(self_t, groups, op, include_self, nan_reading="R_prop", error_variant=None):
    one_d = not isinstance(self_t[0], list)
    out = [Fraction(v) if not _isnan(v) else v for v in self_t] if one_d else \
          [[Fraction(v) if not _isnan(v) else v for v in row] for row in self_t]

    def get(pos):
        return out[pos[0]] if one_d else out[pos[0]][pos[1]]

    def put(pos, v):
        if one_d:
            out[pos[0]] = v
        else:
            out[pos[0]][pos[1]] = v

    for pos, contrib in groups.items():
        base = get(pos)
        if op == "mean":
            vals = ([base] if include_self else []) + contrib
            s = sum((Fraction(v) for v in vals), Fraction(0))
            if error_variant == "mean_divides_by_contrib_count":     # known-wrong variant, never a valid reading
                put(pos, s / len(contrib))
            else:
                put(pos, s / len(vals))
        else:
            vals = ([base] if include_self else []) + contrib
            put(pos, _reduce(vals, op, nan_reading))
    return out


def index_add(self_t, dim, index, source, alpha=1):
    _validate_index_family(self_t, dim, index, source, None, _REDUCES_INDEX)
    groups = _groups_index(self_t, dim, index, source)
    scaled = {k: [Fraction(alpha) * Fraction(v) for v in vs] for k, vs in groups.items()}
    return _apply(self_t, scaled, "sum", include_self=True)


def index_reduce(self_t, dim, index, source, reduce, include_self=True, **readings):
    _validate_index_family(self_t, dim, index, source, reduce, _REDUCES_INDEX)
    return _apply(self_t, _groups_index(self_t, dim, index, source), reduce, include_self, **readings)


def scatter_reduce(self_t, dim, index, src, reduce, include_self=True, **readings):
    _validate_scatter(self_t, dim, index, src, reduce)
    return _apply(self_t, _groups_scatter(self_t, dim, index, src), reduce, include_self, **readings)


def prod_grad_factors(factors):
    """exact d(prod)/d(factor_i) = product of the other factors (no division; correct with zeros)."""
    fs = [Fraction(v) for v in factors]
    return [math.prod(fs[:i] + fs[i + 1:], start=Fraction(1)) for i in range(len(fs))]


def maximisers(vals):
    """set of positions attaining the max (for checking amax/amin gradient routing under IDX-A4)."""
    fs = [Fraction(v) for v in vals]
    m = max(fs)
    return {i for i, v in enumerate(fs) if v == m}


# ---------------------------------------------------------------- reference-free properties (apply to ANY candidate)

def prop_counting(n_dup, self_value=0, include_self=True):
    """B016-type counting property: n_dup ones scattered/added into ONE slot must give self_value + n_dup (sum), and
    exactly 1 for 'mean' when all contributions (and self, if included) equal 1."""
    return Fraction(self_value) + n_dup if include_self else Fraction(n_dup)


def prop_untouched_rows(before, after, touched):
    """rows/positions that receive no contribution must be unchanged (S-D2, I-D2)."""
    bad = []
    one_d = not isinstance(before[0], list)
    if one_d:
        for i, (a, b) in enumerate(zip(before, after)):
            if (i,) not in touched and not (a == b or (_isnan(a) and _isnan(b))):
                bad.append((i,))
    else:
        for r, (ra, rb) in enumerate(zip(before, after)):
            for c, (a, b) in enumerate(zip(ra, rb)):
                if (r, c) not in touched and not (a == b or (_isnan(a) and _isnan(b))):
                    bad.append((r, c))
    return bad


def prop_order_invariance(fn, self_t, dim, index, source, perm, *args, **kw):
    """Precondition: exact arithmetic (the spec) or inputs chosen so every partial result is exactly representable.
    sum/prod/amax/amin/mean are order-independent: permuting the contributions - index entries together with the
    matching source slices along dim (rows for dim 0, columns for dim 1) - must not change the result."""
    idx_p = [index[i] for i in perm]
    if dim == 0 or _ndim(self_t) == 1:
        src_p = [source[i] for i in perm]
    else:
        src_p = [[row[i] for i in perm] for row in source]
    return fn(self_t, dim, index, source, *args, **kw), fn(self_t, dim, idx_p, src_p, *args, **kw)


def check_extremum_subgradient(values, grad_values, upstream):
    """IDX-A4 backward set check for one reduced position: `values` are the participating values, `grad_values` the
    candidate's gradient w.r.t. each of them, `upstream` the gradient flowing into the reduced output.
    Valid iff grad_i = upstream * alpha_i with alpha_i = 0 outside the extremum set, alpha_i >= 0 inside, sum = 1
    (any alpha when upstream == 0 requires all grads 0). Returns None if valid, else a reason."""
    fs = [Fraction(v) for v in values]
    gs = [Fraction(g) for g in grad_values]
    u = Fraction(upstream)
    S = maximisers(fs)
    if u == 0:
        return None if all(g == 0 for g in gs) else "nonzero gradient with zero upstream"
    alpha = [g / u for g in gs]
    if any(alpha[i] != 0 for i in range(len(fs)) if i not in S):
        return "support outside the extremum set"
    if any(alpha[i] < 0 for i in S):
        return "negative coefficient"
    if sum((alpha[i] for i in S), Fraction(0)) != 1:
        return "coefficients do not sum to 1"
    return None
