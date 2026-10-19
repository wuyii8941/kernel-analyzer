"""Element-wise (lazy) evaluation of an ATen graph, for large tensors where only some output elements are needed.

Each node is a LazyTensor: a shape, a dtype and a memoised get(index). Only the elements an output element depends on
are computed (a matrix-product element reads one row and one column; a reduction reads its slice). The scalar
semantics are intended to match kernel_spec.aten (the contracts' references).
Differential tests check the exercised inputs; they do not establish general equivalence.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import aten as A
from . import ieee
from . import real as rl
from .num import Invalid, NotEstablished, Undefined, Unsupported
from .real import R


@dataclass
class EvaluationBudget:
    limit: int
    used: int = 0


_budgets = ContextVar("lazy_evaluation_budgets", default=())


def evaluation_budget_exhausted():
    """Whether any active scalar evaluation budget has no remaining work."""
    return any(b.used >= b.limit for b in _budgets.get())


@contextmanager
def evaluation_budget(limit):
    """Limit uncached scalar evaluations, including all dependency nodes.

    This is a work counter, not a bound on time or arbitrary-precision memory.
    Nested budgets also consume their enclosing budgets; exhaustion is unknown,
    never a claim that the mathematical operation is undefined.
    """
    if type(limit) is not int or limit <= 0:
        raise Invalid("lazy evaluation budget must be a positive integer")
    budget = EvaluationBudget(limit)
    token = _budgets.set((*_budgets.get(), budget))
    try:
        yield budget
    finally:
        _budgets.reset(token)


class Lazy:
    __slots__ = ("shape", "dtype", "_get", "_memo")

    def __init__(self, shape, dtype, get):
        self.shape = tuple(A._layout_sizes(shape, "lazy tensor shape"))
        if any(size < 0 for size in self.shape):
            raise Invalid("lazy tensor shape must be nonnegative")
        self.dtype = A.dt(dtype)
        self._get = get
        self._memo = {}

    def get(self, idx):
        idx = tuple(A._layout_sizes(idx, "lazy coordinate"))
        if len(idx) != self.ndim or any(i < 0 or i >= size for i, size in zip(idx, self.shape)):
            raise Invalid("lazy coordinate is outside the tensor shape")
        if idx not in self._memo:
            budgets = _budgets.get()
            if evaluation_budget_exhausted():
                raise NotEstablished("lazy scalar evaluation budget exhausted")
            for budget in budgets:
                budget.used += 1
            self._memo[idx] = self._get(idx)
        return self._memo[idx]

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return math.prod(self.shape)


def const(value, dtype):
    v = A._elem(R(value) if isinstance(value, (float, Fraction, str)) else value, A.dt(dtype))
    return Lazy((), dtype, lambda idx: v)


def from_array(arr, dtype):
    """A placeholder backed by a numpy array (float64 holding the exact values of the dtype, or ints / bools)."""
    d = A.dt(dtype)
    if d in A.FLOAT:
        return Lazy(arr.shape, d, lambda idx: R(float(arr[idx])))
    if d in A.BOOL:
        return Lazy(arr.shape, d, lambda idx: bool(arr[idx]))
    return Lazy(arr.shape, d, lambda idx: int(arr[idx]))


def as_lazy(x, like=None):
    if isinstance(x, Lazy):
        return x
    if isinstance(x, A.Ten):
        arr = x.a
        return Lazy(arr.shape, x.dtype, lambda idx: arr[idx])
    t = A.as_ten(x, A.Ten(np.empty((), dtype=object), like.dtype) if like is not None else None)
    v = t.a[()]
    return Lazy((), t.dtype, lambda idx: v)


def _bshape(*shapes):
    return A._broadcast_shape(*shapes)


def _bidx(idx, shape):
    off = len(idx) - len(shape)
    return tuple(0 if shape[i] == 1 else idx[i + off] for i in range(len(shape)))


def _conv(v, d):
    return A._elem(v, d) if not (isinstance(v, R) and d in A.FLOAT) else v


# ------------------------------------------------------------------ element-wise
def _map(fn, *xs, out_dtype=None, compute=None):
    xs = [as_lazy(x, next((y for y in xs if isinstance(y, (Lazy, A.Ten))), None) if not isinstance(x, (Lazy, A.Ten)) else None)
          for x in xs]
    d = compute or A.promote(*[x.dtype for x in xs])
    shape = _bshape(*[x.shape for x in xs])
    return Lazy(shape, out_dtype or d, lambda idx: fn(*[_conv(x.get(_bidx(idx, x.shape)), d) for x in xs]))


def _float(x):
    x = as_lazy(x)
    return x if x.dtype in A.FLOAT else Lazy(x.shape, "f32", lambda idx: R(int(x.get(idx))))


def l_add(a, b, alpha=1):
    a, b = A._arithmetic_operands(a, b, convert=as_lazy, tensor_types=(Lazy, A.Ten))
    alpha, al = A._arithmetic_alpha(A.promote(a.dtype, b.dtype), alpha)
    return _map(lambda x, y: x + y * al if isinstance(x, R) else A._iadd(x, y, alpha), a, b)


def l_sub(a, b, alpha=1):
    a, b = A._arithmetic_operands(a, b, convert=as_lazy, tensor_types=(Lazy, A.Ten))
    if a.dtype in A.BOOL or b.dtype in A.BOOL:
        raise Invalid("subtraction of booleans is not supported (torch.sub)")
    alpha, al = A._arithmetic_alpha(A.promote(a.dtype, b.dtype), alpha)
    return _map(lambda x, y: x - y * al if isinstance(x, R) else x - y * int(alpha), a, b)


def l_div(a, b, rounding_mode=None):
    a, b = A._arithmetic_operands(a, b, convert=as_lazy, tensor_types=(Lazy, A.Ten))
    dtype = A._division_dtype(a, b, rounding_mode)
    return _map(lambda x, y: A._division_value(x, y, rounding_mode), a, b, compute=dtype)


def _fl1(fn):
    return lambda a: _map(fn, _float(a))


UNARY = {
    "aten.exp": rl.exp, "aten.log": rl.log, "aten.log1p": rl.log1p, "aten.sqrt": rl.sqrt, "aten.rsqrt": rl.rsqrt,
    "aten.sin": rl.sin, "aten.cos": rl.cos, "aten.tanh": rl.tanh, "aten.sigmoid": rl.sigmoid, "aten.erf": rl.erf,
    "aten.reciprocal": lambda x: R(1) / x,
}


def l_pow(a, b):
    a, b, dtype = A._power_operands(a, b, convert=as_lazy, tensor_types=(Lazy, A.Ten))
    return _map(A._power_value, a, b, compute=dtype)


def _cmp(fn):
    return lambda a, b: _map(lambda x, y: bool(fn(R(x) if not isinstance(x, R) else x, R(y) if not isinstance(y, R) else y))
                             if (isinstance(x, R) or isinstance(y, R)) else bool(fn(x, y)), a, b, out_dtype="b8")


def _truth(x):
    return x if isinstance(x, bool) else (not x.same(0)) if isinstance(x, R) else x != 0


def l_where(c, a, b):
    c = as_lazy(c)
    A._where_condition(c)
    a = as_lazy(a, b if isinstance(b, (Lazy, A.Ten)) else None)
    b = as_lazy(b, a)
    d = A.promote(a.dtype, b.dtype)
    shape = _bshape(c.shape, a.shape, b.shape)
    return Lazy(shape, d, lambda idx: _conv(a.get(_bidx(idx, a.shape)), d) if c.get(_bidx(idx, c.shape))
                else _conv(b.get(_bidx(idx, b.shape)), d))


def l_convert(a, dtype):
    a, d = as_lazy(a), A.dt(dtype)
    t = A.Ten(np.empty((), dtype=object), a.dtype)

    def f(x):
        t.a[()] = x
        return A.convert_element_type(t, d).a[()]
    return Lazy(a.shape, d, lambda idx: f(a.get(idx)))


# ------------------------------------------------------------------ layout
def _strides(shape):
    st, acc = [], 1
    for s in reversed(shape):
        st.append(acc)
        acc *= s
    return list(reversed(st))


def l_view(a, shape):
    a = as_lazy(a)
    shape = A._view_shape(a.shape, shape)
    src_st, dst_st = _strides(a.shape), _strides(shape)

    def g(idx):
        flat = sum(i * s for i, s in zip(idx, dst_st))
        return a.get(tuple((flat // s) % d if d else 0 for s, d in zip(src_st, a.shape)))
    return Lazy(shape, a.dtype, g)


def l_permute(a, dims):
    a = as_lazy(a)
    dims = A._permutation(a.ndim, dims)
    inv = [0] * a.ndim
    for i, d in enumerate(dims):
        inv[d] = i
    return Lazy([a.shape[d] for d in dims], a.dtype, lambda idx: a.get(tuple(idx[inv[k]] for k in range(a.ndim))))


def l_transpose(a, d0, d1):
    a = as_lazy(a)
    d0, d1 = A._layout_axis(d0, max(1, a.ndim)), A._layout_axis(d1, max(1, a.ndim))
    if a.ndim == 0:
        return a
    p = list(range(a.ndim))
    p[d0], p[d1] = p[d1], p[d0]
    return l_permute(a, p)


def l_t(a):
    a = as_lazy(a)
    if a.ndim > 2:
        raise Invalid("t requires a tensor with at most two dimensions")
    return a if a.ndim < 2 else l_transpose(a, 0, 1)


def l_expand(a, shape):
    a = as_lazy(a)
    full = A._expand_shape(a.shape, shape)
    return Lazy(full, a.dtype, lambda idx: a.get(_bidx(idx, a.shape)))


def l_unsqueeze(a, d):
    a = as_lazy(a)
    d = A._layout_axis(d, a.ndim + 1)
    shape = list(a.shape)
    shape.insert(d, 1)
    return Lazy(shape, a.dtype, lambda idx: a.get(idx[:d] + idx[d + 1:]))


def l_squeeze(a, dims=None):
    a = as_lazy(a)
    drop = set(A._squeeze_axes(a.shape, dims))
    keep = [i for i in range(a.ndim) if i not in drop]

    def g(idx):
        full = [0] * a.ndim
        for k, i in enumerate(keep):
            full[i] = idx[k]
        return a.get(tuple(full))
    return Lazy([a.shape[i] for i in keep], a.dtype, g)


def l_select(a, dim, index):
    a = as_lazy(a)
    dim, index = A._select_coordinate(a.shape, dim, index)
    return Lazy(a.shape[:dim] + a.shape[dim + 1:], a.dtype, lambda idx: a.get(idx[:dim] + (index,) + idx[dim:]))


def l_slice(a, dim=0, start=None, end=None, step=1):
    a = as_lazy(a)
    dim = A._layout_axis(dim, a.ndim)
    s, e, st = A._slice_bounds(a.shape[dim], start, end, step)
    n = max(0, (e - s + st - 1) // st)
    shape = list(a.shape)
    shape[dim] = n
    return Lazy(shape, a.dtype, lambda idx: a.get(idx[:dim] + (s + idx[dim] * st,) + idx[dim + 1:]))


def l_cat(tensors, dim=0):
    ts, dim, d = A._cat_layout([as_lazy(t) for t in tensors], dim)
    offs, acc = [], 0
    for t in ts:
        offs.append(acc)
        acc += t.shape[dim]
    shape = list(ts[0].shape)
    shape[dim] = acc

    def g(idx):
        i = idx[dim]
        for t, o in zip(ts, offs):
            if o <= i < o + t.shape[dim]:
                return _conv(t.get(idx[:dim] + (i - o,) + idx[dim + 1:]), d)
        raise IndexError("cat index")
    return Lazy(shape, d, g)


def l_split_with_sizes(a, sizes, dim=0):
    a = as_lazy(a)
    dim, sizes = A._split_sections(a.shape, sizes, dim)
    out, start = [], 0
    for s in sizes:
        out.append(l_slice(a, dim, start, start + s))
        start += s
    return out


def l_split(a, size, dim=0):
    a = as_lazy(a)
    dim, sizes = A._split_size(a.shape, size, dim)
    return l_split_with_sizes(a, sizes, dim)


def l_pad(a, pad, value=0):
    a = as_lazy(a)
    lo, hi, shape = A._padding_layout(a.shape, pad)
    v = A._elem(R(value) if not isinstance(value, (bool, int)) or a.dtype in A.FLOAT else value, a.dtype)

    def g(idx):
        src = tuple(i - lo[d] for d, i in enumerate(idx))
        if all(0 <= s < a.shape[d] for d, s in enumerate(src)):
            return a.get(src)
        return v
    return Lazy(shape, a.dtype, g)


def l_select_scatter(base, src, dim, index):
    base, src = as_lazy(base), as_lazy(src)
    dim, index = A._select_scatter_layout(base.shape, src.shape, dim, index)
    return Lazy(base.shape, base.dtype, lambda idx: _conv(src.get(idx[:dim] + idx[dim + 1:]), base.dtype)
                if idx[dim] == index else base.get(idx))


def l_slice_scatter(base, src, dim=0, start=None, end=None, step=1):
    base, src = as_lazy(base), as_lazy(src)
    dim, s, e, st = A._slice_scatter_layout(base.shape, src.shape, dim, start, end, step)

    def g(idx):
        i = idx[dim]
        if s <= i < e and (i - s) % st == 0:
            return _conv(src.get(idx[:dim] + ((i - s) // st,) + idx[dim + 1:]), base.dtype)
        return base.get(idx)
    return Lazy(base.shape, base.dtype, g)


def l_full(size, value, dtype=None, **kw):
    A._unimplemented_options(kw)
    size = A._creation_shape(size)
    d = A.dt(dtype) if dtype is not None else ("b8" if isinstance(value, bool) else "i64" if isinstance(value, int) else "f32")
    v = A._elem(R(value) if isinstance(value, (float, str, Fraction)) else value, d)
    return Lazy(list(size), d, lambda idx: v)


def l_iota(length, start=0, step=1, dtype="i64", **kw):
    A._unimplemented_options(kw)
    return Lazy([length], dtype, lambda idx: start + idx[0] * step)


def l_copy(dst, src, non_blocking=False):
    A._copy_options(non_blocking)
    dst, src = as_lazy(dst), as_lazy(src)
    s = l_expand(src, list(dst.shape)) if src.shape != dst.shape else src
    return s if s.dtype == dst.dtype else l_convert(s, dst.dtype)


# ------------------------------------------------------------------ indexing
def _index_coordinates(input_shape, ixs):
    """Map coordinates of the indexed result back to the original tensor."""
    bshape = A._advanced_index_metadata(input_shape, ixs)
    adv = [k for k, i in enumerate(ixs) if i is not None]
    contiguous = adv == list(range(adv[0], adv[-1] + 1)) if adv else True
    rest = [k for k in range(len(input_shape)) if k not in adv]
    if contiguous and adv:
        shape = [input_shape[k] for k in range(adv[0])] + list(bshape) + [input_shape[k] for k in range(adv[-1] + 1, len(input_shape))]
    else:
        shape = list(bshape) + [input_shape[k] for k in rest]
    nb = len(bshape)

    def g(idx):
        if contiguous and adv:
            pre = idx[:adv[0]]
            bi = idx[adv[0]:adv[0] + nb]
            post = idx[adv[0] + nb:]
            src = list(pre)
            for k in adv:
                v = ixs[k].get(_bidx(bi, ixs[k].shape))
                n = input_shape[k]
                if not -n <= v < n:
                    raise Invalid("index out of range")
                src.append(v % n)
            src += list(post)
            return tuple(src)
        bi, post = idx[:nb], idx[nb:]
        src = [0] * len(input_shape)
        for k in adv:
            value = ixs[k].get(_bidx(bi, ixs[k].shape))
            size = input_shape[k]
            if not -size <= value < size:
                raise Invalid("index out of range")
            src[k] = value % size
        for j, k in enumerate(rest):
            src[k] = post[j]
        return tuple(src)
    return tuple(shape), g



def l_index(a, indices):
    a = as_lazy(a)
    ixs = [None if i is None else as_lazy(i) for i in indices]
    shape, coordinate = _index_coordinates(a.shape, ixs)
    return Lazy(shape, a.dtype, lambda idx: a.get(coordinate(idx)))

def _materialize_ints(t):
    t = as_lazy(t)
    return np.array([[t.get(i) for i in np.ndindex(*t.shape)]]).reshape(t.shape) if t.shape else np.array(t.get(()))


def l_gather(a, dim, idx):
    a, idx = as_lazy(a), as_lazy(idx)
    dim = A.index_dimension(a.shape, dim)
    if idx.numel() == 0:
        return from_array(np.empty(idx.shape), a.dtype)
    A.require_integer_indices(idx)
    dim = A.gather_dimension(a.shape, idx.shape, dim)
    source_shape = a.shape or (1,)
    def get(i):
        index = idx.get(i)
        if not 0 <= index < source_shape[dim]:
            raise Invalid("gather: index out of range")
        coordinate = i or (0,)
        return a.get(coordinate[:dim] + (index,) + coordinate[dim + 1:] if a.ndim else ())
    return Lazy(idx.shape, a.dtype, get)


def l_embedding(weight, indices, padding_idx=-1, **kw):
    w, ix = as_lazy(weight), as_lazy(indices)
    A.require_integer_indices(ix)
    if w.ndim != 2:
        raise Invalid("embedding: weight must be a matrix")
    def get(i):
        index = ix.get(i[:ix.ndim])
        if not 0 <= index < w.shape[0]:
            raise Invalid("embedding: index out of range")
        return w.get((index,) + i[ix.ndim:])
    return Lazy(ix.shape + w.shape[1:], w.dtype, get)


def _writes(index_positions):
    m = {}
    for target, src in index_positions:
        m.setdefault(target, []).append(src)
    return m


def l_scatter(a, dim, idx, src=None, value=None, reduce=None):
    a, idx = as_lazy(a), as_lazy(idx)
    dim = A.index_dimension(a.shape, dim)
    if idx.numel() == 0:
        return a
    A.require_integer_indices(idx)
    srcl = as_lazy(src) if src is not None else None
    A.scatter_shape(a.shape, idx.shape, None if srcl is None else srcl.shape)
    A.gather_dimension(a.shape, idx.shape, dim)
    source_shape = a.shape or (1,)
    positions = []
    for i in np.ndindex(*idx.shape):
        index = idx.get(i)
        if not 0 <= index < source_shape[dim]:
            raise Invalid("scatter: index out of range")
        coordinate = i or (0,)
        target = coordinate[:dim] + (index,) + coordinate[dim + 1:] if a.ndim else ()
        positions.append((target, coordinate if srcl is not None and srcl.ndim else ()))
    writes = _writes(positions)

    def g(t):
        if t not in writes:
            return a.get(t)
        vals = [_conv(srcl.get(i), a.dtype) if srcl is not None else A._elem(R(value) if not isinstance(value, bool) else value,
                                                                             a.dtype) for i in writes[t]]
        if reduce in ("add", "sum"):
            acc = a.get(t)
            for v in vals:
                acc = acc + v
            return acc
        if reduce in ("multiply", "prod"):
            acc = a.get(t)
            for v in vals:
                acc = acc * v
            return acc
        if len(vals) == 1:
            return vals[0]
        exact = [v.v if isinstance(v, R) and v.exact else v for v in vals]
        if all(not isinstance(v, R) or v.exact for v in vals) and all(v == exact[0] for v in exact):
            return vals[0]
        raise NotEstablished("lazy scatter requires a shared graph choice at duplicated indices")
    return Lazy(a.shape, a.dtype, g)


def l_scatter_add(a, dim, idx, src):
    return l_scatter(a, dim, idx, src=src, reduce="add")


def l_index_put(a, indices, values, accumulate=False):
    a, vals = as_lazy(a), as_lazy(values)
    accumulate = A._index_put_parameters(a, vals, accumulate)
    ixs = [None if i is None else as_lazy(i) for i in indices]
    vshape, coordinate = _index_coordinates(a.shape, ixs)
    A._index_put_values_shape(vals.shape, vshape)
    # INDEX requires in-range indices even when an unindexed axis is empty.
    for axis, ix in enumerate(ixs):
        if ix is not None:
            for idx in np.ndindex(*ix.shape):
                if not -a.shape[axis] <= ix.get(idx) < a.shape[axis]:
                    raise Invalid("index out of range")
    writes = {}
    for source in np.ndindex(*vshape):
        writes.setdefault(coordinate(source), []).append(source)
    if not accumulate and any(len(sources) > 1 for sources in writes.values()):
        raise Undefined("index_put duplicates with accumulate=False")

    def g(t):
        if t not in writes:
            return a.get(t)
        vs = [_conv(vals.get(_bidx(s, vals.shape)), a.dtype) for s in writes[t]]
        if accumulate:
            acc = a.get(t)
            for v in vs:
                acc = A._iadd(acc, v, 1)
            return acc
        return vs[0]
    return Lazy(a.shape, a.dtype, g)


# ------------------------------------------------------------------ reductions
def _slices(a, dims):
    dims = A._axes(a, dims)
    keep = [i for i in range(a.ndim) if i not in dims]
    return dims, keep


def _reduce(a, dim, keepdim, fn, out_dtype=None):
    a = as_lazy(a)
    dims, keep = _slices(a, dim)
    shape = [a.shape[i] if i in keep else 1 for i in range(a.ndim)] if keepdim else [a.shape[i] for i in keep]

    def g(idx):
        full = [0] * a.ndim
        if keepdim:
            for i in keep:
                full[i] = idx[i]
        else:
            for k, i in enumerate(keep):
                full[i] = idx[k]
        vals = []
        for red in np.ndindex(*[a.shape[i] for i in dims]):
            for k, i in enumerate(dims):
                full[i] = red[k]
            vals.append(a.get(tuple(full)))
        return fn(vals)
    return Lazy(shape, out_dtype or a.dtype, g)


def _sum(vals, zero):
    s = zero
    for v in vals:
        s = s + v
    return s


def _reduction_input(a, dtype, *, floating=False):
    a = as_lazy(a)
    d = A._reduction_dtype(a, dtype, floating=floating)
    if dtype is not None:
        return l_convert(a, d)
    return Lazy(a.shape, d, lambda idx: _conv(a.get(idx), d)) if d != a.dtype else a


def l_sum(a, dim=None, keepdim=False, dtype=None):
    a = _reduction_input(a, dtype)
    d = a.dtype
    return _reduce(a, dim, keepdim, any if d in A.BOOL else
                   lambda vs: _sum(vs, R(0) if d in A.FLOAT else 0), d)


def l_mean(a, dim=None, keepdim=False, dtype=None):
    a = _reduction_input(a, dtype, floating=True)
    def m(vs):
        if not vs:
            raise Undefined("mean of an empty reduction")
        return _sum(vs, R(0)) / len(vs)
    return _reduce(a, dim, keepdim, m)


def _ext(pick):
    def f(vs):
        if not vs:
            raise Invalid("extreme of an empty reduction")
        out = vs[0]
        for v in vs[1:]:
            out = pick(out, v)
        return out
    return f


def l_amax(a, dim=None, keepdim=False):
    a = as_lazy(a)
    A._extremum_axes(a, dim)
    return _reduce(a, dim, keepdim, _ext(lambda x, y: rl.max_(x, y) if isinstance(x, R) else max(x, y)))


def l_amin(a, dim=None, keepdim=False):
    a = as_lazy(a)
    A._extremum_axes(a, dim)
    return _reduce(a, dim, keepdim, _ext(lambda x, y: rl.min_(x, y) if isinstance(x, R) else min(x, y)))


def l_any(a, dim=None, keepdim=False):
    return _reduce(a, dim, keepdim, lambda vs: any(_truth(v) for v in vs), "b8")


def l_var_mean(a, dim=None, correction=1, keepdim=False):
    a = _float(a)

    def var(vs):
        n = len(vs)
        if n - correction <= 0:
            raise Undefined("variance with non-positive degrees of freedom")
        m = _sum(vs, R(0)) / n
        return _sum([(v - m) * (v - m) for v in vs], R(0)) / (n - correction)
    return [_reduce(a, dim, keepdim, var), l_mean(a, dim, keepdim)]


def l_max_dim(a, dim, keepdim=False):
    a = as_lazy(a)
    A._layout_axis(dim, max(1, a.ndim))
    A._extremum_axes(a, dim)

    def first_max(vs):
        best, k = vs[0], 0
        for i, v in enumerate(vs[1:], 1):
            if v > best:
                best, k = v, i
        return (best, k)
    pair = _reduce(a, [dim], keepdim, first_max)
    return [Lazy(pair.shape, a.dtype, lambda idx: pair.get(idx)[0]), Lazy(pair.shape, "i64", lambda idx: pair.get(idx)[1])]


def l_cumsum(a, dim, dtype=None):
    a = _reduction_input(a, dtype)
    d = a.dtype
    dim = A._layout_axis(dim, max(1, a.ndim))
    if a.ndim == 0 or 0 in a.shape or a.shape[dim] == 1:
        return a
    if d in A.BOOL:
        raise NotEstablished("boolean-output cumsum accumulation semantics have not been established")
    out = Lazy(a.shape, d, None)
    completed = {}

    def g(idx):
        row, position = idx[:dim] + idx[dim + 1:], idx[dim]
        # Build missing earlier prefixes through get(), preserving per-coordinate
        # budget accounting and caching, without a chain of recursive calls.
        for j in range(completed.get(row, -1) + 1, position):
            out.get(idx[:dim] + (j,) + idx[dim + 1:])
        cur = a.get(idx)
        value = cur if position == 0 else out._memo[idx[:dim] + (position - 1,) + idx[dim + 1:]] + cur
        completed[row] = position
        return value
    out._get = g
    return out


def l_prepare_softmax_online(a, dim):
    a = _float(a)
    m = l_amax(a, [dim], keepdim=True)
    e = _map(lambda x, mm: rl.exp(x - mm), a, m)
    return [m, l_sum(e, [dim], keepdim=True)]


# ------------------------------------------------------------------ matrix products
def _dot(a_row, b_col, k, zero):
    s = zero
    for i in range(k):
        s = s + a_row(i) * b_col(i)
    return s


def l_mm(a, b):
    a, b = as_lazy(a), as_lazy(b)
    A._product_layout(a, b, 2)
    k = a.shape[1]
    zero = R(0) if a.dtype in A.FLOAT else 0
    return Lazy((a.shape[0], b.shape[1]), A.promote(a.dtype, b.dtype),
                lambda idx: _dot(lambda t: a.get((idx[0], t)), lambda t: b.get((t, idx[1])), k, zero))


def l_bmm(a, b):
    a, b = as_lazy(a), as_lazy(b)
    A._product_layout(a, b, 3)
    k = a.shape[2]
    zero = R(0) if a.dtype in A.FLOAT else 0
    return Lazy((a.shape[0], a.shape[1], b.shape[2]), A.promote(a.dtype, b.dtype),
                lambda idx: _dot(lambda t: a.get((idx[0], idx[1], t)), lambda t: b.get((idx[0], t, idx[2])), k, zero))


def _scaled_product(c, a, b, beta, alpha, rank):
    c, a, b = as_lazy(c), as_lazy(a), as_lazy(b)
    A._scaled_product_layout(c, a, b, rank)

    def scalar(value, name):
        if isinstance(value, Lazy):
            if value.shape != ():
                raise Invalid(f"{name} must be scalar")
            return value.get(())
        return value

    beta, alpha = A._product_scales(c, scalar(beta, "beta"), scalar(alpha, "alpha"))
    p = l_mm(a, b) if rank == 2 else l_bmm(a, b)
    if alpha != 1:
        p = _map(lambda x: x * alpha, p)
    if beta == 0:
        return p
    cc = c if beta == 1 else _map(lambda x: x * beta, c)
    return l_add(cc, p)


def l_addmm(c, a, b, *, beta=1, alpha=1):
    return _scaled_product(c, a, b, beta, alpha, 2)


def l_baddbmm(c, a, b, *, beta=1, alpha=1):
    return _scaled_product(c, a, b, beta, alpha, 3)


def l_mm_plus_mm(mat1, mat2, mat3, mat4):
    return l_add(l_mm(mat1, mat2), l_mm(mat3, mat4))


def l_convolution(x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    x, w = as_lazy(x), as_lazy(w)
    bl = as_lazy(bias) if bias is not None else None
    A._convolution_dtypes(x, w, bl)
    stride, padding, dilation, shape, groups = A._convolution_layout(
        x.shape, w.shape, stride, padding, dilation, transposed, output_padding, groups)
    nsp = x.ndim - 2
    N, Cin = x.shape[:2]
    Cout, cpg = w.shape[:2]
    ksz = w.shape[2:]
    osz = shape[2:]
    opg = Cout // groups
    A._convolution_bias(bl, Cout)

    def g(idx):
        n, co, o = idx[0], idx[1], idx[2:]
        grp = co // opg
        s = (R(0) if x.dtype in A.FLOAT else 0) if bl is None else bl.get((co,))
        for ci in range(cpg):
            for k in np.ndindex(*ksz):
                pos = tuple(o[d] * stride[d] + k[d] * dilation[d] - padding[d] for d in range(nsp))
                if all(0 <= pos[d] < x.shape[2 + d] for d in range(nsp)):
                    s = s + x.get((n, grp * cpg + ci) + pos) * w.get((co, ci) + k)
        return s
    return Lazy(shape, x.dtype, g)


# ------------------------------------------------------------------ dispatch
OPS = {
    "aten.add": l_add, "aten.sub": l_sub, "aten.mul": lambda a, b: _map(lambda x, y: (x and y) if isinstance(x, bool) else x * y, a, b),
    "aten.div": l_div, "aten.neg": lambda a: _map(lambda x: -x, a), "aten.abs": lambda a: _map(lambda x: rl.abs_(x) if isinstance(x, R) else abs(x), a),
    "aten.sign": lambda a: _map(lambda x: rl.sign(x) if isinstance(x, R) else (x > 0) - (x < 0), a),
    "aten.pow": l_pow, "prims.fma": lambda a, b, c: l_add(OPS["aten.mul"](a, b), c),
    "aten.floor": lambda a: a if as_lazy(a).dtype not in A.FLOAT else _map(rl.floor, a),
    "aten.ceil": lambda a: a if as_lazy(a).dtype not in A.FLOAT else _map(rl.ceil, a),
    "aten.relu": lambda a: _map(lambda x: rl.relu(x) if isinstance(x, R) else max(x, 0), a),
    "aten.maximum": lambda a, b: _map(lambda x, y: rl.max_(x, y) if isinstance(x, R) else max(x, y), a, b),
    "aten.minimum": lambda a, b: _map(lambda x, y: rl.min_(x, y) if isinstance(x, R) else min(x, y), a, b),
    "aten.clamp_min": lambda a, m: _map(lambda x, y: rl.max_(x, y) if isinstance(x, R) else max(x, y), a, m),
    "aten.clamp_max": lambda a, m: _map(lambda x, y: rl.min_(x, y) if isinstance(x, R) else min(x, y), a, m),
    "aten.where": l_where,
    "aten.eq": _cmp(lambda x, y: x.same(y) if isinstance(x, R) else x == y),
    "aten.ne": _cmp(lambda x, y: (not x.same(y)) if isinstance(x, R) else x != y),
    "aten.lt": _cmp(lambda x, y: x < y), "aten.le": _cmp(lambda x, y: x <= y), "aten.gt": _cmp(lambda x, y: x > y),
    "aten.ge": _cmp(lambda x, y: x >= y),
    "aten.logical_and": lambda a, b: _map(lambda x, y: _truth(x) and _truth(y), a, b, out_dtype="b8"),
    "aten.logical_not": lambda a: _map(lambda x: not _truth(x), a, out_dtype="b8"),
    "aten.bitwise_and": lambda a, b: _map(lambda x, y: x & y, a, b),
    "aten.bitwise_not": lambda a: _map(lambda x: (not x) if isinstance(x, bool) else ~x, a),
    "aten.any": l_any,
    "prims.convert_element_type": l_convert, "aten._to_copy": lambda a, dtype=None, **k: l_convert(a, dtype) if dtype else a,
    "aten.clone": lambda a, **k: as_lazy(a), "aten.copy": l_copy, "aten.lift_fresh_copy": lambda a: as_lazy(a),
    "aten.view": l_view, "aten.reshape": l_view, "aten._unsafe_view": l_view, "aten.permute": l_permute,
    "aten.transpose": l_transpose, "aten.t": l_t, "aten.expand": l_expand, "aten.unsqueeze": l_unsqueeze,
    "aten.squeeze": l_squeeze, "aten.select": l_select, "aten.slice": l_slice, "aten.cat": l_cat, "aten.split": l_split,
    "aten.split_with_sizes": l_split_with_sizes, "aten.constant_pad_nd": l_pad, "aten.select_scatter": l_select_scatter,
    "aten.slice_scatter": l_slice_scatter, "aten.full": l_full, "aten.scalar_tensor": lambda v, dtype=None, **k: l_full([], v, dtype),
    "prims.iota": l_iota, "aten.index": l_index, "aten.index_put": l_index_put, "aten.index_put_": l_index_put,
    "aten.gather": l_gather, "aten.scatter": l_scatter, "aten.scatter_add": l_scatter_add, "aten.embedding": l_embedding,
    "aten.sum": l_sum, "aten.mean": l_mean, "aten.amax": l_amax, "aten.amin": l_amin, "aten.max": l_max_dim,
    "aten.var_mean": l_var_mean, "aten.cumsum": l_cumsum, "prims.prepare_softmax_online": l_prepare_softmax_online,
    "aten.mm": l_mm, "aten.bmm": l_bmm, "aten.addmm": l_addmm, "aten.baddbmm": l_baddbmm, "inductor.mm_plus_mm": l_mm_plus_mm,
    "aten.convolution": l_convolution,
}
OPS.update({k: _fl1(v) for k, v in UNARY.items()})


def evaluate_lazy(graph, inputs, elide_intermediate_casts=False):
    """Evaluate available lazy helpers for a GRAPH-shaped record.

    This function is also used for development references whose operations are
    outside the admitted base contracts. Semantic GRAPH callers must first use
    graph.covered_ops(); helper availability alone is not contract admission.
    """
    from .graph import functional_value_graph, select_result
    graph, _ = functional_value_graph(graph, input_names=inputs)
    env = dict(inputs)
    outputs = {o["ref"] if isinstance(o, dict) else o for o in graph.get("outputs", [])}
    from .aten_adapter import decode, bind_reference_call
    import inspect

    def res(x):
        if isinstance(x, dict) and "ref" in x:
            v = env[x["ref"]]
            if isinstance(v, Exception):
                raise v
            return select_result(v, x["index"]) if "index" in x else v
        if isinstance(x, list):
            return [res(v) for v in x]
        return decode(x)
    for n in graph["nodes"]:
        op = n["op"]
        try:
            args = [res(a) for a in n.get("args", [])]
            kwargs = {k: res(v) for k, v in (n.get("kwargs") or {}).items()}
            if op == "operator.getitem":
                env[n["name"]] = select_result(args[0], args[1])
                continue
            if op in A.OPS:
                bound = bind_reference_call(op, inspect.signature(A.OPS[op]), args, kwargs)
                args, kwargs = bound.args, bound.kwargs
            if (elide_intermediate_casts and op in ("prims.convert_element_type", "aten._to_copy")
                    and n["name"] not in outputs and isinstance(args[0], Lazy) and args[0].dtype in A.FLOAT):
                target = A.dt(args[1] if len(args) > 1 else kwargs.get("dtype"))
                if target in A.FLOAT:
                    src = args[0]
                    env[n["name"]] = Lazy(src.shape, target, src.get)
                    continue
            if op not in OPS:
                raise Unsupported(f"{op} has no lazy evaluation")
            env[n["name"]] = OPS[op](*args, **kwargs)
        except (Undefined, Invalid, Unsupported, NotEstablished) as e:
            env[n["name"]] = e
    for output in graph.get("outputs", []):
        if isinstance(output, dict) and "index" in output:
            value = env[output["ref"]]
            if not isinstance(value, Exception):
                select_result(value, output["index"])
    return env
