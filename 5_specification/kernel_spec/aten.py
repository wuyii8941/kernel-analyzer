"""Executable real-number semantics of the ATen / prims operations that occur in the tested corpus.

Each operation follows the PyTorch 2.10 documentation of the corresponding public function (the contracts cite the
pages); prims and Inductor-internal operations follow their definition in the torch 2.10 sources (also cited).
Tensors are `Ten`: a numpy object array plus a dtype name. Floating elements are `real.R` (exact rationals,
enclosures, infinities); integer elements are Python ints; boolean elements are Python bools.

Real-number semantics means: no rounding except where the program asks for a conversion (`convert_element_type`,
`_to_copy` to another dtype), which rounds as the conversion contract (CAST) says. Undefined real operations raise
num.Undefined; undecidable branches on enclosures raise num.NotEstablished; arguments outside the documented domain
raise num.Invalid; documented-but-unsupported variants raise num.Unsupported.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np

from . import real as rl
from .num import Invalid, NotEstablished, Undefined, Unsupported
from .real import R

FLOAT = {"f64", "f32", "f16", "bf16", "f8e4m3", "f8e5m2"}
INT = {"i64", "i32", "i16", "i8", "u8"}
BOOL = {"b8"}
TORCH_DTYPES = {"torch.float64": "f64", "torch.float32": "f32", "torch.float16": "f16", "torch.bfloat16": "bf16",
                "torch.int64": "i64", "torch.int32": "i32", "torch.int16": "i16", "torch.int8": "i8", "torch.uint8": "u8",
                "torch.bool": "b8", "torch.float8_e4m3fn": "f8e4m3", "torch.float8_e5m2": "f8e5m2"}
INT_RANGE = {"i64": (-2 ** 63, 2 ** 63 - 1), "i32": (-2 ** 31, 2 ** 31 - 1), "i16": (-2 ** 15, 2 ** 15 - 1),
             "i8": (-128, 127), "u8": (0, 255)}


def dt(d):
    """Canonical dtype name from 'f32', torch.float32, 'torch.float32' or 'float32'."""
    s = str(d)
    if s in FLOAT | INT | BOOL:
        return s
    if s in TORCH_DTYPES:
        return TORCH_DTYPES[s]
    s2 = "torch." + s
    if s2 in TORCH_DTYPES:
        return TORCH_DTYPES[s2]
    raise Invalid(f"unknown dtype {d!r}")


class Ten:
    __slots__ = ("a", "dtype")

    def __init__(self, a, dtype):
        self.a = a if isinstance(a, np.ndarray) else np.array(a, dtype=object)
        if self.a.dtype != object:
            self.a = self.a.astype(object)
        self.dtype = dt(dtype)

    @property
    def shape(self):
        return self.a.shape

    def __repr__(self):
        return f"Ten({self.dtype}, {self.a.shape})"


def tensor(values, dtype):
    """Ten from nested lists / numpy / torch values; floats become exact rationals (their binary value)."""
    d = dt(dtype)
    try:
        import torch
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu()
            values = (values.double() if values.is_floating_point() else values).numpy()
    except ImportError:
        pass
    arr = np.array(values, dtype=object)
    conv = (lambda x: R(x)) if d in FLOAT else (lambda x: bool(x)) if d in BOOL else (lambda x: _int(x))
    out = np.empty(arr.shape, dtype=object)
    for i in np.ndindex(arr.shape):
        out[i] = conv(arr[i])
    return Ten(out, d)


def _int(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, Fraction) and x.denominator == 1:
        return int(x)
    raise Invalid(f"{x!r} is not an integer")


def plain(t):
    """Nested lists of plain values (Fraction / IV / infinity strings / ints / bools) for results."""
    if isinstance(t, Ten):
        t = t.a
    if isinstance(t, np.ndarray):
        if t.ndim == 0:
            return plain(t[()])
        return [plain(x) for x in t]
    if isinstance(t, R):
        return t.v
    if isinstance(t, (list, tuple)):
        return [plain(x) for x in t]
    return t


# ------------------------------------------------------------------ promotion and element maps
def _kind(d):
    return "f" if d in FLOAT else "i" if d in INT else "b"


def promote(*ds):
    ds = [dt(d) for d in ds]
    fl = [d for d in ds if d in FLOAT]
    if fl:
        order = ["f8e5m2", "f8e4m3", "bf16", "f16", "f32", "f64"]
        top = max(fl, key=order.index)
        if {"bf16", "f16"} <= set(fl) and top in ("bf16", "f16"):
            return "f32"
        return top
    if any(d in INT for d in ds):
        return "i64" if "i64" in ds else max((d for d in ds if d in INT), key=lambda x: INT_RANGE[x][1])
    return "b8"


def _scalar_dtype(x):
    return "b8" if isinstance(x, bool) else "i64" if isinstance(x, int) else "f32"


def as_ten(x, like=None):
    if isinstance(x, Ten):
        return x
    if isinstance(x, (bool, int, float, Fraction, R, str)):
        if isinstance(x, str):
            x = R(rl.POS if x in ("inf", "+inf") else rl.NEG if x == "-inf" else x)
        d = _scalar_dtype(x) if not isinstance(x, (Fraction, R)) else "f32"
        if like is not None and _kind(like.dtype) == "f" and d != "b8":
            d = like.dtype                       # a Python scalar does not promote a tensor's floating dtype
        elif like is not None and _kind(like.dtype) == "i" and d == "i64":
            d = like.dtype
        return Ten(np.array(_elem(x, d), dtype=object), d)
    raise Invalid(f"cannot use {type(x).__name__} as a tensor")


def _elem(x, d):
    if d in FLOAT:
        return x if isinstance(x, R) else R(x)
    if d in BOOL:
        return bool(x) if not isinstance(x, R) else (not x.same(0))
    if isinstance(x, R):
        if not x.exact or not x.finite or x.v.denominator != 1:
            raise Invalid("non-integer value for an integer tensor")
        return int(x.v)
    return int(x)


def _convert_elems(t: Ten, d):
    if t.dtype == d or (_kind(t.dtype) == _kind(d) == "f"):
        return t.a
    return np.vectorize(lambda x: _elem(x, d), otypes=[object])(t.a) if t.a.size else t.a.copy()


def binary(fn, a, b, out_dtype=None, compute=None):
    """Broadcasting binary map. compute: dtype the operands are converted to (default: promoted dtype)."""
    if not isinstance(a, Ten) and not isinstance(b, Ten):
        a = as_ten(a)
    if not isinstance(a, Ten):
        a = as_ten(a, b)
    if not isinstance(b, Ten):
        b = as_ten(b, a)
    d = compute or promote(a.dtype, b.dtype)
    x, y = _convert_elems(a, d), _convert_elems(b, d)
    try:
        x, y = np.broadcast_arrays(x, y)
    except ValueError:
        raise Invalid(f"shapes {a.shape} and {b.shape} do not broadcast") from None
    out = np.empty(x.shape, dtype=object)
    for i in np.ndindex(x.shape):
        out[i] = fn(x[i], y[i])
    return Ten(out, out_dtype or d)


def unary(fn, a, out_dtype=None, compute=None):
    a = as_ten(a)
    d = compute or a.dtype
    x = _convert_elems(a, d)
    out = np.empty(x.shape, dtype=object)
    for i in np.ndindex(x.shape):
        out[i] = fn(x[i])
    return Ten(out, out_dtype or d)


def _floatify(a):
    a = as_ten(a)
    return a if _kind(a.dtype) == "f" else Ten(_convert_elems(a, "f32"), "f32")


# ------------------------------------------------------------------ arithmetic (torch.add, sub, mul, div, ...)
def add(a, b, alpha=1):
    al = R(alpha) if not isinstance(alpha, bool) else R(int(alpha))
    return binary(lambda x, y: x + (y * al if isinstance(y, R) else y * int(alpha)) if isinstance(x, R) else
                  _iadd(x, y, alpha), a, b)


def _iadd(x, y, alpha):
    if isinstance(x, bool) and isinstance(y, bool):
        return x or y
    return x + int(alpha) * y


def sub(a, b, alpha=1):
    if isinstance(a, Ten) and a.dtype in BOOL:
        raise Invalid("subtraction of booleans is not supported (torch.sub)")
    return add(a, neg(b) if isinstance(b, Ten) else -_num(b), alpha)


def _num(x):
    return R(x) if isinstance(x, (Fraction, float, str)) else x


def rsub(a, b, alpha=1):
    return sub(b, a, alpha)


def mul(a, b):
    return binary(lambda x, y: (x and y) if isinstance(x, bool) and isinstance(y, bool) else x * y, a, b)


def div(a, b, rounding_mode=None):
    a, b = (as_ten(a) if isinstance(a, Ten) or not isinstance(b, Ten) else as_ten(a, b)), (
        as_ten(b, a if isinstance(a, Ten) else None))
    if rounding_mode is None:                    # true division: integer inputs are promoted to the default float
        a, b = _floatify(a), _floatify(b)
        return binary(lambda x, y: x / y, a, b)
    if rounding_mode not in ("trunc", "floor"):
        raise Invalid(f"rounding_mode {rounding_mode!r}")
    d = promote(a.dtype, b.dtype)
    if _kind(d) == "f":
        f = rl.trunc if rounding_mode == "trunc" else rl.floor
        return binary(lambda x, y: f(x / y), a, b)

    def idiv(x, y):
        if y == 0:
            raise Invalid("integer division by zero (torch raises)")
        q = Fraction(x, y)
        return math.trunc(q) if rounding_mode == "trunc" else math.floor(q)
    return binary(idiv, a, b)


def neg(a):
    a = as_ten(a)
    if a.dtype in BOOL:
        raise Invalid("negation of a boolean tensor is not supported (torch.neg)")
    return unary(lambda x: -x, a)


def reciprocal(a):
    return unary(lambda x: R(1) / x, _floatify(a))


def abs_(a):
    a = as_ten(a)
    return unary(lambda x: rl.abs_(x) if isinstance(x, R) else abs(x), a)


def sign(a):
    a = as_ten(a)
    return unary(lambda x: rl.sign(x) if isinstance(x, R) else (x > 0) - (x < 0), a)


def pow_(a, b):
    """torch.pow: 0**0 = 1; a negative base with a non-integer exponent has no real value."""
    if isinstance(a, Ten) and _kind(a.dtype) == "i" and not isinstance(b, Ten) and isinstance(b, int):
        if b < 0:
            raise Invalid("integers to negative integer powers are not allowed (torch.pow)")
        return unary(lambda x: x ** b, a)
    if not isinstance(a, Ten):
        a = as_ten(a, b if isinstance(b, Ten) else None)
    a = _floatify(a)
    return binary(lambda x, y: x ** y, a, b)


def fma(a, b, c):
    """prims.fma: a * b + c, a single exact operation in real arithmetic."""
    return add(mul(a, b), c)


def _fl(fn):
    return lambda a: unary(fn, _floatify(a))


exp, log, log1p, expm1 = _fl(rl.exp), _fl(rl.log), _fl(rl.log1p), _fl(rl.expm1)
sqrt, rsqrt, sin, cos = _fl(rl.sqrt), _fl(rl.rsqrt), _fl(rl.sin), _fl(rl.cos)
tanh, sigmoid, erf = _fl(rl.tanh), _fl(rl.sigmoid), _fl(rl.erf)


def floor(a):
    a = as_ten(a)
    return a if _kind(a.dtype) != "f" else unary(rl.floor, a)


def ceil(a):
    a = as_ten(a)
    return a if _kind(a.dtype) != "f" else unary(rl.ceil, a)


def relu(a):
    a = as_ten(a)
    return unary(lambda x: rl.relu(x) if isinstance(x, R) else max(x, 0), a)


def _maxv(x, y):
    return rl.max_(x, y) if isinstance(x, R) or isinstance(y, R) else max(x, y)


def _minv(x, y):
    return rl.min_(x, y) if isinstance(x, R) or isinstance(y, R) else min(x, y)


def maximum(a, b):
    return binary(_maxv, a, b)


def minimum(a, b):
    return binary(_minv, a, b)


def clamp(a, min=None, max=None):
    """torch.clamp: min(max(x, min), max); with min > max every element becomes max (documented)."""
    out = as_ten(a)
    if min is not None:
        out = maximum(out, min)
    if max is not None:
        out = minimum(out, max)
    return out


def clamp_min(a, m):
    return maximum(a, m)


def clamp_max(a, m):
    return minimum(a, m)


def where(cond, a, b):
    cond = as_ten(cond)
    if cond.dtype not in BOOL:
        raise Invalid("where: condition must be boolean")
    if not isinstance(a, Ten) and not isinstance(b, Ten):
        a, b = as_ten(a), as_ten(b)
    a = a if isinstance(a, Ten) else as_ten(a, b)
    b = b if isinstance(b, Ten) else as_ten(b, a)
    d = promote(a.dtype, b.dtype)
    c, x, y = np.broadcast_arrays(cond.a, _convert_elems(a, d), _convert_elems(b, d))
    out = np.empty(c.shape, dtype=object)
    for i in np.ndindex(c.shape):
        out[i] = x[i] if c[i] else y[i]
    return Ten(out, d)


def masked_fill(a, mask, value):
    return where(mask, value if isinstance(value, Ten) else as_ten(value, a), a)


# ------------------------------------------------------------------ comparison and logic (exact; undecided -> NotEstablished)
def _cmp(op):
    def f(a, b):
        return binary(lambda x, y: bool(op(R(x) if not isinstance(x, R) else x, R(y) if not isinstance(y, R) else y))
                      if (isinstance(x, R) or isinstance(y, R)) else bool(op(x, y)), a, b, out_dtype="b8")
    return f


eq = _cmp(lambda x, y: x.same(y) if isinstance(x, R) else x == y)
ne = _cmp(lambda x, y: (not x.same(y)) if isinstance(x, R) else x != y)
lt, le = _cmp(lambda x, y: x < y), _cmp(lambda x, y: x <= y)
gt, ge = _cmp(lambda x, y: x > y), _cmp(lambda x, y: x >= y)


def _truth(x):
    return x if isinstance(x, bool) else (not x.same(0)) if isinstance(x, R) else x != 0


def logical_and(a, b):
    return binary(lambda x, y: _truth(x) and _truth(y), a, b, out_dtype="b8")


def logical_or(a, b):
    return binary(lambda x, y: _truth(x) or _truth(y), a, b, out_dtype="b8")


def logical_not(a):
    return unary(lambda x: not _truth(x), as_ten(a), out_dtype="b8")


def bitwise_and(a, b):
    return binary(lambda x, y: x & y, a, b)


def bitwise_or(a, b):
    return binary(lambda x, y: x | y, a, b)


def bitwise_not(a):
    a = as_ten(a)
    return unary(lambda x: (not x) if isinstance(x, bool) else ~x, a)


def any_(a, dim=None, keepdim=False):
    return _reduce(as_ten(a), dim, keepdim, lambda xs: any(_truth(x) for x in xs), "b8")


def all_(a, dim=None, keepdim=False):
    return _reduce(as_ten(a), dim, keepdim, lambda xs: all(_truth(x) for x in xs), "b8")


# ------------------------------------------------------------------ conversion (CAST contract)
def convert_element_type(a, dtype):
    from . import ieee
    a, d = as_ten(a), dt(dtype)
    if d in FLOAT:
        if _kind(a.dtype) == "f":
            return unary(lambda x: ieee.round_to(x, d), a, out_dtype=d)
        return unary(lambda x: ieee.round_to(R(int(x)), d), a, out_dtype=d)
    if d in BOOL:
        return unary(_truth, a, out_dtype="b8")
    lo, hi = INT_RANGE[d]

    def to_int(x):
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, R):
            if not x.finite:
                raise Undefined("conversion of an infinity to an integer")
            v = rl.trunc(x)                        # C conversion: toward zero
            v = int(v.v)
        else:
            v = int(x)
        if not lo <= v <= hi:
            raise Undefined(f"{v} is outside the range of {d}")
        return v
    return unary(to_int, a, out_dtype=d)


def _to_copy(a, dtype=None, **kw):
    return convert_element_type(a, dtype) if dtype is not None else clone(a)


# ------------------------------------------------------------------ layout and creation
def clone(a, **kw):
    a = as_ten(a)
    return Ten(a.a.copy(), a.dtype)


copy_ = clone


def view(a, shape):
    a = as_ten(a)
    shape = list(shape)
    n = a.a.size
    if shape.count(-1) > 1:
        raise Invalid("view: more than one -1")
    if -1 in shape:
        known = int(np.prod([s for s in shape if s != -1])) if len(shape) > 1 else 1
        shape[shape.index(-1)] = n // known if known else 0
    if int(np.prod(shape)) != n:
        raise Invalid(f"view: {a.shape} -> {shape}")
    return Ten(a.a.reshape(shape), a.dtype)


reshape = view


def permute(a, dims):
    a = as_ten(a)
    return Ten(np.transpose(a.a, [d % a.a.ndim for d in dims]), a.dtype)


def transpose(a, d0, d1):
    a = as_ten(a)
    return Ten(np.swapaxes(a.a, d0, d1), a.dtype)


def t(a):
    a = as_ten(a)
    return a if a.a.ndim < 2 else transpose(a, 0, 1)


def expand(a, shape):
    a = as_ten(a)
    shape = list(shape)
    off = len(shape) - a.a.ndim
    full = [a.a.shape[i - off] if (s == -1) else s for i, s in enumerate(shape)]
    try:
        return Ten(np.broadcast_to(a.a, full).copy(), a.dtype)
    except ValueError:
        raise Invalid(f"expand: {a.shape} -> {shape}") from None


def unsqueeze(a, d):
    a = as_ten(a)
    return Ten(np.expand_dims(a.a, d if d >= 0 else d + a.a.ndim + 1), a.dtype)


def squeeze(a, dims=None):
    a = as_ten(a)
    if dims is None:
        return Ten(np.squeeze(a.a), a.dtype)
    dims = [dims] if isinstance(dims, int) else list(dims)
    keep = tuple(d % a.a.ndim for d in dims if a.a.shape[d % a.a.ndim] == 1)
    return Ten(np.squeeze(a.a, axis=keep) if keep else a.a, a.dtype)


def select(a, dim, index):
    a = as_ten(a)
    n = a.a.shape[dim]
    if not -n <= index < n:
        raise Invalid(f"select: index {index} out of range for size {n}")
    return Ten(np.take(a.a, index % n, axis=dim), a.dtype)


def _slice_bounds(n, start, end, step):
    start = 0 if start is None else start
    end = n if end is None else end
    start = start + n if start < 0 else start
    end = end + n if end < 0 else end
    return max(0, min(start, n)), max(0, min(end, n)), step


def slice_(a, dim=0, start=None, end=None, step=1):
    a = as_ten(a)
    if step <= 0:
        raise Invalid("slice step must be positive")
    s, e, st = _slice_bounds(a.a.shape[dim], start, end, step)
    idx = [slice(None)] * a.a.ndim
    idx[dim] = slice(s, e, st)
    return Ten(a.a[tuple(idx)], a.dtype)


def cat(tensors, dim=0):
    ts = [as_ten(x) for x in tensors]
    ts = [x for x in ts if not (x.a.ndim == 1 and x.a.size == 0)] or ts[:1]    # legacy: 1-d empty tensors are skipped
    d = promote(*[x.dtype for x in ts])
    try:
        return Ten(np.concatenate([_convert_elems(x, d) for x in ts], axis=dim), d)
    except ValueError:
        raise Invalid("cat: shapes do not match") from None


def stack(tensors, dim=0):
    return cat([unsqueeze(x, dim) for x in tensors], dim)


def split(a, size, dim=0):
    a = as_ten(a)
    n = a.a.shape[dim]
    return split_with_sizes(a, [min(size, n - i) for i in range(0, n, size)] or [0], dim)


def split_with_sizes(a, sizes, dim=0):
    a = as_ten(a)
    if sum(sizes) != a.a.shape[dim]:
        raise Invalid("split_with_sizes: sizes do not add up")
    out, start = [], 0
    for s in sizes:
        out.append(slice_(a, dim, start, start + s))
        start += s
    return out


def constant_pad_nd(a, pad, value=0):
    """torch.nn.functional.pad (constant): pad pairs from the last dimension backwards; negative pads crop."""
    a = as_ten(a)
    arr = a.a
    v = _elem(value, a.dtype)
    for k in range(len(pad) // 2):
        dim = arr.ndim - 1 - k
        lo, hi = pad[2 * k], pad[2 * k + 1]
        if lo < 0:
            arr = np.take(arr, range(-lo, arr.shape[dim]), axis=dim)
            lo = 0
        if hi < 0:
            arr = np.take(arr, range(0, arr.shape[dim] + hi), axis=dim)
            hi = 0
        shp_lo, shp_hi = list(arr.shape), list(arr.shape)
        shp_lo[dim], shp_hi[dim] = lo, hi
        fill = lambda s: np.full(s, v, dtype=object)
        arr = np.concatenate([fill(shp_lo), arr, fill(shp_hi)], axis=dim)
    return Ten(arr, a.dtype)


def select_scatter(base, src, dim, index):
    base, src = as_ten(base), as_ten(src)
    out = base.a.copy()
    idx = [slice(None)] * out.ndim
    idx[dim] = index % out.shape[dim]
    out[tuple(idx)] = _convert_elems(src, base.dtype)
    return Ten(out, base.dtype)


def slice_scatter(base, src, dim=0, start=None, end=None, step=1):
    base, src = as_ten(base), as_ten(src)
    out = base.a.copy()
    s, e, st = _slice_bounds(out.shape[dim], start, end, step)
    idx = [slice(None)] * out.ndim
    idx[dim] = slice(s, e, st)
    if out[tuple(idx)].shape != src.a.shape:
        raise Invalid("slice_scatter: source shape does not match the slice")
    out[tuple(idx)] = _convert_elems(src, base.dtype)
    return Ten(out, base.dtype)


def full(size, value, dtype=None, **kw):
    d = dt(dtype) if dtype is not None else ("b8" if isinstance(value, bool) else "i64" if isinstance(value, int)
                                             else "f32")
    if isinstance(value, float) and math.isnan(value):
        raise Undefined("NaN fill value")
    v = _elem(R(value) if isinstance(value, (float, str, Fraction)) else value, d)
    return Ten(np.full(list(size), v, dtype=object), d)


def zeros(size, dtype=None, **kw):
    return full(size, 0, dtype or "f32")


def ones(size, dtype=None, **kw):
    return full(size, 1, dtype or "f32")


def full_like(a, value, dtype=None, **kw):
    a = as_ten(a)
    return full(a.shape, value, dtype or a.dtype)


def zeros_like(a, dtype=None, **kw):
    return full_like(a, 0, dtype)


def new_zeros(a, size, dtype=None, **kw):
    return full(size, 0, dtype or as_ten(a).dtype)


def scalar_tensor(value, dtype=None, **kw):
    return full([], value, dtype or ("f32" if not isinstance(value, bool) else "b8"))


def arange(start, end=None, step=1, dtype=None, **kw):
    if end is None:
        start, end = 0, start
    vals, x = [], Fraction(start)
    if step == 0:
        raise Invalid("arange: step is zero")
    n = max(0, math.ceil((Fraction(end) - Fraction(start)) / Fraction(step)))
    vals = [Fraction(start) + i * Fraction(step) for i in range(n)]
    d = dt(dtype) if dtype else ("i64" if all(isinstance(v, int) for v in (start, end, step)) else "f32")
    return tensor([v if d in FLOAT else int(v) for v in vals], d)


def iota(length, start=0, step=1, dtype="i64", **kw):
    return tensor([start + i * step for i in range(length)], dtype)


def lift_fresh_copy(a):
    return clone(a)


def triu(a, diagonal=0):
    a = as_ten(a)
    out = a.a.copy()
    zero = _elem(0, a.dtype)
    for i in np.ndindex(out.shape):
        if i[-1] - i[-2] < diagonal:
            out[i] = zero
    return Ten(out, a.dtype)


def tril(a, diagonal=0):
    a = as_ten(a)
    out = a.a.copy()
    zero = _elem(0, a.dtype)
    for i in np.ndindex(out.shape):
        if i[-1] - i[-2] > diagonal:
            out[i] = zero
    return Ten(out, a.dtype)


def fill(a, value):
    return full_like(a, value)


# ------------------------------------------------------------------ indexing (GATHER / INDEXREDUCE / EMB contracts)
def _ints(t):
    t = as_ten(t)
    if t.dtype in BOOL:
        raise Unsupported("boolean mask indexing")
    return np.vectorize(int, otypes=[object])(t.a) if t.a.size else t.a


def index(a, indices):
    """aten.index: advanced indexing with a list of index tensors (None = full slice). Out-of-range indices are
    invalid (torch raises)."""
    a = as_ten(a)
    idx, adv = [], []
    for k, ix in enumerate(indices):
        if ix is None:
            idx.append(slice(None))
            continue
        arr = _ints(ix).astype(np.int64)
        n = a.a.shape[k]
        if arr.size and (arr.min() < -n or arr.max() >= n):
            raise Invalid("index out of range")
        idx.append(arr % n if n else arr)
    return Ten(a.a[tuple(idx)], a.dtype)


def index_put(a, indices, values, accumulate=False):
    """aten.index_put: with accumulate, duplicates add (exactly, order irrelevant); without, a position written
    more than once may hold any of the written values (documented as undefined) -> an allowed set."""
    a, values = as_ten(a), as_ten(values)
    out = a.a.copy()
    idx = []
    for k, ix in enumerate(indices):
        if ix is None:
            idx.append(slice(None))
        else:
            n = a.a.shape[k]
            arr = _ints(ix).astype(np.int64)
            if arr.size and (arr.min() < -n or arr.max() >= n):
                raise Invalid("index out of range")
            idx.append(arr % n)
    target_shape = a.a[tuple(idx)].shape
    vals = np.broadcast_to(_convert_elems(values, a.dtype), target_shape)
    pos = np.empty(a.a.shape, dtype=object)
    for i in np.ndindex(pos.shape):
        pos[i] = i
    targets = pos[tuple(idx)]
    writes = {}
    for j in np.ndindex(targets.shape):
        writes.setdefault(targets[j], []).append(vals[j])
    for p, vs in writes.items():
        if accumulate:
            s = out[p]
            for v in vs:
                s = s + v
            out[p] = s
        elif len(vs) == 1:
            out[p] = vs[0]
        else:
            uniq = []
            for v in vs:
                if not any(isinstance(v, R) and isinstance(u, R) and v.exact and u.exact and v.v == u.v or v is u
                           or (not isinstance(v, R) and v == u) for u in uniq):
                    uniq.append(v)
            out[p] = uniq[0] if len(uniq) == 1 else AllowedSet(uniq)
    return Ten(out, a.dtype)


class AllowedSet:
    """An element whose value is any member of a finite set (documented nondeterminism)."""
    __slots__ = ("members",)

    def __init__(self, members):
        self.members = list(members)

    def __repr__(self):
        return f"AllowedSet({self.members})"


def gather(a, dim, idx):
    a = as_ten(a)
    ix = _ints(idx)
    n = a.a.shape[dim]
    out = np.empty(ix.shape, dtype=object)
    for i in np.ndindex(ix.shape):
        k = ix[i]
        if not -n <= k < n:
            raise Invalid("gather: index out of range")
        j = list(i)
        j[dim] = k % n
        out[i] = a.a[tuple(j)]
    return Ten(out, a.dtype)


def scatter(a, dim, idx, src=None, value=None, reduce=None):
    a = as_ten(a)
    ix = _ints(idx)
    out = a.a.copy()
    if src is not None:
        s = as_ten(src)
        sv = _convert_elems(s, a.dtype)
    writes = {}
    for i in np.ndindex(ix.shape):
        j = list(i)
        n = a.a.shape[dim]
        if not -n <= ix[i] < n:
            raise Invalid("scatter: index out of range")
        j[dim] = ix[i] % n
        v = sv[i] if src is not None else _elem(R(value) if isinstance(value, (float, Fraction)) else value, a.dtype)
        writes.setdefault(tuple(j), []).append(v)
    for p, vs in writes.items():
        if reduce in ("add", "sum"):
            acc = out[p]
            for v in vs:
                acc = acc + v
            out[p] = acc
        elif reduce in ("multiply", "prod"):
            acc = out[p]
            for v in vs:
                acc = acc * v
            out[p] = acc
        elif reduce is None:
            out[p] = vs[0] if len(vs) == 1 else AllowedSet(vs)
        else:
            raise Unsupported(f"scatter reduce={reduce!r}")
    return Ten(out, a.dtype)


def scatter_add(a, dim, idx, src):
    return scatter(a, dim, idx, src=src, reduce="add")


def index_select(a, dim, idx):
    a = as_ten(a)
    ix = _ints(idx)
    return Ten(np.take(a.a, [int(k) % a.a.shape[dim] for k in ix.reshape(-1)], axis=dim), a.dtype)


def index_add(a, dim, idx, src, alpha=1):
    a, src = as_ten(a), as_ten(src)
    out = a.a.copy()
    ix = _ints(idx).reshape(-1)
    sv = _convert_elems(src, a.dtype)
    for k, j in enumerate(ix):
        sl = [slice(None)] * out.ndim
        sl[dim] = int(j) % out.shape[dim]
        ssl = [slice(None)] * sv.ndim
        ssl[dim] = k
        tgt, add_ = out[tuple(sl)], sv[tuple(ssl)]
        out[tuple(sl)] = np.vectorize(lambda x, y: x + y * alpha, otypes=[object])(tgt, add_)
    return Ten(out, a.dtype)


def embedding(weight, indices, padding_idx=-1, **kw):
    return index_select(weight, 0, indices) if as_ten(indices).a.ndim <= 1 else \
        Ten(np.stack([index_select(weight, 0, Ten(row, "i64")).a for row in as_ten(indices).a]), as_ten(weight).dtype)


def embedding_dense_backward(grad, indices, num_weights, padding_idx=-1, scale_grad_by_freq=False):
    """Gradient of embedding: sum of the output gradients of each index; padding_idx rows get zero; with
    scale_grad_by_freq each index's sum is divided by its count in the batch (torch.nn.Embedding docs)."""
    grad, ix = as_ten(grad), _ints(indices).reshape(-1)
    D = grad.a.shape[-1]
    g = grad.a.reshape(-1, D)
    counts = {}
    for k in ix:
        counts[int(k)] = counts.get(int(k), 0) + 1
    out = np.full((num_weights, D), R(0), dtype=object)
    for r, k in enumerate(ix):
        k = int(k)
        if k == padding_idx:
            continue
        for c in range(D):
            v = g[r, c]
            out[k, c] = out[k, c] + (v / counts[k] if scale_grad_by_freq else v)
    return Ten(out, grad.dtype)


# ------------------------------------------------------------------ reductions (REDUCE / SOFTMAX contracts)
def _axes(a, dim):
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        return tuple(range(a.a.ndim))
    dims = [dim] if isinstance(dim, int) else list(dim)
    return tuple(sorted({d % a.a.ndim for d in dims})) if a.a.ndim else ()


def _reduce(a, dim, keepdim, fn, out_dtype=None):
    axes = _axes(a, dim)
    rest = [i for i in range(a.a.ndim) if i not in axes]
    moved = np.transpose(a.a, rest + list(axes)) if a.a.ndim else a.a.reshape(())
    lead = [a.a.shape[i] for i in rest]
    flat = moved.reshape(lead + [-1]) if a.a.ndim else moved.reshape(1)
    out = np.empty(lead, dtype=object)
    for i in np.ndindex(*lead) if lead else [()]:
        out[i] = fn(list(flat[i]) if lead else list(flat))
    if keepdim:
        out = out.reshape([1 if k in axes else a.a.shape[k] for k in range(a.a.ndim)])
    return Ten(out, out_dtype or a.dtype)


def _rsum(xs, zero):
    s = zero
    for x in xs:
        s = s + x
    return s


def sum_(a, dim=None, keepdim=False, dtype=None):
    a = as_ten(a)
    d = dt(dtype) if dtype else ("i64" if a.dtype in INT | BOOL else a.dtype)
    a = Ten(_convert_elems(a, d), d) if d != a.dtype else a
    return _reduce(a, dim, keepdim, lambda xs: _rsum(xs, R(0) if d in FLOAT else 0), d)


def mean(a, dim=None, keepdim=False, dtype=None):
    a = _floatify(a) if dtype is None else Ten(_convert_elems(as_ten(a), dt(dtype)), dt(dtype))

    def m(xs):
        if not xs:
            raise Undefined("mean of an empty reduction")
        return _rsum(xs, R(0)) / len(xs)
    return _reduce(a, dim, keepdim, m)


def prod(a, dim=None, keepdim=False, dtype=None):
    a = as_ten(a)

    def p(xs):
        s = R(1) if a.dtype in FLOAT else 1
        for x in xs:
            s = s * x
        return s
    return _reduce(a, dim, keepdim, p)


def amax(a, dim=None, keepdim=False):
    a = as_ten(a)

    def m(xs):
        if not xs:
            raise Invalid("amax of an empty reduction (torch raises)")
        out = xs[0]
        for x in xs[1:]:
            out = _maxv(out, x)
        return out
    return _reduce(a, dim, keepdim, m)


def amin(a, dim=None, keepdim=False):
    a = as_ten(a)

    def m(xs):
        if not xs:
            raise Invalid("amin of an empty reduction (torch raises)")
        out = xs[0]
        for x in xs[1:]:
            out = _minv(out, x)
        return out
    return _reduce(a, dim, keepdim, m)


def _arg(xs, better):
    if not xs:
        raise Invalid("arg-reduction of an empty set")
    best, k = xs[0], 0
    for i, x in enumerate(xs[1:], 1):
        if better(x, best):
            best, k = x, i
    return k


def argmax(a, dim=None, keepdim=False):
    """torch.argmax: the index of the first maximal value (documented for ties)."""
    a = as_ten(a)
    if dim is None:
        flat = Ten(a.a.reshape(-1), a.dtype)
        return Ten(np.array(_arg(list(flat.a), lambda x, y: x > y), dtype=object), "i64")
    return _reduce(a, dim, keepdim, lambda xs: _arg(xs, lambda x, y: x > y), "i64")


def argmin(a, dim=None, keepdim=False):
    a = as_ten(a)
    if dim is None:
        return Ten(np.array(_arg(list(a.a.reshape(-1)), lambda x, y: x < y), dtype=object), "i64")
    return _reduce(a, dim, keepdim, lambda xs: _arg(xs, lambda x, y: x < y), "i64")


def max_dim(a, dim, keepdim=False):
    """torch.max(input, dim): (values, indices of the first maximal value)."""
    return amax(a, dim, keepdim), argmax(a, dim, keepdim)


def min_dim(a, dim, keepdim=False):
    return amin(a, dim, keepdim), argmin(a, dim, keepdim)


def var(a, dim=None, correction=1, keepdim=False):
    a = _floatify(a)

    def v(xs):
        n = len(xs)
        if n - correction <= 0:
            raise Undefined("variance with non-positive degrees of freedom")
        m = _rsum(xs, R(0)) / n
        return _rsum([(x - m) * (x - m) for x in xs], R(0)) / (n - correction)
    return _reduce(a, dim, keepdim, v)


def var_mean(a, dim=None, correction=1, keepdim=False):
    return var(a, dim, correction, keepdim), mean(a, dim, keepdim)


def std(a, dim=None, correction=1, keepdim=False):
    return unary(rl.sqrt, var(a, dim, correction, keepdim))


def cumsum(a, dim, dtype=None):
    a = as_ten(a)
    d = dt(dtype) if dtype else ("i64" if a.dtype in INT | BOOL else a.dtype)
    arr = _convert_elems(a, d)
    out = np.empty(arr.shape, dtype=object)
    dim = dim % arr.ndim
    for i in np.ndindex(*[s for k, s in enumerate(arr.shape) if k != dim]):
        acc = R(0) if d in FLOAT else 0
        for j in range(arr.shape[dim]):
            ix = list(i)
            ix.insert(dim, j)
            acc = acc + arr[tuple(ix)]
            out[tuple(ix)] = acc
    return Ten(out, d)


def logsumexp(a, dim, keepdim=False):
    a = _floatify(a)

    def lse(xs):
        if not xs:
            return R(rl.NEG)                      # log of an empty sum of exponentials: log 0 (documented -inf)
        m = xs[0]
        for x in xs[1:]:
            m = rl.max_(m, x)
        if m.v == rl.NEG:
            return R(rl.NEG)
        if m.v == rl.POS:
            return R(rl.POS)
        return m + rl.log(_rsum([rl.exp(x - m) for x in xs], R(0)))
    return _reduce(a, dim, keepdim, lse)


def norm(a, p=2, dim=None, keepdim=False):
    a = _floatify(a)
    if p == 2:
        return _reduce(a, dim, keepdim, lambda xs: rl.sqrt(_rsum([x * x for x in xs], R(0))))
    if p == 1:
        return _reduce(a, dim, keepdim, lambda xs: _rsum([rl.abs_(x) for x in xs], R(0)))
    if p in ("inf", float("inf")):
        return amax(abs_(a), dim, keepdim)
    return _reduce(a, dim, keepdim, lambda xs: _rsum([rl.abs_(x) ** p for x in xs], R(0)) ** R(Fraction(1, 1) / p))


def _softmax_row(xs):
    m = xs[0]
    for x in xs[1:]:
        m = rl.max_(m, x)
    if m.v == rl.NEG:
        raise Undefined("softmax of a row whose entries are all -inf")
    if m.v == rl.POS:
        raise Undefined("softmax of a row with +inf")
    es = [rl.exp(x - m) for x in xs]
    s = _rsum(es, R(0))
    return [e / s for e in es]


def _rowwise(a, dim, fn):
    a = _floatify(a)
    dim = dim % a.a.ndim
    out = np.empty(a.a.shape, dtype=object)
    for i in np.ndindex(*[s for k, s in enumerate(a.a.shape) if k != dim]):
        idx = [list(i[:dim]) + [j] + list(i[dim:]) for j in range(a.a.shape[dim])]
        vals = fn([a.a[tuple(ix)] for ix in idx])
        for ix, v in zip(idx, vals):
            out[tuple(ix)] = v
    return Ten(out, a.dtype)


def softmax(a, dim, half_to_float=False):
    return _rowwise(a, dim, _softmax_row)


def safe_softmax(a, dim):
    """aten._safe_softmax: softmax, except rows whose entries are all -inf give zeros (SDPA's documented choice)."""
    def row(xs):
        if all(x.v == rl.NEG for x in xs):
            return [R(0)] * len(xs)
        return _softmax_row(xs)
    return _rowwise(a, dim, row)


def log_softmax(a, dim, half_to_float=False):
    def row(xs):
        m = xs[0]
        for x in xs[1:]:
            m = rl.max_(m, x)
        if not m.finite:
            raise Undefined("log_softmax of a row that is all -inf or has +inf")
        lse = m + rl.log(_rsum([rl.exp(x - m) for x in xs], R(0)))
        return [x - lse for x in xs]
    return _rowwise(a, dim, row)


def softmax_backward_data(grad, out, dim, input_dtype=None):
    """d softmax: g_in = y * (g - sum(g * y)) along dim."""
    g, y = _floatify(grad), _floatify(out)
    s = sum_(mul(g, y), dim, keepdim=True)
    return mul(y, sub(g, s))


def log_softmax_backward_data(grad, out, dim, input_dtype=None):
    """d log_softmax: g_in = g - exp(y) * sum(g) along dim."""
    g, y = _floatify(grad), _floatify(out)
    return sub(g, mul(exp(y), sum_(g, dim, keepdim=True)))


def prepare_softmax_online(a, dim):
    """prims.prepare_softmax_online (Inductor, torch 2.10): (amax(x, dim, keepdim), sum(exp(x - amax), dim, keepdim))."""
    m = amax(a, dim, keepdim=True)
    return m, sum_(exp(sub(a, m)), dim, keepdim=True)


# ------------------------------------------------------------------ matrix products (MATMUL / CONV contracts)
def _dot(xs, ys):
    s = R(0)
    for x, y in zip(xs, ys):
        s = s + x * y
    return s


def mm(a, b):
    a, b = _floatify(a), _floatify(b)
    if a.a.ndim != 2 or b.a.ndim != 2 or a.a.shape[1] != b.a.shape[0]:
        raise Invalid(f"mm: shapes {a.shape} x {b.shape}")
    m, k = a.a.shape
    n = b.a.shape[1]
    out = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            out[i, j] = _dot(a.a[i, :], b.a[:, j]) if k else R(0)
    return Ten(out, promote(a.dtype, b.dtype))


def bmm(a, b):
    a, b = _floatify(a), _floatify(b)
    if a.a.ndim != 3 or b.a.ndim != 3 or a.a.shape[0] != b.a.shape[0]:
        raise Invalid(f"bmm: shapes {a.shape} x {b.shape}")
    return Ten(np.stack([mm(Ten(a.a[i], a.dtype), Ten(b.a[i], b.dtype)).a for i in range(a.a.shape[0])])
               if a.a.shape[0] else np.empty((0, a.a.shape[1], b.a.shape[2]), dtype=object), promote(a.dtype, b.dtype))


def matmul(a, b):
    """torch.matmul: 1-d operands are promoted and the added dimension removed; batch dimensions broadcast."""
    a, b = _floatify(a), _floatify(b)
    if a.a.ndim == 1 and b.a.ndim == 1:
        if a.a.shape != b.a.shape:
            raise Invalid("matmul: vector lengths differ")
        return Ten(np.array(_dot(a.a, b.a), dtype=object), promote(a.dtype, b.dtype))
    va, vb = a.a.ndim == 1, b.a.ndim == 1
    A = a.a.reshape(1, -1) if va else a.a
    B = b.a.reshape(-1, 1) if vb else b.a
    batch = np.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    A = np.broadcast_to(A, batch + A.shape[-2:])
    B = np.broadcast_to(B, batch + B.shape[-2:])
    out = np.empty(batch + (A.shape[-2], B.shape[-1]), dtype=object)
    for i in np.ndindex(*batch) if batch else [()]:
        out[i] = mm(Ten(A[i], a.dtype), Ten(B[i], b.dtype)).a
    if va:
        out = out[..., 0, :]
    if vb:
        out = out[..., 0]
    return Ten(out, promote(a.dtype, b.dtype))


def addmm(c, a, b, beta=1, alpha=1):
    """torch.addmm: beta * c + alpha * (a @ b); with beta = 0 the input c is ignored (documented, even inf/nan)."""
    p = mm(a, b)
    p = mul(p, alpha) if alpha != 1 else p
    if beta == 0:
        return p
    return add(mul(c, beta) if beta != 1 else c, p)


def baddbmm(c, a, b, beta=1, alpha=1):
    p = bmm(a, b)
    p = mul(p, alpha) if alpha != 1 else p
    return p if beta == 0 else add(mul(c, beta) if beta != 1 else c, p)


def convolution(x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    """aten.convolution (non-transposed): cross-correlation over the spatial dimensions with zero padding,
    stride, dilation and groups (torch.nn.functional.conv1d/2d/3d docs)."""
    if transposed:
        raise Unsupported("transposed convolution")
    x, w = _floatify(x), _floatify(w)
    nsp = x.a.ndim - 2
    stride, padding, dilation = ([v] * nsp if isinstance(v, int) else list(v) for v in (stride, padding, dilation))
    N, Cin = x.a.shape[:2]
    Cout, cpg = w.a.shape[:2]
    if Cin != cpg * groups or Cout % groups:
        raise Invalid("convolution: channel counts do not match groups")
    xp = constant_pad_nd(x, [p for pd in reversed(padding) for p in (pd, pd)], 0).a
    ksz = w.a.shape[2:]
    osz = [(xp.shape[2 + d] - dilation[d] * (ksz[d] - 1) - 1) // stride[d] + 1 for d in range(nsp)]
    out = np.empty([N, Cout] + osz, dtype=object)
    opg = Cout // groups
    for n in range(N):
        for co in range(Cout):
            g = co // opg
            for o in np.ndindex(*osz):
                s = R(0) if bias is None else _floatify(bias).a[co]
                for ci in range(cpg):
                    for k in np.ndindex(*ksz):
                        pos = tuple(o[d] * stride[d] + k[d] * dilation[d] for d in range(nsp))
                        s = s + xp[(n, g * cpg + ci) + pos] * w.a[(co, ci) + k]
                out[(n, co) + o] = s
    return Ten(out, x.dtype)


# ------------------------------------------------------------------ activations (ACT contract)
def gelu(a, approximate="none"):
    """torch.nn.GELU: x * Phi(x) with Phi(x) = (1 + erf(x / sqrt 2)) / 2; approximate='tanh':
    0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). The tanh form is its own documented function, not an
    approximation of the erf form, when selected."""
    a = _floatify(a)
    if approximate == "none":
        return unary(lambda x: x * (1 + rl.erf(x / rl.sqrt(R(2)))) / 2, a)
    if approximate == "tanh":
        c = rl.sqrt(R(2) / R(_pi()))
        return unary(lambda x: x * (1 + rl.tanh(c * (x + R(Fraction(44715, 1000000)) * x * x * x))) / 2, a)
    raise Invalid(f"gelu approximate={approximate!r}")


def _pi():
    from mpmath import iv
    return iv.mpf(1) * iv.pi


def gelu_backward(grad, a, approximate="none"):
    g, a = _floatify(grad), _floatify(a)
    if approximate == "none":
        def d(x):
            cdf = (1 + rl.erf(x / rl.sqrt(R(2)))) / 2
            pdf = rl.exp(-(x * x) / 2) / rl.sqrt(2 * R(_pi()))
            return cdf + x * pdf
    else:
        c = rl.sqrt(R(2) / R(_pi()))
        k = R(Fraction(44715, 1000000))

        def d(x):
            u = c * (x + k * x * x * x)
            th = rl.tanh(u)
            return (1 + th) / 2 + x * (1 - th * th) * c * (1 + 3 * k * x * x) / 2
    return mul(g, unary(d, a))


def silu(a):
    return unary(lambda x: x * rl.sigmoid(x), _floatify(a))


def silu_backward(grad, a):
    def d(x):
        s = rl.sigmoid(x)
        return s * (1 + x * (1 - s))
    return mul(_floatify(grad), unary(d, _floatify(a)))


def threshold_backward(grad, a, threshold):
    """Gradient of threshold / relu: grad where input > threshold, else 0 (so relu'(0) = 0)."""
    a = as_ten(a)
    return where(gt(a, threshold), grad, as_ten(0, as_ten(grad)))


def tanh_backward(grad, y):
    return mul(grad, unary(lambda v: 1 - v * v, _floatify(y)))


def sigmoid_backward(grad, y):
    return mul(grad, unary(lambda v: v * (1 - v), _floatify(y)))


def softplus(a, beta=1, threshold=20):
    return unary(lambda x: rl.softplus(x, beta, threshold), _floatify(a))


def softplus_backward(grad, a, beta=1, threshold=20):
    def d(x):
        bx = R(beta) * x
        return R(1) if bx > R(threshold) else rl.sigmoid(bx)
    return mul(grad, unary(d, _floatify(a)))


# ------------------------------------------------------------------ randomness (RANDOM contract): conditioned on the kernel's numbers
def inductor_lookup_seed(seeds, i):
    s = as_ten(seeds)
    return Ten(np.array(s.a[i], dtype=object), s.dtype)


def inductor_random(*args, **kw):
    raise Unsupported("the random stream is implementation-defined: the RANDOM contract conditions on the kernel's own "
                      "random numbers, which the binding supplies as an input")


# ------------------------------------------------------------------ dispatch by ATen name (graph fragments)
OPS = {
    "aten.add": add, "aten.sub": sub, "aten.rsub": rsub, "aten.mul": mul, "aten.div": div, "aten.neg": neg,
    "aten.reciprocal": reciprocal, "aten.abs": abs_, "aten.sign": sign, "aten.pow": pow_, "prims.fma": fma,
    "aten.exp": exp, "aten.log": log, "aten.log1p": log1p, "aten.expm1": expm1, "aten.sqrt": sqrt, "aten.rsqrt": rsqrt,
    "aten.sin": sin, "aten.cos": cos, "aten.tanh": tanh, "aten.sigmoid": sigmoid, "aten.erf": erf, "aten.floor": floor,
    "aten.ceil": ceil, "aten.relu": relu, "aten.maximum": maximum, "aten.minimum": minimum, "aten.clamp": clamp,
    "aten.clamp_min": clamp_min, "aten.clamp_max": clamp_max, "aten.where": where, "aten.masked_fill": masked_fill,
    "aten.eq": eq, "aten.ne": ne, "aten.lt": lt, "aten.le": le, "aten.gt": gt, "aten.ge": ge,
    "aten.logical_and": logical_and, "aten.logical_or": logical_or, "aten.logical_not": logical_not,
    "aten.bitwise_and": bitwise_and, "aten.bitwise_or": bitwise_or, "aten.bitwise_not": bitwise_not,
    "aten.any": any_, "aten.all": all_,
    "prims.convert_element_type": convert_element_type, "aten._to_copy": _to_copy,
    "aten.clone": clone, "aten.copy": lambda dst, src, *a: clone(src), "aten.view": view, "aten.reshape": reshape,
    "aten._unsafe_view": view, "aten.permute": permute, "aten.transpose": transpose, "aten.t": t, "aten.expand": expand,
    "aten.unsqueeze": unsqueeze, "aten.squeeze": squeeze, "aten.select": select, "aten.slice": slice_, "aten.cat": cat,
    "aten.stack": stack, "aten.split": split, "aten.split_with_sizes": split_with_sizes,
    "aten.constant_pad_nd": constant_pad_nd, "aten.select_scatter": select_scatter, "aten.slice_scatter": slice_scatter,
    "aten.full": full, "aten.zeros": zeros, "aten.ones": ones, "aten.full_like": full_like, "aten.zeros_like": zeros_like,
    "aten.new_zeros": new_zeros, "aten.scalar_tensor": scalar_tensor, "aten.arange": arange, "prims.iota": iota,
    "aten.lift_fresh_copy": lift_fresh_copy, "aten.triu": triu, "aten.tril": tril, "aten.fill": fill,
    "aten.index": index, "aten.index_put": index_put, "aten.index_put_": index_put, "aten.gather": gather,
    "aten.scatter": scatter, "aten.scatter_add": scatter_add, "aten.index_select": index_select,
    "aten.index_add": index_add, "aten.embedding": embedding, "aten.embedding_dense_backward": embedding_dense_backward,
    "aten.sum": sum_, "aten.mean": mean, "aten.prod": prod, "aten.amax": amax, "aten.amin": amin, "aten.argmax": argmax,
    "aten.argmin": argmin, "aten.max": max_dim, "aten.min": min_dim, "aten.var": var, "aten.var_mean": var_mean,
    "aten.std": std, "aten.cumsum": cumsum, "aten.logsumexp": logsumexp, "aten.norm": norm,
    "aten._softmax": softmax, "aten._safe_softmax": safe_softmax, "aten._log_softmax": log_softmax,
    "aten._softmax_backward_data": softmax_backward_data, "aten._log_softmax_backward_data": log_softmax_backward_data,
    "prims.prepare_softmax_online": prepare_softmax_online,
    "aten.mm": mm, "aten.bmm": bmm, "aten.matmul": matmul, "aten.addmm": addmm, "aten.baddbmm": baddbmm,
    "aten.convolution": convolution,
    "aten.gelu": gelu, "aten.gelu_backward": gelu_backward, "aten.silu": silu, "aten.silu_backward": silu_backward,
    "aten.threshold_backward": threshold_backward, "aten.tanh_backward": tanh_backward,
    "aten.sigmoid_backward": sigmoid_backward, "aten.softplus": softplus, "aten.softplus_backward": softplus_backward,
    "prims.inductor_lookup_seed": inductor_lookup_seed, "prims.inductor_random": inductor_random,
}
