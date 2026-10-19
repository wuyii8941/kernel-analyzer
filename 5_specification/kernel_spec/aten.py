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
from .num import Invalid, NotEstablished, Undefined, Unsupported, integer
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


def tensor(values, dtype, *, allow_ignored_nan=False):
    """Convert tensor values to exact numbers. NaN storage may be retained only
    when the caller has established that the operation ignores these values.
    """
    d = dt(dtype)
    try:
        import torch
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu()
            values = (values.double() if values.is_floating_point() else values).numpy()
    except ImportError:
        pass
    arr = np.array(values, dtype=object)
    def real_input(x):
        if isinstance(x, (float, np.floating)) and math.isnan(x):
            if allow_ignored_nan:
                return float("nan")  # Retained input storage, never a real-number value.
            raise Invalid("NaN is not a valid real tensor input")
        return R(x)
    conv = real_input if d in FLOAT else (lambda x: bool(x)) if d in BOOL else (lambda x: _int(x))
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
        return x if isinstance(x, R) else R(int(x)) if isinstance(x, bool) else R(x)
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


def _broadcast_shape(*shapes):
    shapes = [_layout_sizes(shape, "broadcast shape") for shape in shapes]
    if any(n < 0 for shape in shapes for n in shape):
        raise Invalid("broadcast shapes must have nonnegative dimensions")
    common = [1] * max((len(shape) for shape in shapes), default=0)
    for shape in shapes:
        for axis in range(-1, -len(shape)-1, -1):
            if common[axis] == 1:
                common[axis] = shape[axis]
            elif shape[axis] != 1 and shape[axis] != common[axis]:
                raise Invalid(f"shapes {shapes} do not broadcast")
    return tuple(common)


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
def _arithmetic_operands(a, b, *, convert=as_ten, tensor_types=(Ten,)):
    if not isinstance(a, tensor_types) and not isinstance(b, tensor_types):
        a = convert(a)
    if not isinstance(a, tensor_types):
        a = convert(a, b)
    if not isinstance(b, tensor_types):
        b = convert(b, a)
    return convert(a), convert(b)


def _arithmetic_alpha(dtype, alpha):
    if isinstance(alpha, bool) and dtype not in BOOL:
        raise Invalid("boolean alpha requires a boolean result")
    if dtype not in FLOAT:
        if isinstance(alpha, (float, np.floating)):
            raise Invalid("floating alpha is invalid for integer or boolean results")
        alpha = int(alpha) if isinstance(alpha, bool) else integer(alpha, "alpha")
    al = R(alpha) if not isinstance(alpha, bool) else R(int(alpha))
    return alpha, al


def add(a, b, alpha=1):
    a, b = _arithmetic_operands(a, b)
    alpha, al = _arithmetic_alpha(promote(a.dtype, b.dtype), alpha)
    return binary(lambda x, y: x + y * al if isinstance(x, R) else _iadd(x, y, alpha), a, b)


def _iadd(x, y, alpha):
    if isinstance(x, bool) and isinstance(y, bool):
        return x or (bool(alpha) and y)
    return x + int(alpha) * y


def sub(a, b, alpha=1):
    a, b = _arithmetic_operands(a, b)
    if a.dtype in BOOL or b.dtype in BOOL:
        raise Invalid("subtraction of booleans is not supported (torch.sub)")
    return add(a, neg(b), alpha)


def _num(x):
    return R(x) if isinstance(x, (Fraction, float, str)) else x


def rsub(a, b, alpha=1):
    return sub(b, a, alpha)


def mul(a, b):
    return binary(lambda x, y: (x and y) if isinstance(x, bool) and isinstance(y, bool) else x * y, a, b)


def _division_dtype(a, b, rounding_mode):
    if rounding_mode is not None and (not isinstance(rounding_mode, str) or rounding_mode not in ("trunc", "floor")):
        raise Invalid(f"rounding_mode {rounding_mode!r}")
    dtype = promote(a.dtype, b.dtype)
    if rounding_mode is None:
        return dtype if dtype in FLOAT else "f32"
    if dtype in BOOL:
        raise Unsupported("rounded boolean division has no implemented semantics")
    return dtype


def _division_value(x, y, rounding_mode):
    if isinstance(x, R) or isinstance(y, R):
        quotient = R(x) / R(y)
        if rounding_mode is None:
            return quotient
        return rl.trunc(quotient) if rounding_mode == "trunc" else rl.floor(quotient)
    if y == 0:
        raise Invalid("integer division by zero (torch raises)")
    quotient = Fraction(x, y)
    return math.trunc(quotient) if rounding_mode == "trunc" else math.floor(quotient)


def div(a, b, rounding_mode=None):
    a, b = _arithmetic_operands(a, b)
    dtype = _division_dtype(a, b, rounding_mode)
    return binary(lambda x, y: _division_value(x, y, rounding_mode), a, b, compute=dtype)


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


def _power_operands(a, b, *, convert=as_ten, tensor_types=(Ten,)):
    integer_scalar = isinstance(b, (int, np.integer)) and not isinstance(b, (bool, np.bool_))
    scalar_exponent = int(b) if integer_scalar else None
    a, b = _arithmetic_operands(a, b, convert=convert, tensor_types=tensor_types)
    dtype = promote(a.dtype, b.dtype)
    if dtype in BOOL:
        raise Unsupported("boolean power has no implemented semantics")
    if dtype in INT and scalar_exponent is not None and scalar_exponent < 0:
        raise Invalid("integers to negative integer powers are not allowed (torch.pow)")
    return a, b, dtype


def _power_value(x, y):
    if isinstance(x, R) or isinstance(y, R):
        return R(x) ** R(y)
    if y < 0:
        raise NotEstablished("negative integer tensor exponents require an explicit integer-result policy")
    return int(x) ** int(y)


def pow_(a, b):
    """torch.pow: 0**0 = 1; a negative base with a non-integer exponent has no real value."""
    a, b, dtype = _power_operands(a, b)
    return binary(_power_value, a, b, compute=dtype)


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


def _where_condition(cond):
    if cond.dtype not in BOOL:
        raise Invalid("where: condition must be boolean")


def where(cond, a, b):
    cond = as_ten(cond)
    _where_condition(cond)
    if not isinstance(a, Ten) and not isinstance(b, Ten):
        a, b = as_ten(a), as_ten(b)
    a = a if isinstance(a, Ten) else as_ten(a, b)
    b = b if isinstance(b, Ten) else as_ten(b, a)
    d = promote(a.dtype, b.dtype)
    _broadcast_shape(cond.shape, a.shape, b.shape)
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
    _unimplemented_options(kw)
    return convert_element_type(a, dtype) if dtype is not None else clone(a)


# ------------------------------------------------------------------ layout and creation
def clone(a, **kw):
    _unimplemented_options(kw)
    a = as_ten(a)
    return Ten(a.a.copy(), a.dtype)


def _copy_options(non_blocking):
    # The logical values do not model scheduling; the schema still requires bool.
    if not isinstance(non_blocking, (bool, np.bool_)):
        raise Invalid("copy non_blocking must be boolean")


def copy(dst, src, non_blocking=False):
    """aten.copy (functional Tensor.copy_): the values of src broadcast to dst's shape and converted to dst's dtype."""
    _copy_options(non_blocking)
    dst, src = as_ten(dst), as_ten(src)
    s = expand(src, list(dst.shape)) if src.shape != dst.shape else src
    return s if s.dtype == dst.dtype else convert_element_type(s, dst.dtype)


def _layout_sizes(values, what):
    if not isinstance(values, (list, tuple)):
        raise Invalid(f"{what}: expected a sequence of integers")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in values):
        raise Invalid(f"{what}: sizes and dimensions must have integer types")
    return [int(v) for v in values]


def _layout_axis(dim, rank):
    dim = _layout_sizes([dim], "layout axis")[0]
    if not -rank <= dim < rank:
        raise Invalid("layout axis out of range")
    return dim % rank


def _view_shape(source_shape, shape):
    shape = _layout_sizes(shape, "view shape")
    n = math.prod(source_shape)
    if any(s < -1 for s in shape):
        raise Invalid("view: sizes must be nonnegative or -1")
    if shape.count(-1) > 1:
        raise Invalid("view: more than one -1")
    if -1 in shape:
        known = math.prod(s for s in shape if s != -1)
        if known == 0:
            raise Invalid("view: cannot infer a size when the other sizes have zero product")
        if n % known:
            raise Invalid("view: inferred size is not an integer")
        shape[shape.index(-1)] = n // known
    if math.prod(shape) != n:
        raise Invalid(f"view: {source_shape} -> {shape}")
    return shape


def view(a, shape):
    a = as_ten(a)
    return Ten(a.a.reshape(_view_shape(a.shape, shape)), a.dtype)


reshape = view


def _permutation(rank, dims):
    dims = _layout_sizes(dims, "permute dimensions")
    if len(dims) != rank or any(not -rank <= d < rank for d in dims):
        raise Invalid("permute: dimensions must enumerate the existing axes")
    axes = [d % rank for d in dims]
    if len(set(axes)) != rank:
        raise Invalid("permute: repeated dimension")
    return axes


def permute(a, dims):
    a = as_ten(a)
    return Ten(np.transpose(a.a, _permutation(a.a.ndim, dims)), a.dtype)


def transpose(a, d0, d1):
    a = as_ten(a)
    rank = max(1, a.a.ndim)
    d0, d1 = _layout_axis(d0, rank), _layout_axis(d1, rank)
    if a.a.ndim == 0:
        return a
    return Ten(np.swapaxes(a.a, d0, d1), a.dtype)


def t(a):
    a = as_ten(a)
    if a.a.ndim > 2:
        raise Invalid("t requires a tensor with at most two dimensions")
    return a if a.a.ndim < 2 else transpose(a, 0, 1)


def _expand_shape(source_shape, shape):
    shape = _layout_sizes(shape, "expand shape")
    off = len(shape) - len(source_shape)
    if off < 0 or any(s < -1 for s in shape) or any(s == -1 for s in shape[:off]):
        raise Invalid("expand: cannot remove axes or infer the size of a new axis")
    full = [source_shape[i - off] if s == -1 else s for i, s in enumerate(shape)]
    if any(old != 1 and old != new for old, new in zip(source_shape, full[off:])):
        raise Invalid(f"expand: {source_shape} -> {shape}")
    return full


def expand(a, shape):
    a = as_ten(a)
    return Ten(np.broadcast_to(a.a, _expand_shape(a.shape, shape)).copy(), a.dtype)


def unsqueeze(a, d):
    a = as_ten(a)
    return Ten(np.expand_dims(a.a, _layout_axis(d, a.a.ndim + 1)), a.dtype)


def _squeeze_axes(shape, dims):
    if dims is None:
        return tuple(i for i, size in enumerate(shape) if size == 1)
    dims = dims if isinstance(dims, (list, tuple)) else [dims]
    axes = [_layout_axis(d, max(1, len(shape))) for d in dims]
    if len(set(axes)) != len(axes):
        raise Invalid("squeeze: repeated dimension")
    return tuple(d for d in axes if shape and shape[d] == 1)


def squeeze(a, dims=None):
    a = as_ten(a)
    axes = _squeeze_axes(a.shape, dims)
    return Ten(np.squeeze(a.a, axis=axes), a.dtype)


def _select_coordinate(shape, dim, index):
    dim = _layout_axis(dim, len(shape))
    index = _layout_sizes([index], "select index")[0]
    n = shape[dim]
    if not -n <= index < n:
        raise Invalid(f"select: index {index} out of range for size {n}")
    return dim, index % n


def select(a, dim, index):
    a = as_ten(a)
    dim, index = _select_coordinate(a.shape, dim, index)
    return Ten(np.take(a.a, index, axis=dim), a.dtype)


def _slice_bounds(n, start, end, step):
    start = 0 if start is None else start
    end = n if end is None else end
    start, end, step = _layout_sizes([start, end, step], "slice bounds")
    if step <= 0:
        raise Invalid("slice step must be positive")
    start = start + n if start < 0 else start
    end = end + n if end < 0 else end
    return max(0, min(start, n)), max(0, min(end, n)), step


def slice_(a, dim=0, start=None, end=None, step=1):
    a = as_ten(a)
    dim = _layout_axis(dim, a.a.ndim)
    s, e, st = _slice_bounds(a.a.shape[dim], start, end, step)
    idx = [slice(None)] * a.a.ndim
    idx[dim] = slice(s, e, st)
    return Ten(a.a[tuple(idx)], a.dtype)


def _cat_layout(tensors, dim):
    if not tensors:
        raise Invalid("cat requires at least one tensor")
    # Empty 1-D inputs contribute dtype even though they contribute no coordinates.
    dtype = promote(*[t.dtype for t in tensors])
    dim = _layout_sizes([dim], "cat axis")[0]
    ts = [t for t in tensors if tuple(t.shape) != (0,)]
    if not ts:
        # TS_REFS.cat returns the empty result before canonicalizing the axis.
        return tensors[:1], 0, dtype
    rank = len(ts[0].shape)
    dim = _layout_axis(dim, rank)
    for t in ts:
        if len(t.shape) != rank or any(t.shape[i] != ts[0].shape[i] for i in range(rank) if i != dim):
            raise Invalid("cat: shapes do not match outside the concatenation axis")
    return ts, dim, dtype


def cat(tensors, dim=0):
    ts, dim, dtype = _cat_layout([as_ten(x) for x in tensors], dim)
    return Ten(np.concatenate([_convert_elems(x, dtype) for x in ts], axis=dim), dtype)


def stack(tensors, dim=0):
    return cat([unsqueeze(x, dim) for x in tensors], dim)


def _split_sections(shape, sizes, dim):
    dim = _layout_axis(dim, len(shape))
    sizes = _layout_sizes(sizes, "split sizes")
    if any(n < 0 for n in sizes) or sum(sizes) != shape[dim]:
        raise Invalid("split_with_sizes: nonnegative sizes must sum to the axis size")
    return dim, sizes


def _split_size(shape, size, dim):
    dim = _layout_axis(dim, len(shape))
    size = _layout_sizes([size], "split size")[0]
    n = shape[dim]
    if size < 0 or (size == 0 and n != 0):
        raise Invalid("split: size must be positive, or zero for an empty axis")
    return dim, ([min(size, n - i) for i in range(0, n, size)] if n else [0])


def split(a, size, dim=0):
    a = as_ten(a)
    dim, sizes = _split_size(a.shape, size, dim)
    return split_with_sizes(a, sizes, dim)


def split_with_sizes(a, sizes, dim=0):
    a = as_ten(a)
    dim, sizes = _split_sections(a.shape, sizes, dim)
    out, start = [], 0
    for s in sizes:
        out.append(slice_(a, dim, start, start + s))
        start += s
    return out


def _padding_layout(input_shape, pad):
    pad = _layout_sizes(pad, "padding")
    rank = len(input_shape)
    if len(pad) % 2 or len(pad) > 2 * rank:
        raise Invalid("padding must contain pairs for existing dimensions")
    lo, hi = [0] * rank, [0] * rank
    for k in range(len(pad) // 2):
        dim = rank - 1 - k
        lo[dim], hi[dim] = pad[2 * k:2 * k + 2]
        # Cropping happens before padding; a positive opposite pad cannot
        # make an out-of-range crop valid (TS_REFS.CONSTANT_PAD).
        if input_shape[dim] + min(lo[dim], 0) + min(hi[dim], 0) < 0:
            raise Invalid("negative padding crops beyond the input dimension")
    shape = [n + left + right for n, left, right in zip(input_shape, lo, hi)]
    return lo, hi, shape


def constant_pad_nd(a, pad, value=0):
    """torch.nn.functional.pad (constant): pad pairs from the last dimension backwards; negative pads crop."""
    a = as_ten(a)
    _padding_layout(a.shape, pad)
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


def _select_scatter_layout(shape, source_shape, dim, index):
    dim, index = _select_coordinate(shape, dim, index)
    if tuple(source_shape) != tuple(shape[:dim]) + tuple(shape[dim + 1:]):
        raise Invalid("select_scatter: source shape does not match the selected slice")
    return dim, index


def _slice_scatter_layout(shape, source_shape, dim, start, end, step):
    dim = _layout_axis(dim, len(shape))
    start, end, step = _slice_bounds(shape[dim], start, end, step)
    expected = list(shape)
    expected[dim] = max(0, (end - start + step - 1) // step)
    if tuple(source_shape) != tuple(expected):
        raise Invalid("slice_scatter: source shape does not match the slice")
    return dim, start, end, step


def select_scatter(base, src, dim, index):
    base, src = as_ten(base), as_ten(src)
    dim, index = _select_scatter_layout(base.shape, src.shape, dim, index)
    out = base.a.copy()
    idx = [slice(None)] * out.ndim
    idx[dim] = index
    out[tuple(idx)] = _convert_elems(src, base.dtype)
    return Ten(out, base.dtype)


def slice_scatter(base, src, dim=0, start=None, end=None, step=1):
    base, src = as_ten(base), as_ten(src)
    dim, s, e, st = _slice_scatter_layout(base.shape, src.shape, dim, start, end, step)
    out = base.a.copy()
    idx = [slice(None)] * out.ndim
    idx[dim] = slice(s, e, st)
    out[tuple(idx)] = _convert_elems(src, base.dtype)
    return Ten(out, base.dtype)


def _unimplemented_options(options):
    if options:
        raise Unsupported("keyword options have no implemented semantics: " + ", ".join(sorted(options)))


def _creation_shape(size):
    size = _layout_sizes(size, "creation shape")
    if any(n < 0 for n in size):
        raise Invalid("creation shape requires nonnegative dimensions")
    return size


def full(size, value, dtype=None, **kw):
    _unimplemented_options(kw)
    size = _creation_shape(size)
    d = dt(dtype) if dtype is not None else ("b8" if isinstance(value, bool) else "i64" if isinstance(value, int)
                                             else "f32")
    if isinstance(value, float) and math.isnan(value):
        raise Undefined("NaN fill value")
    v = _elem(R(value) if isinstance(value, (float, str, Fraction)) else value, d)
    return Ten(np.full(list(size), v, dtype=object), d)


def zeros(size, dtype=None, **kw):
    _unimplemented_options(kw)
    return full(size, 0, dtype or "f32")


def ones(size, dtype=None, **kw):
    _unimplemented_options(kw)
    return full(size, 1, dtype or "f32")


def full_like(a, value, dtype=None, **kw):
    _unimplemented_options(kw)
    a = as_ten(a)
    return full(a.shape, value, dtype or a.dtype)


def zeros_like(a, dtype=None, **kw):
    _unimplemented_options(kw)
    return full_like(a, 0, dtype)


def new_zeros(a, size, dtype=None, **kw):
    _unimplemented_options(kw)
    return full(size, 0, dtype or as_ten(a).dtype)


def scalar_tensor(value, dtype=None, **kw):
    _unimplemented_options(kw)
    return full([], value, dtype or ("f32" if not isinstance(value, bool) else "b8"))


def arange(start, end=None, step=1, dtype=None, **kw):
    _unimplemented_options(kw)
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
    _unimplemented_options(kw)
    return tensor([start + i * step for i in range(length)], dtype)


def lift_fresh_copy(a):
    return clone(a)


def linspace(start, end, steps, dtype=None, **kw):
    """torch.linspace: steps values evenly spaced from start to end inclusive, start + i (end - start)/(steps - 1)."""
    _unimplemented_options(kw)
    if steps < 0:
        raise Invalid("linspace: negative steps")
    s, e = Fraction(start), Fraction(end)
    vals = [s] if steps == 1 else [s + i * (e - s) / (steps - 1) for i in range(steps)]
    return tensor(vals, dtype or "f32")


def narrow(a, dim, start, length):
    a = as_ten(a)
    n = a.a.shape[dim]
    start = start + n if start < 0 else start
    if not (0 <= start and start + length <= n and length >= 0):
        raise Invalid("narrow: range outside the dimension")
    return slice_(a, dim, start, start + length)


def take_along_dim(a, idx, dim=None):
    """torch.take_along_dim (numpy take_along_axis): indices broadcast against input in every dimension except dim;
    with dim None the input is flattened."""
    a, idx = as_ten(a), as_ten(idx)
    if dim is None:
        return gather(view(a, [-1]), 0, view(idx, [-1]))
    dim = dim % a.a.ndim
    if idx.a.ndim != a.a.ndim:
        raise Invalid("take_along_dim: input and indices must have the same number of dimensions")
    other = [max(a.shape[d], idx.shape[d]) if d != dim else None for d in range(a.a.ndim)]
    ishape = [idx.shape[d] if d == dim else other[d] for d in range(a.a.ndim)]
    ashape = [a.shape[d] if d == dim else other[d] for d in range(a.a.ndim)]
    return gather(expand(a, ashape), dim, expand(idx, ishape))


def floor_divide(a, b):
    return div(a, b, rounding_mode="floor")


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
def require_integer_indices(t):
    if t.dtype in BOOL:
        raise Unsupported("boolean mask indexing")
    if t.dtype not in INT:
        raise Invalid("index tensors must have integer dtype; implicit conversion is not allowed")


def _ints(t):
    t = as_ten(t)
    require_integer_indices(t)
    return np.vectorize(int, otypes=[object])(t.a) if t.a.size else t.a


def _advanced_index_metadata(shape, indices):
    if len(indices) > len(shape):
        raise Invalid("advanced indexing: too many indices")
    shapes = []
    for ix in indices:
        if ix is None:
            continue
        if ix.dtype in BOOL or ix.dtype == "u8":
            raise Unsupported("boolean and byte mask indexing")
        if ix.dtype not in {"i32", "i64"}:
            raise Invalid("advanced indexing requires int32 or int64 indices")
        shapes.append(ix.shape)
    return _broadcast_shape(*shapes)


def index(a, indices):
    """aten.index: advanced indexing with a list of index tensors (None = full slice). Out-of-range indices are
    invalid (torch raises)."""
    a = as_ten(a)
    indices = [None if ix is None else as_ten(ix) for ix in indices]
    _advanced_index_metadata(a.shape, indices)
    idx = []
    for k, ix in enumerate(indices):
        if ix is None:
            idx.append(slice(None))
            continue
        arr = _ints(ix)
        n = a.a.shape[k]
        if arr.size and (arr.min() < -n or arr.max() >= n):
            raise Invalid("index out of range")
        idx.append(np.asarray(arr % n if n else arr, dtype=np.intp))
    return Ten(a.a[tuple(idx)], a.dtype)


def _index_put_parameters(a, values, accumulate):
    if a.dtype != values.dtype:
        raise Invalid("index_put: values and destination must have the same dtype")
    if not isinstance(accumulate, (bool, np.bool_)):
        raise Invalid("index_put: accumulate must be boolean")
    return bool(accumulate)


def _index_put_values_shape(source_shape, target_shape):
    if len(source_shape) > len(target_shape) or _broadcast_shape(source_shape, target_shape) != tuple(target_shape):
        raise Invalid("index_put: values cannot broadcast to the indexed shape")


def index_put(a, indices, values, accumulate=False, pick_one=False):
    """aten.index_put: with accumulate, duplicates add (exactly, order irrelevant); without, duplicate indices are
    documented as undefined (Undefined). pick_one=True is the diagnostic reading in which such a position holds one
    of the written values (an allowed set), as scatter documents for itself."""
    a, values = as_ten(a), as_ten(values)
    accumulate = _index_put_parameters(a, values, accumulate)
    indices = [None if ix is None else as_ten(ix) for ix in indices]
    _advanced_index_metadata(a.shape, indices)
    out = a.a.copy()
    idx = []
    for k, ix in enumerate(indices):
        if ix is None:
            idx.append(slice(None))
        else:
            n = a.a.shape[k]
            arr = _ints(ix)
            if arr.size and (arr.min() < -n or arr.max() >= n):
                raise Invalid("index out of range")
            idx.append(np.asarray(arr % n if n else arr, dtype=np.intp))
    target_shape = np.shape(a.a[tuple(idx)])
    _index_put_values_shape(values.shape, target_shape)
    vals = np.broadcast_to(_convert_elems(values, a.dtype), target_shape)
    pos = np.empty(a.a.shape, dtype=object)
    for i in np.ndindex(pos.shape):
        pos[i] = i
    targets = pos[tuple(idx)]
    if target_shape == ():
        scalar_target = targets
        targets = np.empty((), dtype=object)
        targets[()] = scalar_target
    writes = {}
    for j in np.ndindex(targets.shape):
        writes.setdefault(targets[j], []).append(vals[j])
    for p, vs in writes.items():
        if accumulate:
            s = out[p]
            for v in vs:
                s = _iadd(s, v, 1)
            out[p] = s
        elif len(vs) == 1:
            out[p] = vs[0]
        elif not pick_one:
            raise Undefined("index_put with duplicate indices and accumulate=False is undefined (torch docs)")
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


def index_dimension(shape, dim):
    rank = max(1, len(shape))
    dim = integer(dim, "index dimension")
    if not -rank <= dim < rank:
        raise Invalid("index dimension out of range")
    return dim % rank


def gather_dimension(shape, index_shape, dim):
    dim = index_dimension(shape, dim)
    shape, index_shape = shape or (1,), index_shape or (1,)
    if len(shape) != len(index_shape):
        raise Invalid("gather: input and index ranks must agree")
    if any(index_shape[d] > shape[d] for d in range(len(shape)) if d != dim):
        raise Invalid("gather: index shape exceeds input outside the gather dimension")
    return dim


def gather(a, dim, idx):
    a, idx = as_ten(a), as_ten(idx)
    dim = index_dimension(a.shape, dim)
    if idx.a.size == 0:
        return Ten(np.empty(idx.shape, dtype=object), a.dtype)
    dim = gather_dimension(a.shape, idx.shape, dim)
    ix = _ints(idx)
    source = a.a.reshape(a.shape or (1,))
    n = source.shape[dim]
    out = np.empty(ix.shape, dtype=object)
    for i in np.ndindex(ix.shape):
        k = ix[i]
        if not 0 <= k < n:
            raise Invalid("gather: index out of range")
        j = list(i or (0,))
        j[dim] = k
        out[i] = source[tuple(j)]
    return Ten(out, a.dtype)


def scatter_shape(shape, index_shape, source_shape=None):
    shape, index_shape = shape or (1,), index_shape or (1,)
    if len(shape) != len(index_shape):
        raise NotEstablished("scatter rank broadcasting: retained PyTorch 2.10 documentation and local implementation disagree")
    if source_shape is not None:
        source_shape = source_shape or (1,)
        if len(source_shape) != len(index_shape) or any(i > s for i, s in zip(index_shape, source_shape)):
            raise Invalid("scatter: index shape must not exceed source shape")


def scatter(a, dim, idx, src=None, value=None, reduce=None):
    a, idx = as_ten(a), as_ten(idx)
    dim = index_dimension(a.shape, dim)
    if idx.a.size == 0:
        return Ten(a.a.copy(), a.dtype)
    ix = _ints(idx)
    s = as_ten(src) if src is not None else None
    scatter_shape(a.shape, idx.shape, None if s is None else s.shape)
    gather_dimension(a.shape, idx.shape, dim)
    out = a.a.copy()
    if s is not None:
        sv = _convert_elems(s, a.dtype).reshape(s.shape or (1,))
    writes = {}
    for i in np.ndindex(ix.shape):
        coordinate = i or (0,)
        j = list(coordinate)
        n = (a.shape or (1,))[dim]
        if not 0 <= ix[i] < n:
            raise Invalid("scatter: index out of range")
        j[dim] = ix[i]
        v = sv[coordinate] if src is not None else _elem(R(value) if isinstance(value, (float, Fraction)) else value, a.dtype)
        writes.setdefault(tuple(j) if a.shape else (), []).append(v)
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
    if ix.ndim > 1:
        raise Invalid("index_select: index must be a scalar or vector")
    if any(not 0 <= k < a.a.shape[dim] for k in ix.reshape(-1)):
        raise Invalid("index_select: index out of range")
    return Ten(np.take(a.a, [int(k) for k in ix.reshape(-1)], axis=dim), a.dtype)


def index_add(a, dim, idx, src, alpha=1):
    a, src = as_ten(a), as_ten(src)
    out = a.a.copy()
    ix = _ints(idx)
    if ix.ndim > 1:
        raise Invalid("index_add: index must be a scalar or vector")
    ix = ix.reshape(-1)
    if any(not 0 <= j < a.a.shape[dim] for j in ix):
        raise Invalid("index_add: index out of range")
    expected = list(a.shape)
    expected[dim] = ix.size
    if tuple(src.shape) != tuple(expected):
        raise Invalid("index_add: source shape must match the input except for the index length along dim")
    sv = _convert_elems(src, a.dtype)
    for k, j in enumerate(ix):
        sl = [slice(None)] * out.ndim
        sl[dim] = int(j)
        ssl = [slice(None)] * sv.ndim
        ssl[dim] = k
        tgt, add_ = out[tuple(sl)], sv[tuple(ssl)]
        out[tuple(sl)] = np.vectorize(lambda x, y: x + y * alpha, otypes=[object])(tgt, add_)
    return Ten(out, a.dtype)


def vjp_gather(g, a, dim, idx):
    """d gather: the upstream gradient added back at the gathered positions (repeated indices accumulate)."""
    return scatter_add(zeros_like(_floatify(a)), dim, idx, _floatify(g))


def vjp_index_select(g, a, dim, idx):
    """d index_select: the upstream slices added back at the selected indices (repeated indices accumulate)."""
    return index_add(zeros_like(_floatify(a)), dim, idx, _floatify(g))


def embedding(weight, indices, padding_idx=-1, **kw):
    _unimplemented_options(kw)
    weight = as_ten(weight)
    if weight.a.ndim != 2:
        raise Invalid("embedding: weight must be a matrix")
    ix = _ints(indices)
    if any(not 0 <= k < weight.shape[0] for k in ix.reshape(-1)):
        raise Invalid("embedding: index out of range")
    return Ten(np.take(weight.a, np.asarray(ix, dtype=np.int64), axis=0), weight.dtype)


def embedding_dense_backward(grad, indices, num_weights, padding_idx=-1, scale_grad_by_freq=False):
    """Gradient of embedding: sum of the output gradients of each index; padding_idx rows get zero; with
    scale_grad_by_freq each index's sum is divided by its count in the batch (torch.nn.Embedding docs)."""
    grad, ix = as_ten(grad), _ints(indices)
    num_weights = integer(num_weights, "num_weights", lo=0)
    if grad.a.ndim < 1 or grad.shape[:-1] != ix.shape:
        raise Invalid("embedding backward: gradient shape must be index shape plus embedding dimension")
    ix = ix.reshape(-1)
    if any(not 0 <= k < num_weights for k in ix):
        raise Invalid("embedding backward: index out of range")
    D = grad.a.shape[-1]
    g = grad.a.reshape(ix.size, D)
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
    rank = len(a.shape)
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        return tuple(range(rank))
    dims = dim if isinstance(dim, (list, tuple)) else [dim]
    axes = [_layout_axis(d, max(1, rank)) for d in dims]
    if len(set(axes)) != len(axes):
        raise Invalid("reduction: repeated dimension")
    return tuple(sorted(axes)) if rank else ()


def _reduce(a, dim, keepdim, fn, out_dtype=None):
    axes = _axes(a, dim)
    rest = [i for i in range(a.a.ndim) if i not in axes]
    moved = np.transpose(a.a, rest + list(axes)) if a.a.ndim else a.a.reshape(())
    lead = [a.a.shape[i] for i in rest]
    count = math.prod(a.a.shape[i] for i in axes)
    flat = moved.reshape(lead + [count]) if a.a.ndim else moved.reshape(1)
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


def _reduction_dtype(a, dtype, *, floating=False):
    d = dt(dtype) if dtype is not None else (a.dtype if floating or a.dtype in FLOAT else "i64")
    if floating and d not in FLOAT:
        raise Invalid("mean requires a floating input or explicit floating dtype")
    return d


def _reduction_input(a, dtype, *, floating=False):
    a = as_ten(a)
    d = _reduction_dtype(a, dtype, floating=floating)
    if dtype is not None:
        return convert_element_type(a, d)
    return Ten(_convert_elems(a, d), d) if d != a.dtype else a


def sum_(a, dim=None, keepdim=False, dtype=None):
    a = _reduction_input(a, dtype)
    d = a.dtype
    if d in BOOL:
        return _reduce(a, dim, keepdim, any, d)
    return _reduce(a, dim, keepdim, lambda xs: _rsum(xs, R(0) if d in FLOAT else 0), d)


def mean(a, dim=None, keepdim=False, dtype=None):
    a = _reduction_input(a, dtype, floating=True)

    def m(xs):
        if not xs:
            raise Undefined("mean of an empty reduction")
        return _rsum(xs, R(0)) / len(xs)
    return _reduce(a, dim, keepdim, m)


def _pointwise_loss_inputs(a, target, reduction):
    a, target = _floatify(as_ten(a)), _floatify(as_ten(target))
    if a.shape != target.shape:
        raise Invalid(f"loss requires equal shapes, got {a.shape} and {target.shape}")
    if a.a.size == 0:
        raise Invalid("the reviewed loss task requires at least one element")
    if reduction not in ("none", "sum", "mean"):
        raise Invalid("loss reduction must be none, sum or mean")
    return a, target


def _pointwise_loss_reduce(values, reduction):
    return values if reduction == "none" else sum_(values) if reduction == "sum" else mean(values)


def _pointwise_loss_vjp(grad, derivative, reduction):
    grad = _floatify(as_ten(grad))
    expected = derivative.shape if reduction == "none" else ()
    if grad.shape != expected:
        raise Invalid(f"loss VJP requires upstream shape {expected}, got {grad.shape}")
    dx = mul(derivative, grad)
    if reduction == "mean":
        dx = mul(dx, Fraction(1, derivative.a.size))
    return [dx, neg(dx)]


def mse_loss(a, target, reduction="mean"):
    """Unweighted same-shape squared error, preserving the selected reduction."""
    a, target = _pointwise_loss_inputs(a, target, reduction)
    difference = sub(a, target)
    return _pointwise_loss_reduce(mul(difference, difference), reduction)


def vjp_mse_loss(grad, a, target, reduction="mean"):
    """Complete VJP of the selected MSE reduction with respect to both arguments."""
    a, target = _pointwise_loss_inputs(a, target, reduction)
    return _pointwise_loss_vjp(grad, mul(sub(a, target), 2), reduction)


def l1_loss(a, target, reduction="mean"):
    """Unweighted same-shape absolute error, preserving the selected reduction."""
    a, target = _pointwise_loss_inputs(a, target, reduction)
    return _pointwise_loss_reduce(abs_(sub(a, target)), reduction)


def vjp_l1_loss(grad, a, target, reduction="mean"):
    """Complete selected L1 VJP with the PyTorch zero subgradient at equality."""
    a, target = _pointwise_loss_inputs(a, target, reduction)
    return _pointwise_loss_vjp(grad, sign(sub(a, target)), reduction)


def prod(a, dim=None, keepdim=False, dtype=None):
    a = _reduction_input(a, dtype)
    if a.dtype in BOOL:
        return _reduce(a, dim, keepdim, all, a.dtype)

    def p(xs):
        s = R(1) if a.dtype in FLOAT else 1
        for x in xs:
            s = s * x
        return s
    return _reduce(a, dim, keepdim, p)


def _extremum_input(a, dim, *, single_axis=False):
    a = as_ten(a)
    _extremum_axes(a, dim, single_axis=single_axis)
    return a


def _extremum_axes(a, dim, *, single_axis=False):
    if single_axis and dim is not None:
        _layout_axis(dim, max(1, len(a.shape)))
    axes = _axes(a, dim)
    if any(a.shape[axis] == 0 for axis in axes):
        raise Invalid("extremum of an empty reduction (no identity)")
    return axes


def amax(a, dim=None, keepdim=False):
    a = _extremum_input(a, dim)

    def m(xs):
        if not xs:
            raise Invalid("amax of an empty reduction (torch raises)")
        out = xs[0]
        for x in xs[1:]:
            out = _maxv(out, x)
        return out
    return _reduce(a, dim, keepdim, m)


def amin(a, dim=None, keepdim=False):
    a = _extremum_input(a, dim)

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
    a = _extremum_input(a, dim, single_axis=True)
    return _reduce(a, dim, keepdim, lambda xs: _arg(xs, lambda x, y: x > y), "i64")


def argmin(a, dim=None, keepdim=False):
    a = _extremum_input(a, dim, single_axis=True)
    return _reduce(a, dim, keepdim, lambda xs: _arg(xs, lambda x, y: x < y), "i64")


def max_dim(a, dim, keepdim=False):
    """torch.max(input, dim): (values, indices of the first maximal value)."""
    a = as_ten(a)
    dim = _layout_axis(dim, max(1, a.a.ndim))
    return amax(a, dim, keepdim), argmax(a, dim, keepdim)


def min_dim(a, dim, keepdim=False):
    a = as_ten(a)
    dim = _layout_axis(dim, max(1, a.a.ndim))
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
    a = _reduction_input(a, dtype)
    d, arr = a.dtype, a.a
    dim = _layout_axis(dim, max(1, arr.ndim))
    if arr.ndim == 0 or arr.size == 0 or arr.shape[dim] == 1:
        return a
    if d in BOOL:
        raise NotEstablished("boolean-output cumsum accumulation semantics have not been established")
    out = np.empty(arr.shape, dtype=object)
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


def linalg_vector_norm(a, ord=2, dim=None, keepdim=False):
    """Selected real ``torch.linalg.vector_norm`` value operation."""
    return norm(a, ord, dim, keepdim)


def _softmax_row(xs):
    if not xs:
        return []                                 # softmax over an empty dimension is the empty tensor
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
    dim = _layout_axis(dim, max(1, a.a.ndim))
    arr = a.a.reshape(1) if a.a.ndim == 0 else a.a
    out = np.empty(arr.shape, dtype=object)
    for i in np.ndindex(*[s for k, s in enumerate(arr.shape) if k != dim]):
        idx = [list(i[:dim]) + [j] + list(i[dim:]) for j in range(arr.shape[dim])]
        vals = fn([arr[tuple(ix)] for ix in idx])
        if len(vals) != len(idx):
            raise ValueError("row evaluator returned an incomplete or oversized result")
        for ix, v in zip(idx, vals):
            out[tuple(ix)] = v
    return Ten(out.reshape(a.shape), a.dtype)


def _softmax_input(a, half_to_float):
    a = as_ten(a)
    if type(half_to_float) is not bool:
        raise Invalid("half_to_float must be a boolean")
    if a.dtype not in FLOAT:
        raise Invalid("softmax requires a floating input")
    if half_to_float:
        if a.dtype != "f16":
            raise Invalid("half_to_float requires float16 input")
        # The half values are already exact inputs; widening changes no value.
        return Ten(a.a, "f32")
    return a


def softmax(a, dim, half_to_float=False):
    return _rowwise(_softmax_input(a, half_to_float), dim, _softmax_row)


def safe_softmax(a, dim, dtype=None):
    """aten._safe_softmax: softmax, except rows whose entries are all -inf give zeros (SDPA's documented choice)."""
    a = as_ten(a)
    target = a.dtype if dtype is None else dt(dtype)
    if target not in FLOAT:
        raise Invalid("softmax requires a floating result dtype")

    def row(xs):
        # attention.cpp computes the all-masked predicate on the original input.
        if all(x.v == rl.NEG for x in xs):
            return [R(0)] * len(xs)
        if dtype is not None and target != a.dtype:
            from .ieee import round_to
            xs = [round_to(x, target) for x in xs]
        return _softmax_row(xs)
    result = _rowwise(a, dim, row)
    return Ten(result.a, target)


def log_softmax(a, dim, half_to_float=False):
    a = _softmax_input(a, half_to_float)
    def row(xs):
        if not xs:
            return []
        m = xs[0]
        for x in xs[1:]:
            m = rl.max_(m, x)
        if not m.finite:
            raise Undefined("log_softmax of a row that is all -inf or has +inf")
        shifted = [x - m for x in xs]
        log_mass = rl.log(_rsum([rl.exp(x) for x in shifted], R(0)))
        return [x - log_mass for x in shifted]
    return _rowwise(a, dim, row)


def _softmax_backward_inputs(grad, out, dim, input_dtype):
    g, y = as_ten(grad), as_ten(out)
    if g.shape != y.shape:
        raise Invalid("softmax backward gradient and output must have the same shape")
    if g.dtype not in FLOAT or y.dtype != g.dtype:
        raise Invalid("softmax backward requires matching floating gradient and output dtypes")
    target = dt(input_dtype)
    if target not in FLOAT:
        raise Invalid("softmax backward input_dtype must be floating")
    axis = _layout_axis(dim, max(1, g.a.ndim))
    return g, y, axis, target


def softmax_backward_data(grad, out, dim, input_dtype=None):
    """d softmax: g_in = y * (g - sum(g * y)) along dim."""
    g, y, dim, target = _softmax_backward_inputs(grad, out, dim, input_dtype)
    s = sum_(mul(g, y), dim, keepdim=True)
    result = mul(y, sub(g, s))
    # Real profile: retain the derivative, without the operation's final rounding.
    return Ten(result.a, target)


def vjp_softmax(grad, a, dim):
    """Public same-dtype softmax VJP from original finite logits, without saved device output."""
    g, x = as_ten(grad), as_ten(a)
    g, x, dim, target = _softmax_backward_inputs(g, x, dim, x.dtype)
    if any(not value.finite for tensor in (g, x) for value in tensor.a.flat):
        raise Unsupported("original-input softmax VJP currently requires finite logits and upstream")
    probabilities = softmax(x, dim)
    return softmax_backward_data(g, probabilities, dim, target)


def log_softmax_backward_data(grad, out, dim, input_dtype=None):
    """d log_softmax: g_in = g - exp(y) * sum(g) along dim."""
    g, y, dim, target = _softmax_backward_inputs(grad, out, dim, input_dtype)
    result = sub(g, mul(exp(y), sum_(g, dim, keepdim=True)))
    return Ten(result.a, target)


def prepare_softmax_online(a, dim):
    """prims.prepare_softmax_online (Inductor, torch 2.10): (amax(x, dim, keepdim), sum(exp(x - amax), dim, keepdim))."""
    a = as_ten(a)
    dim = _layout_axis(dim, max(1, a.a.ndim))
    m = amax(a, dim, keepdim=True)
    return m, sum_(exp(sub(a, m)), dim, keepdim=True)


# ------------------------------------------------------------------ vector-Jacobian products of the reductions (REDUCE contract)
def _reduction_vjp_axes(g, x, dim, keepdim):
    if not isinstance(keepdim, (bool, np.bool_)):
        raise Invalid("reduction VJP: keepdim must be boolean")
    axes = _axes(x, dim)
    expected = tuple(1 if i in axes else n for i, n in enumerate(x.shape)) if keepdim else tuple(
        n for i, n in enumerate(x.shape) if i not in axes)
    if g.shape != expected:
        raise Invalid("reduction VJP: upstream gradient must match the complete forward output shape")
    return axes


def _bcast_back(g, x, dim, keepdim):
    """The upstream gradient of a reduction, broadcast back to the input's shape."""
    g, x = as_ten(g), as_ten(x)
    axes = _reduction_vjp_axes(g, x, dim, keepdim)
    if not keepdim:
        for ax in axes:
            g = unsqueeze(g, ax)
    return expand(g, list(x.shape))


def vjp_sum(g, x, dim=None, keepdim=False):
    return _bcast_back(_floatify(g), x, dim, keepdim)


def vjp_mean(g, x, dim=None, keepdim=False):
    x = as_ten(x)
    n = 1
    for ax in _axes(x, dim):
        n *= x.shape[ax]
    if n == 0:
        raise Undefined("mean of an empty reduction")
    return div(_bcast_back(_floatify(g), x, dim, keepdim), n)


def vjp_prod(g, x, dim=None, keepdim=False):
    """d prod / d x_i = product of the other elements (exact also when some x_j are zero)."""
    x = _floatify(x)
    gb = _bcast_back(_floatify(g), x, dim, keepdim)
    axes = _axes(x, dim)
    rest = [i for i in range(x.a.ndim) if i not in axes]
    out = np.empty(x.shape, dtype=object)
    for lead in np.ndindex(*[x.shape[i] for i in rest]) if rest else [()]:
        idxs = []
        for red in np.ndindex(*[x.shape[i] for i in axes]):
            full_ = [0] * x.a.ndim
            for k, i in enumerate(rest):
                full_[i] = lead[k]
            for k, i in enumerate(axes):
                full_[i] = red[k]
            idxs.append(tuple(full_))
        for j in idxs:
            p_ = R(1)
            for k in idxs:
                if k != j:
                    p_ = p_ * x.a[k]
            out[j] = gb.a[j] * p_
    return Ten(out, x.dtype)


def _vjp_extreme(g, x, dim, keepdim, pick):
    """amax / amin and full max / min: the upstream gradient is split evenly among the elements equal to the extreme."""
    x = _floatify(x)
    m = pick(x, dim, keepdim=True)
    ties = eq(x, expand(m, list(x.shape)))
    count = sum_(convert_element_type(ties, x.dtype), dim, keepdim=True)
    gb = _bcast_back(_floatify(g), x, dim, keepdim)
    return where(ties, div(gb, expand(count, list(x.shape))), as_ten(0, x))


def vjp_amax(g, x, dim=None, keepdim=False):
    return _vjp_extreme(g, x, dim, keepdim, amax)


def vjp_amin(g, x, dim=None, keepdim=False):
    return _vjp_extreme(g, x, dim, keepdim, amin)


def vjp_max_dim(g, x, dim, keepdim=False):
    """max / min along one dim: only the returned (first maximal) index receives the gradient."""
    x = _floatify(x)
    _reduction_vjp_axes(as_ten(g), x, dim, keepdim)
    idx = argmax(x, dim, keepdim=True)
    gb = _floatify(g) if keepdim else unsqueeze(_floatify(g), dim)
    return scatter(zeros_like(x), dim, idx, src=gb)


def vjp_min_dim(g, x, dim, keepdim=False):
    x = _floatify(x)
    _reduction_vjp_axes(as_ten(g), x, dim, keepdim)
    idx = argmin(x, dim, keepdim=True)
    gb = _floatify(g) if keepdim else unsqueeze(_floatify(g), dim)
    return scatter(zeros_like(x), dim, idx, src=gb)


def vjp_var(g, x, dim=None, correction=1, keepdim=False):
    x = _floatify(x)
    n = 1
    for ax in _axes(x, dim):
        n *= x.shape[ax]
    if n - correction <= 0:
        raise Undefined("variance with non-positive degrees of freedom")
    m = mean(x, dim, keepdim=True)
    gb = _bcast_back(_floatify(g), x, dim, keepdim)
    return mul(gb, div(mul(sub(x, expand(m, list(x.shape))), 2), n - correction))


def vjp_std(g, x, dim=None, correction=1, keepdim=False):
    x = _floatify(x)
    sd = std(x, dim, correction, keepdim=True)
    gb = _bcast_back(_floatify(g), x, dim, keepdim)
    n = 1
    for ax in _axes(x, dim):
        n *= x.shape[ax]
    if n == 1:
        # With one finite sample and positive degrees of freedom, std is
        # identically zero. Differentiating its quotient formula gives 0/0.
        # std above still validates the domain before this constant case.
        return mul(gb, zeros_like(x))
    m = mean(x, dim, keepdim=True)
    return div(mul(gb, sub(x, expand(m, list(x.shape)))), mul(expand(sd, list(x.shape)), n - correction))


def vjp_var_mean(g_var, g_mean, x, dim=None, correction=1, keepdim=False):
    """Input VJP of the ordered ``(variance, mean)`` result."""
    return add(
        vjp_var(g_var, x, dim, correction, keepdim),
        vjp_mean(g_mean, x, dim, keepdim),
    )


def vjp_logsumexp(g, x, dim, keepdim=False):
    """d logsumexp / d x_i = exp(x_i - logsumexp(x)) over the reduced dims (the softmax of the reduced slice); a slice of
    -inf only has lse = -inf and the gradient is undefined there."""
    x = _floatify(x)
    gb = _bcast_back(_floatify(g), x, dim, keepdim)
    if x.a.size == 0:
        return Ten(np.empty(x.shape, dtype=object), x.dtype)
    lse = expand(logsumexp(x, dim, keepdim=True), list(x.shape))
    for v in lse.a.reshape(-1):
        if v.v == rl.NEG:
            raise Undefined("softmax of a row whose entries are all -inf")
    return mul(gb, exp(sub(x, lse)))


def vjp_cumsum(g, x, dim):
    """d cumsum: the reverse cumulative sum of the upstream gradient."""
    g, x = _floatify(g), as_ten(x)
    if g.shape != x.shape:
        raise Invalid("cumsum VJP: upstream gradient must match the complete input shape")
    dim = _layout_axis(dim, max(1, len(x.shape)))
    if not x.shape:
        return g
    flipped = Ten(np.flip(g.a, dim), g.dtype)
    return Ten(np.flip(cumsum(flipped, dim).a, dim), g.dtype)


def vjp_norm(g, x, p=2, dim=None, keepdim=False):
    """d ||x||_2 = x / ||x|| (0 where the norm is 0, decision RD.NORM0); d ||x||_1 = sign(x)."""
    x = _floatify(x)
    gb = _bcast_back(_floatify(g), x, dim, keepdim)
    if p == 1:
        return mul(gb, sign(x))
    if p != 2:
        raise Unsupported(f"norm gradient for p={p}")
    nrm = expand(norm(x, 2, dim, keepdim=True), list(x.shape))
    safe = where(eq(nrm, 0), as_ten(1, nrm), nrm)
    return where(eq(nrm, 0), as_ten(0, x), mul(gb, div(x, safe)))


def vjp_linalg_vector_norm(g, x, ord=2, dim=None, keepdim=False):
    """Input VJP for the selected smooth real vector 2-norm task."""
    return vjp_norm(g, x, ord, dim, keepdim)


# ------------------------------------------------------------------ matrix products (MATMUL / CONV contracts)
def _dot(xs, ys, zero):
    s = zero
    for x, y in zip(xs, ys):
        s = s + x * y
    return s


def _product_layout(a, b, rank=None):
    """Validate product metadata without reading or allocating tensor values."""
    na, nb = len(a.shape), len(b.shape)
    if rank is not None and (na != rank or nb != rank):
        raise Invalid(f"matrix product requires rank {rank}: {a.shape} x {b.shape}")
    if na == 0 or nb == 0:
        raise Invalid("matmul does not accept scalar operands")
    if rank == 3 and a.shape[0] != b.shape[0]:
        raise Invalid("bmm batch sizes must match without broadcasting")
    if a.shape[-1] != (b.shape[-2] if nb > 1 else b.shape[0]):
        raise Invalid(f"matrix product inner dimensions differ: {a.shape} x {b.shape}")
    try:
        batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    except ValueError as error:
        raise Invalid("matmul batch dimensions are not broadcastable") from error
    output = batch + (a.shape[-2:-1] if na > 1 else ()) + (b.shape[-1:] if nb > 1 else ())
    if a.dtype != b.dtype:
        if (na >= 3 or nb >= 3) and (0 in a.shape or 0 in b.shape):
            raise NotEstablished("empty batched products with mixed dtypes require a reviewed result-type convention")
        raise Invalid("matrix product operands must have matching dtypes")
    if a.dtype in BOOL:
        raise Unsupported("boolean matrix-product semantics are not established by the real profile")
    return batch, output


def mm(a, b):
    a, b = as_ten(a), as_ten(b)
    _product_layout(a, b, 2)
    m, k = a.a.shape
    n = b.a.shape[1]
    out = np.empty((m, n), dtype=object)
    zero = R(0) if a.dtype in FLOAT else 0
    for i in range(m):
        for j in range(n):
            out[i, j] = _dot(a.a[i, :], b.a[:, j], zero)
    return Ten(out, promote(a.dtype, b.dtype))


def bmm(a, b):
    a, b = as_ten(a), as_ten(b)
    _product_layout(a, b, 3)
    return Ten(np.stack([mm(Ten(a.a[i], a.dtype), Ten(b.a[i], b.dtype)).a for i in range(a.a.shape[0])])
               if a.a.shape[0] else np.empty((0, a.a.shape[1], b.a.shape[2]), dtype=object), promote(a.dtype, b.dtype))


def matmul(a, b):
    """torch.matmul: 1-d operands are promoted and the added dimension removed; batch dimensions broadcast."""
    a, b = as_ten(a), as_ten(b)
    batch, _ = _product_layout(a, b)
    if a.a.ndim == 1 and b.a.ndim == 1:
        if a.a.shape != b.a.shape:
            raise Invalid("matmul: vector lengths differ")
        zero = R(0) if a.dtype in FLOAT else 0
        return Ten(np.array(_dot(a.a, b.a, zero), dtype=object), a.dtype)
    va, vb = a.a.ndim == 1, b.a.ndim == 1
    A = a.a.reshape(1, -1) if va else a.a
    B = b.a.reshape(-1, 1) if vb else b.a
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


def _scaled_product_inputs(c, a, b, rank):
    c, a, b = as_ten(c), as_ten(a), as_ten(b)
    _scaled_product_layout(c, a, b, rank)
    return c, a, b


def _scaled_product_layout(c, a, b, rank):
    if c.dtype != a.dtype or c.dtype != b.dtype:
        raise Invalid("scaled matrix product requires matching addend and operand dtypes")
    _, output = _product_layout(a, b, rank)
    try:
        combined = np.broadcast_shapes(c.shape, output)
    except ValueError as error:
        raise Invalid("matrix product addend cannot broadcast to the output") from error
    if combined != output:
        raise Invalid("matrix product addend would enlarge the output shape")
    return output


def _product_scalar(value, name):
    """Normalize scalar parameters before any zero/one branch or tensor arithmetic."""
    if isinstance(value, Ten):
        if value.shape != ():
            raise Invalid(f"{name} must be a scalar, not a tensor of shape {value.shape}")
        value = value.a[()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float, Fraction, R)):
        raise Invalid(f"{name} must be a real scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise Invalid(f"{name} must be finite in the real matrix-product profile")
    value = R(value)
    if not value.finite:
        raise Invalid(f"{name} must be finite in the real matrix-product profile")
    if not value.exact:
        lo, hi = value.bounds()
        if lo != hi:
            raise NotEstablished(f"{name} requires an exact scalar for the selected task")
        return lo
    return value.v


def _product_scales(c, beta, alpha):
    beta, alpha = _product_scalar(beta, "beta"), _product_scalar(alpha, "alpha")
    if c.dtype in INT and (beta.denominator != 1 or alpha.denominator != 1):
        raise NotEstablished("integer matrix-product scaling: documentation requires integers but the source "
                             "truncates nonintegers; a task choice has not been established")
    if c.dtype in INT:
        return int(beta), int(alpha)
    return beta, alpha


def addmm(c, a, b, *, beta=1, alpha=1):
    """torch.addmm: beta * c + alpha * (a @ b); with beta = 0 the input c is ignored (documented, even inf/nan)."""
    c, a, b = _scaled_product_inputs(c, a, b, 2)
    beta, alpha = _product_scales(c, beta, alpha)
    p = mm(a, b)
    p = mul(p, alpha) if alpha != 1 else p
    if beta == 0:
        return p
    return add(mul(c, beta) if beta != 1 else c, p)


def baddbmm(c, a, b, *, beta=1, alpha=1):
    c, a, b = _scaled_product_inputs(c, a, b, 3)
    beta, alpha = _product_scales(c, beta, alpha)
    p = bmm(a, b)
    p = mul(p, alpha) if alpha != 1 else p
    return p if beta == 0 else add(mul(c, beta) if beta != 1 else c, p)


def _linear_inputs(x, w, b):
    x, w = as_ten(x), as_ten(w)
    if x.a.ndim == 0 or w.a.ndim not in (1, 2):
        raise Invalid("linear requires a nonscalar input and one- or two-dimensional weight")
    _, output = _product_layout(x, t(w))
    if b is not None:
        b = as_ten(b)
        try:
            combined = np.broadcast_shapes(b.shape, output)
        except ValueError as error:
            raise Invalid("linear bias cannot broadcast to the output") from error
        if combined != output:
            raise Invalid("linear bias would enlarge the output shape")
        if b.dtype != x.dtype:
            raise NotEstablished("linear with mixed bias and operand dtypes requires a reviewed conversion convention")
        if b.shape != () and not (w.a.ndim == 2 and b.shape == (w.shape[0],)):
            raise NotEstablished("linear bias outside documented scalar or output-feature vector shapes")
    return x, w, b


def linear(x, w, b=None):
    """torch.nn.functional.linear: y = x W^T + b over the last dimension of x."""
    x, w, b = _linear_inputs(x, w, b)
    y = matmul(x, t(w))
    return y if b is None else add(y, b)


def sum_to_shape(g, shape):
    """Sum a broadcast gradient back to the shape of the operand it came from."""
    g = as_ten(g)
    shape = list(shape)
    while g.a.ndim > len(shape):
        g = sum_(g, [0])
    for d, n in enumerate(shape):
        if n == 1 and g.shape[d] != 1:
            g = sum_(g, [d], keepdim=True)
    return g


def mm_plus_mm(mat1, mat2, mat3, mat4):
    """Inductor's lowering of aten.add(aten.mm(mat1, mat2), aten.mm(mat3, mat4)) (fx_passes/post_grad.py)."""
    return add(mm(mat1, mat2), mm(mat3, mat4))


def _product_vjp_inputs(g, a, b, rank=None):
    g, a, b = as_ten(g), as_ten(a), as_ten(b)
    _, output = _product_layout(a, b, rank)
    if a.dtype not in FLOAT:
        raise Unsupported("matrix-product VJP requires floating operands")
    if g.dtype != a.dtype:
        raise Invalid("upstream gradient and matrix product operands must have matching dtypes")
    if g.shape != output:
        raise Invalid(f"upstream shape {g.shape} does not match product output {output}")
    return g, a, b


def vjp_mm(g, a, b):
    """d(A @ B): dA = G B^T, dB = A^T G."""
    g, a, b = _product_vjp_inputs(g, a, b, 2)
    return [mm(g, t(b)), mm(t(a), g)]


def vjp_bmm(g, a, b):
    g, a, b = _product_vjp_inputs(g, a, b, 3)
    return [bmm(g, transpose(b, 1, 2)), bmm(transpose(a, 1, 2), g)]


def vjp_matmul(g, a, b):
    """d matmul with broadcasting: the batch dimensions of each operand's gradient are summed back to its shape."""
    g, a, b = _product_vjp_inputs(g, a, b)
    if a.a.ndim == 1 or b.a.ndim == 1:
        A2 = unsqueeze(a, 0) if a.a.ndim == 1 else a
        B2 = unsqueeze(b, -1) if b.a.ndim == 1 else b
        G2 = g
        if a.a.ndim == b.a.ndim == 1:
            G2 = view(g, [1, 1])
        elif a.a.ndim == 1:
            G2 = unsqueeze(G2, -2)
        elif b.a.ndim == 1:
            G2 = unsqueeze(G2, -1)
        da, db = vjp_matmul(G2, A2, B2)
        return [view(da, list(a.shape)) if a.a.ndim == 1 else da, view(db, list(b.shape)) if b.a.ndim == 1 else db]
    da = matmul(g, transpose(b, -1, -2))
    db = matmul(transpose(a, -1, -2), g)
    return [sum_to_shape(da, a.shape), sum_to_shape(db, b.shape)]


def vjp_linear(g, x, w, b=None):
    """Differentiate the documented product and sum the bias gradient to its shape."""
    x, w, b = _linear_inputs(x, w, b)
    dx, dwt = vjp_matmul(g, x, t(w))
    dw = t(dwt)
    db = None if b is None else sum_to_shape(g, b.shape)
    return [dx, dw, db]


def vjp_addmm(g, c, a, b, *, beta=1, alpha=1):
    c, a, b = _scaled_product_inputs(c, a, b, 2)
    beta, alpha = _product_scales(c, beta, alpha)
    g, a, b = _product_vjp_inputs(g, a, b, 2)
    dc = mul(sum_to_shape(g, as_ten(c).shape), beta)
    da, db = vjp_mm(mul(g, alpha), a, b)
    return [dc, da, db]


def _convolution_dtypes(*tensors, backward=False):
    types = {t.dtype for t in tensors if t is not None}
    if len(types) != 1:
        raise NotEstablished("mixed convolution dtypes require an explicit conversion policy")
    dtype = next(iter(types))
    if dtype in BOOL:
        raise Unsupported("boolean convolution is not a real cross-correlation")
    if backward and dtype not in FLOAT:
        raise Unsupported("convolution gradients require floating tensor inputs")
    return dtype


def _convolution_layout(xshape, wshape, stride, padding, dilation, transposed, output_padding, groups):
    if not isinstance(transposed, (bool, np.bool_)):
        raise Invalid("convolution transposed must be boolean")
    if transposed:
        raise Unsupported("transposed convolution")
    nsp = len(xshape) - 2
    if nsp < 1 or len(wshape) != len(xshape):
        raise Invalid("convolution input and weight require matching spatial ranks")
    if nsp > 3:
        raise Unsupported("convolution supports one to three spatial dimensions")
    def spatial(value, name, minimum):
        values = _layout_sizes([value] if isinstance(value, (int, np.integer)) else value, name)
        if len(values) == 1:
            values *= nsp
        if len(values) != nsp or any(v < minimum for v in values):
            raise Invalid(f"convolution {name}: invalid length or value")
        return values
    stride = spatial(stride, "stride", 1)
    padding = spatial(padding, "padding", 0)
    dilation = spatial(dilation, "dilation", 0)
    # Non-transposed convolution does not use output_padding in its formula,
    # but the parameter still has a declared nonnegative spatial shape.
    spatial(output_padding, "output_padding", 0)
    groups = _layout_sizes([groups], "convolution groups")[0]
    if groups <= 0:
        raise Invalid("convolution groups must be positive")
    N, Cin = xshape[:2]
    Cout, cpg = wshape[:2]
    if Cin != cpg * groups or Cout % groups or Cout <= 0:
        raise Invalid("convolution channel counts do not match groups")
    if Cin == 0:
        raise NotEstablished("zero input-channel convolution semantics are not established")
    if any(k <= 0 for k in wshape[2:]):
        raise Invalid("convolution kernel sizes must be positive")
    if any(d == 0 for d in dilation):
        raise NotEstablished("zero dilation has unresolved empty/nonempty execution domains")
    osz = [(xshape[2+d] + 2*padding[d] - dilation[d]*(wshape[2+d]-1) - 1)//stride[d] + 1
           for d in range(nsp)]
    if any(n <= 0 for n in osz):
        raise Invalid("convolution effective kernel exceeds the padded input")
    return stride, padding, dilation, (N, Cout, *osz), groups


def _convolution_bias(bias, cout):
    if bias is not None and tuple(bias.shape) != (cout,):
        raise Invalid("convolution bias must contain exactly one value per output channel")


def convolution(x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    """aten.convolution (non-transposed): cross-correlation over the spatial dimensions with zero padding,
    stride, dilation and groups (torch.nn.functional.conv1d/2d/3d docs)."""
    x, w = as_ten(x), as_ten(w)
    bias = as_ten(bias) if bias is not None else None
    _convolution_dtypes(x, w, bias)
    stride, padding, dilation, shape, groups = _convolution_layout(
        x.shape, w.shape, stride, padding, dilation, transposed, output_padding, groups)
    nsp = x.a.ndim - 2
    N, Cin = x.a.shape[:2]
    Cout, cpg = w.a.shape[:2]
    _convolution_bias(bias, Cout)
    xp = constant_pad_nd(x, [p for pd in reversed(padding) for p in (pd, pd)], 0).a
    ksz = w.a.shape[2:]
    osz = shape[2:]
    out = np.empty(shape, dtype=object)
    opg = Cout // groups
    for n in range(N):
        for co in range(Cout):
            g = co // opg
            for o in np.ndindex(*osz):
                s = (R(0) if x.dtype in FLOAT else 0) if bias is None else bias.a[co]
                for ci in range(cpg):
                    for k in np.ndindex(*ksz):
                        pos = tuple(o[d] * stride[d] + k[d] * dilation[d] for d in range(nsp))
                        s = s + xp[(n, g * cpg + ci) + pos] * w.a[(co, ci) + k]
                out[(n, co) + o] = s
    return Ten(out, x.dtype)


def _output_mask(mask, count):
    if not isinstance(mask, (list, tuple)) or len(mask) != count:
        raise Invalid(f"output mask must contain exactly {count} entries")
    if any(not isinstance(flag, (bool, np.bool_)) for flag in mask):
        raise Invalid("output mask entries must be boolean")
    return tuple(bool(flag) for flag in mask)


def convolution_backward(g, x, w, bias_sizes, stride, padding, dilation, transposed, output_padding, groups,
                         output_mask):
    """aten.convolution_backward: (grad_input, grad_weight, grad_bias) of the non-transposed convolution, the VJP of the
    cross-correlation defined above (entries not requested in output_mask are None)."""
    output_mask = _output_mask(output_mask, 3)
    g, x, w = as_ten(g), as_ten(x), as_ten(w)
    _convolution_dtypes(g, x, w, backward=True)
    stride, padding, dilation, shape, groups = _convolution_layout(
        x.shape, w.shape, stride, padding, dilation, transposed, output_padding, groups)
    if tuple(g.shape) != shape:
        raise Invalid("convolution backward gradient must match the complete forward output shape")
    nsp = x.a.ndim - 2
    N, Cin = x.a.shape[:2]
    Cout, cpg = w.a.shape[:2]
    if bias_sizes is not None:
        declared_bias_shape = _layout_sizes(bias_sizes, "convolution backward bias_sizes")
        if declared_bias_shape != [Cout]:
            raise NotEstablished("bias_sizes does not establish the declared per-output-channel bias shape")
    ksz = w.a.shape[2:]
    osz = shape[2:]
    opg = Cout // groups
    xp = constant_pad_nd(x, [p for pd in reversed(padding) for p in (pd, pd)], 0).a
    gin = np.full(xp.shape, R(0), dtype=object)
    gw = np.full(w.a.shape, R(0), dtype=object)
    gb = np.full((Cout,), R(0), dtype=object)
    for n in range(N):
        for co in range(Cout):
            grp = co // opg
            for o in np.ndindex(*osz):
                gv = g.a[(n, co) + o]
                gb[co] = gb[co] + gv
                for ci in range(cpg):
                    for k in np.ndindex(*ksz):
                        pos = tuple(o[d] * stride[d] + k[d] * dilation[d] for d in range(nsp))
                        gin[(n, grp * cpg + ci) + pos] = gin[(n, grp * cpg + ci) + pos] + gv * w.a[(co, ci) + k]
                        gw[(co, ci) + k] = gw[(co, ci) + k] + gv * xp[(n, grp * cpg + ci) + pos]
    crop = tuple([slice(None), slice(None)] + [slice(padding[d], padding[d] + x.a.shape[2 + d]) for d in range(nsp)])
    out = [Ten(gin[crop], x.dtype), Ten(gw, w.dtype), Ten(gb, w.dtype)]
    return [out[i] if output_mask[i] else None for i in range(3)]


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
    if approximate not in ("none", "tanh"):
        raise Invalid(f"gelu approximate={approximate!r}")
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


def vjp_tanh(grad, a):
    """VJP of the public tanh function from its original input."""
    y = tanh(a)
    return mul(_floatify(grad), unary(lambda v: 1 - v * v, y))


def vjp_sigmoid(grad, a):
    """VJP of the public sigmoid function from its original input."""
    y = sigmoid(a)
    return mul(_floatify(grad), unary(lambda v: v * (1 - v), y))


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


def inductor_random(size, seed=None, mode="rand", values=None):
    """prims.inductor_random, conditioned: the random stream is implementation-defined, so the value is the kernel's own
    random tensor, supplied by the binding as `values` (checked for shape and, for 'rand', the range [0, 1))."""
    if values is None:
        raise Unsupported("inductor_random needs the kernel's own random values (conditioning)")
    v = _floatify(values)
    if list(v.shape) != list(size):
        raise Invalid(f"conditioning values have shape {v.shape}, expected {list(size)}")
    if mode == "rand":
        for x in v.a.reshape(-1):
            if not (R(0) <= x < R(1)):
                raise Invalid("a 'rand' value outside [0, 1)")
    elif mode != "randn":
        raise Invalid(f"inductor_random mode {mode!r}")
    return v


def native_dropout(x, p, train=True, keep=None):
    """aten.native_dropout conditioned on the kernel's keep mask: (x * keep / (1 - p), keep); p = 1 gives zeros and an
    all-False mask; train False or p = 0 gives (x, all True) -- the torch 2.10 decomposition."""
    x = _floatify(x)
    p = Fraction(p) if not isinstance(p, R) else p.v
    if not 0 <= p <= 1:
        raise Invalid("dropout probability outside [0, 1]")
    if not train or p == 0:
        return [x, full(x.shape, True, "b8")]
    if p == 1:
        return [zeros_like(x), full(x.shape, False, "b8")]
    if keep is None:
        raise Unsupported("dropout needs the kernel's own keep mask (conditioning)")
    k = as_ten(keep)
    if k.dtype not in BOOL or list(k.shape) != list(x.shape):
        raise Invalid("keep mask must be boolean with the input's shape")
    return [mul(mul(x, k), R(1) / R(1 - p)), k]


def dropout(x, p=0.5, train=True, keep=None):
    return native_dropout(x, p, train, keep)[0]


def native_dropout_backward(grad, mask, scale):
    """grad * mask * scale (torch 2.10 decomposition)."""
    return mul(mul(_floatify(grad), convert_element_type(as_ten(mask), _floatify(grad).dtype)), scale)


def keep_from_uniform(u, p):
    """The decomposition's mask rule: keep where the uniform value exceeds p."""
    return gt(u, p)


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
    "aten.clone": clone, "aten.copy": copy, "aten.view": view, "aten.reshape": reshape,
    "aten._unsafe_view": view, "aten.permute": permute, "aten.transpose": transpose, "aten.t": t, "aten.expand": expand,
    "aten.unsqueeze": unsqueeze, "aten.squeeze": squeeze, "aten.select": select, "aten.slice": slice_, "aten.cat": cat,
    "aten.stack": stack, "aten.split": split, "aten.split_with_sizes": split_with_sizes,
    "aten.constant_pad_nd": constant_pad_nd, "aten.select_scatter": select_scatter, "aten.slice_scatter": slice_scatter,
    "aten.full": full, "aten.zeros": zeros, "aten.ones": ones, "aten.full_like": full_like, "aten.zeros_like": zeros_like,
    "aten.new_zeros": new_zeros, "aten.scalar_tensor": scalar_tensor, "aten.arange": arange, "prims.iota": iota,
    "aten.lift_fresh_copy": lift_fresh_copy, "aten.lift_fresh": clone, "aten.alias": clone, "aten.triu": triu,
    "aten.tril": tril, "aten.fill": fill, "aten.linspace": linspace, "aten.narrow": narrow,
    "aten.take_along_dim": take_along_dim, "aten.floor_divide": floor_divide,
    "aten.index": index, "aten.index_put": index_put, "aten.index_put_": index_put, "aten.gather": gather,
    "aten.scatter": scatter, "aten.scatter_add": scatter_add, "aten.index_select": index_select,
    "aten.index_add": index_add, "aten.embedding": embedding, "aten.embedding_dense_backward": embedding_dense_backward,
    "vjp.aten.gather": vjp_gather, "vjp.aten.index_select": vjp_index_select,
    "aten.sum": sum_, "aten.mean": mean, "aten.prod": prod, "aten.amax": amax, "aten.amin": amin, "aten.argmax": argmax,
    "aten.argmin": argmin, "aten.max": max_dim, "aten.min": min_dim, "aten.var": var, "aten.var_mean": var_mean,
    "aten.std": std, "aten.cumsum": cumsum, "aten.logsumexp": logsumexp, "aten.norm": norm,
    "aten.linalg_vector_norm": linalg_vector_norm,
    "aten.mse_loss": mse_loss, "vjp.aten.mse_loss": vjp_mse_loss,
    "aten.l1_loss": l1_loss, "vjp.aten.l1_loss": vjp_l1_loss,
    "vjp.aten.sum": vjp_sum, "vjp.aten.mean": vjp_mean, "vjp.aten.prod": vjp_prod, "vjp.aten.amax": vjp_amax,
    "vjp.aten.amin": vjp_amin, "vjp.aten.max": vjp_max_dim, "vjp.aten.min": vjp_min_dim, "vjp.aten.var": vjp_var,
    "vjp.aten.std": vjp_std, "vjp.aten.var_mean": vjp_var_mean,
    "vjp.aten.logsumexp": vjp_logsumexp, "vjp.aten.cumsum": vjp_cumsum, "vjp.aten.norm": vjp_norm,
    "vjp.aten.linalg_vector_norm": vjp_linalg_vector_norm,
    "aten._softmax": lambda a, dim, half_to_float: softmax(a, dim, half_to_float),
    "aten._safe_softmax": safe_softmax,
    "aten._log_softmax": lambda a, dim, half_to_float: log_softmax(a, dim, half_to_float),
    "aten._softmax_backward_data": softmax_backward_data, "aten._log_softmax_backward_data": log_softmax_backward_data,
    "vjp.aten.softmax": vjp_softmax,
    "prims.prepare_softmax_online": prepare_softmax_online,
    "aten.mm": mm, "aten.bmm": bmm, "aten.matmul": matmul, "aten.addmm": addmm, "aten.baddbmm": baddbmm,
    "aten.convolution": convolution, "aten.convolution_backward": convolution_backward, "aten.linear": linear,
    "inductor.mm_plus_mm": mm_plus_mm, "vjp.aten.mm": vjp_mm, "vjp.aten.bmm": vjp_bmm,
    "vjp.aten.matmul": vjp_matmul, "vjp.aten.linear": vjp_linear, "vjp.aten.addmm": vjp_addmm,
    "aten.gelu": gelu, "aten.gelu_backward": gelu_backward, "aten.silu": silu, "aten.silu_backward": silu_backward,
    "aten.threshold_backward": threshold_backward, "aten.tanh_backward": tanh_backward,
    "aten.sigmoid_backward": sigmoid_backward, "vjp.aten.tanh": vjp_tanh,
    "vjp.aten.sigmoid": vjp_sigmoid, "aten.softplus": softplus, "aten.softplus_backward": softplus_backward,
    "prims.inductor_lookup_seed": inductor_lookup_seed, "prims.inductor_random": inductor_random,
    "aten.native_dropout": native_dropout, "aten.dropout": dropout, "aten.native_dropout_backward": native_dropout_backward,
    "keep_from_uniform": keep_from_uniform,
}
