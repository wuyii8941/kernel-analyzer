"""Array evaluation of selected admitted ATen-call profiles, for tensors too large for element-wise R objects.

The contract adapters (kernel_spec.aten through make_reference) define f. They build one Python real-number object per
tensor element, which is exact but too slow for model-sized calls (a 768x128x768 product does not finish in 600 s).
This module evaluates the *same* real-number result for a small set of (contract, profile, operation) entries on
NumPy arrays, and returns it as binary64 lower / upper arrays:

- exact outputs (MATMUL real, ARITH real, INDEX real): the exact rational result is computed (integer matrix product
  through python-flint, error-free two_sum / products), and lo / hi are its binary64 roundings toward -inf / +inf,
  the same information the measurement layer extracts from an exact Fraction;
- enclosure outputs (SOFTMAX real, ACT torch): Arb ball arithmetic (python-flint, 128-bit working precision); the
  ball endpoints are moved one binary64 ulp outward after conversion.

Scope rule: an entry accepts only the plain domain it can evaluate without a decision (finite same-dtype floating
tensors, default options). Anything else returns None and the caller uses the contract adapter, which then decides the
outcome (value, unsupported, invalid_input, ...). Where an entry does raise, it mirrors the adapter's own check for the
same condition with the same exception class. tests/test_bulk_reference.py compares every entry with the adapter on
generated inputs, including the refusals; evidence/bulk_reference.md records the scope.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .num import Invalid, Result

VERSION = "bulk-reference v1"
FLOATS = ("f16", "bf16", "f32", "f64")
_TORCH = {"f16": "float16", "bf16": "bfloat16", "f32": "float32", "f64": "float64"}


class BulkTensor:
    """A tensor result as binary64 endpoint arrays. exact=True: lo / hi are the directed roundings of an exact
    rational value (lo == hi where that value is a binary64 number); exact=False: [lo, hi] is a rigorous enclosure."""
    __slots__ = ("lo", "hi", "dtype", "exact")

    def __init__(self, lo, hi, dtype, exact):
        self.lo = np.ascontiguousarray(lo, dtype=np.float64)
        self.hi = np.ascontiguousarray(hi, dtype=np.float64)
        if self.lo.shape != self.hi.shape:
            raise ValueError("bulk endpoint arrays differ in shape")
        self.dtype, self.exact = dtype, bool(exact)

    @property
    def shape(self):
        return self.lo.shape

    def bulk_bounds(self):
        """(lo, hi) binary64 arrays; read by the measurement layer without importing this module."""
        return self.lo, self.hi


# ---------------------------------------------------------------------------------------------- input records

class _Decline(Exception):
    """The entry does not evaluate this call; the adapter does."""


def _record(value, *, floating=True):
    """A tensor record {t, v, shape} -> (array, dtype). Floating records become float64 arrays (exact: every
    f16/bf16/f32/f64 value is a binary64 number); i64 records become int64 arrays."""
    if not isinstance(value, dict) or set(value) != {"t", "v", "shape"}:
        raise _Decline("not a complete tensor record")
    dtype, shape = value["t"], value["shape"]
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise _Decline("malformed shape")
    if floating:
        if dtype not in FLOATS:
            raise _Decline(f"dtype {dtype!r}")
        try:
            arr = np.asarray(value["v"], dtype=np.float64)
        except (TypeError, ValueError):
            raise _Decline("non-numeric values") from None
    else:
        if dtype != "i64":
            raise _Decline(f"dtype {dtype!r}")
        flat = _leaves(value["v"])
        if any(type(x) is not int for x in flat):
            raise _Decline("index values must be literal integers")
        arr = np.asarray(flat, dtype=np.int64)
    if arr.size != math.prod(shape):
        raise _Decline("values do not have the declared shape")
    if floating and tuple(arr.shape) != tuple(shape) and arr.size:
        raise _Decline("values do not have the declared shape")
    return arr.reshape(shape), dtype


def _leaves(v):
    out, stack = [], [v]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(reversed(x))
        else:
            out.append(x)
    return out


def _require_finite(arr):
    if not np.all(np.isfinite(arr)):
        raise _Decline("non-finite value")


def _require_representable(arr, dtype):
    if dtype == "f64":
        return
    import torch
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if not torch.equal(t.to(getattr(torch, _TORCH[dtype])).to(torch.float64), t):
        raise _Decline(f"a value is not representable as {dtype}")


def _float_tensor(value):
    arr, dtype = _record(value)
    _require_finite(arr)
    _require_representable(arr, dtype)
    return arr, dtype


# ---------------------------------------------------------------------------------------------- exact helpers

def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _two_sum_bounds(a, b):
    """Directed binary64 roundings of the exact a + b (a, b binary64 arrays, no overflow)."""
    with np.errstate(over="raise", invalid="raise"):
        s = a + b
        bb = s - a
        err = (a - (s - bb)) + (b - bb)
    lo = np.where(err < 0, _down(s), s)
    hi = np.where(err > 0, _up(s), s)
    return lo, hi


def _product_bounds(a, b):
    """Directed roundings of the exact a * b for binary64 values each with at most 24 significant bits (f16, bf16,
    f32): the product has at most 48 bits and is exact in binary64 unless it underflows below 2^-1022."""
    with np.errstate(over="raise", invalid="raise"):
        p = a * b
    tiny = (p != 0) & (np.abs(p) < 2.0 ** -1022) | ((p == 0) & (a != 0) & (b != 0))
    if np.any(tiny):
        raise _Decline("product below the binary64 normal range")
    return p, p.copy()


def _ints(arr):
    """arr (float64) == M * 2^e with integer M (Python ints, row-major list) and one exponent e."""
    m, e = np.frexp(arr)
    mi = (m * 2.0 ** 53).astype(np.int64)
    e = e.astype(np.int64) - 53
    nz = mi != 0
    if not np.any(nz):
        return [0] * arr.size, 0
    emin = int(e[nz].min())
    shift = np.where(nz, e - emin, 0)
    return [int(x) << int(s) for x, s in zip(mi.ravel().tolist(), shift.ravel().tolist())], emin


def _dyadic_bounds(nums, exp, shape):
    """Directed binary64 roundings of nums[i] * 2^exp (Python ints)."""
    lo = np.empty(len(nums), dtype=np.float64)
    hi = np.empty(len(nums), dtype=np.float64)
    scale_ok = -1000 < exp < 900
    for i, n in enumerate(nums):
        if n == 0:
            lo[i] = hi[i] = 0.0
            continue
        f = float(n)                      # correctly rounded (Python int -> float)
        back = int(f)
        if scale_ok:
            v = math.ldexp(f, exp)
            if v != 0 and abs(v) >= 2.0 ** -1022 and math.isfinite(v):
                if back == n:
                    lo[i] = hi[i] = v
                elif back > n:
                    hi[i], lo[i] = v, math.nextafter(v, -math.inf)
                else:
                    lo[i], hi[i] = v, math.nextafter(v, math.inf)
                continue
        lo[i], hi[i] = _rational_bounds(Fraction(n) * Fraction(2) ** exp)
    return lo.reshape(shape), hi.reshape(shape)


def _rational_bounds(q):
    f = float(q)                          # Fraction -> float is correctly rounded
    if not math.isfinite(f):
        raise _Decline("result outside the binary64 range")
    fq = Fraction(f)
    if fq == q:
        return f, f
    return (f, math.nextafter(f, math.inf)) if fq < q else (math.nextafter(f, -math.inf), f)


def _exact_matmul(a, b):
    """Exact a @ b of float64 matrices -> (Python ints row-major, exponent)."""
    import flint
    (m, k), (_, n) = a.shape, b.shape
    la, ea = _ints(a)
    lb, eb = _ints(b)
    c = flint.fmpz_mat(m, k, la) * flint.fmpz_mat(k, n, lb)
    return [int(x) for x in c.entries()], ea + eb


# ---------------------------------------------------------------------------------------------- Arb helpers

_PRECISIONS = (128, 512, 2048)
_TARGET_ULPS = 4


def _narrow(bounds):
    return all(h - l <= _TARGET_ULPS * max(math.ulp(max(abs(l), abs(h))), 2.0 ** -1074) for l, h in bounds)


def _refined(fn, item):
    """fn(item) -> list of (lo, hi) under increasing Arb working precision until every pair is within a few binary64
    ulps; the last level's (still rigorous) result stands otherwise."""
    import flint
    old = flint.ctx.prec
    try:
        for prec in _PRECISIONS:
            flint.ctx.prec = prec
            bounds = fn(item)
            if _narrow(bounds):
                break
        return bounds
    finally:
        flint.ctx.prec = old


def _arb_bounds(r):
    if not r.is_finite():
        raise _Decline("non-finite ball")
    return math.nextafter(float(r.lower()), -math.inf), math.nextafter(float(r.upper()), math.inf)


# ---------------------------------------------------------------------------------------------- entries

def _scalar(value, name):
    """MATMUL beta / alpha in the forms the adapter reads as one exact finite real; anything else declines."""
    if isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _Decline(f"{name} not finite")
        return Fraction(value)
    raise _Decline(f"{name} form")


def _matmul(inp):
    op, args, kwargs = inp["op"], list(inp.get("args") or []), dict(inp.get("kwargs") or {})
    names = {"aten.mm": ("self", "mat2"), "aten.bmm": ("self", "mat2"), "aten.addmm": ("self", "mat1", "mat2")}
    if op not in names:
        raise _Decline("operation")
    params = names[op]
    if len(args) != len(params):
        raise _Decline("positional form")
    if set(kwargs) - ({"beta", "alpha"} if op == "aten.addmm" else set()):
        raise _Decline("keywords")
    tensors = [_float_tensor(v) for v in args]
    if any(0 in t.shape for t, _ in tensors):
        raise _Decline("empty operand")
    dtypes = {d for _, d in tensors}
    if len(dtypes) != 1:
        raise _Decline("mixed dtypes")
    dtype = dtypes.pop()
    if op == "aten.addmm":
        (c, _), (a, _), (b, _) = tensors
        beta, alpha = _scalar(kwargs.get("beta", 1), "beta"), _scalar(kwargs.get("alpha", 1), "alpha")
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise _Decline("shapes")
        out_shape = (a.shape[0], b.shape[1])
        try:
            if np.broadcast_shapes(c.shape, out_shape) != out_shape:
                raise _Decline("addend enlarges the output")
        except ValueError:
            raise _Decline("addend does not broadcast") from None
        nums, e = _exact_matmul(a, b)
        if beta == 0 and alpha == 1:
            return BulkTensor(*_dyadic_bounds(nums, e, out_shape), dtype, True)
        cb = np.broadcast_to(c, out_shape).ravel()
        lc, ec = _ints(np.ascontiguousarray(cb))
        lo = np.empty(len(nums))
        hi = np.empty(len(nums))
        two_e, two_ec = Fraction(2) ** e, Fraction(2) ** ec
        for i in range(len(nums)):
            q = alpha * nums[i] * two_e + (beta * lc[i] * two_ec if beta else 0)
            lo[i], hi[i] = _rational_bounds(Fraction(q))
        return BulkTensor(lo.reshape(out_shape), hi.reshape(out_shape), dtype, True)
    (a, _), (b, _) = tensors
    if op == "aten.mm":
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise _Decline("shapes")
        nums, e = _exact_matmul(a, b)
        return BulkTensor(*_dyadic_bounds(nums, e, (a.shape[0], b.shape[1])), dtype, True)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise _Decline("shapes")
    los, his = [], []
    for i in range(a.shape[0]):
        nums, e = _exact_matmul(a[i], b[i])
        lo, hi = _dyadic_bounds(nums, e, (a.shape[1], b.shape[2]))
        los.append(lo)
        his.append(hi)
    shape = (a.shape[0], a.shape[1], b.shape[2])
    return BulkTensor(np.stack(los) if los else np.empty(shape), np.stack(his) if his else np.empty(shape), dtype, True)


def _arith(inp):
    op, args, kwargs = inp["op"], list(inp.get("args") or []), dict(inp.get("kwargs") or {})
    if op not in ("aten.add", "aten.sub", "aten.mul") or len(args) != 2:
        raise _Decline("operation")
    if kwargs and not (op in ("aten.add", "aten.sub") and set(kwargs) == {"alpha"} and kwargs["alpha"] == 1
                       and type(kwargs["alpha"]) is int):
        raise _Decline("keywords")
    (a, da), (b, db) = _float_tensor(args[0]), _float_tensor(args[1])
    if da != db:
        raise _Decline("mixed dtypes")
    if a.shape != b.shape:
        raise _Decline("broadcasting")
    if op == "aten.mul":
        if da == "f64":
            raise _Decline("f64 products are not exact in binary64")
        lo, hi = _product_bounds(a, b)
    else:
        lo, hi = _two_sum_bounds(a, b if op == "aten.add" else -b)
    return BulkTensor(lo, hi, da, True)


def _index(inp):
    op, args, kwargs = inp["op"], list(inp.get("args") or []), dict(inp.get("kwargs") or {})
    if op != "aten.embedding" or len(args) != 2 or kwargs:
        raise _Decline("operation")
    weight, dtype = _float_tensor(args[0])
    if np.any((weight == 0) & np.signbit(weight)):
        raise Invalid("weight contains negative zero outside the logical-value profile")
    idx, _ = _record(args[1], floating=False)
    if weight.ndim != 2:
        raise _Decline("weight rank")
    if idx.size and (idx.min() < 0 or idx.max() >= weight.shape[0]):
        raise _Decline("index out of range")
    out = np.take(weight, idx, axis=0)
    return BulkTensor(out, out.copy(), dtype, True)


def _softmax(inp):
    import flint
    from . import aten as A
    from .aten_adapter import decode
    op, args, kwargs = inp["op"], list(inp.get("args") or []), dict(inp.get("kwargs") or {})
    if kwargs:
        raise _Decline("keywords")
    if op == "aten._log_softmax" and len(args) == 3:
        x, dtype = _float_tensor(args[0])
        dim, half = args[1], args[2]
        if half is not False or type(dim) is not int or x.ndim == 0 or not -x.ndim <= dim < x.ndim:
            raise _Decline("options")
        rows = np.moveaxis(x, dim, -1)
        lo, hi = np.empty(rows.shape), np.empty(rows.shape)
        flo, fhi = lo.reshape(-1, rows.shape[-1]), hi.reshape(-1, rows.shape[-1])

        def row_bounds(row):
            # log sum_j exp(x_j - m) = log(c + s) = log(c) + log1p(s / c), c = count of maxima, s = the rest:
            # keeps full relative accuracy when the maximum dominates (the result is then tiny)
            mx = max(row)
            m = flint.arb(mx)
            shifted = [flint.arb(v) - m for v in row]
            c = sum(1 for v in row if v == mx)
            rest = flint.arb(0)
            for v, d in zip(row, shifted):
                if v != mx:
                    rest += d.exp()
            log_mass = flint.arb(c).log() + (rest / c).log1p()
            return [_arb_bounds(d - log_mass) for d in shifted]

        for r, row in enumerate(rows.reshape(-1, rows.shape[-1]).tolist()):
            if row:
                flo[r], fhi[r] = zip(*_refined(row_bounds, row))
        return BulkTensor(np.moveaxis(lo, -1, dim), np.moveaxis(hi, -1, dim), dtype, False)
    if op == "aten._log_softmax_backward_data" and len(args) == 4:
        (g, dg), (y, dy) = _float_tensor(args[0]), _float_tensor(args[1])
        if g.shape != y.shape or dg != dy:
            raise _Decline("operands")
        dim = args[2]
        target = A.dt(decode(args[3])) if isinstance(args[3], (dict, str)) else None
        if target not in FLOATS or type(dim) is not int or g.ndim == 0 or not -g.ndim <= dim < g.ndim:
            raise _Decline("options")
        gr, yr = np.moveaxis(g, dim, -1), np.moveaxis(y, dim, -1)
        lo, hi = np.empty(gr.shape), np.empty(gr.shape)
        flo, fhi = lo.reshape(-1, gr.shape[-1]), hi.reshape(-1, gr.shape[-1])

        def row_bounds(pair):
            grow, yrow = pair
            s = flint.arb(0)
            for v in grow:
                s += flint.arb(v)
            return [_arb_bounds(flint.arb(gv) - flint.arb(yv).exp() * s) for gv, yv in zip(grow, yrow)]

        for r, pair in enumerate(zip(gr.reshape(-1, gr.shape[-1]).tolist(), yr.reshape(-1, yr.shape[-1]).tolist())):
            if pair[0]:
                flo[r], fhi[r] = zip(*_refined(row_bounds, pair))
        return BulkTensor(np.moveaxis(lo, -1, dim), np.moveaxis(hi, -1, dim), target, False)
    raise _Decline("operation")


def _act(inp):
    import flint
    op, args, kwargs = inp["op"], list(inp.get("args") or []), dict(inp.get("kwargs") or {})
    if op != "aten.silu_backward" or len(args) != 2 or kwargs:
        raise _Decline("operation")
    (g, dg), (x, dx) = _float_tensor(args[0]), _float_tensor(args[1])
    if g.shape != x.shape:
        raise _Decline("shapes")
    if dg != dx:
        raise _Decline("mixed dtypes")
    lo, hi = np.empty(g.shape), np.empty(g.shape)
    flo, fhi = lo.reshape(-1), hi.reshape(-1)

    def element(pair):
        gv, xv = pair
        one, xa = flint.arb(1), flint.arb(xv)
        s = one / (one + (-xa).exp())
        return [_arb_bounds(flint.arb(gv) * (s * (one + xa * (one - s))))]

    for i, pair in enumerate(zip(g.ravel().tolist(), x.ravel().tolist())):
        (flo[i], fhi[i]), = _refined(element, pair)
    return BulkTensor(lo, hi, dg, False)


_FMT = {"f16": (11, -14, 15), "bf16": (8, -126, 127), "f32": (24, -126, 127), "f64": (53, -1022, 1023)}


def _rne(x, target):
    """Exact roundTiesToEven of binary64 values into target (precision p, minimum normal exponent emin): the value is
    scaled by a power of two so that its target-ulp is 1, rounded with rint (ties to even, exact on binary64
    integers below 2^53), and scaled back. Overflow beyond the target's largest finite value declines."""
    p, emin, emax = _FMT[target]
    if target == "f64":
        return x.copy()
    _, e = np.frexp(x)                        # x = m 2^e, 0.5 <= |m| < 1 (e = 0 for x = 0)
    q = np.maximum(e - 1, emin) - (p - 1)     # exponent of the target ulp (subnormals share emin's ulp)
    scaled = np.ldexp(x, -q)
    if not np.all(np.abs(scaled) < 2.0 ** 53):
        raise _Decline("scaling exceeded binary64 integer range")
    y = np.ldexp(np.rint(scaled), q)
    largest = (2.0 - 2.0 ** (1 - p)) * 2.0 ** emax
    if np.any(np.abs(y) > largest):
        raise _Decline("conversion overflows the target")
    return y


def _cast(inp):
    from . import aten as A
    from .aten_adapter import decode
    op, args, kwargs = inp["op"], list(inp.get("args") or []), dict(inp.get("kwargs") or {})
    if op == "aten._to_copy" and len(args) == 1 and set(kwargs) == {"dtype"}:
        target = kwargs["dtype"]
    elif op == "prims.convert_element_type" and len(args) == 2 and not kwargs:
        target = args[1]
    else:
        raise _Decline("operation")
    try:
        target = A.dt(decode(target))
    except Exception:  # noqa: BLE001 -- the adapter decides malformed dtypes
        raise _Decline("target dtype") from None
    if target not in FLOATS:
        raise _Decline("target dtype")
    x, _ = _float_tensor(args[0])
    y = _rne(x, target)
    return BulkTensor(y, y.copy(), target, True)


def _int_scalar(x):
    if isinstance(x, bool) or not isinstance(x, int):
        raise _Decline("integer argument form")
    return x


def _nll(inp):
    """NLL aten_real (kernel_spec.norm_emb_nll_aten): only the N target coordinates of self are read; weight None."""
    op, args, kwargs = inp["op"], list(inp.get("args") or []), dict(inp.get("kwargs") or {})
    if kwargs:
        raise _Decline("keywords")
    if op == "aten.nll_loss_forward" and len(args) == 5:
        self_, target, weight, reduction, ignore = args
    elif op == "aten.nll_loss_backward" and len(args) == 7:
        grad, self_, target, weight, reduction, ignore, total = args
    else:
        raise _Decline("operation")
    if weight is not None:
        raise _Decline("weight")
    red, ign = _int_scalar(reduction), _int_scalar(ignore)
    if red not in (0, 1, 2):
        raise _Decline("reduction")
    if op == "aten.nll_loss_forward":
        x, dtype = _float_tensor(self_)
    else:
        x, dtype = _record(self_)             # values not read (NaN storage allowed by the adapter)
    t, _ = _record(target, floating=False)
    if x.ndim != 2 or t.ndim != 1 or t.shape[0] != x.shape[0]:
        raise _Decline("shapes")
    n, c = x.shape
    keep = t != ign
    if np.any(keep & ((t < 0) | (t >= c))):
        raise _Decline("target out of range")
    rows, cols = np.nonzero(keep)[0], t[keep]
    if op == "aten.nll_loss_forward":
        picked = [-Fraction(float(v)) for v in x[rows, cols]]
        tw = Fraction(len(picked))
        if red == 0:
            losses = [Fraction(0)] * n
            for r, v in zip(rows.tolist(), picked):
                losses[r] = v
            return [losses, Fraction(0)]
        total_loss = sum(picked, Fraction(0))
        if red == 1:
            if tw == 0:
                raise _Decline("mean over total weight 0")
            total_loss /= tw
        return [total_loss, tw]
    g, _ = _float_tensor(grad)
    tws, _ = _float_tensor(total)
    if tws.size != 1:
        raise _Decline("total_weight size")
    if red == 0:
        if g.shape != (n,):
            raise _Decline("grad_output shape")
        gv = [Fraction(float(g[r])) for r in rows.tolist()]
    else:
        if g.size != 1 or g.ndim > 1:
            raise _Decline("grad_output shape")
        gv = [Fraction(float(g.ravel()[0]))] * len(rows)
    if red == 1:
        twv = Fraction(float(tws.ravel()[0]))
        if twv == 0 and len(rows):
            raise _Decline("mean backward with supplied total_weight 0")
        gv = [v / twv for v in gv]
    lo, hi = np.zeros((n, c)), np.zeros((n, c))
    for r, col, v in zip(rows.tolist(), cols.tolist(), gv):
        lo[r, col], hi[r, col] = _rational_bounds(-v)
    return BulkTensor(lo, hi, dtype, True)


ENTRIES = {("CAST", "finite_rne"): _cast, ("NLL", "aten_real"): _nll, ("MATMUL", "real"): _matmul, ("ARITH", "real"): _arith, ("INDEX", "real"): _index,
           ("SOFTMAX", "real"): _softmax, ("ACT", "torch"): _act}

# Below this many scalar operations the adapter itself is fast enough and stays the evaluator.
MIN_WORK = 200_000


def work(inp):
    """Rough scalar-operation count of one ATen call (the product dimension for matrix products)."""
    sizes = [math.prod(v["shape"]) for v in (inp.get("args") or []) if isinstance(v, dict) and "shape" in v]
    if not sizes:
        return 0
    if inp.get("op") in ("aten.mm", "aten.addmm", "aten.bmm"):
        shapes = [v["shape"] for v in inp["args"] if isinstance(v, dict) and "shape" in v]
        a, b = shapes[-2], shapes[-1]
        return math.prod(a) * (b[-1] if b else 1)
    return max(sizes)


def evaluate(cid, profile, inp, *, force=False):
    """Result('value', {'out': BulkTensor}) for an accepted call; None where the adapter must evaluate.
    Exceptions raised here are the adapter's own refusals for the same condition (num.Invalid / Unsupported)."""
    entry = ENTRIES.get((cid, profile))
    if entry is None or not isinstance(inp, dict) or not isinstance(inp.get("op"), str):
        return None
    if set(inp) - {"op", "args", "kwargs"}:
        return None
    if not force and work(inp) < MIN_WORK:
        return None
    try:
        out = entry(inp)
    except _Decline:
        return None
    except FloatingPointError:
        return None
    return Result("value", {"out": out}, note=VERSION)
