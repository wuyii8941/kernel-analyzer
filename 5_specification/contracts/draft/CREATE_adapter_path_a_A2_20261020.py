"""Tensor factory semantics from contracts/CREATE.yaml (aten.arange, zeros, ones, full, zeros_like, ones_like,
full_like, scalar_tensor).

The factory rules here are written from the PyTorch 2.10 documentation and the retained native schemas; they do not
call the shared kernel_spec.aten creation helpers. Float results are exact real task values (start + i*step, the fill
value); the stored rounding of a device result is the measured deviation, not part of f.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import yaml

from kernel_spec import aten as A
from kernel_spec import ieee
from kernel_spec.aten_adapter import decode, encode, make_reference
from kernel_spec.num import Invalid, Result, Undefined, Unsupported
from kernel_spec.real import NEG, POS, R
from kernel_spec.registry import CONTRACTS

COVERS = yaml.safe_load((CONTRACTS / "CREATE.yaml").read_text())["covers"]
FLOATS = {"f16", "bf16", "f32", "f64"}
INTS = {"i64": (-2 ** 63, 2 ** 63 - 1), "i32": (-2 ** 31, 2 ** 31 - 1), "i16": (-2 ** 15, 2 ** 15 - 1),
        "i8": (-2 ** 7, 2 ** 7 - 1), "u8": (0, 255)}
BOOL = "b8"
# profile -> (output field, default floating dtype or None, admitted result dtypes)
PROFILES = {
    "explicit_numeric": ("out", None, FLOATS | set(INTS)),
    "explicit_boolean": ("out_bool", None, {BOOL}),
    "default_float32_numeric": ("out", "f32", FLOATS | set(INTS)),
}
# Band around an integer quotient inside which a non-integer arange length is refused (decision CR.ARANGE_LENGTH).
ARANGE_BAND = Fraction(1, 2 ** 30)
DOUBLE_EXACT = 2 ** 53


# ---------------------------------------------------------------- scalar arguments
def _kind(value):
    """'bool', 'int' or 'float' for a decoded scalar argument (a 'p/q' string or an infinity counts as floating)."""
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating, Fraction, R)):
        return "float"
    raise Invalid(f"{value!r} is not a number")


def _real(value, name):
    """Exact real value of a scalar argument; NaN has no real value."""
    if isinstance(value, (bool, np.bool_)):
        return R(int(value))
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        raise Unsupported(f"{name} is NaN, which has no real value")
    try:
        result = value if isinstance(value, R) else R(value if not isinstance(value, np.generic) else value.item())
    except Undefined:
        raise Unsupported(f"{name} has no real value") from None
    except (TypeError, ValueError):
        raise Invalid(f"{name} must be a number") from None
    if not result.exact:
        raise Invalid(f"{name} must be exact")
    return result


def _dtype(value, name="dtype"):
    if value is None:
        return None
    try:
        d = A.dt(value)
    except Invalid:
        raise Invalid(f"{name} {value!r} is not a dtype") from None
    if d not in FLOATS and d not in INTS and d != BOOL:
        raise Unsupported(f"{name} {d!r} is outside the CREATE profiles")
    return d


def _element(value, dtype, name):
    """The task value of a fill value in the result dtype (decision CR.CONVERSION)."""
    kind = _kind(value)
    if dtype in FLOATS:
        x = _real(value, name)
        if x.finite and not ieee.round_to(x, dtype).finite:
            raise Unsupported(f"{name}={x!r} overflows {dtype}; overflow is outside the CREATE profiles")
        return x
    if dtype == BOOL:
        if kind == "bool":
            return bool(value)
        x = _real(value, name)
        if x.finite and x.v in (0, 1):
            return x.v == 1
        raise Unsupported(f"{name}={value!r} for a Boolean result is outside the CREATE profiles")
    x = _real(value, name)
    if not x.finite:
        raise Invalid(f"{name} is infinite and cannot be stored in {dtype}")
    if x.v.denominator != 1:
        raise Unsupported(f"non-integer {name} for an integer result is outside the CREATE profiles")
    lo, hi = INTS[dtype]
    if not lo <= x.v <= hi:
        raise Invalid(f"{name}={x.v} is outside the range of {dtype}")
    return int(x.v)


def _size(size):
    if isinstance(size, (int, np.integer)) and not isinstance(size, (bool, np.bool_)):
        size = [size]
    if not isinstance(size, (list, tuple)):
        raise Invalid("size must be a list of integers")
    out = []
    for n in size:
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            raise Invalid("size entries must be integers")
        if n < 0:
            raise Invalid("size entries must be nonnegative")
        out.append(int(n))
    return tuple(out)


def _filled(shape, value, dtype):
    a = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        a[index] = value
    return A.Ten(a, dtype)


def _default(default):
    if default is None:
        raise Unsupported("the result dtype is the process default floating dtype, which the call does not record; "
                          "profile default_float32_numeric declares it as float32")
    return default


# ---------------------------------------------------------------- operations (built per profile)
def _operations(default):
    def arange(start_or_end, end=None, step=None, *, dtype=None):
        if end is None:
            if step is not None:
                raise Invalid("arange: a step requires start and end")
            start, end, step = 0, start_or_end, 1
        else:
            start = start_or_end
            step = 1 if step is None else step
        kinds = [_kind(v) for v in (start, end, step)]
        if "bool" in kinds:
            raise Unsupported("Boolean arange arguments are outside the CREATE profiles")
        reals = [_real(v, n) for v, n in zip((start, end, step), ("start", "end", "step"))]
        if not all(x.finite for x in reals):
            raise Invalid("arange: start, end and step must be finite")
        s, e, st = (x.v for x in reals)
        if st == 0:
            raise Invalid("arange: step must be nonzero")
        if (st > 0 and e < s) or (st < 0 and e > s):
            raise Invalid("arange: upper bound and lower bound inconsistent with step sign")
        d = _dtype(dtype)
        if d is None:
            d = _default(default) if "float" in kinds else "i64"
        if d == BOOL:
            raise Unsupported("Boolean arange results are outside the CREATE profiles")
        q = (e - s) / st
        if d in INTS:
            if not all(x.denominator == 1 for x in (s, e, st)):
                raise Unsupported("non-integer arange arguments with an integer result dtype are outside the profiles")
            lo, hi = INTS["i64"]
            if not all(lo <= x <= hi for x in (s, e, st)):
                raise Invalid("arange: integer arguments outside the int64 range of an integer arange")
        elif not all(x.denominator == 1 and abs(x) < DOUBLE_EXACT for x in (s, e, st)):
            distance = min(q - math.floor(q), math.ceil(q) - q)
            if distance <= ARANGE_BAND * max(1, abs(q)):
                raise Unsupported("arange: (end - start) / step lies at or next to an integer with non-integer "
                                  "arguments; the documented length is subject to rounding there")
        n = math.ceil(q)
        values = [s + i * st for i in range(n)]
        out = np.empty((n,), dtype=object)
        for i, v in enumerate(values):
            out[i] = _element(v, d, f"arange value {i}")
        return A.Ten(out, d)

    def zeros(size, *, dtype=None):
        d = _dtype(dtype) or _default(default)
        return _filled(_size(size), _element(0, d, "zero"), d)

    def ones(size, *, dtype=None):
        d = _dtype(dtype) or _default(default)
        return _filled(_size(size), _element(1, d, "one"), d)

    def full(size, fill_value, *, dtype=None):
        d = _dtype(dtype)
        if d is None:
            kind = _kind(fill_value)
            d = BOOL if kind == "bool" else "i64" if kind == "int" else _default(default)
        return _filled(_size(size), _element(fill_value, d, "fill_value"), d)

    def like(input, value, dtype):
        if not isinstance(input, A.Ten):
            raise Invalid("input must be a tensor")
        d = _dtype(dtype) or input.dtype
        if d not in FLOATS and d not in INTS and d != BOOL:
            raise Unsupported(f"dtype {d!r} is outside the CREATE profiles")
        return _filled(input.shape, _element(value, d, "fill_value"), d)

    def zeros_like(input, *, dtype=None):
        return like(input, 0, dtype)

    def ones_like(input, *, dtype=None):
        return like(input, 1, dtype)

    def full_like(input, fill_value, *, dtype=None):
        return like(input, fill_value, dtype)

    def scalar_tensor(s, *, dtype=None):
        d = _dtype(dtype)
        if d is None:
            raise Unsupported("scalar_tensor without dtype: the sources disagree on the result dtype (CR.SCALAR_DTYPE)")
        return _filled((), _element(s, d, "s"), d)

    return {"aten.arange": arange, "aten.zeros": zeros, "aten.ones": ones, "aten.full": full,
            "aten.zeros_like": zeros_like, "aten.ones_like": ones_like, "aten.full_like": full_like,
            "aten.scalar_tensor": scalar_tensor}


def _decode_argument(op, name, value, arguments):
    if name == "input":
        # Only the shape and dtype of a *_like input are read (CR.LIKE); its values, including NaN, are ignored.
        if not isinstance(value, dict) or "t" not in value:
            raise Invalid("input must be a tensor")
        tensor = decode(value, allow_ignored_nan=True)
        if not isinstance(tensor, A.Ten):
            raise Invalid("input must be a tensor")
        return tensor
    return decode(value)


def _build(table, profile_names, check_dtype=True):
    raw = {p: make_reference(COVERS, extra=table(PROFILES[p][1]), profiles=(p,), encode_output=False,
                             argument_decoder=_decode_argument) for p in profile_names}

    def reference(profile, inp):
        if profile not in raw:
            raise Unsupported(f"profile {profile!r} is not covered by this contract")
        field, _, admitted = PROFILES[profile]
        out = raw[profile](profile, inp).fields["out"]
        if check_dtype and out.dtype not in admitted:
            raise Unsupported(f"result dtype {out.dtype} belongs to another CREATE profile")
        return Result("value", {field: encode(out)})
    return reference


reference = _build(_operations, tuple(PROFILES))


def check_output(profile, inp, observed):
    """Complete observed result {field: {t, v, shape}}: dtype, shape and every value equal the reference."""
    from kernel_spec.compare import check_enclosed_output
    field = PROFILES[profile][0]
    raw = make_reference(COVERS, extra=_operations(PROFILES[profile][1]), profiles=(profile,), encode_output=False,
                         argument_decoder=_decode_argument)
    expected = raw(profile, inp).fields["out"]
    if expected.dtype not in PROFILES[profile][2]:
        raise Unsupported(f"result dtype {expected.dtype} belongs to another CREATE profile")
    if not isinstance(observed, dict) or set(observed) != {field}:
        return False
    candidate = observed[field]
    if not isinstance(candidate, dict) or set(candidate) != {"t", "v", "shape"}:
        return False
    try:
        if A.dt(candidate["t"]) != expected.dtype or tuple(candidate["shape"]) != expected.shape:
            return False
        got = decode(candidate)
    except (Invalid, Unsupported, Undefined):
        return False
    if got.shape != expected.shape:
        return False
    return check_enclosed_output(encode(expected), encode(got))


# ---------------------------------------------------------------- mutants (independent wrong implementations)
def _mutant_table(replacements):
    def table(default):
        ops = _operations(default)
        ops.update({k: f(default) for k, f in replacements.items()})
        return ops
    return table


def _m_arange_includes_end(default):
    def arange(start_or_end, end=None, step=None, *, dtype=None):
        start, stop = (0, start_or_end) if end is None else (start_or_end, end)
        st = Fraction(1 if step is None else step)
        n = math.floor((Fraction(stop) - Fraction(start)) / st) + 1
        return A.Ten(np.array([Fraction(start) + i * st for i in range(n)], dtype=object), A.dt(dtype or "i64"))
    return arange


def _m_arange_floor_length(default):
    def arange(start_or_end, end=None, step=None, *, dtype=None):
        start, stop = (0, start_or_end) if end is None else (start_or_end, end)
        st = Fraction(1 if step is None else step)
        n = math.floor((Fraction(stop) - Fraction(start)) / st)
        return A.Ten(np.array([Fraction(start) + i * st for i in range(n)], dtype=object), A.dt(dtype or "i64"))
    return arange


def _m_arange_from_one(default):
    def arange(start_or_end, end=None, step=None, *, dtype=None):
        start, stop = (1, start_or_end + 1) if end is None else (start_or_end, end)
        st = Fraction(1 if step is None else step)
        n = math.ceil((Fraction(stop) - Fraction(start)) / st)
        return A.Ten(np.array([Fraction(start) + i * st for i in range(n)], dtype=object), A.dt(dtype or "i64"))
    return arange


def _m_arange_float_default(default):
    def arange(start_or_end, end=None, step=None, *, dtype=None):
        start, stop = (0, start_or_end) if end is None else (start_or_end, end)
        st = Fraction(1 if step is None else step)
        n = math.ceil((Fraction(stop) - Fraction(start)) / st)
        d = A.dt(dtype or "f32")          # misreading: integer arguments also give the default floating dtype
        return A.Ten(np.array([R(Fraction(start) + i * st) for i in range(n)], dtype=object), d)
    return arange


def _m_ones_like_float(default):
    def ones_like(input, *, dtype=None):
        a = np.empty(input.shape, dtype=object)   # misreading: always the default float dtype
        for index in np.ndindex(input.shape):
            a[index] = R(1)
        return A.Ten(a, "f32")
    return ones_like


def _m_scalar_finite(default):
    maxima = {"f32": (2 - Fraction(1, 2 ** 23)) * 2 ** 127, "bf16": (2 - Fraction(1, 2 ** 7)) * 2 ** 127,
              "f16": Fraction(65504), "f64": (2 - Fraction(1, 2 ** 52)) * 2 ** 1023}

    def scalar_tensor(s, *, dtype=None):
        d = A.dt(dtype)
        x = R(s) if not isinstance(s, R) else s
        if x.v == NEG:                       # misreading: infinity replaced by the largest finite magnitude
            x = R(-maxima[d])
        elif x.v == POS:
            x = R(maxima[d])
        return A.Ten(np.array(x, dtype=object), d)
    return scalar_tensor


def _m_zeros_no_empty(default):
    def zeros(size, *, dtype=None):
        shape = tuple(max(1, int(n)) for n in size)   # misreading: an empty dimension becomes size 1
        a = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            a[index] = R(0)
        return A.Ten(a, A.dt(dtype or "f32"))
    return zeros


MUTANTS = {
    "CR.M1": _build(_mutant_table({"aten.arange": _m_arange_includes_end}), tuple(PROFILES), check_dtype=False),
    "CR.M2": _build(_mutant_table({"aten.arange": _m_arange_floor_length}), tuple(PROFILES), check_dtype=False),
    "CR.M3": _build(_mutant_table({"aten.arange": _m_arange_float_default}), tuple(PROFILES), check_dtype=False),
    "CR.M4": _build(_mutant_table({"aten.ones_like": _m_ones_like_float}), tuple(PROFILES), check_dtype=False),
    "CR.M5": _build(_mutant_table({"aten.scalar_tensor": _m_scalar_finite}), tuple(PROFILES), check_dtype=False),
    "CR.M6": _build(_mutant_table({"aten.zeros": _m_zeros_no_empty}), tuple(PROFILES), check_dtype=False),
    "CR.M7": _build(_mutant_table({"aten.arange": _m_arange_from_one}), tuple(PROFILES), check_dtype=False),
}
READINGS = {}
