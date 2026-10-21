"""Shared adapter for the contracts whose clauses are ATen operations (ARITH, MATHFN, LOGIC, CAST, LAYOUT, INDEX,
REDUCE, SOFTMAX, MATMUL, CONV, ACT, RANDOM).

Example inputs: {"op": "aten.add", "args": [...], "kwargs": {...}} where a tensor argument is written
{"t": "f32", "v": nested list} (values: integers, "p/q" strings, "inf" / "-inf", booleans; an optional "shape" reshapes
the values, needed for empty tensors such as shape [0, 3]), a list of tensors is a
list of such objects, and other arguments are plain YAML values ("p/q" strings become exact rationals).
The output field `out` is the operation's result as nested lists; operations with several results give a list.
"""
from __future__ import annotations

from fractions import Fraction
import inspect

from . import aten as A
from .num import Invalid, Result, Unsupported, rational, require_fields
from .real import NEG, POS, R


def decode(x, *, allow_ignored_nan=False):
    if isinstance(x, dict) and ("t" in x or "v" in x):
        require_fields(x, {"t", "v"}, "tensor argument")
        if set(x) - {"t", "v", "shape"}:
            raise Unsupported("tensor argument contains unsupported layout or metadata fields")
        t = A.tensor(_vals(x["v"], x["t"], allow_ignored_nan=allow_ignored_nan), x["t"],
                     allow_ignored_nan=allow_ignored_nan)
        return A.view(t, x["shape"]) if "shape" in x else t
    if isinstance(x, dict) and "dtype" in x:
        require_fields(x, {"dtype"}, "dtype argument")
        if set(x) != {"dtype"}:
            raise Unsupported("dtype argument contains unsupported fields")
        return A.dt(x["dtype"])
    if isinstance(x, list):
        return [decode(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if s in ("inf", "+inf"):
            return R(POS)
        if s == "-inf":
            return R(NEG)
        if s.startswith("torch.") or s in A.FLOAT | A.INT | A.BOOL:
            return s
        try:
            return rational(s)
        except Invalid:
            return s
    return x


def _vals(v, d, *, allow_ignored_nan=False):
    if hasattr(v, "tolist") and hasattr(v, "dtype"):      # a NumPy-array record (large tensors): same values
        v = v.tolist()
    if isinstance(v, list):
        return [_vals(x, d, allow_ignored_nan=allow_ignored_nan) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if allow_ignored_nan and s.lower() == "nan":
            return float("nan")
        if s in ("inf", "+inf"):
            return float("inf")
        if s == "-inf":
            return float("-inf")
        return rational(s)
    return v


def encode(r):
    if isinstance(r, A.Ten):
        return _enc(r.a)
    if isinstance(r, (tuple, list)):
        return [encode(x) for x in r]
    return _enc(r)


def _enc(a):
    import numpy as np
    if isinstance(a, np.ndarray):
        return _enc(a[()]) if a.ndim == 0 else [_enc(x) for x in a]
    if isinstance(a, A.AllowedSet):
        return frozenset(_enc(m) for m in a.members)
    if isinstance(a, R):
        return a.v
    if isinstance(a, Fraction):
        return a
    return a


def bind_reference_call(op, signature, args, kwargs):
    """Validate a call before decoding or choosing a full/lazy evaluator."""
    named = {name for name, p in signature.parameters.items()
             if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    unknown = set(kwargs) - named
    if unknown:
        raise Unsupported(f"{op}: keyword arguments not supported by the reference: {sorted(unknown)}")
    try:
        return signature.bind(*args, **kwargs)
    except TypeError as error:
        raise Invalid(f"{op}: arguments do not match the reference signature: {error}") from error


def make_reference(covers, extra=None, *, profiles=None, encode_output=True, argument_decoder=None,
                   argument_aliases=None):
    """reference(profile, inp) for a contract that covers the given ATen operation names."""
    table = dict(A.OPS)
    table.update(extra or {})
    signatures = {}

    def reference(profile, inp):
        if profiles is not None and (not isinstance(profile, str) or profile not in profiles):
            raise Unsupported(f"profile {profile!r} is not covered by this contract")
        require_fields(inp, {"op"}, "ATen call")
        if set(inp) - {"op", "args", "kwargs"}:
            raise Unsupported("ATen call contains fields outside op, args and kwargs")
        op = inp.get("op")
        if not isinstance(op, str):
            raise Invalid("ATen operation must be a string")
        if op not in covers:
            raise Unsupported(f"{op} is not covered by this contract")
        fn = table.get(op)
        if fn is None:
            raise Unsupported(f"{op} has no executable semantics")
        raw_args, raw_kwargs = inp.get("args", []), inp.get("kwargs", {})
        if not isinstance(raw_args, (list, tuple)):
            raise Invalid("ATen args must be a list or tuple")
        if raw_kwargs is None:
            raw_kwargs = {}
        require_fields(raw_kwargs, (), "ATen kwargs")
        # Public schema names may differ from a reference function's internal
        # names. Only the contract's explicit per-operation mapping is used;
        # duplicate bindings remain errors even when the values agree.
        aliases = (argument_aliases or {}).get(op, {})
        if aliases:
            normalized = {}
            for name, value in raw_kwargs.items():
                target = aliases.get(name, name)
                if target in normalized:
                    raise Invalid(f"{op}: multiple keyword arguments for {target!r}")
                normalized[target] = value
            raw_kwargs = normalized
        if op not in signatures:
            signatures[op] = inspect.signature(fn)
        signature = signatures[op]
        bound = bind_reference_call(op, signature, raw_args, raw_kwargs)
        if argument_decoder is None:
            args = [decode(a) for a in raw_args]
            kwargs = {k: decode(v) for k, v in raw_kwargs.items()}
        else:
            # Contract-specific decoding sees validated argument names and the
            # original parameters. It cannot bypass signature validation.
            raw = dict(bound.arguments)
            bound.arguments = {k: argument_decoder(op, k, v, raw) for k, v in raw.items()}
            args, kwargs = bound.args, bound.kwargs
        output = fn(*args, **kwargs)  # An internal TypeError stays an implementation error.
        return Result("value", {"out": encode(output) if encode_output else output})
    return reference


def make_output_checker(covers, extra=None, *, profiles=None, argument_decoder=None, argument_aliases=None):
    """Complete tensor results, with explicit observed dtype/shape even for empty tensors.

    Observations use the same tensor records as inputs, with all of t/v/shape
    required. This checks the supplied output against a task, not kernel binding.
    """
    from .compare import check_enclosed_output
    reference = make_reference(covers, extra, profiles=profiles, encode_output=False,
                               argument_decoder=argument_decoder, argument_aliases=argument_aliases)

    def check(profile, inp, observed):
        expected = reference(profile, inp).fields["out"]
        if not isinstance(observed, dict) or set(observed) != {"out"}:
            return False

        def prepare(result, candidate):
            if result is None:
                return candidate is None, None
            if isinstance(result, A.Ten):
                if not isinstance(candidate, dict) or set(candidate) != {"t", "v", "shape"}:
                    return False, None
                try:
                    shape = A._layout_sizes(candidate["shape"], "observed tensor shape")
                    if any(size < 0 for size in shape):
                        return False, None
                    if result.dtype != A.dt(candidate["t"]) or result.shape != tuple(shape):
                        return False, None
                    decoded = decode(candidate)
                except (Invalid, Unsupported):
                    return False, None
                if result.dtype != decoded.dtype or result.shape != decoded.shape:
                    return False, None
                return True, encode(decoded)
            if isinstance(result, (tuple, list)):
                if not isinstance(candidate, (tuple, list)) or len(result) != len(candidate):
                    return False, None
                children = [prepare(r, c) for r, c in zip(result, candidate)]
                return all(ok for ok, _ in children), [value for _, value in children]
            raise Unsupported("complete output comparison requires tensor results")

        complete, values = prepare(expected, observed["out"])
        return complete and check_enclosed_output(encode(expected), values)

    return check
