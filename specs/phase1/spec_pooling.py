"""Independent specification f for avg_pool{1,2,3}d and max_pool{1,2,3}d.  (v0.4)

Written ONLY from the official documentation (no PyTorch source code consulted):
  https://docs.pytorch.org/docs/main/generated/torch.nn.AvgPool2d.html   (fetched 2026-10-07)
  https://docs.pytorch.org/docs/main/generated/torch.nn.MaxPool2d.html   (fetched 2026-10-07)
The 1d/3d variants are taken to follow the same per-dimension rules (assumption, listed in ambiguities.md as POOL-A0).

Documented semantics used:
  P-D1  avg "simplest case": out = 1/(kH*kW) * sum_{m,n} input(stride*h + m, stride*w + n)
  P-D2  avg: "the input is implicitly zero-padded on both sides for padding number of points"
  P-D3  max: "implicitly padded with negative infinity on both sides"; dilation "controls the spacing between the kernel points"
  P-D4  ceil_mode: "will use ceil instead of floor to compute the output shape"; "sliding windows are allowed to go
        off-bounds if they start within the left padding or the input. Sliding windows that would start in the right
        padded region are ignored." Shape: if ceil_mode and (out-1)*stride >= L_in + padding, out is reduced by one.
  P-D5  output length (avg): floor((L + 2p - k)/s + 1); (max): floor((L + 2p - d(k-1) - 1)/s + 1)
  P-D6  count_include_pad: "when True, will include the zero-padding in the averaging calculation"
  P-D7  divisor_override: "if specified, it will be used as divisor, otherwise size of the pooling region will be used"
  P-D8  "pad should be at most half of effective kernel size" (precondition; violating inputs are out of scope)

Ambiguities (every reading evaluated):
  POOL-A1  count_include_pad=True and a ceil_mode window that runs past the right padding ("overhang"):
           R2 divisor = |W_o ∩ P|, the number of window positions inside the padded input P = [-p, L+p) - MAIN reading
           (adopted 2026-10-07); R1 divisor = product of kernel sizes (P-D1 formula) - kept as the tail reading for
           comparison. The two coincide whenever the whole window lies inside P. AvgPool3d's divisor_override text still
           calls the default divisor kernel_size: this documentation tension is recorded, and a difference caused only by
           the overhang divisor is a "difference under declared interpretation", not a confirmed bug.
  POOL-A2  max pooling tie-breaking for return_indices and for routing the gradient: not documented.
           The spec returns the SET of argmax positions; a candidate is consistent iff its index is in the set and its
           gradient goes to exactly one member of the set (or is split - recorded as a different reading).
  POOL-A3  NaN inside a max window: not documented. R_prop (output NaN; only NaNs that actually participate in the
           window propagate) is the project's diagnostic profile; R_ignore (NaN skipped) is an auxiliary reading. A window
           whose real values are ALL NaN under R_ignore has no numeric value left: this is NOT the geometric empty-window
           case (POOL-A5) and the spec returns "not established" (SpecNotEstablished) for it rather than -inf.
  POOL-A4  the max formula in the documentation omits dilation even though dilation is a parameter (documentation
           inconsistency); the spec uses position = s*o - p + d*m.
Empty windows (v0.2): with dilation (or large padding) a legal window may sample only padding positions. For max pooling the
padding is -inf (P-D3), so the output is -inf and the argmax set is empty; this is part of the spec, not a failure.
Under POOL-A3 R_ignore, a window whose real values are all NaN is a different case and the spec declines (SpecNotEstablished). Avg pooling with non-finite
inputs is outside the scope of this spec (raises). Max pooling accepts +/-inf inputs (extended reals; inf compares with
Fraction correctly in Python). Everything else is exact (Fraction). Inputs are nested lists [C][L...] without batch.
"""
from fractions import Fraction
from itertools import product
import math


class SpecInputError(ValueError):
    """illegal or unsupported input (e.g. padding above half the effective kernel, P-D8): refused, not answered."""


class SpecNotEstablished(Exception):
    """the spec declines for this condition (e.g. a max window with only NaN values under R_ignore)."""


def _validate(Ls, k, s, p, d, divisor_override=None):
    n = len(Ls)
    for name, v in (("kernel", k), ("stride", s), ("dilation", d)):
        if len(v) != n or any(int(x) != x or x < 1 for x in v):
            raise SpecInputError(f"{name} must be {n} positive integers")
    if len(p) != n or any(int(x) != x or x < 0 for x in p):
        raise SpecInputError("padding must be non-negative integers")
    for i in range(n):
        if 2 * p[i] > d[i] * (k[i] - 1) + 1:           # P-D8: pad at most half of the effective kernel size
            raise SpecInputError("padding exceeds half of the effective kernel size (P-D8)")
        if Ls[i] + 2 * p[i] < d[i] * (k[i] - 1) + 1:
            raise SpecInputError("effective kernel larger than padded input")
    if divisor_override is not None and (int(divisor_override) != divisor_override or divisor_override == 0):
        raise SpecInputError("divisor_override must be a non-zero integer")


def _out_len(L, k, s, p, d, ceil_mode):
    num = L + 2 * p - d * (k - 1) - 1
    out = (math.ceil(Fraction(num, s)) if ceil_mode else num // s) + 1
    if ceil_mode and (out - 1) * s >= L + p:
        out -= 1
    return out


def _shape(t):
    s = []
    while isinstance(t, list):
        s.append(len(t)); t = t[0]
    return s


def _get(t, idx):
    for i in idx:
        t = t[i]
    return t


def _build(shape, fn, prefix=()):
    if not shape:
        return fn(prefix)
    return [_build(shape[1:], fn, prefix + (i,)) for i in range(shape[0])]


def _norm(v, n):
    return list(v) if isinstance(v, (list, tuple)) else [v] * n


def avg_pool(x, kernel, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None,
             reading="R2"):
    """x: [C][L1]...[Ln] (n = 1,2,3). Returns exact Fractions."""
    shape = _shape(x)
    C, Ls = shape[0], shape[1:]
    n = len(Ls)
    k = _norm(kernel, n); s = _norm(stride if stride is not None else kernel, n); p = _norm(padding, n)
    _validate(Ls, k, s, p, [1] * n, divisor_override)
    outs = [_out_len(Ls[i], k[i], s[i], p[i], 1, ceil_mode) for i in range(n)]

    def one(idx):
        c, o = idx[0], idx[1:]
        total = Fraction(0)
        n_real = n_padded = 0
        for m in product(*[range(k[i]) for i in range(n)]):
            pos = [s[i] * o[i] - p[i] + m[i] for i in range(n)]
            inside = all(0 <= pos[i] < Ls[i] for i in range(n))
            in_padded = all(-p[i] <= pos[i] < Ls[i] + p[i] for i in range(n))
            if inside:
                v = _get(x, [c] + pos)
                if isinstance(v, float) and not math.isfinite(v):
                    raise ValueError("avg_pool spec: non-finite inputs are out of scope")
                total += Fraction(v); n_real += 1
            if in_padded:
                n_padded += 1
        if divisor_override is not None:
            div = divisor_override
        elif not count_include_pad:
            div = n_real
        elif reading == "R2":
            div = n_padded
        else:
            div = math.prod(k)
        return total / div

    return _build([C] + outs, one)


def avg_pool_backward(grad_out, x_shape, kernel, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                      divisor_override=None, reading="R2"):
    """exact adjoint of avg_pool (it is linear): returns d<avg_pool(x), g>/dx."""
    C, Ls = x_shape[0], x_shape[1:]
    n = len(Ls)
    k = _norm(kernel, n); s = _norm(stride if stride is not None else kernel, n); p = _norm(padding, n)
    _validate(Ls, k, s, p, [1] * n, divisor_override)
    outs = _shape(grad_out)[1:]
    expected = [_out_len(Ls[i], k[i], s[i], p[i], 1, ceil_mode) for i in range(n)]
    if outs != expected or _shape(grad_out)[0] != C:
        raise SpecInputError(f"grad_out shape {_shape(grad_out)} does not match the output shape {[C] + expected}")
    gx = {}
    for c in range(C):
        for o in product(*[range(v) for v in outs]):
            n_real = n_padded = 0
            cells = []
            for m in product(*[range(k[i]) for i in range(n)]):
                pos = tuple(s[i] * o[i] - p[i] + m[i] for i in range(n))
                if all(0 <= pos[i] < Ls[i] for i in range(n)):
                    cells.append(pos); n_real += 1
                if all(-p[i] <= pos[i] < Ls[i] + p[i] for i in range(n)):
                    n_padded += 1
            div = (divisor_override if divisor_override is not None else
                   n_real if not count_include_pad else n_padded if reading == "R2" else math.prod(k))
            g = Fraction(_get(grad_out, (c,) + o)) / div
            for pos in cells:
                gx[(c,) + pos] = gx.get((c,) + pos, Fraction(0)) + g
    return _build([C] + list(Ls), lambda idx: gx.get(idx, Fraction(0)))


def _ext(v):
    """extended-real value: Fraction for finite numbers, the float itself for +/-inf (NaN handled separately)."""
    if isinstance(v, float) and math.isinf(v):
        return v
    return Fraction(v)


def max_pool(x, kernel, stride=None, padding=0, dilation=1, ceil_mode=False, nan_reading="R_prop"):
    """Returns (values, argmax_sets). argmax_sets[c][o...] is the set of input positions attaining the max (empty for
    an empty window). Values are Fractions, +/-inf floats, or NaN (POOL-A3 R_prop). Padding is -inf (P-D3), so a window
    that samples no real position evaluates to -inf (see module docstring)."""
    shape = _shape(x)
    C, Ls = shape[0], shape[1:]
    n = len(Ls)
    k = _norm(kernel, n); s = _norm(stride if stride is not None else kernel, n)
    p = _norm(padding, n); d = _norm(dilation, n)
    _validate(Ls, k, s, p, d)
    outs = [_out_len(Ls[i], k[i], s[i], p[i], d[i], ceil_mode) for i in range(n)]
    vals, sets = {}, {}
    for c in range(C):
        for o in product(*[range(v) for v in outs]):
            cand = []
            for m in product(*[range(k[i]) for i in range(n)]):
                pos = tuple(s[i] * o[i] - p[i] + d[i] * m[i] for i in range(n))
                if all(0 <= pos[i] < Ls[i] for i in range(n)):
                    cand.append((pos, _get(x, (c,) + pos)))
            nans = [pos for pos, v in cand if isinstance(v, float) and math.isnan(v)]
            if nans and nan_reading == "R_prop":
                vals[(c,) + o] = float("nan"); sets[(c,) + o] = set(nans)
                continue
            had_real = bool(cand)
            cand = [(pos, _ext(v)) for pos, v in cand if not (isinstance(v, float) and math.isnan(v))]
            if not cand:
                if had_real:                               # real values existed but were all NaN and got filtered
                    raise SpecNotEstablished(f"window {(c,) + o}: all real values NaN under R_ignore; no numeric value")
                vals[(c,) + o] = float("-inf"); sets[(c,) + o] = set()   # geometric empty window: -inf padding only
                continue
            best = max(v for _, v in cand)
            vals[(c,) + o] = best
            sets[(c,) + o] = {pos for pos, v in cand if v == best}
    return (_build([C] + outs, lambda idx: vals[idx]), _build([C] + outs, lambda idx: sets[idx]))


# ---------------------------------------------------------------- reference-free properties (apply to ANY candidate)

def prop_avg_linearity(pool, x, y, a, b, **kw):
    """avg_pool(a x + b y) == a avg_pool(x) + b avg_pool(y) exactly in real arithmetic."""
    shape = _shape(x)
    z = _build(shape, lambda idx: Fraction(a) * Fraction(_get(x, idx)) + Fraction(b) * Fraction(_get(y, idx)))
    return pool(z, **kw), pool(x, **kw), pool(y, **kw)


def prop_adjoint(fwd_out, grad_out, x, grad_x):
    """Precondition: shapes match exactly (checked). <avg_pool(x), g> == <x, avg_pool_backward(g)>, exact for any
    linear forward/backward pair."""
    if _shape(fwd_out) != _shape(grad_out) or _shape(x) != _shape(grad_x):
        raise SpecInputError("adjoint check: shape mismatch between forward output and upstream gradient, or x and grad_x")
    def flat(t):
        return [t] if not isinstance(t, list) else [v for e in t for v in flat(e)]
    lhs = sum(Fraction(a) * Fraction(b) for a, b in zip(flat(fwd_out), flat(grad_out)))
    rhs = sum(Fraction(a) * Fraction(b) for a, b in zip(flat(x), flat(grad_x)))
    return lhs, rhs


def prop_max_shift(pool, x, c, **kw):
    """Precondition: finite inputs. max_pool(x + c) == max_pool(x) + c exactly in real arithmetic, for any tie-breaking.
    (-inf outputs of empty windows stay -inf on both sides.)"""
    shape = _shape(x)
    xc = _build(shape, lambda idx: Fraction(_get(x, idx)) + Fraction(c))
    return pool(xc, **kw)[0], pool(x, **kw)[0]


def prop_max_grad_mass(grad_out, grad_x, argmax_sets):
    """Precondition: shapes match; only windows with at least one real input can route a gradient. Each non-empty
    window routes its upstream gradient to exactly one input, so sum(grad_x) == sum of grad_out over NON-EMPTY windows.
    Returns (routed_mass, input_mass, n_empty_windows); empty windows are reported separately, not as violations."""
    if _shape(grad_out) != _shape(argmax_sets):
        raise SpecInputError("grad mass check: grad_out and argmax_sets shapes differ")
    def flat(t):
        return [t] if not isinstance(t, list) else [v for e in t for v in flat(e)]
    def flat_sets(t):
        return [t] if isinstance(t, set) else [v for e in t for v in flat_sets(e)]
    gos, sets = flat(grad_out), flat_sets(argmax_sets)
    routed = sum((Fraction(g) for g, st in zip(gos, sets) if st), Fraction(0))
    n_empty = sum(1 for st in sets if not st)
    return routed, sum(Fraction(v) for v in flat(grad_x)), n_empty


def check_max_indices(indices, argmax_sets):
    """POOL-A2 forward check: a candidate's returned index for every non-empty window must belong to the argmax set.
    indices: nested list of input positions (tuples) with the output shape. Returns the list of offending outputs."""
    if _shape(indices) != _shape(argmax_sets):
        raise SpecInputError("indices and argmax_sets shapes differ")
    def flat(t):
        return [t] if (isinstance(t, tuple) or t is None) else [v for e in t for v in flat(e)]
    def flat_sets(t):
        return [t] if isinstance(t, set) else [v for e in t for v in flat_sets(e)]
    bad = []
    for n, (idx, st) in enumerate(zip(flat(indices), flat_sets(argmax_sets))):
        if st and idx not in st:
            bad.append((n, idx, sorted(st)))
    return bad


def check_max_subgradient(grad_out, grad_x, argmax_sets, kernel, stride=None, padding=0, dilation=1, ceil_mode=False):
    """POOL-A2 / IDX-A4 style backward check for NON-OVERLAPPING windows (stride >= dilated kernel extent per dim):
    for every non-empty window with upstream v != 0, the input gradient restricted to the window must equal v*alpha with
    alpha = 0 outside the argmax set, alpha >= 0 on it and sum(alpha) = 1; grad_x must be 0 everywhere else.
    Single-member routing and even splitting both pass; (2, -1) does not. Overlapping windows cannot be checked this way
    and raise SpecInputError (the protocol then uses returned indices, if the task declares them)."""
    shape = _shape(grad_x)
    Ls = shape[1:]
    n = len(Ls)
    k = _norm(kernel, n); s = _norm(stride if stride is not None else kernel, n)
    p = _norm(padding, n); d = _norm(dilation, n)
    if any(s[i] < d[i] * (k[i] - 1) + 1 for i in range(n)):
        raise SpecInputError("subgradient check needs non-overlapping windows (stride >= dilated kernel extent)")
    if _shape(grad_out) != _shape(argmax_sets):
        raise SpecInputError("grad_out and argmax_sets shapes differ")
    C = shape[0]
    outs = _shape(grad_out)[1:]
    covered = set()
    bad = []
    for c in range(C):
        for o in product(*[range(v) for v in outs]):
            st = _get(argmax_sets, (c,) + o)
            v = Fraction(_get(grad_out, (c,) + o))
            window = [tuple(s[i] * o[i] - p[i] + d[i] * m[i] for i in range(n)) for m in product(*[range(k[i]) for i in range(n)])]
            window = [pos for pos in window if all(0 <= pos[i] < Ls[i] for i in range(n))]
            covered.update((c,) + pos for pos in window)
            g = {pos: Fraction(_get(grad_x, (c,) + pos)) for pos in window}
            if not st:
                if any(val != 0 for val in g.values()):
                    bad.append(((c,) + o, "gradient routed into an empty window"))
                continue
            if v == 0:
                if any(val != 0 for val in g.values()):
                    bad.append(((c,) + o, "nonzero gradient with zero upstream"))
                continue
            alpha = {pos: val / v for pos, val in g.items()}
            if any(alpha[pos] != 0 for pos in alpha if pos not in st):
                bad.append(((c,) + o, "support outside the argmax set")); continue
            if any(alpha[pos] < 0 for pos in st):
                bad.append(((c,) + o, "negative coefficient")); continue
            if sum((alpha[pos] for pos in st), Fraction(0)) != 1:
                bad.append(((c,) + o, "coefficients do not sum to 1"))
    for c in range(C):
        for pos in product(*[range(L) for L in Ls]):
            if (c,) + pos not in covered and Fraction(_get(grad_x, (c,) + pos)) != 0:
                bad.append(((c,) + pos, "gradient on an input no window samples"))
    return bad
