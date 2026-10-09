"""Independent specification f for torch.nn.functional.cross_entropy / nn.CrossEntropyLoss.   (v0.4)

Written ONLY from the official documentation (no PyTorch source code consulted):
  https://docs.pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html  (main, 2.16.0a0, fetched 2026-10-07)

Documented semantics used (quoted or paraphrased from the page):
  D1  class-index target, unreduced:  l_n = - w_{y_n} * log( exp(x_{n,y_n}) / sum_c exp(x_{n,c}) ) * 1{y_n != ignore_index}
  D2  class-index target, 'mean':     sum_n l_n / sum_n ( w_{y_n} * 1{y_n != ignore_index} );   'sum': sum_n l_n
  D3  probability target, unreduced:  l_n = - sum_c w_c * log( exp(x_{n,c}) / sum_i exp(x_{n,i}) ) * y_{n,c}
  D4  probability target, 'mean':     sum_n l_n / N;  'sum': sum_n l_n
  D5  ignore_index: "Specifies a target value that is ignored and does not contribute to the input gradient";
      "this index may not necessarily be in the class range"; "only applicable when the target contains class indices".
  D6  label_smoothing in [0,1]: "The targets become a mixture of the original ground truth and a uniform distribution",
      i.e. q = (1 - eps) * target + eps / C.

Ambiguities (see ambiguities.md; every reading is evaluated, and a candidate must follow ONE reading across all inputs):
  CE-A1  class-index target + label_smoothing + class weights: R_A (per-class weight w_c), R_B (target-class weight w_{y_n}).
  CE-A2  'mean' denominator in that case: R_A/R_B use D2's denominator; R_C uses sum_n sum_c w_c q_{n,c} 1{not ignored}.
  CE-A3  'mean' with D2's denominator equal to zero (all targets ignored, or every present target class has weight 0):
         the finite real-valued objective is UNDEFINED. Note this is not always 0/0 - with label smoothing the numerator
         can be positive (weights (0,1), target 0, eps=1/2 gives ln2/4 over 0). The spec returns loss = NaN and
         grad_defined = False; a candidate's output here is a convention to be recorded, not judged. reduction='none'
         and 'sum' remain defined (zero) in the all-ignored case.
  Adopted decisions (2026-10-07 review): R_A is the main interpretation for CE-A1, with D2's denominator (reading
  'R_A' as implemented) for CE-A2; R_B and R_C are kept as labelled alternative normalisations for diagnosis only, and
  a difference between them is a "difference under declared interpretation", never a confirmed bug by itself.

Numerics (v0.3): RIGOROUS interval arithmetic in decimal with directed rounding, on a FIXED context (PREC digits,
Emin/Emax = -/+999999) in which Overflow, Underflow, Subnormal and Clamped are TRAPPED. decimal's exp() and ln() are
correctly rounded (ROUND_HALF_EVEN), so [r - ulp(r), r + ulp(r)] encloses the true value as long as r is a normal
number of the context; +,-,*,/ use ROUND_FLOOR / ROUND_CEILING on the endpoints. The one place where leaving the
normal range is expected - exp() of a very negative argument - is enclosed by [0, smallest normal], which is a valid
enclosure (the v0.2 audit's counterexample, a gradient of ~1.9e-1000058, now lies inside [0, 1e-999999]). Anywhere else
the trap fires and the spec returns established=False instead of a value: the supported domain is "every other
intermediate quantity is a normal number of the context"; outside it the spec declines rather than returning a
non-enclosing interval. log-softmax is evaluated in the
stable form log p_k = (x_k - m) - ln(sum_j exp(x_j - m)) and p_k = exp(x_k - m) / sum (no exp of a wide interval), so
e.g. logits [1e100, 1e100] give exactly ln 2. Inputs (logits, weights, targets, eps) are exact rationals; floats are
converted exactly. Interface rounding of scalars (0.1 vs float32(0.1)) is the caller's choice: evaluate twice.
Input validation (v0.3): class-index targets must lie in [0, C) or equal ignore_index; probability targets must have
the logits' shape; weights must have length C; label_smoothing in [0, 1]; otherwise SpecInputError is raised - a spec
must refuse illegal inputs, not return a plausible number.
"""
from decimal import (Decimal, Context, localcontext, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN,
                     Overflow, Underflow, Subnormal, Clamped, InvalidOperation, DivisionByZero)
from fractions import Fraction
import math

PREC = 60
_TRAPS = [Overflow, Underflow, Subnormal, Clamped, InvalidOperation, DivisionByZero]


def _ctx(rounding=ROUND_HALF_EVEN, prec=PREC):
    c = Context(prec=prec, rounding=rounding, Emin=-999999, Emax=999999)
    for t in _TRAPS:
        c.traps[t] = True
    return c


_SMALLEST_NORMAL = Decimal("1E-999999")


class SpecNotEstablished(Exception):
    """the spec declines: an intermediate quantity left the normal range of the fixed context."""


class SpecInputError(ValueError):
    """illegal input for the task (e.g. target out of range): refused, not answered."""


class Interval:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = lo
        self.hi = lo if hi is None else hi

    @staticmethod
    def exact(v):
        """exact rational -> enclosing interval (floats/ints are exact in Decimal; other rationals are rounded outward)."""
        if isinstance(v, Interval):
            return v
        if isinstance(v, Decimal):
            return Interval(v, v)
        if isinstance(v, (int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError("non-finite logits/weights are outside the scope of this spec")
            d = Decimal(v)
            return Interval(d, d)
        f = Fraction(v)
        n, d = Decimal(f.numerator), Decimal(f.denominator)
        with localcontext(_ctx(ROUND_FLOOR)):
            lo = n / d
        with localcontext(_ctx(ROUND_CEILING)):
            hi = n / d
        return Interval(lo, hi)

    def _bin(self, other, op, signed_tiny=False):
        """directed-rounding evaluation over the four endpoint pairs. For * and / (signed_tiny=True) a result that
        underflows below the normal range is enclosed by [0, smallest normal] with the sign of the exact result;
        for + and - underflow is not expected and lets the trap propagate (the spec then declines)."""
        other = Interval.exact(other)
        pairs = [(a, b) for a in (self.lo, self.hi) for b in (other.lo, other.hi)]

        def ev(a, b, side):
            try:
                with localcontext(_ctx(ROUND_FLOOR if side == "lo" else ROUND_CEILING)):
                    return op(a, b)
            except (Underflow, Subnormal):
                if not signed_tiny:
                    raise
                sign = (1 if a >= 0 else -1) * (1 if b >= 0 else -1)
                if sign >= 0:
                    return Decimal(0) if side == "lo" else _SMALLEST_NORMAL
                return -_SMALLEST_NORMAL if side == "lo" else Decimal(0)

        return Interval(min(ev(a, b, "lo") for a, b in pairs), max(ev(a, b, "hi") for a, b in pairs))

    def __add__(self, o): return self._bin(o, lambda a, b: a + b)
    def __sub__(self, o): return self._bin(o, lambda a, b: a - b)
    def __mul__(self, o): return self._bin(o, lambda a, b: a * b, signed_tiny=True)

    def __truediv__(self, o):
        o = Interval.exact(o)
        if o.lo <= 0 <= o.hi:
            raise ZeroDivisionError("interval division by an interval containing zero")
        return self._bin(o, lambda a, b: a / b, signed_tiny=True)

    def __neg__(self): return Interval(self.hi.copy_negate(), self.lo.copy_negate())   # exact (unary minus rounds)

    def exp(self):
        """exp of an interval. An endpoint whose exp() underflows below the normal range is enclosed by
        [0, smallest normal] - a valid (if wide) enclosure - so very negative logits do not make the spec decline."""
        def one(d, side):
            try:
                with localcontext(_ctx()):
                    r = d.exp()
            except (Underflow, Subnormal):
                return Decimal(0) if side == "lo" else _SMALLEST_NORMAL
            return _down(r) if side == "lo" else _up(r)
        return Interval(one(self.lo, "lo"), one(self.hi, "hi"))

    def ln(self):
        if self.lo <= 0:
            raise SpecNotEstablished("ln of an interval touching zero")
        with localcontext(_ctx()):
            lo, hi = self.lo.ln(), self.hi.ln()
        return Interval(_down(lo), _up(hi))

    def contains(self, v):
        v = Decimal(v) if not isinstance(v, Decimal) else v
        return self.lo <= v <= self.hi

    def width(self):
        with localcontext(_ctx(ROUND_CEILING, PREC + 5)):
            return self.hi - self.lo

    def mid(self):
        with localcontext(_ctx(ROUND_HALF_EVEN, PREC + 5)):
            return (self.lo + self.hi) / 2
    def __repr__(self): return f"[{self.lo}, {self.hi}]"


def _ulp(d):
    """one unit in the last place of a correctly rounded PREC-digit result d. Valid only for NORMAL d: if d is zero,
    subnormal or so small that the ulp itself is below the context's tiny exponent, the traps fire and the spec declines."""
    with localcontext(_ctx()):
        if d == 0:
            raise SpecNotEstablished("correctly rounded result is zero: true value below the normal range")
        return Decimal(10) ** (d.adjusted() - PREC + 1)


def _down(d):
    with localcontext(_ctx(ROUND_FLOOR, PREC + 5)):
        return d - _ulp(d)


def _up(d):
    with localcontext(_ctx(ROUND_CEILING, PREC + 5)):
        return d + _ulp(d)


def _isum(items, zero=None):
    acc = Interval(Decimal(0), Decimal(0)) if zero is None else zero
    for it in items:
        acc = acc + it
    return acc


def _softmax_row(row):
    """stable form: t_k = x_k - m (m = max x), s = sum exp(t_k); log p_k = t_k - ln s; p_k = exp(t_k) / s."""
    xs = [Interval.exact(v) for v in row]
    m = max(x.hi for x in xs)
    mI = Interval(m, m)
    ts = [x - mI for x in xs]
    es = [t.exp() for t in ts]
    s = _isum(es)
    lns = s.ln()
    return [t - lns for t in ts], [e / s for e in es]


def _coeffs(target_row, n_classes, weights, eps, reading, prob_target):
    """coefficients a_{n,c} (exact rationals) with l_n = - sum_c a_{n,c} log p_{n,c}, and the mean-denominator term."""
    C = n_classes
    w = [Fraction(1)] * C if weights is None else [Fraction(v) for v in weights]
    eps = Fraction(eps)
    if prob_target:
        q = [(1 - eps) * Fraction(t) + eps / C for t in target_row]
        return [w[c] * q[c] for c in range(C)], Fraction(1)          # D4: mean divides by N
    y = target_row
    q = [(1 - eps) * (1 if c == y else 0) + eps / C for c in range(C)]
    if reading == "R_B":
        return [w[y] * q[c] for c in range(C)], w[y]
    a = [w[c] * q[c] for c in range(C)]
    return a, (sum(a) if reading == "R_C" else w[y])


def cross_entropy(logits, target, weights=None, ignore_index=-100, reduction="mean", label_smoothing=0,
                  reading="R_A", prob_target=False):
    """Returns dict(loss, grad, grad_defined, nan).

    loss : Interval (or list of Interval for reduction='none'); grad : [N][C] Intervals of d loss / d logits
           (for 'none', the gradient of sum_n l_n). All intervals are rigorous enclosures of the exact real values.
    grad_defined : False only in the all-ignored 'mean' case (CE-A3), where loss is NaN and grad is None.
    established  : False when an intermediate quantity left the normal range of the fixed context (reason given);
                   then loss and grad are None and the condition is counted as "spec not established", never as a
                   candidate failure. Illegal inputs raise SpecInputError.
    """
    N, C = len(logits), len(logits[0])
    _validate(logits, target, weights, ignore_index, reduction, label_smoothing, reading, prob_target, N, C)
    try:
        return _cross_entropy(logits, target, weights, ignore_index, reduction, label_smoothing, reading, prob_target, N, C)
    except (SpecNotEstablished, Overflow, Underflow, Subnormal, Clamped, InvalidOperation, DivisionByZero) as e:
        return dict(loss=None, grad=None, grad_defined=False, nan=False, established=False, reason=repr(e))


def _validate(logits, target, weights, ignore_index, reduction, label_smoothing, reading, prob_target, N, C):
    if any(len(row) != C for row in logits) or C == 0:
        raise SpecInputError("logits must be a non-empty [N][C] matrix")
    if reduction not in ("none", "mean", "sum"):
        raise SpecInputError(f"unknown reduction {reduction!r}")
    if reading not in ("R_A", "R_B", "R_C"):
        raise SpecInputError(f"unknown reading {reading!r}")
    eps = Fraction(label_smoothing)
    if not (0 <= eps <= 1):
        raise SpecInputError("label_smoothing must lie in [0, 1]")
    if weights is not None and len(weights) != C:
        raise SpecInputError("weights must have length C")
    if len(target) != N:
        raise SpecInputError("target must have N entries")
    if prob_target:
        if any(not isinstance(t, (list, tuple)) or len(t) != C for t in target):
            raise SpecInputError("probability targets must have the logits' shape")
    else:
        for t in target:
            if isinstance(t, (list, tuple)) or int(t) != t:
                raise SpecInputError("class-index targets must be integers")
            if t != ignore_index and not (0 <= t < C):
                raise SpecInputError(f"target {t} outside [0, {C}) and not equal to ignore_index")


def _cross_entropy(logits, target, weights, ignore_index, reduction, label_smoothing, reading, prob_target, N, C):
    per_row, grads, dens = [], [], []
    zero = Interval(Decimal(0), Decimal(0))
    for n in range(N):
        tgt = target[n]
        if not prob_target and tgt == ignore_index:
            per_row.append(zero); grads.append([zero] * C); dens.append(Fraction(0))
            continue
        logp, p = _softmax_row(logits[n])
        a, den = _coeffs(tgt, C, weights, label_smoothing, reading, prob_target)
        aI = [Interval.exact(v) for v in a]
        ln = -_isum(aI[k] * logp[k] for k in range(C))
        A = Interval.exact(sum(a))
        g = [p[k] * A - aI[k] for k in range(C)]          # d l_n / d x_{n,k} = p_k * sum_c a_c - a_k
        per_row.append(ln); grads.append(g); dens.append(den)
    if reduction == "none":
        return dict(loss=per_row, grad=grads, grad_defined=True, nan=False, established=True)
    total = _isum(per_row)
    if reduction == "sum":
        return dict(loss=total, grad=grads, grad_defined=True, nan=False, established=True)
    D = sum(dens, Fraction(0))
    if D == 0:                                          # CE-A3: 0/0, undefined
        return dict(loss=Decimal("NaN"), grad=None, grad_defined=False, nan=True, established=True)
    DI = Interval.exact(D)
    return dict(loss=total / DI, grad=[[v / DI for v in row] for row in grads], grad_defined=True, nan=False,
                established=True)


# ---------------------------------------------------------------- reference-free properties
# Each property states its precondition. They are identities of the REAL-valued task; a floating-point candidate
# satisfies them up to its rounding, so the protocol compares residuals against a declared bound, except where the
# inputs are chosen so that every intermediate value is exactly representable (then equality is required).

def prop_shift_invariance(f, logits, target, c, **kw):
    """Precondition: none. f(x + c*1_row) == f(x) for loss and gradient (log-softmax is shift invariant)."""
    shifted = [[Fraction(v) + Fraction(c) for v in row] for row in logits]
    return f(logits, target, **kw), f(shifted, target, **kw)


def prop_grad_row_sums(grad_rows):
    """Precondition: gradient defined. For every row, sum_k d loss / d x_{n,k} == 0 (all readings, any weights/eps)."""
    return [_isum(row) for row in grad_rows]


def prop_ignored_rows_zero_grad(result, target, ignore_index=-100):
    """Precondition: result['grad_defined'] (skipped in the all-ignored case, CE-A3). Returns offending row indices."""
    if not result.get("established", True) or not result["grad_defined"]:
        return []
    return [n for n, t in enumerate(target) if t == ignore_index and any(not v.contains(0) for v in result["grad"][n])]


def prop_class_permutation(f, logits, target, perm, weights=None, ignore_index=-100, prob_target=False, **kw):
    """Precondition: none. Renaming classes consistently leaves the loss unchanged. perm[new] = old.
    Everything indexed by class is transformed together: logits columns, class weights, index targets
    (including ignore_index when it lies inside [0, C)), and the class axis of probability targets."""
    C = len(perm)
    inv = {old: new for new, old in enumerate(perm)}
    lp = [[row[perm[k]] for k in range(C)] for row in logits]
    wp = None if weights is None else [weights[perm[k]] for k in range(C)]
    if prob_target:
        tp = [[row[perm[k]] for k in range(C)] for row in target]
        ii = ignore_index
    else:
        tp = [inv[t] if t in inv else t for t in target]
        ii = inv[ignore_index] if ignore_index in inv else ignore_index
    base = f(logits, target, weights=weights, ignore_index=ignore_index, prob_target=prob_target, **kw)
    permuted = f(lp, tp, weights=wp, ignore_index=ii, prob_target=prob_target, **kw)
    return base, permuted
