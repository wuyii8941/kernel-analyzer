"""Independent review tests for path A package D (ARITH / MATHFN / INDEX `real` domain extensions, 2026-10-20 drafts).

Reviewer-owned. The draft contract/adapter pairs are staged under .cache/spec_a2_review/D/review_stage and loaded with
kernel_spec.registry.load(cid, contracts=..., adapters=...). Expected values are computed here, without the adapters'
helpers, kernel_spec.aten or kernel_spec.real:
- exact operations with Fraction / int and a small extended-real model written below;
- transcendental values with python-flint Arb balls at 1200 bits (rigorous: the adapter interval must contain the
  whole Arb ball, and the Arb ball contains the true value), with an mpmath 3000-bit point value as a crosscheck;
- lookups by NumPy coordinate construction.
"""
from __future__ import annotations

import math
import shutil
import unittest
from fractions import Fraction
from pathlib import Path

import flint
import mpmath
import numpy as np

from kernel_spec import registry
from kernel_spec.num import evaluate

SPEC = Path(__file__).resolve().parents[1]
REPO = SPEC.parent
DRAFT = SPEC / "contracts" / "draft"
STAMP = "path_a_A2_20261020"
STAGE = REPO / ".cache" / "spec_a2_review" / "D" / "review_stage"
INF = float("inf")
CIDS = ("ARITH", "MATHFN", "INDEX")
INT_RANGE = {"i64": (-2 ** 63, 2 ** 63 - 1), "i32": (-2 ** 31, 2 ** 31 - 1), "i16": (-2 ** 15, 2 ** 15 - 1),
             "i8": (-2 ** 7, 2 ** 7 - 1), "u8": (0, 255)}

DRAFTS = {}
VALIDATED = {}


def setUpModule():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "contracts").mkdir(parents=True)
    (STAGE / "adapters").mkdir()
    for cid in CIDS:
        shutil.copy(DRAFT / f"{cid}_{STAMP}.yaml", STAGE / "contracts" / f"{cid}.yaml")
        shutil.copy(DRAFT / f"{cid}_adapter_{STAMP}.py", STAGE / "adapters" / f"{cid}.py")
    for cid in CIDS:
        DRAFTS[cid] = registry.load(cid, contracts=STAGE / "contracts", adapters=STAGE / "adapters").adapter
        VALIDATED[cid] = registry.load(cid).adapter      # top-level validated pair


# ----------------------------------------------------------------------------------------------- helpers
def T(dtype, values, shape=None):
    rec = {"t": dtype, "v": values}
    if shape is not None:
        rec["shape"] = shape
    return rec


def run(cid, op, args, kwargs=None, profile="real", adapter=None):
    inp = {"op": op, "args": args}
    if kwargs:
        inp["kwargs"] = kwargs
    ad = adapter or DRAFTS[cid]
    return evaluate(ad.reference, profile, inp)


def norm(x):
    """Adapter exact output leaf -> Fraction or +-inf float."""
    if isinstance(x, list):
        return [norm(y) for y in x]
    if isinstance(x, str):
        if x in ("+inf", "inf"):
            return INF
        if x == "-inf":
            return -INF
        return Fraction(x)
    if isinstance(x, (bool, np.bool_)):
        return x
    if isinstance(x, (int, np.integer, Fraction)):
        return Fraction(int(x)) if isinstance(x, (int, np.integer)) else x
    if isinstance(x, float):
        return x if math.isinf(x) else Fraction(x)
    v = getattr(x, "v", None)       # an exact R-like object
    if isinstance(v, Fraction):
        return v
    if v in ("+inf", "-inf"):
        return INF if v == "+inf" else -INF
    raise AssertionError(f"not an exact leaf: {x!r}")


def flat(x):
    if isinstance(x, list):
        for y in x:
            yield from flat(y)
    else:
        yield x


def mpf_fraction(m):
    sign, man, exp, _ = m if isinstance(m, tuple) else m._mpf_
    if man == 0 and exp == 0:
        return Fraction(0)
    q = Fraction(int(man)) * (Fraction(2) ** int(exp))
    return -q if sign else q


def iv_ends(x):
    """Exact endpoints of an adapter interval leaf (mpmath ivmpf), or of an exact leaf."""
    if hasattr(x, "_mpi_"):
        return mpf_fraction(x._mpi_[0]), mpf_fraction(x._mpi_[1])
    q = norm(x)
    return q, q


def to_list(a):
    return a.tolist()


def bf16_round(values):
    import torch
    return [float(v) for v in torch.tensor(values, dtype=torch.float64).to(torch.bfloat16).double().reshape(-1)]


def f32_round(values):
    return [float(v) for v in np.asarray(values, dtype=np.float64).astype(np.float32).reshape(-1)]


# -------------------------------------------------- independent extended-real model (reviewer's own arithmetic)
class U(Exception):
    """undefined point"""


def sgn(x):
    return (x > 0) - (x < 0)


def xmul(x, y):
    xi, yi = isinstance(x, float) and math.isinf(x), isinstance(y, float) and math.isinf(y)
    if xi or yi:
        if (not xi and x == 0) or (not yi and y == 0):
            raise U("0*inf")
        return INF if sgn(x) * sgn(y) > 0 else -INF
    return Fraction(x) * Fraction(y)


def xadd(x, y):
    xi, yi = isinstance(x, float) and math.isinf(x), isinstance(y, float) and math.isinf(y)
    if xi and yi:
        if x != y:
            raise U("inf-inf")
        return x
    if xi:
        return x
    if yi:
        return y
    return Fraction(x) + Fraction(y)


def xdiv(x, y):
    xi, yi = isinstance(x, float) and math.isinf(x), isinstance(y, float) and math.isinf(y)
    if not yi and y == 0:
        raise U("x/0")
    if xi and yi:
        raise U("inf/inf")
    if xi:
        return INF if sgn(x) * sgn(y) > 0 else -INF
    if yi:
        return Fraction(0)
    return Fraction(x) / Fraction(y)


def xfloor(q):
    return q if isinstance(q, float) else Fraction(math.floor(q))


def xtrunc(q):
    return q if isinstance(q, float) else Fraction(int(q))


def outcome_of(fn):
    try:
        return ("value", fn())
    except U:
        return ("domain_undefined", None)


def fx(v):
    """reviewer value -> record leaf"""
    if isinstance(v, float) and math.isinf(v):
        return v
    v = Fraction(v)
    return int(v) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


# --------------------------------------------------------------------- Arb / mpmath transcendental references
def arb_ball(op, q, prec=1200):
    saved = flint.ctx.prec
    flint.ctx.prec = prec
    try:
        a = flint.arb(flint.fmpq(q.numerator, q.denominator))
        ball = {"aten.sin": a.sin, "aten.cos": a.cos, "aten.tanh": a.tanh, "aten.exp": a.exp,
                "aten.expm1": a.expm1, "aten.log": a.log, "aten.log1p": a.log1p,
                "aten.sigmoid": lambda: 1 / (1 + (-a).exp())}[op]()
        lo, hi = ball.lower(), ball.upper()
        lo_q = Fraction(int(lo.man_exp()[0])) * Fraction(2) ** int(lo.man_exp()[1])
        hi_q = Fraction(int(hi.man_exp()[0])) * Fraction(2) ** int(hi.man_exp()[1])
    finally:
        flint.ctx.prec = saved
    return lo_q, hi_q


def mp_point(op, q, prec=3000):
    with mpmath.workprec(prec):
        x = mpmath.mpf(q.numerator) / q.denominator
        v = {"aten.sin": mpmath.sin, "aten.cos": mpmath.cos, "aten.tanh": mpmath.tanh, "aten.exp": mpmath.exp,
             "aten.expm1": mpmath.expm1, "aten.log": mpmath.log, "aten.log1p": mpmath.log1p,
             "aten.sigmoid": lambda t: 1 / (1 + mpmath.exp(-t))}[op](x)
        return mpf_fraction(v)


class Base(unittest.TestCase):
    def assertValue(self, res, expected):
        self.assertEqual(res.outcome, "value", res.note)
        self.assertEqual(norm(res.fields["out"]), expected)

    def assertOutcome(self, res, outcome):
        self.assertEqual(res.outcome, outcome, res.note)


# ===================================================================================================== ARITH
class ArithRecordedStructures(Base):
    def test_recorded_python_float_scalars_are_exact_binary64(self):
        rng = np.random.default_rng(1020)
        for dtype, shape, op, scalar in (("bf16", (1, 4, 16), "aten.mul", 0.044715),
                                         ("f32", (1, 4, 16), "aten.mul", 0.044715),
                                         ("bf16", (1, 4, 16), "aten.mul", 0.7978845608028654),
                                         ("f32", (1, 4, 16), "aten.mul", 0.7978845608028654),
                                         ("f32", (1, 2, 4, 8), "aten.mul", 0.3535533905932738),
                                         ("f32", (1, 4, 1), "aten.add", 1e-06)):
            raw = rng.standard_normal(int(np.prod(shape))) * 4
            vals = bf16_round(raw) if dtype == "bf16" else f32_round(raw)
            q = Fraction(scalar)       # exact binary64 value received
            arr = np.array(vals, dtype=object).reshape(shape)
            exp = np.vectorize(lambda v: Fraction(v) * q if op == "aten.mul" else Fraction(v) + q,
                               otypes=[object])(arr)
            with self.subTest(dtype=dtype, op=op, scalar=scalar):
                self.assertValue(run("ARITH", op, [T(dtype, to_list(arr)), scalar]), to_list(exp))
                # the rounded-scalar readings are not the task: for at least one coordinate they differ
                for target in ("f32", "bf16"):
                    rq = Fraction(f32_round([scalar])[0] if target == "f32" else bf16_round([scalar])[0])
                    self.assertNotEqual(rq, q)

    def test_scalar_readings_round_the_scalar_only(self):
        ad = DRAFTS["ARITH"]
        x = T("bf16", [2, -3])
        for reading, target in (("scalar_rounded_to_dtype", "bf16"), ("scalar_rounded_to_opmath", "f32")):
            rq = Fraction(bf16_round([0.044715])[0] if target == "bf16" else f32_round([0.044715])[0])
            res = evaluate(ad.READINGS[reading], "real", {"op": "aten.mul", "args": [x, 0.044715]})
            with self.subTest(reading=reading):
                self.assertEqual(norm(res.fields["out"]), [2 * rq, -3 * rq])
        # f64 tensors: opmath is f64, so the opmath reading equals the task; dtype reading equals it as well
        res = evaluate(ad.READINGS["scalar_rounded_to_opmath"], "real",
                       {"op": "aten.mul", "args": [T("f64", [3]), 0.1]})
        self.assertEqual(norm(res.fields["out"]), [3 * Fraction(0.1)])
        # pow exponent is never rounded by a reading
        res = evaluate(ad.READINGS["scalar_rounded_to_dtype"], "real",
                       {"op": "aten.pow", "args": [T("bf16", [-2]), 3.0]})
        self.assertEqual(norm(res.fields["out"]), [-8])

    def test_recorded_integral_pow_exponents(self):
        rng = np.random.default_rng(3)
        for dtype, shape, n in (("f32", (1, 4, 8), 3.0), ("bf16", (1, 4, 8), 3.0), ("f32", (1, 4, 8), 2.0),
                                ("bf16", (1, 4, 8), 2.0), ("f32", (1, 4, 8), 2), ("f32", (1, 4, 1), 3),
                                ("f32", (1, 4, 8), 1.0)):
            raw = rng.standard_normal(int(np.prod(shape))) * 3
            vals = bf16_round(raw) if dtype == "bf16" else f32_round(raw)
            vals[0] = -abs(vals[0]) or -1.0     # a negative base is present
            arr = np.array(vals, dtype=object).reshape(shape)
            k = int(n)
            exp = np.vectorize(lambda v: Fraction(v) ** k, otypes=[object])(arr)
            with self.subTest(dtype=dtype, n=n):
                self.assertValue(run("ARITH", "aten.pow", [T(dtype, to_list(arr)), n]), to_list(exp))

    def test_recorded_additive_minus_inf_mask_and_where_self(self):
        cond = np.tril(np.ones((4, 4), dtype=bool))
        res = run("ARITH", "aten.where", [T("b8", (~cond).tolist()), T("f32", -INF, []), T("f32", 0, [])])
        mask = [[-INF if not cond[i, j] else Fraction(0) for j in range(4)] for i in range(4)]
        self.assertValue(res, mask)
        rng = np.random.default_rng(5)
        scores = np.array(f32_round(rng.standard_normal(32) * 2), dtype=object).reshape(1, 2, 4, 4)
        exp = [[[[xadd(Fraction(scores[0, h, i, j]), mask[i][j]) for j in range(4)] for i in range(4)]
                for h in range(2)]]
        rec_mask = [[fx(v) for v in row] for row in mask]
        self.assertValue(run("ARITH", "aten.add", [T("f32", to_list(scores)), T("f32", rec_mask)]), exp)

    def test_recorded_int64_position_add_zero(self):
        ids = list(range(0, 128, 9))
        self.assertValue(run("ARITH", "aten.add", [T("i64", ids), 0]), [Fraction(i) for i in ids])


class ArithExtendedReal(Base):
    VALUES = (-INF, Fraction(-2), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(3), INF)

    def test_binary_special_value_grid(self):
        for x in self.VALUES:
            for y in self.VALUES:
                for op, fn, kw in (("aten.add", lambda: xadd(x, y), None),
                                   ("aten.sub", lambda: xadd(x, xmul(-1, y)), None),
                                   ("aten.mul", lambda: xmul(x, y), None),
                                   ("aten.div", lambda: xdiv(x, y), None),
                                   ("aten.div", lambda: xfloor(xdiv(x, y)), {"rounding_mode": "floor"}),
                                   ("aten.div", lambda: xtrunc(xdiv(x, y)), {"rounding_mode": "trunc"}),
                                   ("aten.floor_divide", lambda: xfloor(xdiv(x, y)), None),
                                   ("aten.maximum", lambda: max(x, y), None),
                                   ("aten.minimum", lambda: min(x, y), None)):
                    floor_like = op == "aten.floor_divide" or (kw or {}).get("rounding_mode") == "floor"
                    if floor_like and any(isinstance(v, float) for v in (x, y)) and y != 0:
                        # Review issue A1, resolved by AR.FLOOR_INFINITE: the two retained pages disagree here
                        # (Python // vs floor(x/y)), so the profile leaves these points unsupported.
                        with self.subTest(op=op, kw=kw, x=x, y=y, floor_infinite=True):
                            res = run("ARITH", op, [T("f32", [fx(x)]), T("f32", [fx(y)])], kw)
                            self.assertOutcome(res, "unsupported")
                        continue
                    want = outcome_of(fn)
                    with self.subTest(op=op, kw=kw, x=x, y=y):
                        res = run("ARITH", op, [T("f32", [fx(x)]), T("f32", [fx(y)])], kw)
                        self.assertOutcome(res, want[0])
                        if want[0] == "value":
                            self.assertEqual(norm(res.fields["out"]), [want[1]])

    def test_alpha_with_infinities(self):
        for x in self.VALUES:
            for y in self.VALUES:
                for alpha in (0, -2, 0.5, INF):
                    for op, fn in (("aten.add", lambda: xadd(x, xmul(alpha, y))),
                                   ("aten.sub", lambda: xadd(x, xmul(-1, xmul(alpha, y)))),
                                   ("aten.rsub", lambda: xadd(y, xmul(-1, xmul(alpha, x))))):
                        want = outcome_of(fn)
                        with self.subTest(op=op, x=x, y=y, alpha=alpha):
                            res = run("ARITH", op, [T("f32", [fx(x)]), T("f32", [fx(y)])], {"alpha": alpha})
                            self.assertOutcome(res, want[0])
                            if want[0] == "value":
                                self.assertEqual(norm(res.fields["out"]), [want[1]])

    def test_unary_and_selection_with_infinities(self):
        vals = [fx(v) for v in self.VALUES]
        ext = list(self.VALUES)
        self.assertValue(run("ARITH", "aten.neg", [T("f32", vals)]), [v if v == 0 else -v for v in ext])
        self.assertValue(run("ARITH", "aten.abs", [T("bf16", vals)]), [abs(v) for v in ext])
        self.assertValue(run("ARITH", "aten.sign", [T("f16", vals)]), [Fraction(sgn(v)) for v in ext])
        self.assertValue(run("ARITH", "aten.floor", [T("f64", vals)]), [xfloor(v) for v in ext])
        self.assertValue(run("ARITH", "aten.ceil", [T("f64", vals)]),
                         [v if isinstance(v, float) else Fraction(math.ceil(v)) for v in ext])
        self.assertOutcome(run("ARITH", "aten.reciprocal", [T("f32", vals)]), "domain_undefined")   # contains 0
        nz = [v for v in ext if v != 0]
        self.assertValue(run("ARITH", "aten.reciprocal", [T("f32", [fx(v) for v in nz])]),
                         [xdiv(Fraction(1), v) for v in nz])
        self.assertValue(run("ARITH", "aten.clamp", [T("f32", vals)], {"min": -1, "max": 1}),
                         [min(max(v, Fraction(-1)), Fraction(1)) for v in ext])
        self.assertValue(run("ARITH", "aten.clamp", [T("f32", vals)], {"min": -INF, "max": 0.5}),
                         [min(v, Fraction(1, 2)) for v in ext])
        mask = [i % 2 == 0 for i in range(len(vals))]
        self.assertValue(run("ARITH", "aten.masked_fill", [T("f32", vals), T("b8", mask), -INF]),
                         [-INF if m else v for m, v in zip(mask, ext)])
        self.assertValue(run("ARITH", "aten.where", [T("b8", mask), T("f32", vals), T("f32", vals[::-1])]),
                         [a if m else b for m, a, b in zip(mask, ext, ext[::-1])])

    def test_fma_with_infinities(self):
        for a, b, c in ((INF, Fraction(0), Fraction(1)), (INF, Fraction(1), -INF), (INF, Fraction(-1), -INF),
                        (Fraction(2), Fraction(3), INF), (-INF, -INF, INF)):
            want = outcome_of(lambda: xadd(xmul(a, b), c))
            with self.subTest(a=a, b=b, c=c):
                res = run("ARITH", "prims.fma", [T("f32", [fx(a)]), T("f32", [fx(b)]), T("f32", [fx(c)])])
                self.assertOutcome(res, want[0])
                if want[0] == "value":
                    self.assertEqual(norm(res.fields["out"]), [want[1]])

    def test_one_undefined_coordinate_makes_the_call_undefined_and_nan_is_invalid(self):
        self.assertOutcome(run("ARITH", "aten.add", [T("f32", [1, INF, 2]), T("f32", [1, -INF, 2])]),
                           "domain_undefined")
        for op, args in (("aten.add", [T("f32", [float("nan")]), 1]), ("aten.add", [T("f32", [1]), float("nan")]),
                         ("aten.pow", [T("f32", [2]), float("nan")]), ("aten.where", [T("b8", [True]),
                          T("f32", [float("nan")]), T("f32", [1])])):
            with self.subTest(op=op):
                self.assertOutcome(run("ARITH", op, args), "invalid_input")


class ArithPow(Base):
    def test_integer_exponents_exact_including_negative_bases_and_zero_power(self):
        bases = [Fraction(-3), Fraction(-1, 2), Fraction(0), Fraction(1, 4), Fraction(7)]
        for n in (0, 1, 2, 3, 5, 64, -1, -2, -64, 3.0, 2.0, -3.0, 0.0, "4"):
            k = int(Fraction(n))
            use = [b for b in bases if not (b == 0 and k < 0)]
            with self.subTest(n=n):
                self.assertValue(run("ARITH", "aten.pow", [T("f64", [fx(b) for b in use]), n]),
                                 [Fraction(1) if k == 0 else b ** k for b in use])

    def test_infinite_bases(self):
        for n in (0, 1, 2, 3, -1, -2, -3):
            want = []
            for b in (INF, -INF):
                if n == 0:
                    want.append(Fraction(1))
                elif n > 0:
                    want.append(INF if b > 0 or n % 2 == 0 else -INF)
                else:
                    want.append(Fraction(0))
            with self.subTest(n=n):
                self.assertValue(run("ARITH", "aten.pow", [T("f32", [INF, -INF]), n]), want)

    def test_refusals(self):
        for args, outcome in (([T("f32", [0]), -1], "domain_undefined"), ([T("f32", [0]), -2.0], "domain_undefined"),
                              ([T("f32", [4]), 0.5], "unsupported"), ([T("f32", [4]), 65], "unsupported"),
                              ([T("f32", [4]), -65], "unsupported"), ([T("f32", [4]), 1e20], "unsupported"),
                              ([T("f32", [4]), INF], "unsupported"), ([T("i64", [2]), 2], "unsupported"),
                              ([T("f32", [2]), T("f32", [2])], "unsupported"), ([2, T("f32", [2])], "unsupported"),
                              ([T("f32", [2]), 1 / 3], "unsupported"), ([T("f32", [2]), "1/3"], "invalid_input")):
            with self.subTest(args=args):
                self.assertOutcome(run("ARITH", "aten.pow", args), outcome)


class ArithIntegers(Base):
    def test_overflow_boundaries_for_every_integer_dtype(self):
        for dtype, (lo, hi) in INT_RANGE.items():
            with self.subTest(dtype=dtype):
                self.assertValue(run("ARITH", "aten.add", [T(dtype, [hi, lo]), 0]), [hi, lo])
                self.assertOutcome(run("ARITH", "aten.add", [T(dtype, [hi]), 1]), "domain_undefined")
                self.assertOutcome(run("ARITH", "aten.sub", [T(dtype, [lo]), 1]), "domain_undefined")
                self.assertValue(run("ARITH", "aten.sub", [T(dtype, [hi]), T(dtype, [hi])]), [0])
                self.assertValue(run("ARITH", "aten.rsub", [T(dtype, [lo]), lo]), [0])
                self.assertOutcome(run("ARITH", "aten.mul", [T(dtype, [hi]), 2]), "domain_undefined")
                self.assertValue(run("ARITH", "aten.mul", [T(dtype, [hi // 2]), 2]), [hi // 2 * 2])
                # through alpha: exact result inside the range is a value even if alpha*b alone is not
                self.assertValue(run("ARITH", "aten.add", [T(dtype, [hi]), T(dtype, [hi])], {"alpha": -1}), [0])
                self.assertValue(run("ARITH", "aten.maximum", [T(dtype, [lo, hi]), T(dtype, [hi, lo])]), [hi, hi])
                self.assertValue(run("ARITH", "aten.clamp", [T(dtype, [lo, hi])], {"min": lo + 1, "max": hi - 1}),
                                 [lo + 1, hi - 1])
                self.assertValue(run("ARITH", "aten.sign", [T(dtype, [lo, 0, hi])]), [sgn(lo), 0, 1])
                if lo < 0:
                    self.assertOutcome(run("ARITH", "aten.neg", [T(dtype, [lo])]), "domain_undefined")
                    self.assertOutcome(run("ARITH", "aten.abs", [T(dtype, [lo])]), "domain_undefined")
                    self.assertValue(run("ARITH", "aten.neg", [T(dtype, [lo + 1, hi])]), [-(lo + 1), -hi])
                else:
                    self.assertValue(run("ARITH", "aten.neg", [T(dtype, [0])]), [0])
                    self.assertOutcome(run("ARITH", "aten.neg", [T(dtype, [1])]), "domain_undefined")

    def test_out_of_range_int_scalars_only_matter_when_selected_or_in_the_result(self):
        self.assertOutcome(run("ARITH", "aten.add", [T("i8", [1]), 1000]), "domain_undefined")
        self.assertValue(run("ARITH", "aten.add", [T("i8", [-100]), 200]), [100])
        self.assertValue(run("ARITH", "aten.clamp", [T("i8", [5, -100])], {"min": -1000, "max": 300}), [5, -100])
        self.assertOutcome(run("ARITH", "aten.clamp", [T("i8", [5])], {"min": 200}), "domain_undefined")
        self.assertOutcome(run("ARITH", "aten.masked_fill", [T("i8", [1]), T("b8", [True]), 1000]),
                           "domain_undefined")
        self.assertValue(run("ARITH", "aten.masked_fill", [T("i8", [1]), T("b8", [False]), 1000]), [1])

    def test_integer_refusal_classes(self):
        for op, args, kw, outcome in (
                ("aten.add", [T("i64", [1]), 0.5], None, "unsupported"),
                ("aten.add", [T("i64", [1]), 1], {"alpha": 0.5}, "unsupported"),
                ("aten.add", [T("i64", [1]), T("i32", [1])], None, "unsupported"),
                ("aten.div", [T("i64", [7]), 2], None, "unsupported"),
                ("aten.floor_divide", [T("i64", [7]), 2], None, "unsupported"),
                ("aten.reciprocal", [T("i32", [1])], None, "unsupported"),
                ("aten.floor", [T("i32", [1])], None, "unsupported"),
                ("aten.add", [T("i64", ["3/2"]), 1], None, "invalid_input"),
                ("aten.add", [T("i8", [128]), 1], None, "invalid_input"),
                ("aten.add", [T("u8", [-1]), 1], None, "invalid_input"),
                ("aten.add", [T("b8", [True]), 1], None, "unsupported")):
            with self.subTest(op=op, args=args, kw=kw):
                self.assertOutcome(run("ARITH", op, args, kw), outcome)

    def test_random_int_arith_matches_exact_integers(self):
        rng = np.random.default_rng(11)
        for dtype, (lo, hi) in INT_RANGE.items():
            a = [int(v) for v in rng.integers(lo // 4, hi // 4 + 1, 12)]
            b = [int(v) for v in rng.integers(lo // 4, hi // 4 + 1, 12)]
            with self.subTest(dtype=dtype):
                for op, f in (("aten.add", lambda x, y: x + y), ("aten.sub", lambda x, y: x - y)):
                    want = [f(x, y) for x, y in zip(a, b)]
                    res = run("ARITH", op, [T(dtype, a), T(dtype, b)])
                    if all(lo <= w <= hi for w in want):
                        self.assertValue(res, want)
                    else:
                        self.assertOutcome(res, "domain_undefined")


class ArithScalarsReceived(Base):
    def test_received_scalar_classes(self):
        self.assertValue(run("ARITH", "aten.add", [T("f32", [1]), 0.1]), [1 + Fraction(0.1)])
        self.assertOutcome(run("ARITH", "aten.add", [T("f32", [1]), "1/10"]), "invalid_input")
        self.assertOutcome(run("ARITH", "aten.add", [T("f32", [1]), "1/3"]), "invalid_input")
        self.assertValue(run("ARITH", "aten.add", [T("f64", [0]), 2 ** 53 + 1]), [2 ** 53 + 1])
        self.assertValue(run("ARITH", "aten.add", [T("f32", [1]), 2.0 ** 100]), [1 + 2 ** 100])
        self.assertValue(run("ARITH", "aten.add", [T("f16", [1]), 1e308]), [1 + Fraction(1e308)])

    def test_recheck_floor_infinite_and_scalar_classes(self):
        # AR.FLOOR_INFINITE: zero divisor first (domain_undefined), else any infinite operand -> unsupported
        self.assertOutcome(run("ARITH", "aten.floor_divide", [T("f32", [INF]), 0]), "domain_undefined")
        self.assertOutcome(run("ARITH", "aten.div", [T("f32", [INF, 1]), T("f32", [2, 0])],
                               {"rounding_mode": "floor"}), "domain_undefined")
        self.assertOutcome(run("ARITH", "aten.floor_divide", [T("f32", [1, 2]), T("f32", [2, -INF])]), "unsupported")
        self.assertValue(run("ARITH", "aten.floor_divide", [T("f32", [-7, 7]), 2]), [-4, 3])
        self.assertValue(run("ARITH", "aten.div", [T("f32", [INF, -7]), 2], {"rounding_mode": "trunc"}), [INF, -3])
        # AR.SCALAR_VALUE classes: uint64 range exact, bool and integers outside [-2^63, 2^64) unsupported
        for scalar, outcome, want in ((2 ** 63, "value", [1 + 2 ** 63]), (2 ** 64 - 1, "value", [2 ** 64]),
                                      (-2 ** 63, "value", [1 - 2 ** 63]), (2 ** 64, "unsupported", None),
                                      (-2 ** 63 - 1, "unsupported", None), (True, "unsupported", None),
                                      ("1/3", "invalid_input", None), (str(2 ** 64), "unsupported", None)):
            with self.subTest(scalar=scalar):
                res = run("ARITH", "aten.add", [T("f64", [1]), scalar])
                self.assertOutcome(res, outcome)
                if want:
                    self.assertEqual(norm(res.fields["out"]), want)
        self.assertOutcome(run("ARITH", "aten.add", [T("i64", [1]), 2 ** 63]), "domain_undefined")
        self.assertValue(run("ARITH", "aten.add", [T("i64", [-2 ** 63]), 2 ** 63]), [0])
        self.assertOutcome(run("ARITH", "aten.pow", [T("f32", [2]), True]), "unsupported")
        self.assertOutcome(run("ARITH", "aten.clamp", [T("i8", [1])], {"min": True}), "unsupported")

    def test_issue_A2_legal_scalars_are_not_invalid_input(self):
        # PyTorch 2.10 eager accepts a bool Scalar (tensor([1.]) + True == 2) and a Python int in [2^63, 2^64)
        # (dispatched as aten.add.Tensor with the int, uint64 Scalar). invalid_input claims a violated documented
        # condition; the draft returns invalid_input for both. Flips to a pass once the draft is fixed.
        for scalar in (True, 2 ** 63, 2 ** 64 - 1):
            with self.subTest(scalar=scalar):
                self.assertNotEqual(run("ARITH", "aten.add", [T("f64", [1]), scalar]).outcome, "invalid_input")


class AutogradProfilesUnchanged(Base):
    """The four autograd profiles of the draft agree with the validated top-level adapter on random inputs."""

    def compare(self, profile, inp):
        new = evaluate(DRAFTS["ARITH"].reference, profile, inp)
        old = evaluate(VALIDATED["ARITH"].reference, profile, inp)
        self.assertEqual(new.outcome, old.outcome, (new.note, old.note))
        if new.outcome == "value":
            self.assertEqual(set(new.fields), set(old.fields))
            for k in new.fields:
                self.assertEqual(norm(new.fields[k]), norm(old.fields[k]), k)

    @staticmethod
    def rnd(rng, shape, dtype):
        vals = (rng.integers(-8, 9, int(np.prod(shape))) / 4.0).reshape(shape)
        return T(dtype, vals.tolist(), list(shape))

    def test_random_inputs(self):
        rng = np.random.default_rng(42)
        for trial in range(30):
            dtype = ["f16", "bf16", "f32", "f64"][trial % 4]
            with self.subTest(trial=trial):
                self.compare("torch210_masked_fill_scalar_autograd", {
                    "op": "torch.Tensor.masked_fill", "input": self.rnd(rng, (1, 3), dtype),
                    "mask": T("b8", (rng.random((2, 3)) < 0.5).tolist(), [2, 3]),
                    "value": float(rng.integers(-4, 5)) / 2, "upstream": self.rnd(rng, (2, 3), dtype)})
                self.compare("torch210_where_tensor_autograd", {
                    "op": "torch.where", "condition": T("b8", (rng.random((2, 3)) < 0.5).tolist(), [2, 3]),
                    "input": self.rnd(rng, (1, 3), dtype), "other": self.rnd(rng, (2, 1), dtype),
                    "upstream": self.rnd(rng, (2, 3), dtype)})
                lo = float(rng.integers(-4, 2)) / 2
                self.compare("torch210_clamp_scalar_autograd", {
                    "op": "torch.clamp", "input": self.rnd(rng, (5,), dtype), "min": lo,
                    "max": [None, lo + 1, lo - 1][trial % 3], "upstream": self.rnd(rng, (5,), dtype)})
                self.compare("torch210_extrema_tensor_autograd", {
                    "op": ["torch.maximum", "torch.minimum"][trial % 2], "input": self.rnd(rng, (2, 1), dtype),
                    "other": self.rnd(rng, (1, 3), dtype), "upstream": self.rnd(rng, (2, 3), dtype)})

    def test_finite_scope_is_kept_and_real_profile_agrees_on_finite_inputs(self):
        # autograd profiles still refuse infinities and unrepresentable scalars
        self.compare("torch210_clamp_scalar_autograd", {"op": "torch.clamp", "input": T("f32", [1, INF], [2]),
                     "min": 0, "max": None, "upstream": T("f32", [1, 1], [2])})
        self.compare("torch210_masked_fill_scalar_autograd", {
            "op": "torch.Tensor.masked_fill", "input": T("bf16", [1], [1]), "mask": T("b8", [True], [1]),
            "value": 0.044715, "upstream": T("bf16", [1], [1])})
        res = evaluate(DRAFTS["ARITH"].reference, "torch210_clamp_scalar_autograd",
                       {"op": "torch.clamp", "input": T("f32", [1, INF], [2]), "min": 0, "max": None,
                        "upstream": T("f32", [1, 1], [2])})
        self.assertIn(res.outcome, ("unsupported", "invalid_input"))
        rng = np.random.default_rng(7)
        for op in ("aten.add", "aten.sub", "aten.mul", "aten.div", "aten.maximum", "aten.minimum"):
            a = (rng.integers(-8, 9, 6) / 4.0).tolist()
            b = (rng.integers(1, 9, 6) / 4.0).tolist()
            with self.subTest(op=op):
                new = run("ARITH", op, [T("f32", a), T("f32", b)])
                old = run("ARITH", op, [T("f32", a), T("f32", b)], adapter=VALIDATED["ARITH"])
                self.assertEqual((new.outcome, norm(new.fields["out"])), (old.outcome, norm(old.fields["out"])))


# ==================================================================================================== MATHFN
class MathfnEnclosures(Base):
    def assertEncloses(self, op, dtype, q, rel=Fraction(1, 2 ** 200), absw=Fraction(1, 2 ** 240)):
        res = run("MATHFN", op, [T(dtype, [f"{q.numerator}/{q.denominator}"])])
        self.assertEqual(res.outcome, "value", res.note)
        lo, hi = iv_ends(res.fields["out"][0])
        blo, bhi = arb_ball(op, q)
        if op in ("aten.sin", "aten.cos", "aten.tanh"):
            blo, bhi = max(blo, Fraction(-1)), min(bhi, Fraction(1))
        self.assertLess(bhi - blo, Fraction(1, 2 ** 1000) * max(abs(blo), 1) + Fraction(1, 2 ** 1100),
                        "reviewer Arb ball not tight")
        self.assertLessEqual(lo, blo)          # rigorous: adapter interval contains the Arb ball
        self.assertGreaterEqual(hi, bhi)
        self.assertLess(hi - lo, rel * max(abs(lo), abs(hi)) + absw, "adapter enclosure too wide")
        mp = mp_point(op, q)                   # numerical crosscheck
        self.assertLessEqual(abs(mp - (blo + bhi) / 2), Fraction(1, 2 ** 900) * max(abs(mp), 1) + Fraction(1, 2 ** 1000))

    def test_recorded_rope_angles(self):
        inv = np.float32(1.0) / (np.float32(10000.0) ** (np.arange(0, 64, 2, dtype=np.float32)[:4] / np.float32(64)))
        pos = np.array([0, 5, 63, 127], dtype=np.float32)
        ang = (pos[:, None] * inv[None, :]).astype(np.float32)
        rec = [[[f"{Fraction(float(v)).numerator}/{Fraction(float(v)).denominator}" for v in row] for row in ang]]
        for op in ("aten.cos", "aten.sin"):
            res = run("MATHFN", op, [T("f32", rec)])
            self.assertEqual(res.outcome, "value", res.note)
            out = res.fields["out"]
            self.assertEqual(np.array(out, dtype=object).shape, (1, 4, 4))
            for i in range(4):
                for j in range(4):
                    q = Fraction(float(ang[i, j]))
                    lo, hi = iv_ends(out[0][i][j])
                    blo, bhi = arb_ball(op, q)
                    with self.subTest(op=op, q=float(q)):
                        self.assertTrue(lo <= blo and bhi <= hi)
                        self.assertLess(hi - lo, Fraction(1, 2 ** 200))

    def test_large_and_tiny_arguments_sin_cos_tanh(self):
        f32max = Fraction(float(np.finfo(np.float32).max))
        f64max = Fraction(float(np.finfo(np.float64).max))
        points = [(Fraction(101), "f32"), (Fraction(-269, 1), "f32"), (Fraction(1e30), "f32"), (f32max, "f32"),
                  (-f32max, "f32"), (f64max, "f64"), (Fraction(2) ** 1000 + Fraction(2) ** 948, "f64"), (Fraction(65504), "f16"),
                  (Fraction(355), "f64"), (Fraction(103993), "f64"), (Fraction(2) ** -149, "f32"),
                  (-Fraction(2) ** -126, "f32"), (Fraction(2) ** -1074, "f64"), (Fraction(3, 2 ** 60), "f64"),
                  (Fraction(-1, 2 ** 300), "f64"), (Fraction(2) ** -24, "bf16")]
        for op in ("aten.sin", "aten.cos", "aten.tanh"):
            for q, dtype in points:
                with self.subTest(op=op, q=float(q), dtype=dtype):
                    self.assertEncloses(op, dtype, q)

    def test_unrepresentable_large_input_is_accepted_with_a_vacuous_but_rigorous_enclosure(self):
        # Observation (review note O1): 2^1000+1 is not a binary64 value; MATHFN does not check representability
        # (pre-existing) and the Arb helper converts it at 320 bits, so sin comes back as the whole [-1, 1].
        q = Fraction(2) ** 1000 + 1
        res = run("MATHFN", "aten.sin", [T("f64", [str(q)])])
        self.assertEqual(res.outcome, "value")
        lo, hi = iv_ends(res.fields["out"][0])
        blo, bhi = arb_ball("aten.sin", q)
        self.assertTrue(lo <= blo and bhi <= hi)
        self.assertGreater(hi - lo, 1)

    def test_tanh_is_not_one_and_sin_is_not_zero_at_large_arguments(self):
        lo, hi = iv_ends(run("MATHFN", "aten.tanh", [T("f32", [50])]).fields["out"][0])
        self.assertLess(hi, 1)
        lo, hi = iv_ends(run("MATHFN", "aten.tanh", [T("f32", [-200])]).fields["out"][0])
        self.assertGreaterEqual(lo, -1)
        lo, hi = iv_ends(run("MATHFN", "aten.sin", [T("f64", [355])]).fields["out"][0])
        self.assertTrue(lo < 0 and hi < 0)      # sin(355) = -3.0e-5

    def test_other_functions_small_and_moderate_arguments(self):
        for op in ("aten.exp", "aten.expm1", "aten.sigmoid", "aten.log1p", "aten.tanh"):
            for q in (Fraction(1, 2 ** 55), -Fraction(1, 2 ** 100), Fraction(1, 2 ** 200), Fraction(7, 3 * 2 ** 3) * 0 + Fraction(7, 8),
                      Fraction(-99), Fraction(100)):
                if op == "aten.log1p" and q <= -1:
                    continue
                with self.subTest(op=op, q=float(q)):
                    self.assertEncloses(op, "f64", q, rel=Fraction(1, 2 ** 200), absw=Fraction(0))
        for q in (Fraction(1, 2 ** 70), Fraction(3), Fraction(100)):
            with self.subTest(op="log", q=float(q)):
                self.assertEncloses("aten.log", "f64", q)

    def test_bounds_and_refusals(self):
        for op, args, outcome in (
                ("aten.exp", [T("f32", [101])], "unsupported"), ("aten.expm1", [T("f32", [-101])], "unsupported"),
                ("aten.sigmoid", [T("f32", [101])], "unsupported"), ("aten.log", [T("f32", [101])], "unsupported"),
                ("aten.sqrt", [T("f32", [101])], "unsupported"), ("aten.erf", [T("f32", [101])], "unsupported"),
                ("aten.sin", [T("f32", [INF])], "unsupported"), ("aten.cos", [T("f32", [-INF])], "unsupported"),
                ("aten.tanh", [T("f32", [INF])], "unsupported"), ("aten.tanh", [T("f32", [float("nan")])], "invalid_input"),
                ("aten.sin", [T("i64", [1])], "unsupported"), ("aten.log", [T("f32", [0])], "domain_undefined")):
            # Not asserted: {t: f32, v: ['1/3']} is accepted as a value by both the validated and the draft MATHFN
            # adapter (no representability check; pre-existing, recorded in the review note as an observation).
            with self.subTest(op=op, args=args):
                self.assertOutcome(run("MATHFN", op, args), outcome)

    def test_exact_points(self):
        for op, v, want in (("aten.sin", 0, 0), ("aten.cos", 0, 1), ("aten.tanh", 0, 0), ("aten.exp", 0, 1),
                            ("aten.sigmoid", 0, Fraction(1, 2)), ("aten.log", 1, 0)):
            with self.subTest(op=op):
                self.assertValue(run("MATHFN", op, [T("f32", [v])]), [Fraction(want)])


# ===================================================================================================== INDEX
class IndexIntegerDataAndPadding(Base):
    def test_recorded_int64_gather_buffer(self):
        buf = (np.arange(32, dtype=np.int64) * 3 + 2 ** 53 + 1).reshape(1, 32)
        idx = np.array([[0, 31, 7, 7, 2 ** 0, 16, 30, 3]], dtype=np.int64)
        want = np.take_along_axis(buf, idx, axis=1)
        res = run("INDEX", "aten.gather", [T("i64", buf.tolist()), 1, T("i64", idx.tolist())])
        self.assertValue(res, [[Fraction(int(v)) for v in want[0]]])

    def test_integer_dtypes_gather_index_select(self):
        rng = np.random.default_rng(9)
        for dtype, (lo, hi) in INT_RANGE.items():
            src = rng.integers(lo, hi, (3, 4), endpoint=True, dtype=np.int64)
            src[0, 0], src[2, 3] = lo, hi
            gidx = rng.integers(0, 3, (2, 4))
            sidx = [3, 0, 0, 2]
            with self.subTest(dtype=dtype):
                self.assertValue(run("INDEX", "aten.gather", [T(dtype, src.tolist()), 0, T("i64", gidx.tolist())]),
                                 [[Fraction(int(v)) for v in r] for r in np.take_along_axis(src, gidx, 0)])
                self.assertValue(run("INDEX", "aten.index_select", [T(dtype, src.tolist()), 1, T("i64", sidx)]),
                                 [[Fraction(int(v)) for v in r] for r in src[:, sidx]])

    def test_integer_data_refusals(self):
        for args, outcome in (([T("i8", [128]), 0, T("i64", [0])], "invalid_input"),
                              ([T("u8", [-1]), 0, T("i64", [0])], "invalid_input"),
                              ([T("i64", [1.0]), 0, T("i64", [0])], "invalid_input"),
                              ([T("i64", ["3/2"]), 0, T("i64", [0])], "invalid_input"),
                              ([T("b8", [True]), 0, T("i64", [0])], "unsupported"),
                              ([T("i32", [1]), 0, T("i32", [0])], "unsupported")):
            with self.subTest(args=args):
                self.assertOutcome(run("INDEX", "aten.gather", args), outcome)

    def test_recorded_embedding_with_padding_idx_zero(self):
        rng = np.random.default_rng(13)
        w = np.array(bf16_round(rng.standard_normal(24) + 0.5), dtype=object).reshape(6, 4)
        w[0, :] = [Fraction(1, 2), -2, 3, Fraction(1, 4)]            # the padding row is nonzero
        idx = np.array([[0, 3, 5, 0, 1, 0, 2, 4]])
        want = [[[Fraction(v) for v in w[k]] for k in idx[0]]]
        rec_w = [[fx(Fraction(v)) for v in row] for row in w]
        self.assertValue(run("INDEX", "aten.embedding", [T("bf16", rec_w), T("i64", idx.tolist()), 0]), want)
        self.assertValue(run("INDEX", "aten.embedding", [T("bf16", rec_w), T("i64", idx.tolist())]), want)

    def test_padding_idx_forms(self):
        w = [[1, 2], [3, 4], [5, 6]]
        rows = [[Fraction(1), Fraction(2)], [Fraction(3), Fraction(4)], [Fraction(5), Fraction(6)]]
        i = [2, 0, 1]
        want = [rows[k] for k in i]
        for p in (-1, 0, 1, 2):
            with self.subTest(native=p):
                self.assertValue(run("INDEX", "aten.embedding", [T("f32", w), T("i64", i), p]), want)
        for p in (3, -2, 100):
            with self.subTest(native=p):
                self.assertOutcome(run("INDEX", "aten.embedding", [T("f32", w), T("i64", i), p]), "unsupported")
        for p in (None, -3, -1, 0, 2):
            with self.subTest(public=p):
                kw = {} if p is None else {"padding_idx": p}
                self.assertValue(run("INDEX", "torch.nn.functional.embedding", [T("i64", i), T("f32", w)], kw), want)
        for p in (3, -4):
            with self.subTest(public=p):
                self.assertOutcome(run("INDEX", "torch.nn.functional.embedding", [T("i64", i), T("f32", w)],
                                       {"padding_idx": p}), "invalid_input")
        for kw in ({"max_norm": 1.0}, {"scale_grad_by_freq": True}, {"sparse": True}):
            with self.subTest(kw=kw):
                self.assertOutcome(run("INDEX", "torch.nn.functional.embedding", [T("i64", i), T("f32", w)], kw),
                                   "unsupported")
        self.assertOutcome(run("INDEX", "aten.embedding", [T("i64", w), T("i64", i)]), "unsupported")

    def test_floating_paths_agree_with_validated_adapter(self):
        rng = np.random.default_rng(17)
        for trial in range(10):
            src = (rng.integers(-8, 9, (4, 3)) / 4.0).tolist()
            for op, args in (("aten.gather", [T("f32", src), 1, T("i64", rng.integers(0, 3, (4, 2)).tolist())]),
                             ("aten.index_select", [T("f32", src), 0, T("i64", rng.integers(0, 4, 5).tolist())]),
                             ("aten.embedding", [T("f32", src), T("i64", rng.integers(0, 4, (2, 3)).tolist())])):
                with self.subTest(trial=trial, op=op):
                    new, old = run("INDEX", op, args), run("INDEX", op, args, adapter=VALIDATED["INDEX"])
                    self.assertEqual((new.outcome, norm(new.fields["out"])), (old.outcome, norm(old.fields["out"])))


if __name__ == "__main__":
    unittest.main()
