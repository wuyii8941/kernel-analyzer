"""Inclusion property: {g(x) : x in I} is contained in Eval(I).

Random intervals and random interior points; the truth is computed at a much
higher MPFR precision than the evaluator uses.
"""

import random

import pytest

gmpy2 = pytest.importorskip("gmpy2")
from gmpy2 import mpq  # noqa: E402

from kernel_analyzer.reference_eval import Interval, ProgramBuilder, ReferenceEvaluator
from kernel_analyzer.reference_eval.numbers import ELEMENTARY

DOMAINS = {
    "exp": (-20.0, 20.0),
    "exp2": (-20.0, 20.0),
    "log": (1e-6, 50.0),
    "log2": (1e-6, 50.0),
    "log1p": (-0.999, 50.0),
    "sqrt": (0.0, 50.0),
    "rsqrt": (1e-6, 50.0),
    "tanh": (-10.0, 10.0),
    "erf": (-5.0, 5.0),
    "atan": (-50.0, 50.0),
    "sin": (-20.0, 20.0),
    "cos": (-20.0, 20.0),
}

TRUTH = {
    "exp": gmpy2.exp, "exp2": gmpy2.exp2, "log": gmpy2.log, "log2": gmpy2.log2,
    "log1p": gmpy2.log1p, "sqrt": gmpy2.sqrt, "rsqrt": gmpy2.rec_sqrt, "tanh": gmpy2.tanh,
    "erf": gmpy2.erf, "atan": gmpy2.atan, "sin": gmpy2.sin, "cos": gmpy2.cos,
}


def _truth(name, q):
    with gmpy2.context(precision=2048):
        return mpq(TRUTH[name](gmpy2.mpfr(q)))


@pytest.mark.parametrize("name", sorted(ELEMENTARY))
def test_elementary_enclosure_contains_every_sampled_value(name):
    rng = random.Random(name)
    lo_d, hi_d = DOMAINS[name]
    for _ in range(40):
        a, b = sorted(rng.uniform(lo_d, hi_d) for _ in range(2))
        if rng.random() < 0.3:
            b = a  # point input
        box = Interval(a, b)
        enclosure = ELEMENTARY[name](box)
        for t in (mpq(a), mpq(b), mpq(a) + (mpq(b) - mpq(a)) * mpq(rng.randrange(1, 1000), 1000)):
            assert enclosure.contains(_truth(name, t)), (name, box, t)
        if box.is_point:
            assert enclosure.width > 0 or name in ("sqrt",)


def test_composed_chain_contains_the_true_value_and_widens_monotonically():
    # exp -> sum -> divide, the step-6 structure without a dedicated reference.
    xs = [0.25, -1.5, 3.0, 0.0, -0.75]
    b = ProgramBuilder(("x",))
    e = b.op("exp", "x", out="e")
    s = b.op("reduce_sum", e, out="s")
    p = b.op("div", e, s, out="p")
    result = ReferenceEvaluator(b.build((p,))).composed({"x": xs})
    with gmpy2.context(precision=2048):
        exps = [gmpy2.exp(gmpy2.mpfr(x)) for x in xs]
        total = sum(exps)
        truths = [mpq(v / total) for v in exps]
    for ref, truth in zip(result.values["p"], truths):
        assert ref.value.contains(truth)
        assert 0 < ref.value.width < mpq(1, 10 ** 60)
    assert result.values["s"][0].value.width >= max(r.value.width for r in result.values["e"])


@pytest.mark.parametrize("seed", range(5))
def test_rational_arithmetic_on_points_is_exact(seed):
    rng = random.Random(seed)
    b = ProgramBuilder(("a", "b", "c"))
    m = b.op("mul", "a", "b")
    d = b.op("div", m, "c")
    f = b.op("fma", "a", "b", "c")
    program = b.build()
    for _ in range(50):
        a, bb = rng.uniform(-1e3, 1e3), rng.uniform(-1e3, 1e3)
        c = rng.choice([-1, 1]) * rng.uniform(1e-3, 1e3)
        result = ReferenceEvaluator(program).composed({"a": a, "b": bb, "c": c})
        assert result.values[d][0].value == Interval(mpq(a) * mpq(bb) / mpq(c))
        assert result.values[f][0].value == Interval(mpq(a) * mpq(bb) + mpq(c))
