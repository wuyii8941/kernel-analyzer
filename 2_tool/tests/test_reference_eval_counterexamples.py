"""Step-1 acceptance: the eight counterexamples of stage summary section 11.

Each test names the wrong conclusion it must prevent.
"""

import pytest

gmpy2 = pytest.importorskip("gmpy2")
from gmpy2 import mpq  # noqa: E402

from kernel_analyzer.reference_eval import (
    UNDEFINED,
    Interval,
    Mode,
    NotObservable,
    OutputClass,
    ProgramBuilder,
    ReferenceEvaluator,
    Special,
    UnsupportedOperation,
    adjoint_residual,
    coverage_report,
    round_nearest_even,
)

TWO24 = 2.0 ** 24


def _single(result, name):
    (ref,) = result.values[name]
    return ref


# 1. Spelling a rounding into the op name must not remove it from the study.
@pytest.mark.parametrize("name", ["add", "add_rn"])
def test_rounding_suffix_does_not_change_numerical_difference_reference(name):
    b = ProgramBuilder(("a", "b"))
    s = b.op(name, "a", "b", format="fp32")
    program = b.build((s,))
    actual_k = TWO24  # FP32 a + b

    nd = ReferenceEvaluator(program).composed({"a": TWO24, "b": 1.0})
    ref = _single(nd, s)
    assert ref.value == Interval(TWO24 + 1)
    assert nd.residual(ref, actual_k) == Interval(-1)

    rc = ReferenceEvaluator(program, Mode.ROUNDING_CHECK).composed({"a": TWO24, "b": 1.0})
    ref = _single(rc, s)
    assert ref.value == Interval(TWO24)
    assert rc.residual(ref, actual_k) == Interval(0)


# 2. Correct rounding or exact accumulation must not be reported as a
#    zero-width true value when the real result is irrational.
def test_sqrt2_is_an_interval_that_contains_the_true_value():
    b = ProgramBuilder(("x",))
    r = b.op("sqrt", "x")
    ref = _single(ReferenceEvaluator(b.build((r,))).composed({"x": 2.0}), r)
    assert ref.value.width > 0
    assert ref.value.lo ** 2 < 2 < ref.value.hi ** 2


def test_exp_then_sum_keeps_a_nonzero_enclosing_width():
    xs = [0.5, -1.25, 2.0, 0.125]
    b = ProgramBuilder(("x",))
    e = b.op("exp", "x")
    s = b.op("reduce_sum", e)
    ref = _single(ReferenceEvaluator(b.build((s,))).composed({"x": xs}), s)
    assert ref.value.width > 0
    with gmpy2.context(precision=1024):
        truth = mpq(sum(gmpy2.exp(gmpy2.mpfr(x)) for x in xs))
    assert ref.value.contains(truth)


# 3. A path-conditioned reference must not be reported as a complete composed
#    reference.
def _branch_program():
    then_b = ProgramBuilder()
    one = then_b.op("const", value=1.0)
    else_b = ProgramBuilder()
    zero = else_b.op("const", value=0.0)

    b = ProgramBuilder(("a", "b", "t"))
    s = b.op("add", "a", "b", out="s", format="fp32")
    c = b.op("cmp", s, "t", out="c", predicate="gt")
    y = b.op("if", c, out="y", regions=(then_b.build((one,)), else_b.build((zero,))))
    return b.build((y,))


def test_branch_is_decided_by_reference_value_not_actual_path():
    program = _branch_program()
    inputs = {"a": TWO24, "b": 1.0, "t": TWO24}
    actual = {"s": [TWO24], "c": [False], "y": [0.0]}

    result = ReferenceEvaluator(program).composed(inputs, actual=actual)
    ref = _single(result, "y")
    assert ref.value == Interval(1)
    assert result.residual(ref, 0.0) == Interval(-1)
    assert result.output_classes(("y",))["y"]["class"] is OutputClass.COMPLETE_COMPOSED
    assert result.path_events == [{"node": "if#2", "reference": True, "actual": False, "flipped": True}]

    along_actual = ReferenceEvaluator(program).composed(inputs, actual=actual, follow_actual_path=True)
    ref = _single(along_actual, "y")
    assert along_actual.residual(ref, 0.0) == Interval(0)
    assert along_actual.output_classes(("y",))["y"]["class"] is OutputClass.CONDITIONAL_LOCAL


def test_undecided_branch_takes_the_union_of_both_branches():
    program = _branch_program()
    # A non-point t makes the comparison undecided.
    result = ReferenceEvaluator(program).composed({"a": TWO24, "b": 1.0, "t": Interval(TWO24, TWO24 + 2)})
    ref = _single(result, "y")
    assert ref.value == Interval(0, 1)
    assert result.output_classes(("y",))["y"]["class"] is OutputClass.COMPLETE_COMPOSED
    assert any(t.startswith("path_union:") for t in ref.taints)


# 4. Reading a captured value back at a load must not erase upstream
#    differences.
def _store_load_program():
    b = ProgramBuilder(("a", "b", "base", "zero"))
    t = b.op("add", "a", "b", format="fp32")
    b.op("store", "zero", t, buffer="tmp")
    u = b.op("load", "zero", out="u", buffer="tmp", node_id="load_u")
    d = b.op("sub", u, "base", out="d", format="fp32")
    return b.build((d,))


def test_store_then_load_keeps_the_reference_value():
    inputs = {"a": TWO24, "b": 1.0, "base": TWO24, "zero": 0}
    memory = {"tmp": [0.0]}
    result = ReferenceEvaluator(_store_load_program()).composed(inputs, memory)
    ref = _single(result, "d")
    assert ref.value == Interval(1)
    assert result.output_classes(("d",))["d"]["class"] is OutputClass.COMPLETE_COMPOSED

    pinned = ReferenceEvaluator(_store_load_program()).composed(
        inputs, memory, actual={"u": [TWO24]}, pin_loads=("load_u",))
    ref = _single(pinned, "d")
    assert ref.value == Interval(0)
    assert pinned.output_classes(("d",))["d"]["class"] is OutputClass.CONDITIONAL_LOCAL


# 5. The sum of all contributions must not be treated as every atomic's output.
def _atomic_program(use_result: bool, order=None):
    b = ProgramBuilder(("offsets", "values", "out_offsets"))
    attrs = {"buffer": "acc"}
    if order is not None:
        attrs["order"] = order
    old = b.op("atomic_add", "offsets", "values", out="old", **attrs)
    if use_result:
        b.op("store", "out_offsets", old, buffer="out")
    return b.build()


def test_atomic_return_value_requires_an_order():
    inputs = {"offsets": [0, 0, 0], "values": [1.0, 2.0, 3.0], "out_offsets": [0, 1, 2]}
    memory = {"acc": [0.0], "out": [0.0, 0.0, 0.0]}

    result = ReferenceEvaluator(_atomic_program(True)).composed(inputs, memory)
    assert result.memory["acc"][0].value == Interval(6)
    classes = result.output_classes()
    assert classes["buffer:acc"]["class"] is OutputClass.COMPLETE_COMPOSED
    assert classes["buffer:out"]["class"] is OutputClass.NOT_ESTABLISHED
    assert all(r.value is UNDEFINED for r in result.memory["out"])

    ordered = ReferenceEvaluator(_atomic_program(True, order="lane")).composed(inputs, memory)
    assert [r.value for r in ordered.memory["out"]] == [Interval(0), Interval(1), Interval(3)]
    assert ordered.output_classes()["buffer:out"]["class"] is OutputClass.COMPLETE_COMPOSED


def test_unused_atomic_return_folds_into_an_exact_sum():
    inputs = {"offsets": [0, 0, 0], "values": [1.0, 2.0, 3.0], "out_offsets": [0, 1, 2]}
    result = ReferenceEvaluator(_atomic_program(False)).composed(inputs, {"acc": [0.0], "out": [0.0] * 3})
    assert result.memory["acc"][0].value == Interval(6)
    assert result.output_classes()["buffer:acc"]["class"] is OutputClass.COMPLETE_COMPOSED


# 6. A fused intermediate must not be captured, and its reference must be
#    composed into the observable consumer.
@pytest.mark.parametrize("fused", [True, False])
def test_fma_fusion_moves_the_local_residual_to_the_region_boundary(fused):
    a = 1 + 2.0 ** -12
    c = -(1 + 2.0 ** -11)
    p_rounded = float(round_nearest_even(mpq(a) * mpq(a), "fp32"))
    assert p_rounded == 1 + 2.0 ** -11
    y_actual = 2.0 ** -24 if fused else 0.0  # RN(a*a + c) vs RN(RN(a*a) + c)

    b = ProgramBuilder(("a", "b", "c"))
    p = b.op("mul", "a", "b", out="p", format="fp32", observable=not fused)
    y = b.op("add", p, "c", out="y", format="fp32")
    program = b.build((y,))
    evaluator = ReferenceEvaluator(program)
    actual = {"a": [a], "b": [a], "c": [c], "y": [y_actual]}
    if not fused:
        actual["p"] = [p_rounded]

    composed = evaluator.composed({"a": a, "b": a, "c": c})
    assert _single(composed, "y").value == Interval(mpq(2) ** -24)

    if fused:
        with pytest.raises(NotObservable):
            evaluator.local_residual(actual, "p")
        assert evaluator.local_residual(actual, "y") == Interval(0)
    else:
        assert evaluator.local_residual(actual, "p") == Interval(-(mpq(2) ** -24))
        assert evaluator.local_residual(actual, "y") == Interval(0)


# 7. Legal values must not be rejected and undefined values must not be
#    accepted.
def test_log1p_domain_is_open_at_minus_one():
    b = ProgramBuilder(("x",))
    r = b.op("log1p", "x")
    program = b.build((r,))
    ok = _single(ReferenceEvaluator(program).composed({"x": -0.5}), r)
    with gmpy2.context(precision=512):
        assert ok.value.contains(mpq(gmpy2.log(gmpy2.mpfr(0.5))))
    bad = ReferenceEvaluator(program).composed({"x": -1.0})
    assert bad.output_classes((r,))[r]["class"] is OutputClass.NOT_ESTABLISHED


def test_nan_rules_are_explicit():
    b = ProgramBuilder(("n", "one"))
    isnan = b.op("isnan", "n")
    maxnum = b.op("maxnum", "n", "one")
    maximum = b.op("maximum", "n", "one")
    result = ReferenceEvaluator(b.build()).composed({"n": float("nan"), "one": 1.0})
    assert _single(result, isnan).value is True
    assert _single(result, maxnum).value == Interval(1)
    assert _single(result, maximum).value is Special.NAN


def test_masked_load_without_other_is_undefined_until_used():
    b = ProgramBuilder(("offs", "mask"))
    v = b.op("load", "offs", "mask", buffer="src")
    b.op("store", "offs", v, "mask", buffer="masked_out")
    b.op("store", "offs", v, buffer="unmasked_out")
    memory = {"src": [5.0, 6.0], "masked_out": [0.0, 0.0], "unmasked_out": [0.0, 0.0]}
    result = ReferenceEvaluator(b.build()).composed({"offs": [0, 1], "mask": [True, False]}, memory)
    assert result.values[v][1].value is UNDEFINED
    classes = result.output_classes()
    assert classes["buffer:masked_out"]["class"] is OutputClass.COMPLETE_COMPOSED
    assert classes["buffer:unmasked_out"]["class"] is OutputClass.NOT_ESTABLISHED


# 8. A central difference has a sign-fixed truncation error; forward-mode
#    differentiation does not.
@pytest.mark.parametrize("x", [0.0, 0.75, -1.5, 3.0])
def test_adjoint_check_of_cube_has_no_false_positive(x):
    b = ProgramBuilder(("x",))
    sq = b.op("mul", "x", "x")
    y = b.op("mul", sq, "x", out="y")
    program = b.build((y,))
    u, v, h = 1.0, 1.0, mpq(1, 64)
    backward = [mpq(3) * mpq(x) ** 2 * mpq(v) * mpq(u)]  # B(x, v) u with B = 3x^2 v

    f = lambda t: t ** 3
    central = (f(mpq(x) + h * mpq(u)) - f(mpq(x) - h * mpq(u))) / (2 * h)
    assert backward[0] - mpq(v) * central == -h ** 2  # same sign for every x

    jvp = ReferenceEvaluator(program).derivative({"x": x}, {"x": u}).values["y"]
    residual = adjoint_residual(backward, [v], jvp)
    assert residual == Interval(0)


# Supporting checks -------------------------------------------------------------


def test_round_nearest_even_matches_the_talk_examples():
    assert round_nearest_even(257, "bf16") == 256
    assert round_nearest_even(1 + mpq(17, 4096), "bf16") - (1 + mpq(17, 4096)) == mpq(15, 4096)
    assert round_nearest_even(1 + mpq(17, 4096), "fp16") - (1 + mpq(17, 4096)) == mpq(-1, 4096)
    assert round_nearest_even(mpq(2) ** -149, "fp32") == mpq(2) ** -149  # smallest subnormal
    assert round_nearest_even(mpq(2) ** -151, "fp32") == 0  # below half of it


def test_sin_enclosure_includes_an_interior_maximum():
    b = ProgramBuilder(("x",))
    r = b.op("sin", "x")
    ref = _single(ReferenceEvaluator(b.build((r,))).composed({"x": Interval(1.5, 1.7)}), r)
    assert ref.value.hi == 1
    assert ref.value.lo < mpq(0.9975) < 1


def test_unknown_op_or_attribute_is_rejected_not_skipped():
    b = ProgramBuilder(("x",))
    b.op("mystery", "x")
    b.op("add", "x", "x", fastmath=True)
    report = coverage_report(b.build())
    assert not report["complete"]
    assert {r["op"] for r in report["rejected"]} == {"mystery", "add"}
    with pytest.raises(UnsupportedOperation):
        ReferenceEvaluator(b.build())
