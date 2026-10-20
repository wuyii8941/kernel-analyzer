"""Independent review tests for path A closure package E (CREATE new, RANDOM native_forward_given_mask), 2026-10-20.

Written by the reviewer, not the author. Expected values are computed here with plain Python integers / Fraction /
numpy only; nothing from the adapters' helpers or kernel_spec.aten is used for the computation. The draft pairs are
staged as <stage>/contracts/<ID>.yaml + <stage>/adapters/<ID>.py under .cache/spec_a2_review/E/ (untracked) and loaded
with kernel_spec.registry.load. PyTorch eager comparisons are supporting evidence only.
"""
from __future__ import annotations

import math
import os
import random
import shutil
import struct
import sys
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import pytest

SPEC = Path(__file__).resolve().parents[1]
REPO = SPEC.parent
sys.path.insert(0, str(SPEC))
sys.path.insert(0, str(SPEC / "tools"))
from kernel_spec import registry  # noqa: E402
from kernel_spec.num import evaluate  # noqa: E402

DRAFT = SPEC / "contracts" / "draft"
FILES = {"CREATE": ("CREATE_path_a_A2_20261020.yaml", "CREATE_adapter_path_a_A2_20261020.py"),
         "RANDOM": ("RANDOM_path_a_A2_20261020.yaml", "RANDOM_adapter_path_a_A2_20261020.py")}
NATIVE = "native_forward_given_mask"
STAGE = REPO / ".cache" / "spec_a2_review" / "E" / f"stage_{os.getpid()}"


@pytest.fixture(scope="module")
def stage():
    shutil.rmtree(STAGE, ignore_errors=True)
    (STAGE / "contracts").mkdir(parents=True)
    (STAGE / "adapters").mkdir()
    for cid, (y, p) in FILES.items():
        shutil.copy(DRAFT / y, STAGE / "contracts" / f"{cid}.yaml")
        shutil.copy(DRAFT / p, STAGE / "adapters" / f"{cid}.py")
    yield STAGE
    shutil.rmtree(STAGE, ignore_errors=True)


@pytest.fixture(scope="module")
def create(stage):
    return registry.load("CREATE", contracts=stage / "contracts", adapters=stage / "adapters").adapter


@pytest.fixture(scope="module")
def rnd(stage):
    return registry.load("RANDOM", contracts=stage / "contracts", adapters=stage / "adapters").adapter


@pytest.fixture(scope="module")
def rnd_validated():
    return registry.load("RANDOM").adapter


# ---------------------------------------------------------------- independent helpers
def norm(x):
    """Reference leaf -> comparable exact value (Fraction or +-inf float); nested lists recursively."""
    if isinstance(x, list):
        return [norm(v) for v in x]
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("inf", "+inf"):
            return math.inf
        if s == "-inf":
            return -math.inf
        return Q(s)
    if isinstance(x, bool):
        return Q(int(x))
    if isinstance(x, float):
        return x if math.isinf(x) else Q(x)
    return Q(x)


def outcome(adapter, profile, call):
    return evaluate(adapter.reference, profile, call).outcome


def value(adapter, profile, call):
    r = evaluate(adapter.reference, profile, call)
    assert r.outcome == "value", (call, r.outcome)
    return norm(r.fields["out"])


def ceil_q(q: Q) -> int:
    return -((-q.numerator) // q.denominator)


def my_arange(s, e, st):
    s, e, st = Q(s), Q(e), Q(st)
    n = ceil_q((e - s) / st)
    return [s + i * st for i in range(n)]


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf16(x):
    """Round-to-nearest-even of a float32 to bfloat16, as a Python float (independent bit arithmetic)."""
    b = struct.unpack("I", struct.pack("f", f32(x)))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return struct.unpack("f", struct.pack("I", b << 16))[0]


def filled(shape, v):
    if not shape:
        return v
    return [filled(shape[1:], v) for _ in range(shape[0])]


# ---------------------------------------------------------------- CREATE: recorded structures
def test_recorded_arange_128(create):
    call = {"op": "aten.arange", "args": [128]}
    assert value(create, "explicit", call) == [Q(i) for i in range(128)]
    assert create.check_output("explicit", call, {"out": {"t": "i64", "v": list(range(128)), "shape": [128]}})
    assert not create.check_output("explicit", call, {"out": {"t": "i32", "v": list(range(128)), "shape": [128]}})
    assert not create.check_output("explicit", call, {"out": {"t": "i64", "v": list(range(1, 129)), "shape": [128]}})


@pytest.mark.parametrize("d", ["bf16", "f32"])
def test_recorded_zeros_empty(create, d):
    call = {"op": "aten.zeros", "args": [[0]], "kwargs": {"dtype": {"dtype": d}}}
    assert value(create, "explicit", call) == []
    assert create.check_output("explicit", call, {"out": {"t": d, "v": [], "shape": [0]}})
    assert not create.check_output("explicit", call, {"out": {"t": d, "v": [0.0], "shape": [1]}})


def test_recorded_ones_bool_full_size(create):
    call = {"op": "aten.ones", "args": [[128, 128]], "kwargs": {"dtype": {"dtype": "b8"}}}
    assert value(create, "explicit", call) == filled([128, 128], Q(1))
    good = {"out": {"t": "b8", "v": filled([128, 128], True), "shape": [128, 128]}}
    assert create.check_output("explicit", call, good)
    bad = filled([128, 128], True)
    bad[77][3] = False
    assert not create.check_output("explicit", call, {"out": {"t": "b8", "v": bad, "shape": [128, 128]}})
    for t, one in (("f32", 1.0), ("u8", 1), ("i64", 1)):     # numeric 1 in a non-bool dtype is not a bool result
        assert not create.check_output("explicit", call, {"out": {"t": t, "v": filled([128, 128], one),
                                                                  "shape": [128, 128]}})


def test_bool_check_rejects_non_boolean_entries(create):
    call = {"op": "aten.ones", "args": [[2]], "kwargs": {"dtype": {"dtype": "b8"}}}
    assert not create.check_output("explicit", call, {"out": {"t": "b8", "v": [2, 1], "shape": [2]}})


@pytest.mark.parametrize("arg", ["-inf", -math.inf])
def test_recorded_scalar_tensor_neg_inf(create, arg):
    call = {"op": "aten.scalar_tensor", "args": [arg], "kwargs": {"dtype": {"dtype": "f32"}}}
    assert value(create, "explicit", call) == -math.inf
    assert create.check_output("explicit", call, {"out": {"t": "f32", "v": -math.inf, "shape": []}})
    big = float(np.finfo(np.float32).max)
    assert not create.check_output("explicit", call, {"out": {"t": "f32", "v": -big, "shape": []}})


def test_recorded_scalar_tensor_zero(create):
    call = {"op": "aten.scalar_tensor", "args": [0.0], "kwargs": {"dtype": {"dtype": "f32"}}}
    assert value(create, "explicit", call) == 0
    assert create.check_output("explicit", call, {"out": {"t": "f32", "v": 0.0, "shape": []}})
    assert not create.check_output("explicit", call, {"out": {"t": "f32", "v": 0.0, "shape": [1]}})


@pytest.mark.parametrize("shape", [[], [1, 1, 2, 2], [3, 0, 2]])
def test_recorded_ones_like(create, shape):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(shape).astype(np.float32)
    rec = {"t": "f32", "v": x.astype(np.float64).tolist(), "shape": shape}
    call = {"op": "aten.ones_like", "args": [rec]}
    assert value(create, "explicit", call) == filled(shape, Q(1))
    assert create.check_output("explicit", call, {"out": {"t": "f32", "v": filled(shape, 1.0), "shape": shape}})
    assert not create.check_output("explicit", call, {"out": {"t": "bf16", "v": filled(shape, 1.0), "shape": shape}})


def test_like_ignores_input_values_and_keeps_dtype(create):
    for t in ("f16", "bf16", "f64", "i32", "u8", "b8"):
        v = [True, False] if t == "b8" else [3, 0]
        for op, fill in (("aten.zeros_like", 0), ("aten.ones_like", 1)):
            call = {"op": op, "args": [{"t": t, "v": v}]}
            assert value(create, "explicit", call) == [Q(fill)] * 2
    call = {"op": "aten.full_like", "args": [{"t": "f32", "v": [math.nan, math.inf]}, -2.5],
            "kwargs": {"dtype": {"dtype": "f64"}}}
    assert value(create, "explicit", call) == [Q(-5, 2)] * 2


# ---------------------------------------------------------------- CREATE: arange rule, independent
def test_arange_integer_random(create):
    rng = random.Random(11)
    for _ in range(300):
        s = rng.randint(-50, 50)
        st = rng.choice([v for v in range(-7, 8) if v])
        e = s + st * rng.randint(0, 20) + rng.randint(-abs(st) + 1, abs(st) - 1) * (1 if st > 0 else -1)
        if (st > 0 and e < s) or (st < 0 and e > s):
            assert outcome(create, "explicit", {"op": "aten.arange", "args": [s, e, st]}) == "invalid_input"
            continue
        got = value(create, "explicit", {"op": "aten.arange", "args": [s, e, st]})
        assert got == my_arange(s, e, st)
        # int64 arithmetic of RangeUtils.h (truncating division) gives the same length for consistent signs
        n_ref = int((e - s + st - (1 if st > 0 else -1)) / st) if (e - s) else 0
        assert len(got) == max(n_ref, 0)


def test_arange_band_rule_and_double_agreement(create):
    """Outside the 2^-30 band the exact length equals the double evaluation of RangeUtils.h; inside, unsupported."""
    rng = random.Random(5)
    band = Q(1, 2 ** 30)
    seen = {"value": 0, "unsupported": 0}
    for _ in range(400):
        s = round(rng.uniform(-10, 10), rng.choice([0, 1, 2, 3]))
        st = rng.choice([0.1, 0.25, 0.3, 0.5, 1.5, 2.0, -0.1, -0.7, 1 / 3])
        e = s + st * rng.choice([0, 1, 2, 3, 5, 7, 10]) + rng.choice([0.0, 1e-12, -1e-12, 0.05, -0.05])
        if (st > 0 and e < s) or (st < 0 and e > s):
            continue
        call = {"op": "aten.arange", "args": [s, e, st], "kwargs": {"dtype": {"dtype": "f64"}}}
        q = (Q(e) - Q(s)) / Q(st)
        dist = min(q - math.floor(q), math.ceil(q) - q)
        all_int = all(Q(v).denominator == 1 and abs(Q(v)) < 2 ** 53 for v in (s, e, st))
        if not all_int and dist <= band * max(1, abs(q)):
            assert outcome(create, "explicit", call) == "unsupported", call
            seen["unsupported"] += 1
            continue
        got = value(create, "explicit", call)
        assert got == my_arange(s, e, st), call
        assert len(got) == math.ceil((e - s) / st), call            # double evaluation agrees outside the band
        seen["value"] += 1
    assert seen["value"] > 50 and seen["unsupported"] > 20


@pytest.mark.parametrize("args,expected", [
    ([0, 2.5], [0, 1, 2]), ([0, 2, 0.75], [0, Q(3, 4), Q(3, 2)]), ([2.5, 0, -1], [Q(5, 2), Q(3, 2), Q(1, 2)]),
])
def test_arange_floating_default(create, args, expected):
    call = {"op": "aten.arange", "args": args}
    assert outcome(create, "explicit", call) == "unsupported"         # default floating dtype not recorded
    assert value(create, "default_float32", call) == [Q(v) for v in expected]


@pytest.mark.parametrize("args,kw,cls", [
    ([5, 1, 1], {}, "invalid_input"), ([0, 5, 0], {}, "invalid_input"), ([0, 0], {}, "value"),
    ([0, math.inf], {"dtype": {"dtype": "f32"}}, "invalid_input"), ([math.nan, 1], {"dtype": {"dtype": "f32"}}, None),
    ([1, 2.5, 0.5], {}, "unsupported"), ([1, 2.5, 0.5], {"dtype": {"dtype": "f32"}}, "unsupported"),  # integer q
    ([0, 5], {"dtype": {"dtype": "b8"}}, "unsupported"), ([0.5, 3], {"dtype": {"dtype": "i64"}}, "unsupported"),
    ([0.0, 3.0], {"dtype": {"dtype": "i64"}}, "value"), ([True, 3], {}, "unsupported"),
    ([0, 300], {"dtype": {"dtype": "u8"}}, "invalid_input"), ([2 ** 64, 2 ** 64 + 2], {}, "invalid_input"),
    ([0, 70000], {"dtype": {"dtype": "f16"}}, "unsupported"),
])
def test_arange_classes(create, args, kw, cls):
    call = {"op": "aten.arange", "args": args, "kwargs": kw}
    got = outcome(create, "explicit", call)
    if cls is None:
        assert got in ("unsupported", "invalid_input")
    else:
        assert got == cls, (args, kw, got)


def test_arange_infinite_step_class_recorded(create):
    """Review issue CR-2 (fixed 2026-10-21): RangeUtils.h only rejects nonfinite start / end and eager returns an
    empty tensor for an infinite step; CR.ARANGE_LENGTH now records the alternatives and makes it unsupported."""
    call = {"op": "aten.arange", "args": [0, 5, math.inf], "kwargs": {"dtype": {"dtype": "f32"}}}
    assert outcome(create, "explicit", call) == "unsupported"


# ---------------------------------------------------------------- CREATE: fills, conversion, default dtype
@pytest.mark.parametrize("d", ["f16", "bf16", "f32", "f64", "i64", "i32", "i16", "i8", "u8", "b8"])
def test_zeros_ones_all_dtypes(create, d):
    for op, v in (("aten.zeros", 0), ("aten.ones", 1)):
        for shape in ([], [3], [2, 0, 4], [1, 2, 3]):
            call = {"op": op, "args": [shape], "kwargs": {"dtype": {"dtype": d}}}
            assert value(create, "explicit", call) == filled(shape, Q(v))
            obs = filled(shape, bool(v) if d == "b8" else (float(v) if d in ("f16", "bf16", "f32", "f64") else v))
            assert create.check_output("explicit", call, {"out": {"t": d, "v": obs, "shape": shape}})


@pytest.mark.parametrize("fill,kw,cls,exp", [
    (3, {}, "value", 3), (True, {}, "value", 1), (False, {}, "value", 0), (2.5, {}, "unsupported", None),
    (-math.inf, {"dtype": {"dtype": "f32"}}, "value", -math.inf), (math.nan, {"dtype": {"dtype": "f32"}}, "unsupported", None),
    (1e39, {"dtype": {"dtype": "f32"}}, "unsupported", None), (65520.0, {"dtype": {"dtype": "f16"}}, "unsupported", None),
    (65504.0, {"dtype": {"dtype": "f16"}}, "value", 65504), (0.1, {"dtype": {"dtype": "f32"}}, "value", Q(0.1)),
    (300, {"dtype": {"dtype": "u8"}}, "invalid_input", None), (-1, {"dtype": {"dtype": "u8"}}, "invalid_input", None),
    (2.5, {"dtype": {"dtype": "i32"}}, "unsupported", None), (math.inf, {"dtype": {"dtype": "i64"}}, "invalid_input", None),
    (2, {"dtype": {"dtype": "b8"}}, "unsupported", None), (1.0, {"dtype": {"dtype": "b8"}}, "value", 1),
    (2 ** 63, {}, "invalid_input", None), (2 ** 63 - 1, {}, "value", 2 ** 63 - 1),
])
def test_full_conversion(create, fill, kw, cls, exp):
    call = {"op": "aten.full", "args": [[2], fill], "kwargs": kw}
    assert outcome(create, "explicit", call) == cls
    if cls == "value":
        e = exp if isinstance(exp, float) and math.isinf(exp) else Q(exp)
        assert value(create, "explicit", call) == [e, e]


def test_default_dtype_profile(create):
    for call in ({"op": "aten.zeros", "args": [[2]]}, {"op": "aten.ones", "args": [[2]]},
                 {"op": "aten.full", "args": [[2], 0.5]}):
        assert outcome(create, "explicit", call) == "unsupported"
        got = value(create, "default_float32", call)
        assert len(got) == 2
    # default_float32 means float32: an overflow of float32 is refused, which a float64 default would accept
    assert outcome(create, "default_float32", {"op": "aten.full", "args": [[1], 1e39]}) == "unsupported"
    obs = {"out": {"t": "f32", "v": [0.0, 0.0], "shape": [2]}}
    assert create.check_output("default_float32", {"op": "aten.zeros", "args": [[2]]}, obs)
    assert not create.check_output("default_float32", {"op": "aten.zeros", "args": [[2]]},
                                   {"out": {"t": "f64", "v": [0.0, 0.0], "shape": [2]}})


def test_scalar_tensor_classes(create):
    assert outcome(create, "explicit", {"op": "aten.scalar_tensor", "args": [1]}) == "unsupported"
    assert outcome(create, "default_float32", {"op": "aten.scalar_tensor", "args": [1]}) == "unsupported"
    assert outcome(create, "explicit", {"op": "aten.scalar_tensor", "args": [-math.inf],
                                        "kwargs": {"dtype": {"dtype": "i64"}}}) == "invalid_input"
    assert outcome(create, "explicit", {"op": "aten.ones", "args": [[-1]],
                                        "kwargs": {"dtype": {"dtype": "f32"}}}) == "invalid_input"
    assert outcome(create, "explicit", {"op": "aten.ones_like", "args": [{"t": "f8e4m3", "v": [1]}]}) == "unsupported"
    # non-semantic keywords are not part of the reference interface: the binding must drop them
    assert outcome(create, "explicit", {"op": "aten.zeros", "args": [[1]],
                                        "kwargs": {"dtype": {"dtype": "f32"}, "pin_memory": False}}) != "value"


# ---------------------------------------------------------------- RANDOM: native_forward_given_mask, independent
def native_call(x, t, p, train, mask=None):
    call = {"op": "aten.native_dropout", "args": [{"t": t, "v": x}, p, train]}
    if mask is not None:
        call["kwargs"] = {"observed_mask": {"t": "b8", "v": mask}}
    return call


def flat(x):
    if isinstance(x, list):
        return [v for item in x for v in flat(item)]
    return [x]


def reshape(values, shape):
    if not shape:
        return values[0]
    step = len(values) // shape[0] if shape[0] else 0
    return [reshape(values[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


@pytest.mark.parametrize("t,round_", [("f32", f32), ("bf16", bf16), ("f64", float)])
def test_native_conditioned_random(rnd, t, round_):
    rng = random.Random(23)
    for trial in range(40):
        shape = rng.choice([[3], [2, 3], [1, 2, 2], [1, 1, 2, 3], [2, 1, 1, 1, 2], []])
        n = math.prod(shape)
        xs = [round_(rng.uniform(-8, 8)) for _ in range(n)]
        xs = [0.0 if x == 0 else x for x in xs]
        ms = [rng.random() < 0.7 for _ in range(n)]
        p = rng.choice([0.1, 0.5, 0.9, 1 / 3, 0.25, rng.random() * 0.98 + 0.01])
        scale = 1 / (1 - Q(p))                                 # p: exact value of the received double
        exp_out = [Q(x) * scale if m else Q(0) for x, m in zip(xs, ms)]
        exp_mask = [Q(int(m)) for m in ms]
        call = native_call(reshape(xs, shape), t, p, True, reshape(ms, shape))
        out, mask = value(rnd, NATIVE, call)
        assert flat(out) == exp_out and flat(mask) == exp_mask
        assert np.shape(np.array(out, dtype=object)) == tuple(shape)


@pytest.mark.parametrize("p,train,mask_kind", [(0.5, False, "all1"), (0.0, True, "all1"), (0, False, "all1"),
                                               (1.0, True, "all0"), (1, False, "all1")])
def test_native_deterministic(rnd, p, train, mask_kind):
    xs = [[1.5, -2.0], [0.25, 4.0]]
    out, mask = value(rnd, NATIVE, native_call(xs, "f32", p, train))
    if mask_kind == "all1":
        assert out == [[Q(v) for v in row] for row in xs] and mask == [[1, 1], [1, 1]]
    else:
        assert out == [[0, 0], [0, 0]] and mask == [[0, 0], [0, 0]]
    # a supplied observed mask does not override the documented deterministic mask
    out2, mask2 = value(rnd, NATIVE, native_call(xs, "f32", p, train, [[True, False], [False, True]]))
    assert (out2, mask2) == (out, mask)


def test_native_recorded_p01_example(rnd):
    p = 0.1
    assert Q(p) == Q(3602879701896397, 36028797018963968)
    out, mask = value(rnd, NATIVE, native_call([[[3.0, -0.5]]], "bf16", p, True, [[[True, False]]]))
    assert out == [[[Q(3) * Q(36028797018963968, 32425917317067571), 0]]] and mask == [[[1, 0]]]


@pytest.mark.parametrize("call,cls", [
    (native_call([2.0, 4.0], "f32", 0.5, True), "unsupported"),                         # no mask, stochastic
    (native_call([2.0, 4.0], "f32", 0.5, None, [True, False]), "unsupported"),          # train None
    (native_call([2.0, 4.0], "f32", 0.5, 1, [True, False]), "invalid_input"),
    (native_call([2.0, 4.0], "f32", 1.5, True, [True, True]), "invalid_input"),
    (native_call([2.0, 4.0], "f32", -0.25, False), "invalid_input"),
    (native_call([2.0, 4.0], "f32", math.nan, True, [True, True]), "invalid_input"),
    (native_call([2.0, 4.0], "f32", 0.5, True, [True]), "invalid_input"),
    (native_call([[2.0, 4.0]], "f32", 0.5, True, [True, False]), "invalid_input"),
    ({"op": "aten.native_dropout", "args": [{"t": "f32", "v": [2.0, 4.0]}, 0.5, True],
      "kwargs": {"observed_mask": {"t": "i64", "v": [1, 0]}}}, "invalid_input"),
    ({"op": "aten.native_dropout", "args": [{"t": "f32", "v": [2.0, 4.0]}, 0.5, True],
      "kwargs": {"observed_mask": {"t": "b8", "v": [1, 0]}}}, "invalid_input"),
    (native_call([2.0, math.inf], "f32", 0.5, True, [True, False]), "unsupported"),
    (native_call([2.0, math.nan], "f32", 0.5, True, [True, False]), "unsupported"),
    (native_call([2, 4], "i64", 0.5, True, [True, False]), "unsupported"),
    (native_call([], "f32", 0.5, True, []), "unsupported"),
    (native_call([0.1], "f32", 0.5, True, [True]), "invalid_input"),                    # not an f32 value
])
def test_native_classes(rnd, call, cls):
    assert outcome(rnd, NATIVE, call) == cls


def test_native_not_in_old_profiles(rnd):
    call = native_call([2.0, 4.0], "f32", 0.0, False)
    assert outcome(rnd, "deterministic_forward", call) == "unsupported"
    assert outcome(rnd, "finite_backward", call) == "unsupported"


def test_native_check_output(rnd):
    call = native_call([2.0, -4.0], "f32", 0.5, True, [True, False])
    good = {"out": [{"t": "f32", "v": [4.0, 0.0], "shape": [2]}, {"t": "b8", "v": [True, False], "shape": [2]}]}
    assert rnd.check_output(NATIVE, call, good)
    variants = [
        [{"t": "f32", "v": [4.0, -4.0], "shape": [2]}, {"t": "b8", "v": [True, False], "shape": [2]}],
        [{"t": "f32", "v": [4.0, 0.0], "shape": [2]}, {"t": "b8", "v": [True, True], "shape": [2]}],
        [{"t": "f32", "v": [4.0, 0.0], "shape": [2]}, {"t": "u8", "v": [1, 0], "shape": [2]}],
        [{"t": "f32", "v": [4.0, 0.0], "shape": [2]}, {"t": "f32", "v": [1.0, 0.0], "shape": [2]}],
        [{"t": "bf16", "v": [4.0, 0.0], "shape": [2]}, {"t": "b8", "v": [True, False], "shape": [2]}],
        [{"t": "f32", "v": [2.0, 0.0], "shape": [2]}, {"t": "b8", "v": [True, False], "shape": [2]}],
        [{"t": "f32", "v": [4.0, 0.0], "shape": [2]}],
    ]
    for v in variants:
        assert not rnd.check_output(NATIVE, call, {"out": v}), v


def test_boolean_fixes_accept_real_booleans(create, rnd):
    """Re-check of the CR-1 / RN-1 fixes: numpy Booleans and 0-dim Boolean records stay accepted, ints 1/0 rejected."""
    call = {"op": "aten.ones", "args": [[2]], "kwargs": {"dtype": {"dtype": "b8"}}}
    assert create.check_output("explicit", call, {"out": {"t": "b8", "v": [np.bool_(True)] * 2, "shape": [2]}})
    assert not create.check_output("explicit", call, {"out": {"t": "b8", "v": [1, 1], "shape": [2]}})
    zcall = {"op": "aten.zeros", "args": [[]], "kwargs": {"dtype": {"dtype": "b8"}}}
    assert create.check_output("explicit", zcall, {"out": {"t": "b8", "v": False, "shape": []}})
    assert not create.check_output("explicit", zcall, {"out": {"t": "b8", "v": 0, "shape": []}})
    ncall = native_call(3.0, "f32", 0.5, True, True)
    assert rnd.check_output(NATIVE, ncall, {"out": [{"t": "f32", "v": 6.0, "shape": []},
                                                    {"t": "b8", "v": np.bool_(True), "shape": []}]})
    assert not rnd.check_output(NATIVE, ncall, {"out": [{"t": "f32", "v": 6.0, "shape": []},
                                                        {"t": "b8", "v": 1, "shape": []}]})


def test_native_check_rejects_non_boolean_mask_entries(rnd):
    call = native_call([2.0, -4.0], "f32", 0.5, True, [True, False])
    obs = {"out": [{"t": "f32", "v": [4.0, 0.0], "shape": [2]}, {"t": "b8", "v": [2, 0], "shape": [2]}]}
    assert not rnd.check_output(NATIVE, call, obs)


def test_keep_fraction_reading(rnd):
    call = native_call([1.0, 2.0, 3.0, 4.0], "f32", 0.25, True, [True, False, True, True])
    r = rnd.READINGS["keep_fraction"](NATIVE, call)
    assert [Q(v) for v in r.fields["out"]] == [Q(3, 4), Q(3, 4)]


# ---------------------------------------------------------------- RANDOM: existing profiles unchanged
def _same(a, b, profile, call):
    ra, rb = evaluate(a.reference, profile, call), evaluate(b.reference, profile, call)
    assert ra.outcome == rb.outcome, call
    if ra.outcome == "value":
        assert norm(ra.fields["out"]) == norm(rb.fields["out"]), call


def test_existing_profiles_equal_validated(rnd, rnd_validated):
    rng = random.Random(3)
    dtypes = [("f32", f32), ("bf16", bf16), ("f16", lambda x: float(np.float16(x))), ("f64", float)]
    for _ in range(150):
        t, r = rng.choice(dtypes)
        shape = rng.choice([[2], [2, 3], [1, 2, 2], [1, 1, 2, 2], [], [1, 1, 1, 1, 2]])
        n = math.prod(shape)
        xs = reshape([r(rng.uniform(-4, 4)) for _ in range(n)], shape)
        p = rng.choice([0, 1, 0.0, 1.0, 0.5, "1/4", 0.1, 1.5, -0.1])
        train = rng.choice([True, False, 1])
        for op, kw in (("aten.dropout", None), ("torch.nn.functional.dropout", rng.choice([False, True]))):
            call = {"op": op, "args": [{"t": t, "v": xs}, p, train]}
            if kw is not None:
                call["kwargs"] = {"inplace": kw}
            _same(rnd, rnd_validated, "deterministic_forward", call)
        ms = reshape([rng.random() < 0.5 for _ in range(n)], shape)
        scale = rng.choice([1, "5/4", -0.5, 1 / 0.9, math.inf])
        call = {"op": "aten.native_dropout_backward", "args": [{"t": t, "v": xs}, {"t": "b8", "v": ms}, scale]}
        _same(rnd, rnd_validated, "finite_backward", call)
        if isinstance(scale, (int, str)) and 1 <= len(shape) <= 4:
            out = value(rnd, "finite_backward", call)
            assert flat(out) == [Q(x) * Q(scale) if m else 0 for x, m in zip(flat(xs), flat(ms))]
        # check_output agreement on an observed result
        obs = {"out": {"t": t, "v": xs, "shape": shape}}
        fcall = {"op": "aten.dropout", "args": [{"t": t, "v": xs}, 0.5, False]}
        try:
            a = rnd.check_output("deterministic_forward", fcall, obs)
        except Exception as e:  # noqa: BLE001
            a = type(e).__name__
        try:
            b = rnd_validated.check_output("deterministic_forward", fcall, obs)
        except Exception as e:  # noqa: BLE001
            b = type(e).__name__
        assert a == b


# ---------------------------------------------------------------- PyTorch 2.10 eager (supporting only)
torch = pytest.importorskip("torch")


def _rec(x):
    x = x.detach().cpu()
    t = {torch.float32: "f32", torch.bfloat16: "bf16", torch.bool: "b8", torch.int64: "i64"}[x.dtype]
    return {"t": t, "v": (x.double() if x.is_floating_point() else x).tolist(), "shape": list(x.shape)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eager_native_dropout_recorded_structure(rnd, dtype):
    g = torch.Generator().manual_seed(17)
    x = torch.randn((1, 8, 128), generator=g).to(dtype)
    out, mask = torch.ops.aten.native_dropout.default(x, 0.1, True)
    assert mask.dtype == torch.bool and out.dtype == dtype and mask.shape == x.shape
    ref_out, ref_mask = value(rnd, NATIVE, {"op": "aten.native_dropout", "args": [_rec(x), 0.1, True],
                                            "kwargs": {"observed_mask": _rec(mask)}})
    scale = 1 / (1 - Q(0.1))
    xs, ms, ys = flat(x.double().tolist()), flat(mask.tolist()), flat(out.double().tolist())
    assert flat(ref_mask) == [Q(int(m)) for m in ms]
    assert flat(ref_out) == [Q(v) * scale if m else 0 for v, m in zip(xs, ms)]
    rel = Q(1, 2 ** 6) if dtype == torch.bfloat16 else Q(1, 2 ** 21)       # eager is close, not equal
    for r, y, m in zip(flat(ref_out), ys, ms):
        assert (y == 0) if not m else abs(Q(y) - r) <= rel * abs(r)


def test_eager_factories(create):
    a = torch.ops.aten
    cases = [({"op": "aten.arange", "args": [128]}, a.arange.default(128)),
             ({"op": "aten.zeros", "args": [[0]], "kwargs": {"dtype": {"dtype": "bf16"}}},
              a.zeros.default([0], dtype=torch.bfloat16)),
             ({"op": "aten.ones", "args": [[128, 128]], "kwargs": {"dtype": {"dtype": "b8"}}},
              a.ones.default([128, 128], dtype=torch.bool)),
             ({"op": "aten.scalar_tensor", "args": [-math.inf], "kwargs": {"dtype": {"dtype": "f32"}}},
              a.scalar_tensor.default(-math.inf, dtype=torch.float32)),
             ({"op": "aten.arange", "args": [0, 2, 0.75], "kwargs": {"dtype": {"dtype": "f32"}}},
              a.arange.start_step(0, 2, 0.75, dtype=torch.float32))]
    for call, got in cases:
        assert create.check_output("explicit", call, {"out": _rec(got)}), call
    # eager returns an empty tensor for an infinite step; the draft says invalid_input (issue CR-2)
    assert a.arange.start_step(0, 5, math.inf, dtype=torch.float32).numel() == 0
