"""Independent review tests for the LAYOUT signed-zero draft (LY.SIGNED_ZERO, 2026-10-22).

Reviewer-written. Expected values come from the reviewer's own index loops over Fractions (a received -0.0 is the
real number 0, IEEE754.ZERO_COMPARE); the draft adapter is only called through kernel_spec.registry.load on a fresh
stage under tmp_path.
"""
from fractions import Fraction
from pathlib import Path
import random
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_spec import registry  # noqa: E402
from kernel_spec.num import evaluate  # noqa: E402

DRAFT = ROOT / "contracts" / "draft"
TAG = "path_a_A2_negzero_20261022"
INT64_MAX = 9223372036854775807
NZ = -0.0


@pytest.fixture(scope="module")
def L(tmp_path_factory):
    root = tmp_path_factory.mktemp("negzero_review")
    (root / "contracts").mkdir()
    (root / "adapters").mkdir()
    shutil.copy(DRAFT / f"LAYOUT_{TAG}.yaml", root / "contracts" / "LAYOUT.yaml")
    shutil.copy(DRAFT / f"LAYOUT_adapter_{TAG}.py", root / "adapters" / "LAYOUT.py")
    return registry.load("LAYOUT", contracts=root / "contracts", adapters=root / "adapters")


def run(c, profile, inp):
    return evaluate(c.adapter.reference, profile, inp)


def aten(op, *args, **kw):
    return {"op": op, "args": list(args), "kwargs": kw}


def shape_of(x):
    s = []
    while isinstance(x, list):
        s.append(len(x))
        if not x:
            break
        x = x[0]
    return s


def rec(dtype, v):
    return {"t": dtype, "v": v, "shape": shape_of(v)}


def build(shape, f, prefix=()):
    if not shape:
        return f(prefix)
    return [build(shape[1:], f, prefix + (i,)) for i in range(shape[0])]


def get(x, idx):
    for i in idx:
        x = x[i]
    return x


def real(x):
    return [real(v) for v in x] if isinstance(x, list) else Fraction(x)   # Fraction(-0.0) == 0


def same(observed, expected):
    if isinstance(expected, list):
        assert isinstance(observed, list) and len(observed) == len(expected)
        for o, e in zip(observed, expected):
            same(o, e)
    else:
        assert Fraction(observed) == Fraction(expected), (observed, expected)


def with_negzeros(shape, seed):
    rng = random.Random(seed)
    return build(shape, lambda i: rng.choice([NZ, NZ, 0.0, 1.0, -2.5, 3.0]))


# ------------------------------------------------------------------------- recorded slice_backward structures
@pytest.mark.parametrize("dtype,gshape,sizes,start,end", [
    ("bf16", [1, 2, 128, 32], [1, 2, 128, 64], 0, 32),           # qwen bf16 key 124
    ("bf16", [1, 14, 128, 32], [1, 14, 128, 64], 0, 32),         # qwen bf16 key 129
    ("f32", [1, 14, 128, 32], [1, 14, 128, 64], 0, 32),          # qwen fp32 key 150
    ("f32", [1, 2, 128, 32], [1, 2, 128, 64], 32, INT64_MAX)])
def test_slice_backward_with_negative_zeros(L, dtype, gshape, sizes, start, end):
    g = with_negzeros(gshape, sum(gshape) + start)
    r = run(L, "real", aten("aten.slice_backward", rec(dtype, g), sizes, 3, start, end, 1))
    assert r.outcome == "value", r
    s = 0 if start == 0 else 32
    exp = build(sizes, lambda i: Fraction(get(g, i[:3] + (i[3] - s,))) if s <= i[3] < s + 32 else Fraction(0))
    same(r.fields["out"], exp)


def test_view_pad_cat_repeat_tril_with_negative_zeros(L):
    x = [[NZ, 1.0], [2.0, NZ]]
    r = run(L, "real", aten("aten.view", rec("f32", x), [4]))
    same(r.fields["out"], [0, 1, 2, 0])
    r = run(L, "real", aten("aten.constant_pad_nd", rec("f32", x), [1, 0], NZ))
    same(r.fields["out"], [[0, 0, 1], [0, 2, 0]])
    r = run(L, "real", aten("aten.pad", rec("bf16", x), [0, 1], "constant", 5.0))
    same(r.fields["out"], [[0, 1, 5], [2, 0, 5]])
    r = run(L, "real", aten("aten.cat", [rec("f64", [NZ]), rec("f64", [7.0, NZ])], 0))
    same(r.fields["out"], [0, 7, 0])
    r = run(L, "real", aten("aten.repeat_interleave", rec("f16", [NZ, 1.0]), 2, 0))
    same(r.fields["out"], [0, 0, 1, 1])
    r = run(L, "real", aten("aten.tril", rec("f32", [[1.0, 2.0], [NZ, 3.0]])))
    same(r.fields["out"], [[1, 0], [0, 3]])


def test_output_check_ignores_sign_of_zero_but_not_values(L):
    inp = aten("aten.slice_backward", rec("f32", [[NZ, 2.0]]), [1, 4], 1, 0, 2, 1)
    chk = L.adapter.check_output

    def obs(v):   # observations are complete tensor records (make_output_checker)
        return {"out": {"t": "f32", "v": v, "shape": [1, 4]}}
    assert chk("real", inp, obs([[NZ, 2.0, 0.0, 0.0]])) is True      # sign kept
    assert chk("real", inp, obs([[0.0, 2.0, 0.0, 0.0]])) is True     # sign flipped: not detected (nonclaim)
    assert chk("real", inp, obs([[NZ, 2.0, NZ, 0.0]])) is True       # -0 fill: not detected (nonclaim)
    assert chk("real", inp, obs([[2.0, NZ, 0.0, 0.0]])) is False     # misplaced value: detected
    assert chk("real", inp, obs([[0.0, 0.0, NZ, 2.0]])) is False     # wrong offset: detected


def test_nan_still_invalid_in_real_profile(L):
    assert run(L, "real", aten("aten.clone", rec("f32", [float("nan")]))).outcome == "invalid_input"


def test_autograd_profiles_still_refuse_negative_zero(L):
    pad = {"op": "torch.nn.functional.pad", "input": {"t": "f32", "shape": [2], "v": [1.0, NZ]}, "pad": [1, 0],
           "mode": "constant", "value": 0, "upstream": {"t": "f32", "shape": [3], "v": [1, 1, 1]}}
    assert run(L, "torch210_functional_constant_pad_autograd", pad).outcome == "invalid_input"
    pad_up = dict(pad, input={"t": "f32", "shape": [2], "v": [1, 2]},
                  upstream={"t": "f32", "shape": [3], "v": [1, NZ, 1]})
    assert run(L, "torch210_functional_constant_pad_autograd", pad_up).outcome == "invalid_input"
    cat = {"op": "torch.cat", "tensors": [{"t": "f32", "shape": [1], "v": [NZ]}, {"t": "f32", "shape": [1], "v": [1]}],
           "dim": 0, "upstream": {"t": "f32", "shape": [2], "v": [1, 1]}}
    assert run(L, "torch210_cat_autograd", cat).outcome == "invalid_input"
    # Control: the same records without -0 are values.
    ok = dict(cat, tensors=[{"t": "f32", "shape": [1], "v": [0]}, {"t": "f32", "shape": [1], "v": [1]}])
    assert run(L, "torch210_cat_autograd", ok).outcome == "value"


def test_boolean_and_integer_profiles_unchanged(L):
    r = run(L, "boolean", aten("aten.tril", rec("b8", [[True, True], [True, True]])))
    assert r.fields["bool_out"] == [[True, False], [True, True]]
    assert run(L, "real", aten("aten.clone", rec("i64", [NZ]))).outcome == "invalid_input"   # not an integer literal
