"""Path A closure package C: ATen-call interfaces of the staged LAYOUT, POOL and INTERP drafts.

The drafts live in contracts/draft/<ID>_path_a_A2_20261020.yaml and <ID>_adapter_path_a_A2_20261020.py. Each test
stages them into tmp_path (<stage>/contracts/<ID>.yaml, <stage>/adapters/<ID>.py) and loads them with
kernel_spec.registry.load. Expected values are computed here by separate index loops, not by the adapters.
"""
from fractions import Fraction
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_spec import registry  # noqa: E402
from kernel_spec.num import evaluate  # noqa: E402

DRAFT = ROOT / "contracts" / "draft"
INT64_MAX = 9223372036854775807


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    root = tmp_path_factory.mktemp("lpi_stage")
    (root / "contracts").mkdir()
    (root / "adapters").mkdir()
    for cid in ("LAYOUT", "POOL", "INTERP"):
        shutil.copy(DRAFT / f"{cid}_path_a_A2_20261020.yaml", root / "contracts" / f"{cid}.yaml")
        shutil.copy(DRAFT / f"{cid}_adapter_path_a_A2_20261020.py", root / "adapters" / f"{cid}.py")
    return root


@pytest.fixture(scope="module")
def contracts(stage):
    return {cid: registry.load(cid, contracts=stage / "contracts", adapters=stage / "adapters")
            for cid in ("LAYOUT", "POOL", "INTERP")}


def call(contract, profile, op, *args, **kwargs):
    return evaluate(contract.adapter.reference, profile, {"op": op, "args": list(args), "kwargs": kwargs})


def rec(dtype, values):
    array = np.array(values, dtype=object)
    return {"t": dtype, "v": array.tolist(), "shape": list(array.shape)}


def frac(x):
    return Fraction(x)


# ----------------------------------------------------------------------------------------------- harness surface
def test_staged_adapters_read_their_own_contracts(contracts):
    for cid, names in {"LAYOUT": ["aten.pad", "aten.slice_backward", "aten.repeat_interleave", "aten.tril"],
                       "POOL": ["aten.avg_pool2d", "aten.avg_pool2d_backward"],
                       "INTERP": ["aten.upsample_nearest2d", "aten.upsample_nearest2d_backward"]}.items():
        c = contracts[cid]
        assert set(names) <= set(c.data["covers"])
        assert set(names) <= set(c.adapter.COVERS)
        assert "make_reference" in c.adapter_path.read_text()
        assert c.data["review"]["status"] == "drafted"
        assert c.data["activation"]["implementation_status"] == "pending"


def test_task_choice_flags(contracts):
    layout = contracts["LAYOUT"].data["profiles"]
    assert layout["real"]["task_choice_required"] is False
    assert layout["boolean"]["task_choice_required"] is False
    assert contracts["POOL"].data["profiles"]["torch210_aten_avg2d"]["task_choice_required"] is True
    assert contracts["INTERP"].data["profiles"]["torch210_aten_upsample_nearest2d"]["task_choice_required"] is False
    for cid, profile in [("LAYOUT", "boolean"), ("POOL", "torch210_aten_avg2d"),
                         ("INTERP", "torch210_aten_upsample_nearest2d")]:
        assert len(contracts[cid].data["profiles"][profile]["outputs"]) == 1


# ----------------------------------------------------------------------------------------------- LAYOUT
def test_pad_recorded_int64_labels(contracts):
    labels = [[(7 * j + 3) % 50257 for j in range(128)]]
    r = call(contracts["LAYOUT"], "real", "aten.pad", rec("i64", labels), [0, 1], "constant", -100.0)
    assert r.outcome == "value"
    assert r.fields["out"] == [labels[0] + [-100]]


@pytest.mark.parametrize("value,outcome", [(0.5, "unsupported"), (-2.7, "unsupported"), (1e30, "unsupported"),
                                           (3.0, "value"), (None, "value")])
def test_integer_fill_conversion(contracts, value, outcome):
    r = call(contracts["LAYOUT"], "real", "aten.pad", rec("i64", [1, 2]), [1, 0], "constant", value)
    assert r.outcome == outcome
    if outcome == "value":
        assert r.fields["out"] == [0 if value is None else int(value), 1, 2]


def test_boolean_profile_is_exclusive(contracts):
    layout = contracts["LAYOUT"]
    mask = rec("b8", [[True, True], [True, True]])
    assert call(layout, "real", "aten.tril", mask).outcome == "unsupported"
    assert call(layout, "boolean", "aten.tril", rec("f32", [[1, 1], [1, 1]])).outcome == "unsupported"
    r = call(layout, "boolean", "aten.tril", mask)
    assert r.outcome == "value" and r.fields == {"bool_out": [[True, False], [True, True]]}


def test_tril_recorded_bool_mask(contracts):
    mask = np.ones((128, 128), dtype=bool).tolist()
    r = call(contracts["LAYOUT"], "boolean", "aten.tril", rec("b8", mask))
    expected = [[j <= i for j in range(128)] for i in range(128)]
    assert r.outcome == "value" and r.fields["bool_out"] == expected


@pytest.mark.parametrize("start,end", [(32, INT64_MAX), (0, 32)])
def test_slice_backward_recorded_structure(contracts, start, end):
    heads = 2
    grad = [[[[(h * 997 + s * 31 + d) % 251 - 125 for d in range(32)] for s in range(128)] for h in range(heads)]]
    r = call(contracts["LAYOUT"], "real", "aten.slice_backward", rec("bf16", grad), [1, heads, 128, 64], 3,
             start, end, 1)
    assert r.outcome == "value"
    out = r.fields["out"]
    offset = 32 if start == 32 else 0
    for h in range(heads):
        for s in (0, 77, 127):
            row = out[0][h][s]
            assert len(row) == 64
            for d in range(64):
                inside = offset <= d < offset + 32
                assert row[d] == (grad[0][h][s][d - offset] if inside else 0)


def test_slice_backward_rejects_mismatched_grad(contracts):
    r = call(contracts["LAYOUT"], "real", "aten.slice_backward", rec("f32", [[1, 2, 3]]), [1, 4], 1, 0, 4, 1)
    assert r.outcome == "invalid_input"


def test_repeat_interleave_recorded_structure(contracts):
    x = [[[[(k * 64 + d) % 97 for d in range(4)] for _ in range(3)] for k in range(2)]]   # [1, 2, 3, 4]
    r = call(contracts["LAYOUT"], "real", "aten.repeat_interleave", rec("f32", x), 7, -3)
    assert r.outcome == "value"
    out = r.fields["out"]
    assert len(out[0]) == 14
    for j in range(14):
        assert out[0][j] == x[0][j // 7]


def test_layout_autograd_profiles_refuse_aten_calls(contracts):
    for profile in ("torch210_functional_constant_pad_autograd", "torch210_cat_autograd"):
        r = call(contracts["LAYOUT"], profile, "aten.pad", rec("f32", [1.0]), [1, 1])
        assert r.outcome == "unsupported"


# ----------------------------------------------------------------------------------------------- POOL
X16 = [[[[4 * i + j + 1 for j in range(4)] for i in range(4)]]]


def avg_expected(x, grad, k, s, ceil):
    """Independent R2 average pooling on one plane with padding 0: divisor = window clipped to the input."""
    h, w = len(x), len(x[0])

    def length(n):
        out = -(-(n - k) // s) + 1 if ceil else (n - k) // s + 1
        if ceil and (out - 1) * s >= n:
            out -= 1
        return out
    oh, ow = length(h), length(w)
    out = [[None] * ow for _ in range(oh)]
    gin = [[Fraction(0)] * w for _ in range(h)]
    for a in range(oh):
        for b in range(ow):
            rows = range(a * s, min(a * s + k, h))
            cols = range(b * s, min(b * s + k, w))
            div = len(rows) * len(cols)
            out[a][b] = Fraction(sum(x[i][j] for i in rows for j in cols), div)
            for i in rows:
                for j in cols:
                    gin[i][j] += Fraction(grad[a][b]) / div
    return out, gin


def test_avg_pool2d_recorded_D8(contracts):
    pool = contracts["POOL"]
    out, gin = avg_expected(X16[0][0], [[1, 2], [3, 4]], 3, 2, True)
    r = call(pool, "torch210_aten_avg2d", "aten.avg_pool2d", rec("f32", X16), [3, 3], [2, 2], [0, 0], True)
    assert r.outcome == "value" and r.fields["out"] == [[out]]
    assert out == [[6, Fraction(15, 2)], [12, Fraction(27, 2)]]
    r = call(pool, "torch210_aten_avg2d", "aten.avg_pool2d_backward", rec("f32", [[[[1, 2], [3, 4]]]]),
             rec("f32", X16), [3, 3], [2, 2], [0, 0], True, True, None)
    assert r.outcome == "value" and r.fields["out"] == [[gin]]


def test_avg_forward_and_backward_agree_with_autograd_profile(contracts):
    pool = contracts["POOL"]
    task = {"op": "avg_pool2d", "x": rec("f32", X16), "upstream": rec("f32", [[[[1, 2], [3, 4]]]]),
            "kernel_size": [3, 3], "stride": [2, 2], "padding": [0, 0], "ceil_mode": True,
            "count_include_pad": True, "divisor_override": None}
    full = evaluate(pool.adapter.reference, "finite_avg2d_autograd", task).fields
    fwd = call(pool, "torch210_aten_avg2d", "aten.avg_pool2d", rec("f32", X16), [3, 3], [2, 2], [0, 0], True)
    bwd = call(pool, "torch210_aten_avg2d", "aten.avg_pool2d_backward", rec("f32", [[[[1, 2], [3, 4]]]]),
               rec("f32", X16), [3, 3], [2, 2], [0, 0], True, True, None)
    assert fwd.fields["out"] == full["out"] and bwd.fields["out"] == full["grad"]


def test_max_profile_does_not_evaluate_average(contracts):
    r = call(contracts["POOL"], "finite_max2d_values", "aten.avg_pool2d", rec("f32", X16), [3, 3], [2, 2], [0, 0], True)
    assert r.outcome == "unsupported"
    r = call(contracts["POOL"], "finite_avg2d_autograd", "aten.avg_pool2d", rec("f32", X16), [3, 3])
    assert r.outcome == "unsupported"


# ----------------------------------------------------------------------------------------------- INTERP
def test_upsample_recorded_D18(contracts):
    x = [[[[100 * n + 10 * c + h + Fraction(w, 8) for w in range(8)] for h in range(10)] for c in range(3)]
         for n in range(2)]
    r = call(contracts["INTERP"], "torch210_aten_upsample_nearest2d", "aten.upsample_nearest2d",
             rec("f32", x), [20, 16], 2.0, 2.0)
    assert r.outcome == "value"
    out = r.fields["out"]
    for n in range(2):
        for c in range(3):
            for oh in range(20):
                for ow in range(16):
                    assert out[n][c][oh][ow] == x[n][c][oh // 2][ow // 2]


def test_upsample_scale_rules(contracts):
    interp = contracts["INTERP"]
    p = "torch210_aten_upsample_nearest2d"
    row = rec("f32", [[[list(range(11))]]])
    assert call(interp, p, "aten.upsample_nearest2d", row, [1, 2], 1.0, 0.25).fields["out"] == [[[[0, 4]]]]
    assert call(interp, p, "aten.upsample_nearest2d", row, [1, 2]).fields["out"] == [[[[0, 5]]]]
    assert call(interp, p, "aten.upsample_nearest2d", row, [1, 2], 1.0, 0.3).outcome == "unsupported"
    short = rec("f32", [[[[0, 1, 2]]]])
    assert call(interp, p, "aten.upsample_nearest2d", short, [1, 6], 1.0, 1.0).outcome == "unsupported"
    assert call(interp, p, "aten.upsample_nearest2d", short, None, [2.0, 2.0]).outcome == "unsupported"
    r = call(interp, "torch210_nearest2d_size_autograd", "aten.upsample_nearest2d", short, [1, 6])
    assert r.outcome == "unsupported"


def test_upsample_backward_is_scatter_add(contracts):
    g = [[[[1, 2, 3, 4], [5, 6, 7, 8]]]]
    r = call(contracts["INTERP"], "torch210_aten_upsample_nearest2d", "aten.upsample_nearest2d_backward",
             rec("f32", g), [2, 4], [1, 1, 1, 2], 2.0, 2.0)
    assert r.outcome == "value" and r.fields["out"] == [[[[1 + 2 + 5 + 6, 3 + 4 + 7 + 8]]]]
