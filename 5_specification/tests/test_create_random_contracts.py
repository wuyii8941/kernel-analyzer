"""Draft CREATE and RANDOM (native_forward_given_mask) contracts of path A closure package E.

The drafts live in contracts/draft/*_path_a_A2_20261020.*; they are staged under tmp_path as <stage>/contracts/<ID>.yaml
and <stage>/adapters/<ID>.py. Expected values below come from integer / Fraction arithmetic written here, not from
the adapters. PyTorch 2.10 eager results on the recorded argument structures are supporting evidence only.
"""
from __future__ import annotations

import math
import shutil
import sys
from fractions import Fraction as Q
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC))
sys.path.insert(0, str(SPEC / "tools"))
from kernel_spec import registry  # noqa: E402
from kernel_spec import compare as C  # noqa: E402

DRAFT = SPEC / "contracts" / "draft"
FILES = {"CREATE": ("CREATE_path_a_A2_20261020.yaml", "CREATE_adapter_path_a_A2_20261020.py"),
         "RANDOM": ("RANDOM_path_a_A2_20261020.yaml", "RANDOM_adapter_path_a_A2_20261020.py")}
NATIVE = "native_forward_given_mask"


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    root = tmp_path_factory.mktemp("spec_stage_E")
    (root / "contracts").mkdir()
    (root / "adapters").mkdir()
    for cid, (yaml_name, py_name) in FILES.items():
        shutil.copy(DRAFT / yaml_name, root / "contracts" / f"{cid}.yaml")
        shutil.copy(DRAFT / py_name, root / "adapters" / f"{cid}.py")
    return root


@pytest.fixture(scope="module")
def create(stage):
    return registry.load("CREATE", contracts=stage / "contracts", adapters=stage / "adapters").adapter


@pytest.fixture(scope="module")
def random_(stage):
    return registry.load("RANDOM", contracts=stage / "contracts", adapters=stage / "adapters").adapter


def test_checker_consistent(stage):
    import json
    import check_contracts
    sources = json.loads(registry.SOURCES.read_text())
    for cid in FILES:
        rep = check_contracts.check_contract(stage / "contracts" / f"{cid}.yaml", sources, stage / "adapters")
        assert rep["verdict"] == C.CONSISTENT, (cid, rep["structure"], [x for x in rep["examples"]
                                                                      if x["verdict"] != C.CONSISTENT],
                                               [m for m in rep["mutations"] if not m["rejected"]])
        assert all(m["rejected"] for m in rep["mutations"])


def test_adapters_read_their_staged_contract(create, random_, stage):
    assert "aten.scalar_tensor" in create.COVERS
    assert "aten.native_dropout" in random_.COVERS


def test_harness_conventions(random_):
    assert random_.ATEN_RETURNS == {"aten.native_dropout": {"positions": 2, "not_compared": {}}}
    assert random_.ATEN_OBSERVED_INPUTS == {"aten.native_dropout": {"argument": "observed_mask", "return_position": 1}}


# ---------------------------------------------------------------- CREATE on the recorded structures
def test_recorded_arange(create):
    r = create.reference("explicit", {"op": "aten.arange", "args": [128]})
    assert r.fields["out"] == list(range(128))
    obs = {"out": {"t": "i64", "v": list(range(128)), "shape": [128]}}
    assert create.check_output("explicit", {"op": "aten.arange", "args": [128]}, obs)
    assert not create.check_output("explicit", {"op": "aten.arange", "args": [128]},
                                   {"out": {"t": "f32", "v": [float(i) for i in range(128)], "shape": [128]}})


def test_arange_length_rule(create):
    for start, end, step in [(0, 7, 3), (2, 11, 4), (10, -3, -4), (0, 1, 5), (-5, 5, 1)]:
        n = -(-(end - start) // step)          # ceil for integers
        expected = [start + i * step for i in range(max(n, 0))]
        got = create.reference("explicit", {"op": "aten.arange", "args": [start, end, step]}).fields["out"]
        assert got == expected


def test_arange_float_inference(create):
    from kernel_spec.num import evaluate
    call = {"op": "aten.arange", "args": [0, 1.25, 0.5]}
    assert evaluate(create.reference, "explicit", call).outcome == "unsupported"
    got = create.reference("default_float32", call).fields["out"]
    assert got == [Q(0), Q(1, 2), Q(1)]                       # ceil(5/2) = 3 values


def test_recorded_zeros_ones_scalar(create):
    for d in ("bf16", "f32"):
        call = {"op": "aten.zeros", "args": [[0]], "kwargs": {"dtype": {"dtype": d}}}
        assert create.reference("explicit", call).fields["out"] == []
        assert create.check_output("explicit", call, {"out": {"t": d, "v": [], "shape": [0]}})
        assert not create.check_output("explicit", call, {"out": {"t": d, "v": [0.0], "shape": [1]}})
    call = {"op": "aten.ones", "args": [[128, 128]], "kwargs": {"dtype": {"dtype": "b8"}}}
    out = create.reference("explicit", call).fields["out"]             # Boolean true in the 1 / 0 encoding
    assert len(out) == 128 and all(len(row) == 128 and all(v == 1 for v in row) for row in out)
    good = {"out": {"t": "b8", "v": [[True] * 128] * 128, "shape": [128, 128]}}
    assert create.check_output("explicit", call, good)
    assert not create.check_output("explicit", call, {"out": {"t": "i64", "v": [[1] * 128] * 128, "shape": [128, 128]}})
    for arg in ("-inf", float("-inf")):
        call = {"op": "aten.scalar_tensor", "args": [arg], "kwargs": {"dtype": {"dtype": "f32"}}}
        assert create.reference("explicit", call).fields["out"] == "-inf"
    call = {"op": "aten.scalar_tensor", "args": [0.0], "kwargs": {"dtype": {"dtype": "f32"}}}
    assert create.reference("explicit", call).fields["out"] == 0


def test_recorded_ones_like(create):
    call = {"op": "aten.ones_like", "args": [{"t": "f32", "v": 2.5}]}
    assert create.reference("explicit", call).fields["out"] == 1
    assert create.check_output("explicit", call, {"out": {"t": "f32", "v": 1.0, "shape": []}})
    assert not create.check_output("explicit", call, {"out": {"t": "bf16", "v": 1.0, "shape": []}})
    call = {"op": "aten.ones_like", "args": [{"t": "f32", "v": [[[[float("nan"), 1.0], [2.0, -3.0]]]]}]}
    assert create.reference("explicit", call).fields["out"] == [[[[1, 1], [1, 1]]]]


# ---------------------------------------------------------------- RANDOM native_dropout given the mask
def native(adapter, x, p, train, mask=None):
    call = {"op": "aten.native_dropout", "args": [x, p, train]}
    if mask is not None:
        call["kwargs"] = {"observed_mask": mask}
    return adapter.reference(NATIVE, call).fields["out"]


def test_native_scaling_independent(random_):
    p = 0.1
    keep = 1 - Q(p)                                          # exact received double
    xs = [1.5, -2.0, 0.25, 3.0]
    ms = [True, False, True, True]
    out, mask = native(random_, {"t": "f32", "v": xs}, p, True, {"t": "b8", "v": ms})
    assert out == [Q(x) / keep if m else 0 for x, m in zip(xs, ms)]
    assert mask == [1, 0, 1, 1]


def test_native_deterministic_cases(random_):
    x = {"t": "bf16", "v": [[1.0, -2.0]]}
    assert native(random_, x, 0.5, False) == [[[1, -2]], [[1, 1]]]
    assert native(random_, x, 1.0, True) == [[[0, 0]], [[0, 0]]]
    assert native(random_, x, 0.0, True) == [[[1, -2]], [[1, 1]]]


def test_native_check_output_compares_mask(random_):
    call = {"op": "aten.native_dropout", "args": [{"t": "f32", "v": [2.0, 4.0]}, 0.5, False]}
    good = {"out": [{"t": "f32", "v": [2.0, 4.0], "shape": [2]}, {"t": "b8", "v": [True, True], "shape": [2]}]}
    bad = {"out": [{"t": "f32", "v": [2.0, 4.0], "shape": [2]}, {"t": "b8", "v": [True, False], "shape": [2]}]}
    assert random_.check_output(NATIVE, call, good)
    assert not random_.check_output(NATIVE, call, bad)


# ---------------------------------------------------------------- PyTorch 2.10 eager cross-check (supporting)
torch = pytest.importorskip("torch")
DT = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16", torch.float64: "f64", torch.int64: "i64",
      torch.bool: "b8"}


def record(t):
    t = t.detach().cpu()
    values = (t.double() if t.is_floating_point() else t).tolist()
    return {"t": DT[t.dtype], "v": values, "shape": list(t.shape)}


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_torch_factories(create, device):
    a = torch.ops.aten
    cases = [
        ("explicit", {"op": "aten.arange", "args": [128]}, a.arange.default(128, device=device)),
        ("explicit", {"op": "aten.arange", "args": [5, 1, -2]}, a.arange.start_step(5, 1, -2, device=device)),
        ("explicit", {"op": "aten.zeros", "args": [[0]], "kwargs": {"dtype": {"dtype": "bf16"}}},
         a.zeros.default([0], dtype=torch.bfloat16, device=device)),
        ("explicit", {"op": "aten.zeros", "args": [[0]], "kwargs": {"dtype": {"dtype": "f32"}}},
         a.zeros.default([0], dtype=torch.float32, device=device)),
        ("explicit", {"op": "aten.ones", "args": [[128, 128]], "kwargs": {"dtype": {"dtype": "b8"}}},
         a.ones.default([128, 128], dtype=torch.bool, device=device)),
        ("explicit", {"op": "aten.scalar_tensor", "args": ["-inf"], "kwargs": {"dtype": {"dtype": "f32"}}},
         a.scalar_tensor.default(float("-inf"), dtype=torch.float32, device=device)),
        ("explicit", {"op": "aten.scalar_tensor", "args": [0.0], "kwargs": {"dtype": {"dtype": "f32"}}},
         a.scalar_tensor.default(0.0, dtype=torch.float32, device=device)),
    ]
    for shape in ([], [1, 1, 2, 2]):
        x = torch.randn(shape, device=device)
        cases.append(("explicit", {"op": "aten.ones_like", "args": [record(x)]}, a.ones_like.default(x)))
    for profile, call, got in cases:
        assert create.check_output(profile, call, {"out": record(got)}), call["op"]


def test_torch_refusals(create):
    from kernel_spec.num import evaluate
    with pytest.raises(RuntimeError):
        torch.ops.aten.arange.start_step(5, 1, 1)
    assert evaluate(create.reference, "explicit", {"op": "aten.arange", "args": [5, 1, 1]}).outcome == \
        "invalid_input"
    with pytest.raises(RuntimeError):
        torch.ops.aten.arange.start_step(0, 5, 0)
    assert evaluate(create.reference, "explicit", {"op": "aten.arange", "args": [0, 5, 0]}).outcome == \
        "invalid_input"


# Supporting tolerance for eager rounding: CPU bf16 eager rounds the scale 1/(1-p) to bfloat16 before multiplying
# (relative error up to 2^-9) and then rounds the product (2^-8 relative at most), so 2^-7 bounds it; f32 rounds the
# scale to float32 and the product (2^-24 each).
TOL = {"f32": Q(1, 2 ** 22), "bf16": Q(1, 2 ** 7)}


@pytest.mark.parametrize("dtype,shape,device", [
    (torch.float32, (1, 128, 128), "cpu"), (torch.bfloat16, (1, 128, 128), "cpu"),
    (torch.float32, (1, 8, 768), "cpu"), (torch.bfloat16, (1, 8, 768), "cpu")]
    + ([(torch.float32, (1, 128, 128), "cuda"), (torch.bfloat16, (1, 16, 768), "cuda")]
       if torch.cuda.is_available() else []))
def test_torch_native_dropout_given_mask(random_, dtype, shape, device):
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(shape, generator=gen).to(dtype).to(device)
    out, mask = torch.ops.aten.native_dropout.default(x, 0.1, True)
    assert mask.dtype == torch.bool and mask.shape == x.shape and out.dtype == x.dtype
    ref_out, ref_mask = native(random_, record(x), 0.1, True, record(mask))
    flat_ref = [v for row in ref_out[0] for v in row]
    flat_mask = [v for row in ref_mask[0] for v in row]
    flat_got = out.double().cpu().flatten().tolist()
    flat_keep = mask.cpu().flatten().tolist()
    assert flat_mask == [1 if k else 0 for k in flat_keep]
    tol = TOL[DT[dtype]]
    for r, g, k in zip(flat_ref, flat_got, flat_keep):
        if not k:
            assert r == 0 and g == 0
        else:
            assert abs(Q(g) - r) <= tol * abs(r)
    kept = sum(flat_keep) / len(flat_keep)
    assert abs(kept - 0.9) < 0.05                               # diagnostic plausibility, not the verdict
