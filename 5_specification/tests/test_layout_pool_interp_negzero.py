"""LAYOUT draft LY.SIGNED_ZERO (2026-10-22): the real profile reads a received -0.0 as the real value 0.

Stages contracts/draft/LAYOUT_path_a_A2_negzero_20261022.{yaml,py} and checks the recorded path A2 slice_backward
structures (non-contiguous grad_output containing -0.0) against an independent placement loop and PyTorch eager.
"""
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
TAG = "path_a_A2_negzero_20261022"
INT64_MAX = 9223372036854775807


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    root = tmp_path_factory.mktemp("lpi_negzero")
    (root / "contracts").mkdir()
    (root / "adapters").mkdir()
    shutil.copy(DRAFT / f"LAYOUT_{TAG}.yaml", root / "contracts" / "LAYOUT.yaml")
    shutil.copy(DRAFT / f"LAYOUT_adapter_{TAG}.py", root / "adapters" / "LAYOUT.py")
    return registry.load("LAYOUT", contracts=root / "contracts", adapters=root / "adapters")


def run(layout, profile, op, *args):
    return evaluate(layout.adapter.reference, profile, {"op": op, "args": list(args), "kwargs": {}})


def test_decision_present(layout):
    ids = {d["id"] for d in layout.data["decisions"]}
    assert "LY.SIGNED_ZERO" in ids and "LY.SIGNED_ZERO" in layout.data["profiles"]["real"]["basis"]
    assert layout.data["review"]["status"] == "drafted"


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("start,end", [(32, INT64_MAX), (0, 32)])
def test_slice_backward_with_negative_zero(layout, dtype, start, end):
    grad = [[[[(-0.0 if (s + d) % 5 == 0 else float((s * 7 + d) % 9 - 4)) for d in range(32)]
              for s in range(4)] for _ in range(2)]]
    r = run(layout, "real", "aten.slice_backward", {"t": dtype, "v": grad, "shape": [1, 2, 4, 32]},
            [1, 2, 4, 64], 3, start, end, 1)
    assert r.outcome == "value", r.note
    offset = 32 if start == 32 else 0
    for h in range(2):
        for s in range(4):
            for d in range(64):
                want = grad[0][h][s][d - offset] if offset <= d < offset + 32 else 0
                assert r.fields["out"][0][h][s][d] == want      # -0.0 == 0: the sign of zero is not judged


def test_autograd_profile_still_refuses_negative_zero(layout):
    inp = {"op": "torch.nn.functional.pad", "input": {"t": "f32", "shape": [1], "v": [-0.0]}, "pad": [1, 0],
           "mode": "constant", "value": 0, "upstream": {"t": "f32", "shape": [2], "v": [1, 1]}}
    assert evaluate(layout.adapter.reference, "torch210_functional_constant_pad_autograd", inp).outcome == "invalid_input"


def test_recorded_noncontiguous_structure_against_eager(layout):
    torch = pytest.importorskip("torch")
    base = torch.randn(16384 * 1 + 64, generator=torch.Generator().manual_seed(7)).bfloat16()
    base[::3] = -0.0
    g = torch.as_strided(base, [1, 2, 128, 32], [16384, 64, 128, 1])
    assert (torch.signbit(g) & (g == 0)).any()
    y = torch.ops.aten.slice_backward.default(g, [1, 2, 128, 64], 3, 0, 32, 1)
    rec = {"t": "bf16", "v": g.float().tolist(), "shape": list(g.shape)}
    r = run(layout, "real", "aten.slice_backward", rec, [1, 2, 128, 64], 3, 0, 32, 1)
    assert r.outcome == "value", r.note
    assert np.array_equal(np.array(r.fields["out"], dtype=object).astype(float), y.float().numpy())
