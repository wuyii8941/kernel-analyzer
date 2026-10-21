"""PyTorch 2.10 eager cross-check (supporting evidence, not the definition) of the recorded path A argument
structures for package C: aten.pad, aten.slice_backward, aten.repeat_interleave.self_int, aten.tril (b8),
aten.avg_pool2d / avg_pool2d_backward (trigger D8) and aten.upsample_nearest2d (trigger D18).

Tensors are built with the recorded sizes and strides. Data-movement results must agree exactly; average pooling is
compared with the exact reference rounded once to float32 (the backward sums may differ by float32 rounding).
"""
from fractions import Fraction
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kernel_spec import registry  # noqa: E402
from kernel_spec.num import evaluate  # noqa: E402

DRAFT = ROOT / "contracts" / "draft"


def draft_tag(cid):
    # LAYOUT_DRAFT selects a later LAYOUT draft pair (e.g. path_a_A2_negzero_20261022); default: the A2 package C pair.
    import os
    return os.environ.get("LAYOUT_DRAFT", "path_a_A2_20261020") if cid == "LAYOUT" else "path_a_A2_20261020"
A = torch.ops.aten
NAMES = {torch.float32: "f32", torch.bfloat16: "bf16", torch.int64: "i64", torch.bool: "b8"}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def contracts(tmp_path_factory):
    root = tmp_path_factory.mktemp("lpi_torch_stage")
    (root / "contracts").mkdir()
    (root / "adapters").mkdir()
    for cid in ("LAYOUT", "POOL", "INTERP"):
        shutil.copy(DRAFT / f"{cid}_{draft_tag(cid)}.yaml", root / "contracts" / f"{cid}.yaml")
        shutil.copy(DRAFT / f"{cid}_adapter_{draft_tag(cid)}.py", root / "adapters" / f"{cid}.py")
    return {cid: registry.load(cid, contracts=root / "contracts", adapters=root / "adapters")
            for cid in ("LAYOUT", "POOL", "INTERP")}


def rec(t):
    t = t.detach().cpu()
    return {"t": NAMES[t.dtype], "v": (t.float() if t.dtype == torch.bfloat16 else t).tolist(), "shape": list(t.shape)}


def strided(shape, stride, dtype, device, gen):
    span = 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    if dtype == torch.int64:
        base = torch.randint(0, 50257, (span,), generator=gen)
    else:
        base = (torch.randint(-64, 64, (span,), generator=gen).float() / 8).to(dtype)
    return torch.as_strided(base.to(device), shape, stride)


def reference(contract, profile, op, args):
    r = evaluate(contract.adapter.reference, profile,
                 {"op": op, "args": [rec(a) if torch.is_tensor(a) else a for a in args], "kwargs": {}})
    assert r.outcome == "value", r.note
    (field, value), = r.fields.items()
    return value


def as_float(nested):
    return np.array(nested, dtype=object).astype(float)


@pytest.mark.parametrize("device", DEVICES)
def test_pad_labels(contracts, device):
    gen = torch.Generator().manual_seed(1)
    x = strided([1, 128], [4302, 1], torch.int64, device, gen)
    y = A.pad.default(x, [0, 1], "constant", -100.0)
    assert y.dtype == torch.int64 and list(y.shape) == [1, 129]
    assert reference(contracts["LAYOUT"], "real", "aten.pad", [x, [0, 1], "constant", -100.0]) == y.cpu().tolist()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype,heads,stride,start,end", [
    (torch.bfloat16, 2, [8192, 32, 64, 1], 32, 9223372036854775807),
    (torch.bfloat16, 2, [16384, 64, 128, 1], 0, 32),
    (torch.float32, 14, [57344, 4096, 32, 1], 32, 9223372036854775807),
    (torch.float32, 14, [114688, 8192, 64, 1], 0, 32)])
def test_slice_backward(contracts, device, dtype, heads, stride, start, end):
    gen = torch.Generator().manual_seed(2)
    g = strided([1, heads, 128, 32], stride, dtype, device, gen)
    y = A.slice_backward.default(g, [1, heads, 128, 64], 3, start, end, 1)
    expected = reference(contracts["LAYOUT"], "real", "aten.slice_backward", [g, [1, heads, 128, 64], 3, start, end, 1])
    assert y.dtype == dtype and np.array_equal(as_float(expected), y.float().cpu().numpy())


@pytest.mark.parametrize("device", DEVICES)
def test_repeat_interleave(contracts, device):
    gen = torch.Generator().manual_seed(3)
    x = strided([1, 2, 128, 64], [16384, 64, 128, 1], torch.float32, device, gen)
    y = A.repeat_interleave.self_int(x, 7, -3)
    expected = reference(contracts["LAYOUT"], "real", "aten.repeat_interleave", [x, 7, -3])
    assert list(y.shape) == [1, 14, 128, 64] and np.array_equal(as_float(expected), y.cpu().numpy())


@pytest.mark.parametrize("device", DEVICES)
def test_tril_bool(contracts, device):
    m = torch.rand(128, 128, generator=torch.Generator().manual_seed(4)).gt(0.3).to(device)
    y = A.tril.default(m)
    assert y.dtype == torch.bool
    assert reference(contracts["LAYOUT"], "boolean", "aten.tril", [m]) == y.cpu().tolist()


@pytest.mark.parametrize("device", DEVICES)
def test_avg_pool2d_D8(contracts, device):
    pool = contracts["POOL"]
    gen = torch.Generator().manual_seed(5)
    x = strided([1, 1, 4, 4], [16, 16, 4, 1], torch.float32, device, gen)
    g = strided([1, 1, 2, 2], [4, 4, 2, 1], torch.float32, device, gen)
    y = A.avg_pool2d.default(x, [3, 3], [2, 2], [0, 0], True)
    gi = A.avg_pool2d_backward.default(g, x, [3, 3], [2, 2], [0, 0], True, True, None)
    fwd = reference(pool, "torch210_aten_avg2d", "aten.avg_pool2d", [x, [3, 3], [2, 2], [0, 0], True])
    bwd = reference(pool, "torch210_aten_avg2d", "aten.avg_pool2d_backward",
                    [g, x, [3, 3], [2, 2], [0, 0], True, True, None])
    rounded = as_float(fwd).astype(np.float32)
    assert np.array_equal(rounded, y.cpu().numpy())
    assert np.allclose(as_float(bwd), gi.cpu().numpy(), rtol=2 ** -21, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_upsample_nearest2d_D18(contracts, device):
    gen = torch.Generator().manual_seed(6)
    x = strided([2, 3, 10, 8], [240, 80, 1, 10], torch.float32, device, gen)
    y = A.upsample_nearest2d.default(x, [20, 16], 2.0, 2.0)
    expected = reference(contracts["INTERP"], "torch210_aten_upsample_nearest2d", "aten.upsample_nearest2d",
                         [x, [20, 16], 2.0, 2.0])
    assert np.array_equal(as_float(expected), y.cpu().numpy())
