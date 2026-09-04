from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_loss_directions.py"
SPEC = importlib.util.spec_from_file_location("evaluate_loss_directions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_normalized_like_matches_requested_norm() -> None:
    value = torch.tensor([3.0, 4.0])
    result = MODULE.normalized_like(value, 2.5)
    assert torch.isclose(torch.linalg.vector_norm(result.double()), torch.tensor(2.5, dtype=torch.float64))


def test_load_tensor_accepts_one_tensor_state_dictionary(tmp_path: Path) -> None:
    path = tmp_path / "value.pt"
    expected = torch.tensor([1.0, 2.0])
    torch.save({"parameter": expected}, path)
    assert torch.equal(MODULE.load_tensor(path), expected)


def test_load_tensor_selects_named_parameter(tmp_path: Path) -> None:
    path = tmp_path / "state.pt"
    expected = torch.tensor([3.0, 4.0])
    torch.save({"first": torch.tensor([1.0]), "target": expected}, path)
    assert torch.equal(MODULE.load_tensor(path, "target"), expected)
