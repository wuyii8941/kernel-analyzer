from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "theory_oracle" / "evaluate_update_aligned_shift_v0_1.py"
SPEC = importlib.util.spec_from_file_location("aligned_shift", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def save_pair(tmp_path: Path, reference: list[float], candidate: list[float]) -> tuple[str, str]:
    import torch
    from safetensors.torch import save_file

    ref_path = tmp_path / "reference.safetensors"
    cand_path = tmp_path / "candidate.safetensors"
    save_file({"weight": torch.tensor(reference)}, ref_path)
    save_file({"weight": torch.tensor(candidate)}, cand_path)
    return str(ref_path), str(cand_path)


class UpdateAlignedShiftTest(unittest.TestCase):
    def test_aligned_and_orthogonal_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref, cand = save_pair(Path(tmp), [1.0, 0.0], [1.5, 0.5])
            result = MODULE.evaluate_pair("s", ref, cand, chunk_elements=1)
        self.assertEqual(result["status"], "VALID")
        self.assertAlmostEqual(result["aligned_shift"], 0.5)
        self.assertAlmostEqual(result["parallel_relative_l2"], 0.5)
        self.assertAlmostEqual(result["orthogonal_relative_l2"], 0.5)
        self.assertAlmostEqual(result["relative_discrepancy_l2"], math.sqrt(0.5))

    def test_opposite_shift_is_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref, cand = save_pair(Path(tmp), [2.0, 0.0], [1.0, 0.0])
            result = MODULE.evaluate_pair("s", ref, cand, chunk_elements=2)
        self.assertAlmostEqual(result["aligned_shift"], -0.5)
        self.assertAlmostEqual(result["orthogonal_relative_l2"], 0.0)

    def test_zero_reference_is_undefined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ref, cand = save_pair(Path(tmp), [0.0, 0.0], [1.0, 0.0])
            result = MODULE.evaluate_pair("s", ref, cand, chunk_elements=2)
        self.assertEqual(result["status"], "UNDEFINED_ZERO_REFERENCE_UPDATE")
        self.assertIsNone(result["aligned_shift"])


if __name__ == "__main__":
    unittest.main()
