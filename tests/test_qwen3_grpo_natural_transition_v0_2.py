import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen3_grpo_natural_transition_v0_2",
    ROOT / "theory_oracle" / "qwen3_grpo_natural_transition_v0_2.py",
)
EXECUTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EXECUTOR)

EVAL_SPEC = importlib.util.spec_from_file_location(
    "evaluate_qwen3_grpo_natural_transition_v0_2",
    ROOT / "theory_oracle" / "evaluate_qwen3_grpo_natural_transition_v0_2.py",
)
EVALUATOR = importlib.util.module_from_spec(EVAL_SPEC)
assert EVAL_SPEC.loader is not None
EVAL_SPEC.loader.exec_module(EVALUATOR)


class TestNaturalTransitionPureLogic(unittest.TestCase):
    def test_grpo_clipping_uses_advantage_direction(self):
        inputs = {
            "advantages": torch.tensor([1.0, -1.0]),
            "old_per_token_logps": torch.zeros((2, 2)),
            "completion_mask": torch.ones((2, 2)),
        }
        logps = torch.log(torch.tensor([[1.3, 0.7], [1.3, 0.7]]))
        decisions = EXECUTOR.clip_decisions(torch, logps, inputs, 0.2)
        self.assertEqual(decisions.tolist(), [[True, False], [False, True]])

    def test_scalar_profile_separates_effect_and_repeat_noise(self):
        profile = EVALUATOR.scalar_profile([1.0, 1.0], [1.2, 0.8])
        self.assertAlmostEqual(profile["B_signed_mean_effect"], 0.0)
        self.assertAlmostEqual(profile["N_paired_repeat_variance"], 0.08)
        self.assertEqual(profile["H_state_heterogeneity"], "UNIDENTIFIABLE_ONE_STATE")

    def test_vector_profile_keeps_direction_and_paired_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                "e1": torch.tensor([1.0, 2.0]),
                "e2": torch.tensor([1.0, 2.0]),
                "c1": torch.tensor([2.0, 1.0]),
                "c2": torch.tensor([2.0, 1.0]),
            }
            paths = {}
            for name, value in values.items():
                path = root / f"{name}.safetensors"
                save_file({"weight": value}, path)
                paths[name] = str(path)
            profile = EVALUATOR.vector_profile(
                {"eager": [paths["e1"], paths["e2"]], "compiled": [paths["c1"], paths["c2"]]}
            )
            self.assertAlmostEqual(profile["B_effect_l2"], math.sqrt(2.0))
            self.assertEqual(profile["N_paired_coordinate_variance_sum"], 0.0)
            self.assertAlmostEqual(profile["B_effect_alignment_with_reference"], -1 / math.sqrt(10.0))

    def test_vector_profile_detects_runtime_effect_variation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = [torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([-1.0])]
            paths = []
            for index, value in enumerate(tensors):
                path = root / f"{index}.safetensors"
                save_file({"weight": value}, path)
                paths.append(str(path))
            profile = EVALUATOR.vector_profile(
                {"eager": paths[:2], "compiled": paths[2:]}
            )
            self.assertEqual(profile["B_effect_l2"], 0.0)
            self.assertAlmostEqual(profile["N_paired_coordinate_variance_sum"], 2.0)

    def test_vector_profile_materializes_paired_u2_and_u1(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, value in {
                "e1": torch.tensor([1.0, 2.0]),
                "e2": torch.tensor([2.0, 0.0]),
                "c1": torch.tensor([2.0, 2.0]),
                "c2": torch.tensor([1.0, 0.0]),
            }.items():
                path = root / f"{name}.safetensors"
                save_file({"weight": value}, path)
                paths[name] = str(path)
            profile = EVALUATOR.vector_profile(
                {
                    "eager": [paths["e1"], paths["e2"]],
                    "compiled": [paths["c1"], paths["c2"]],
                },
                effect_vector_dir=root / "effects",
                artifact_prefix="update",
            )
            self.assertEqual(profile["paired_U1_repeats"], [0.2, -0.5])
            artifacts = profile["paired_effect_vector_artifacts"]
            self.assertEqual(len(artifacts), 2)
            self.assertTrue(all(Path(row["path"]).is_file() for row in artifacts))
            with safe_open(artifacts[0]["path"], framework="pt", device="cpu") as handle:
                self.assertEqual(handle.get_tensor("weight").tolist(), [1.0, 0.0])
            with safe_open(artifacts[1]["path"], framework="pt", device="cpu") as handle:
                self.assertEqual(handle.get_tensor("weight").tolist(), [-1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
