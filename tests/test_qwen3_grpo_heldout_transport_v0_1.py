import importlib.util
import numpy as np
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "theory_oracle/evaluate_qwen3_grpo_heldout_transport_v0_1.py"
SPEC = importlib.util.spec_from_file_location("heldout_transport", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EXECUTOR_PATH = ROOT / "scripts/phase0_grpo_train.py"
EXECUTOR_SPEC = importlib.util.spec_from_file_location("phase0_grpo_train", EXECUTOR_PATH)
assert EXECUTOR_SPEC is not None and EXECUTOR_SPEC.loader is not None
EXECUTOR = importlib.util.module_from_spec(EXECUTOR_SPEC)
EXECUTOR_SPEC.loader.exec_module(EXECUTOR)

SNAPSHOT_VERIFIER_PATH = ROOT / "theory_oracle/verify_qwen3_grpo_transition_snapshot_v0_1.py"
SNAPSHOT_VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "transition_snapshot_verifier", SNAPSHOT_VERIFIER_PATH
)
assert SNAPSHOT_VERIFIER_SPEC is not None and SNAPSHOT_VERIFIER_SPEC.loader is not None
SNAPSHOT_VERIFIER = importlib.util.module_from_spec(SNAPSHOT_VERIFIER_SPEC)
SNAPSHOT_VERIFIER_SPEC.loader.exec_module(SNAPSHOT_VERIFIER)


class HeldoutTransportEvaluatorTest(unittest.TestCase):
    def test_stage_partition_is_frozen(self):
        self.assertEqual(MODULE.stage_for_step(2), "early")
        self.assertEqual(MODULE.stage_for_step(14), "middle")
        self.assertEqual(MODULE.stage_for_step(29), "late")
        with self.assertRaises(ValueError):
            MODULE.stage_for_step(3)

    def test_boundary_bands_are_half_open(self):
        self.assertEqual(MODULE.boundary_band(0.0), "[0,1e-4)")
        self.assertEqual(MODULE.boundary_band(1e-4), "[1e-4,1e-3)")
        self.assertEqual(MODULE.boundary_band(1e-3), "[1e-3,1e-2)")
        self.assertEqual(MODULE.boundary_band(1e-2), "[1e-2,inf)")

    def test_transport_label_does_not_promote_one_run(self):
        self.assertEqual(
            MODULE.transport_label(valid=True, unstable=0, event_runs=1),
            "INDETERMINATE_SINGLE_EVENT_RUN",
        )
        self.assertEqual(
            MODULE.transport_label(valid=True, unstable=0, event_runs=2),
            "SUPPORTED_EVENT_REPLICATION",
        )
        self.assertEqual(
            MODULE.transport_label(valid=False, unstable=0, event_runs=3),
            "INVALID",
        )

    def test_transition_capture_tree_is_detached_cpu_copy(self):
        source = {"x": torch.tensor([1.0, 2.0]), "nested": [torch.tensor([3])], "tag": "a"}
        cloned = EXECUTOR.cpu_clone_tree(source)
        source["x"][0] = 9.0
        self.assertEqual(cloned["x"].device.type, "cpu")
        self.assertEqual(cloned["x"].tolist(), [1.0, 2.0])
        self.assertEqual(cloned["nested"][0].tolist(), [3])
        self.assertEqual(cloned["tag"], "a")

    def test_transition_tensor_manifest_is_path_stable(self):
        value = {"b": torch.zeros(2, dtype=torch.float16), "a": [torch.ones(1)]}
        rows = EXECUTOR.tree_tensor_manifest(value)
        self.assertEqual([row["path"] for row in rows], ["a[0]", "b"])
        self.assertEqual(rows[1]["shape"], [2])
        self.assertEqual(rows[1]["dtype"], "torch.float16")

    def test_capture_and_verifier_tensor_manifests_agree(self):
        value = {"z": (torch.tensor([2], dtype=torch.int64),), "a": torch.tensor([1.5])}
        self.assertEqual(
            EXECUTOR.tree_tensor_manifest(value),
            SNAPSHOT_VERIFIER.tensor_manifest(value),
        )

    def test_numpy_rng_equality_is_value_based(self):
        state = np.random.get_state()
        copied = (state[0], state[1].copy(), *state[2:])
        self.assertTrue(EXECUTOR.numpy_rng_equal(state, copied))
        copied[1][0] ^= 1
        self.assertFalse(EXECUTOR.numpy_rng_equal(state, copied))


if __name__ == "__main__":
    unittest.main()
