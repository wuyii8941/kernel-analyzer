from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from theory_oracle.evaluate_qwen3_grpo_grad_event_bank_v0_4 import trajectory_record


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class QwenGradEventBankEvaluatorTests(unittest.TestCase):
    def fixture(self, root: Path, *, unstable: bool) -> tuple[Path, Path, Path]:
        token_rows = []
        state_rows = []
        for state_index in range(10):
            state_id = f"state-{state_index}"
            state_rows.append(
                {
                    "state_id": state_id,
                    "batch_size": 4,
                    "completion_length": 128,
                    "autograd_enabled": True,
                    "all_outputs_require_grad": True,
                    "accelerate_native_amp": True,
                    "accelerate_mixed_precision": "fp16",
                    "accelerate_forward_wrapped": True,
                    "attention_backend_locked": "MATH",
                    "candidate_identity_valid": True,
                    "gradients_preserved": True,
                    "tensor_versions_preserved": True,
                    "trainer_steps_preserved": True,
                    "rng_restored_exactly": True,
                }
            )
            for flat_index in range(512):
                reference = -0.25 if flat_index == 0 else -0.1
                candidate_first = -0.1
                candidate_second = -0.3 if unstable and flat_index == 0 else candidate_first
                token_rows.append(
                    {
                        "state_id": state_id,
                        "case_id": f"case-{state_index}-{flat_index // 128}",
                        "batch_index": flat_index // 128,
                        "flat_index": flat_index,
                        "token_index": flat_index % 128,
                        "token_id": flat_index + 1,
                        "logp_ref_first": reference,
                        "logp_ref_second": reference,
                        "logp_alt_first": candidate_first,
                        "logp_alt_second": candidate_second,
                        "old_logp": 0.0,
                        "advantage": -1.0,
                        "advantage_sign": -1,
                        "optimizer_step": state_index * 3 + 2,
                        "rollout_batch": state_index,
                        "policy_iteration": 2,
                    }
                )
        token_path, state_path, metadata_path = (
            root / "tokens.jsonl",
            root / "states.jsonl",
            root / "metadata.json",
        )
        write_jsonl(token_path, token_rows)
        write_jsonl(state_path, state_rows)
        metadata_path.write_text(
            json.dumps(
                {
                    "grad_compile_audit": {
                        "backend_compiles": 2,
                        "runtime_invocations": 30,
                        "graph_code_sha256": ["graph"],
                        "graph_node_counts": [1],
                    }
                }
            ),
            encoding="utf-8",
        )
        return token_path, state_path, metadata_path

    def test_stable_grad_context_events_are_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory), unstable=False)
            record, events, unstable = trajectory_record(
                name="A",
                token_path=paths[0],
                state_path=paths[1],
                metadata_path=paths[2],
                epsilon=0.2,
            )
        self.assertTrue(record["mechanics_valid"])
        self.assertEqual(len(events), 10)
        self.assertEqual(unstable, 0)
        self.assertTrue(all(event["flat_index"] == 0 for event in events))
        self.assertEqual(record["direction_1_to_0"], 10)

    def test_repeat_unstable_event_is_runtime_variability_not_stable_shift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory), unstable=True)
            record, events, unstable = trajectory_record(
                name="A",
                token_path=paths[0],
                state_path=paths[1],
                metadata_path=paths[2],
                epsilon=0.2,
            )
        self.assertTrue(record["mechanics_valid"])
        self.assertEqual(events, [])
        self.assertEqual(unstable, 10)
        self.assertEqual(
            record["within_state_runtime_variability"]["candidate_nonzero_repeat_tokens"],
            10,
        )


if __name__ == "__main__":
    unittest.main()
