from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_gates import check_phase0, check_phase15, check_phase3, check_phase5, check_phase6, check_phase6_twin


class CheckGatesTest(unittest.TestCase):
    def test_phase0_gate_rejects_synthetic_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase0.json"
            path.write_text(
                '{"late_minibatches":{"P(margin<0.01)":0.5},'
                '"provenance":{"canonical_real_training":false},'
                '"determinism":{"metadata_present":true,"warn_messages_recorded":true,"settings_verified":true}}\n',
                encoding="utf-8",
            )

            ok, message = check_phase0(path)

        self.assertFalse(ok)
        self.assertIn("canonical_real_training=False", message)

    def test_phase0_gate_accepts_real_grpo_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase0.json"
            path.write_text(
                '{"late_minibatches":{"P(margin<0.01)":0.002},'
                '"provenance":{"canonical_real_training":true},'
                '"determinism":{"metadata_present":true,"warn_messages_recorded":true,"settings_verified":true}}\n',
                encoding="utf-8",
            )

            ok, message = check_phase0(path)

        self.assertTrue(ok)
        self.assertIn("canonical_real_training=True", message)

    def test_phase3_gate_requires_actual_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase3.jsonl"
            path.write_text('{"fork_possible":true,"actual_fork":false}\n', encoding="utf-8")

            ok, message = check_phase3(path)

        self.assertFalse(ok)
        self.assertIn("actual_forks=0", message)

    def test_phase15_rejects_invocation_order_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase15.jsonl"
            path.write_text(
                "".join(f'{{"level":"L{level}","residual_layer_indexed":false}}\n' for level in range(1, 7)),
                encoding="utf-8",
            )

            ok, message = check_phase15(path)

        self.assertFalse(ok)
        self.assertIn("residual_layer_indexed=False", message)

    def test_phase3_gate_passes_with_actual_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase3.jsonl"
            calibration = Path(tmp) / "calibration.json"
            path.write_text(
                '{"fork_possible":true,"actual_fork":false}\n'
                '{"fork_possible":true,"actual_fork":true}\n',
                encoding="utf-8",
            )
            calibration.write_text(
                '{"model_kind":"empirical_independent_margin_delta_convolution",'
                '"margin_count":10,"delta_count":5}\n',
                encoding="utf-8",
            )

            ok, message = check_phase3(path, calibration)

        self.assertTrue(ok)
        self.assertIn("actual_forks=1", message)

    def test_phase6_claim_gate_rejects_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase6.jsonl"
            path.write_text(
                '{"actual_fork":true,"grad_contribution_diff":1.0,"grad_contribution_mode":"branch_proxy"}\n',
                encoding="utf-8",
            )

            ok, message = check_phase6(path, require_autograd=True)

        self.assertFalse(ok)
        self.assertIn("non_autograd=1", message)

    def test_phase5_requires_both_confusion_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase5.jsonl"
            path.write_text(
                '{"region":"bug","metadata":{"phase5_expected_bug":true,'
                '"bug":{"injection_kind":"kernel_execution"},'
                '"rollout_alignment":{"token_id_match":true}}}\n'
                '{"region":"stable","metadata":{"phase5_expected_bug":false,'
                '"rollout_alignment":{"token_id_match":true}}}\n',
                encoding="utf-8",
            )

            ok, message = check_phase5(path, require_token_match=True)

        self.assertTrue(ok)
        self.assertIn("TP=1", message)
        self.assertIn("TN=1", message)

    def test_phase6_claim_gate_accepts_autograd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase6.jsonl"
            path.write_text(
                '{"actual_fork":true,"grad_contribution_diff":2.5,"grad_contribution_mode":"hf_autograd"}\n',
                encoding="utf-8",
            )

            ok, message = check_phase6(path, require_autograd=True)

        self.assertTrue(ok)
        self.assertIn("non_autograd=0", message)

    def test_phase6_twin_required_after_natural_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase4 = Path(tmp) / "phase4.jsonl"
            summary = Path(tmp) / "twin.json"
            phase4.write_text('{"actual_fork":true}\n', encoding="utf-8")
            summary.write_text(
                '{"status":"completed","backend_only_difference":true,'
                '"exact_weight_divergence":true,"weight_scope":"full_model","optimizer_steps":200,'
                '"weight_measurements":41,"total_fork_events":3}\n',
                encoding="utf-8",
            )

            ok, message = check_phase6_twin(summary, phase4)

        self.assertTrue(ok)
        self.assertIn("fork_events=3", message)

    def test_phase6_twin_rejects_missing_coupling_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            phase4 = Path(tmp) / "phase4.jsonl"
            summary = Path(tmp) / "twin.json"
            phase4.write_text('{"actual_fork":true}\n', encoding="utf-8")
            summary.write_text('{"status":"not_triggered"}\n', encoding="utf-8")

            ok, _message = check_phase6_twin(summary, phase4)

        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
