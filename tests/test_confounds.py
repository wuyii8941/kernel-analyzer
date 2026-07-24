from __future__ import annotations

import unittest

from forkcert.confounds import infer_confound_checklist


class ConfoundsTest(unittest.TestCase):
    def test_self_gate_fail_without_metadata(self) -> None:
        cert = {
            "case_id": "c",
            "token_index": 0,
            "token_id": 1,
            "old_logp": -1.0,
            "logp_ref": -0.9,
            "logp_alt": -0.8,
            "advantage_sign": 1,
            "logprob_delta": 0.1,
            "delta_self_ref": 0.0,
            "delta_self_alt": 0.02,
            "metadata": {},
        }
        items = {item.name: item for item in infer_confound_checklist(cert)}
        self.assertEqual(items["delta_self_gate_passed"].status, "FAIL")
        self.assertEqual(items["deterministic_env_recorded"].status, "FAIL")
        self.assertEqual(items["same_token_compared"].status, "PASS")

    def test_self_gate_passes_with_small_self_delta(self) -> None:
        cert = {
            "case_id": "c",
            "token_index": 0,
            "token_id": 1,
            "old_logp": -1.0,
            "logp_ref": -0.9,
            "logp_alt": -0.8,
            "advantage_sign": -1,
            "logprob_delta": 0.1,
            "delta_self_ref": 0.001,
            "delta_self_alt": 0.001,
            "metadata": {
                "phase": "phase4_natural_scan",
                "phase1_metadata": {
                    "env": {
                        "torch": {
                            "cuda_available": True,
                            "deterministic_algorithms": True,
                            "deterministic_warn_only": True,
                            "cudnn_benchmark": False,
                        },
                        "deterministic_env": {
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "PYTHONHASHSEED": "0",
                        },
                    },
                    "phase1_gates": {"delta_self_ref_gate": True, "delta_self_alt_gate": True},
                },
            },
        }
        items = {item.name: item for item in infer_confound_checklist(cert)}
        self.assertEqual(items["delta_self_gate_passed"].status, "PASS")
        self.assertEqual(items["deterministic_env_recorded"].status, "PASS")

    def test_fingerprints_allow_auto_passes(self) -> None:
        cert = {
            "case_id": "c",
            "token_index": 0,
            "token_id": 1,
            "old_logp": -1.0,
            "logp_ref": -0.9,
            "logp_alt": -0.8,
            "advantage_sign": 1,
            "logprob_delta": 0.1,
            "delta_self_ref": 0.001,
            "delta_self_alt": 0.001,
            "metadata": {
                "phase": "phase4_natural_scan",
                "tokenization": {
                    "prompt_token_hash": "p",
                    "response_token_hash": "r",
                    "full_token_hash": "f",
                },
                "phase1_metadata": {
                    "env": {"torch": {"cuda_available": True}},
                    "phase1_gates": {"delta_self_ref_gate": True, "delta_self_alt_gate": True},
                    "model_artifact_fingerprint_ref": {
                        "verified_local_files": True,
                        "aggregate_sha256": "same",
                    },
                    "model_artifact_fingerprint_alt": {
                        "verified_local_files": True,
                        "aggregate_sha256": "same",
                    },
                    "execution_invariants": {
                        "model_eval_called": True,
                        "dropout_disabled_by_eval": True,
                        "default_position_ids_both_paths": True,
                        "default_causal_attention_mask_both_paths": True,
                    },
                    "config": {
                        "same_weights_config_expected": True,
                        "path_ref": {"model_name_or_path": "m", "dtype": "bf16", "device": "cuda"},
                        "path_alt": {"model_name_or_path": "m", "dtype": "bf16", "device": "cuda", "compile_model": True},
                    },
                },
            },
        }
        items = {item.name: item for item in infer_confound_checklist(cert)}
        self.assertEqual(items["tokenizer_identical"].status, "PASS")
        self.assertEqual(items["model_weights_identical"].status, "PASS")
        self.assertEqual(items["prompt_tokens_identical"].status, "PASS")
        self.assertEqual(items["response_tokens_identical"].status, "PASS")
        self.assertEqual(items["dtype_backend_only_intended_change"].status, "PASS")


if __name__ == "__main__":
    unittest.main()
