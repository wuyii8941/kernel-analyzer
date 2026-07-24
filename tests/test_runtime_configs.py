from __future__ import annotations

import unittest

from scripts.resolve_runtime_configs import resolve_config, resolve_value


class RuntimeConfigResolutionTest(unittest.TestCase):
    def test_t4_resolution_switches_precision_backend_and_reduction_flag(self) -> None:
        source = {
            "path_alt": {
                "name": "hf-compile-bf16-flash",
                "dtype": "bf16",
                "attention_backend": "flash",
                "allow_bf16_reduced_precision_reduction": True,
            },
            "model": {"dtype": "bfloat16"},
        }

        resolved = resolve_value(source, use_bf16=False)

        self.assertEqual(resolved["path_alt"]["name"], "hf-compile-fp16-efficient")
        self.assertEqual(resolved["path_alt"]["dtype"], "fp16")
        self.assertEqual(resolved["path_alt"]["attention_backend"], "efficient")
        self.assertTrue(resolved["path_alt"]["allow_fp16_reduced_precision_reduction"])
        self.assertNotIn("allow_bf16_reduced_precision_reduction", resolved["path_alt"])
        self.assertEqual(resolved["model"]["dtype"], "float16")

    def test_ampere_resolution_preserves_bf16(self) -> None:
        source = {"path_alt": {"dtype": "bf16", "attention_backend": "flash"}}
        self.assertEqual(resolve_value(source, use_bf16=True), source)

    def test_t4_sdpa_pair_uses_executable_eager_vs_math_paths(self) -> None:
        source = {
            "path_ref": {
                "name": "hf-eager-bf16-sdpa-math",
                "dtype": "bf16",
                "attn_implementation": "sdpa",
                "attention_backend": "math",
            },
            "path_alt": {
                "name": "hf-eager-bf16-sdpa-flash",
                "dtype": "bf16",
                "attn_implementation": "sdpa",
                "attention_backend": "flash",
            },
        }

        resolved = resolve_config("configs/hf_sdpa_math_flash.example.yaml", source, use_bf16=False)

        self.assertEqual(resolved["path_ref"]["dtype"], "fp16")
        self.assertEqual(resolved["path_ref"]["attn_implementation"], "eager")
        self.assertIsNone(resolved["path_ref"]["attention_backend"])
        self.assertEqual(resolved["path_alt"]["attn_implementation"], "sdpa")
        self.assertEqual(resolved["path_alt"]["attention_backend"], "math")

    def test_materialization_probe_uses_opposite_low_precision_format(self) -> None:
        source = {
            "path_ref": {"dtype": "bf16"},
            "path_alt": {"name": "forced", "dtype": "bf16", "materialize_bf16_outputs": True},
        }

        t4 = resolve_config("configs/hf_materialization.example.yaml", source, use_bf16=False)
        ampere = resolve_config("configs/hf_materialization.example.yaml", source, use_bf16=True)

        self.assertEqual(t4["path_alt"]["dtype"], "fp16")
        self.assertEqual(t4["path_alt"]["materialization_dtype"], "bf16")
        self.assertEqual(ampere["path_alt"]["dtype"], "bf16")
        self.assertEqual(ampere["path_alt"]["materialization_dtype"], "fp16")


if __name__ == "__main__":
    unittest.main()
