from __future__ import annotations

import unittest

from forkcert.config import parse_simple_yaml


class ConfigTest(unittest.TestCase):
    def test_parse_nested_config_subset(self) -> None:
        text = """
seed: 0
same_weights_config_expected: true
path_ref:
  name: hf-eager-bf16
  model_name_or_path: Qwen/Qwen3-0.6B
  dtype: bf16
  device: cuda
  compile_model: false
  logits_upcast_fp32: true
path_alt:
  name: hf-compile-bf16
  model_name_or_path: Qwen/Qwen3-0.6B
  dtype: bf16
  device: cuda
  compile_model: true
  logits_upcast_fp32: true
"""
        cfg = parse_simple_yaml(text)
        self.assertEqual(cfg["seed"], 0)
        self.assertTrue(cfg["same_weights_config_expected"])
        self.assertEqual(cfg["path_ref"]["model_name_or_path"], "Qwen/Qwen3-0.6B")
        self.assertFalse(cfg["path_ref"]["compile_model"])
        self.assertTrue(cfg["path_alt"]["compile_model"])


if __name__ == "__main__":
    unittest.main()
