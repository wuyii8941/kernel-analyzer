from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forkcert.config import load_config
from scripts.write_checkpoint_configs import rewrite_model_paths, write_simple_yaml


class CheckpointConfigsTest(unittest.TestCase):
    def test_rewrite_model_paths(self) -> None:
        cfg = {
            "seed": 0,
            "path_ref": {"name": "ref", "model_name_or_path": "base", "dtype": "bf16"},
            "path_alt": {"name": "alt", "model_name_or_path": "base", "compile_model": True},
        }
        out = rewrite_model_paths(cfg, "data/phase0_policy_final")
        self.assertEqual(out["path_ref"]["model_name_or_path"], "data/phase0_policy_final")
        self.assertEqual(out["path_alt"]["model_name_or_path"], "data/phase0_policy_final")
        self.assertEqual(cfg["path_ref"]["model_name_or_path"], "base")

    def test_write_simple_yaml_roundtrip(self) -> None:
        cfg = {
            "seed": 0,
            "same_weights_config_expected": True,
            "path_ref": {"name": "ref", "model_name_or_path": "data/phase0_policy_final", "compile_model": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            write_simple_yaml(path, cfg)
            loaded = load_config(str(path))
        self.assertEqual(loaded["path_ref"]["model_name_or_path"], "data/phase0_policy_final")
        self.assertFalse(loaded["path_ref"]["compile_model"])


if __name__ == "__main__":
    unittest.main()
