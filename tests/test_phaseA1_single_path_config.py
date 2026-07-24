from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "phaseA1_single_path.py"
    spec = importlib.util.spec_from_file_location("phaseA1_single_path_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_path_config_preserves_training_precision_fields():
    module = load_module()
    config = module.path_config(
        {
            "path_ref": {
                "name": "heldout",
                "model_name_or_path": "checkpoint",
                "dtype": "fp32",
                "autocast_dtype": "fp16",
                "model_training_mode": True,
                "gradient_checkpointing": True,
            }
        },
        "path_ref",
    )
    assert config.dtype == "fp32"
    assert config.autocast_dtype == "fp16"
    assert config.model_training_mode is True
    assert config.gradient_checkpointing is True
