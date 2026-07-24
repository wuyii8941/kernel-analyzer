from types import SimpleNamespace

import pytest

from scripts.enrich_online_scan import online_path_config
from scripts.phase0_grpo_train import trainer_compute_dtype


def test_online_bf16_metadata_is_not_labeled_fp16() -> None:
    training = {"training_compute_dtype": "bf16", "model_parameter_dtype": "bfloat16"}
    config = online_path_config(training)
    assert config["path_ref"]["name"] == "hf-eager-bf16-sdpa-math-online"
    assert config["path_alt"]["name"] == "hf-compile-bf16-sdpa-math-online"
    assert config["path_ref"]["autocast_dtype"] == "bf16"


def test_trainer_compute_dtype_requires_explicit_mixed_precision() -> None:
    assert trainer_compute_dtype(SimpleNamespace(bf16=True, fp16=False)) == "bf16"
    assert trainer_compute_dtype(SimpleNamespace(bf16=False, fp16=True)) == "fp16"
    with pytest.raises(ValueError):
        trainer_compute_dtype(SimpleNamespace(bf16=False, fp16=False))
