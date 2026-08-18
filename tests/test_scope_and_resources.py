import json
import os
from pathlib import Path

from scripts.download_pinned_model import file_sha256
from scripts.resource_preflight import DATA_ROOT, parameter_budget, required_cache_paths


ROOT = Path(__file__).resolve().parents[1]


def test_model_scope_separates_full_step_mechanism_and_paused():
    scope = json.loads((ROOT / "results/coverage/model_scope.json").read_text())
    models = scope["models"]
    assert models["deepseek_r1_0528_qwen3_8b"]["scope"] == "FULL_STEP"
    assert models["deepseek_v4_flash"]["scope"] == "MECHANISM_REGION"
    assert models["granite_3p1_1b_a400m"]["scope"] == "PAUSED_OUT_OF_SCOPE"
    assert scope["rules"]["mechanism_region_never_counts_toward_full_step_coverage"] is True


def test_default_large_cache_paths_live_under_data1(monkeypatch):
    for name in (
        "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
        "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME",
        "PIP_CACHE_DIR", "TMPDIR",
    ):
        monkeypatch.delenv(name, raising=False)
    for value in required_cache_paths().values():
        assert Path(value).is_relative_to(DATA_ROOT)


def test_parameter_budget_counts_weights_gradients_and_reserves():
    budget = parameter_budget(1_000_000_000, 256, 2)
    assert 1.8 < budget["weights_gib"] < 1.9
    assert budget["weights_gib"] == budget["gradients_gib"]
    assert budget["compiler_reserve_gib"] == 6
    assert budget["single_gpu_low_precision_peak_gib"] > 13


def test_model_manifest_hash_is_streaming_and_exact(tmp_path):
    value = tmp_path / "shard.bin"
    value.write_bytes(b"abcdef")
    assert file_sha256(value, chunk_bytes=2) == (
        "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
    )
