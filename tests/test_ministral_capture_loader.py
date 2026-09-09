from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.capture_qwen_inductor_proof_ids import load_model
from scripts.capture_qwen_aot import load_model as load_aot_model
from scripts import run_generated_fp32_screen
from scripts import run_frozen_candidate_fp32_screen


@pytest.mark.parametrize("loader", [load_model, load_aot_model])
def test_ministral_outer_model_is_loaded_without_generic_fallback(
    monkeypatch, tmp_path, loader,
):
    calls = []

    class FakeModel:
        config = SimpleNamespace(
            use_cache=True, text_config=SimpleNamespace(use_cache=True),
        )

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return cls()

        def to(self, device):
            calls.append(("to", device))
            return self

        def train(self):
            calls.append(("train",))
            return self

    monkeypatch.setattr("transformers.Mistral3ForConditionalGeneration", FakeModel)
    model = loader("ministral3", tmp_path, "cuda:0")
    assert calls[0] == (tmp_path, {
        "dtype": torch.bfloat16, "attn_implementation": "eager",
        "local_files_only": True,
    })
    assert calls[1:] == [("to", "cuda:0"), ("train",)]
    assert model.config.use_cache is False
    assert model.config.text_config.use_cache is False


def test_runtime_release_loader_preserves_current_ministral_config(
    monkeypatch, tmp_path,
):
    calls = []

    class FakeModel:
        config = SimpleNamespace(use_cache=True)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return cls()

        def to(self, device):
            calls.append(("to", device))
            return self

        def train(self):
            calls.append(("train",))
            return self

    monkeypatch.setattr(
        run_generated_fp32_screen,
        "Mistral3ForConditionalGeneration",
        FakeModel,
    )
    model = run_generated_fp32_screen.load_model(
        "ministral3", tmp_path, torch.device("cuda:0")
    )
    assert calls[0] == (tmp_path, {
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
        "local_files_only": True,
    })
    assert calls[1:] == [("to", torch.device("cuda:0")), ("train",)]
    assert model.config.use_cache is False


def test_runtime_environment_records_model_implementation_source():
    class FakeModel(torch.nn.Module):
        pass

    result = run_frozen_candidate_fp32_screen.runtime_environment(FakeModel())
    identity = result["model_class_source"]
    source = Path(__file__).resolve()
    assert identity["source_path"] == str(source)
    assert identity["source_sha256"] == __import__("hashlib").sha256(
        source.read_bytes()
    ).hexdigest()
    assert Path(result["transformers_source_path"]).is_file()
    assert len(result["transformers_source_sha256"]) == 64
