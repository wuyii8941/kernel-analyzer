from __future__ import annotations

import torch
from types import SimpleNamespace

from scripts import bind_backward_rescreen_carriers as carrier_binding


class _Mixer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.D = torch.nn.Parameter(torch.zeros(8))
        self.proj = torch.nn.Linear(4, 8, bias=True)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixer = _Mixer()


def test_parameter_binding_prefers_specific_leaf_stack_over_parent(monkeypatch) -> None:
    forward = {
        "nodes": [
            {
                "name": "primals_1",
                "op": "placeholder",
                "tensor_meta": [[8], "torch.float32", True],
                "users": ["add"],
            },
            {
                "name": "add",
                "op": "call_function",
                "nn_module_stack": {
                    "parent": ["L['self'].mixer", "Mixer"],
                    "leaf": ["L['self'].mixer.proj", "Linear"],
                },
            },
        ]
    }
    monkeypatch.setattr(carrier_binding, "build_meta_model", lambda _path: _Model())
    rows, by_primal = carrier_binding.bind_forward_parameters(forward, "unused")
    assert rows[0]["status"] == "EXACT_MODULE_STACK_PARAMETER_BINDING"
    assert rows[0]["name"] == "mixer.proj.bias"
    assert by_primal["primals_1"]["name"] == "mixer.proj.bias"


def test_ministral_outer_config_uses_explicit_declared_architecture(monkeypatch) -> None:
    config = SimpleNamespace(model_type="mistral3")
    expected = _Model()
    calls = []

    monkeypatch.setattr(
        carrier_binding.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        "transformers.Mistral3ForConditionalGeneration",
        lambda value: calls.append(value) or expected,
    )
    monkeypatch.setattr(
        carrier_binding.AutoModelForCausalLM,
        "from_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("generic causal-LM factory must not be used")
        ),
    )

    assert carrier_binding.build_meta_model("ministral") is expected
    assert calls == [config]
