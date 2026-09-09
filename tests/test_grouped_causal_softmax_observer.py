from types import SimpleNamespace

import pytest
import torch

from kernel_analyzer import grouped_causal_softmax_observer as adapter


class Base:
    def __init__(self, modules, **kwargs):
        self.modules = modules

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def buffers(rows=4, width=2):
    return [torch.ones(rows * width, dtype=torch.bfloat16),
            torch.arange(width, dtype=torch.int64), torch.empty(rows),
            torch.empty(rows), torch.empty(rows * width, dtype=torch.bfloat16)]


def test_validates_before_inplace_execution_and_restores(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("placeholder")
    contract = {"function_ast_sha256": "checked", "rows": 4, "width": 2,
                "output_pointers": list(adapter.POINTER_NAMES)}
    monkeypatch.setattr(adapter, "check_source", lambda *args: dict(
        contract, source_sha256="current", runtime_binding_complete=False))
    calls = []
    original = lambda *args, **kwargs: calls.append(True)
    kernel = SimpleNamespace(run=original)
    module = SimpleNamespace(__file__=str(source), softmax=kernel)
    signature = lambda kernel: [*( (name, "*tensor") for name in adapter.POINTER_NAMES),
                                ("xnumel", "i32"), ("r0_numel", "i32")]
    cls = adapter.observer_class(Base, {"softmax": contract}, signature)
    with cls(modules=[module]):
        kernel.run(*buffers(), 4, 2)
        assert calls == [True]
        with pytest.raises(ValueError, match="dimensions differ"):
            kernel.run(*buffers(), 8, 2)
    assert kernel.run is original


def test_storage_alias_rejected_before_execution(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("placeholder")
    contract = {"function_ast_sha256": "checked", "rows": 4, "width": 2,
                "output_pointers": list(adapter.POINTER_NAMES)}
    monkeypatch.setattr(adapter, "check_source", lambda *args: dict(
        contract, source_sha256="current", runtime_binding_complete=False))
    calls = []
    original = lambda *args, **kwargs: calls.append(True)
    kernel = SimpleNamespace(run=original)
    module = SimpleNamespace(__file__=str(source), softmax=kernel)
    signature = lambda kernel: [*( (name, "*tensor") for name in adapter.POINTER_NAMES),
                                ("xnumel", "i32"), ("r0_numel", "i32")]
    cls = adapter.observer_class(Base, {"softmax": contract}, signature)
    values = buffers()
    values[-1] = values[0]
    with cls(modules=[module]):
        with pytest.raises(ValueError, match="Shared argument storage"):
            kernel.run(*values, 4, 2)
    assert not calls


def test_missing_symbol_substitution_forbidden():
    cls = adapter.observer_class(Base, {}, lambda kernel: [])
    with pytest.raises(ValueError):
        cls(modules=[], allow_missing_symbols=True)
