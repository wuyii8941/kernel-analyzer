from types import SimpleNamespace
import pytest
import torch
from kernel_analyzer import residual_rms_forward_observer as adapter


class Base:
    def __init__(self, modules, **kwargs): self.modules = modules
    def __enter__(self): return self
    def __exit__(self, *args): return False


def test_validation_before_kernel_and_restore(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('placeholder')
    contract = dict(function_ast_sha256='checked', rows=2, width=3)
    monkeypatch.setattr(adapter, 'check_source', lambda *args: contract)
    calls = []
    original = lambda *args, **kwargs: calls.append(True)
    kernel = SimpleNamespace(run=original)
    module = SimpleNamespace(__file__=str(source), triton_x=kernel)
    names = ['in_out_ptr0', 'in_out_ptr1', 'in_ptr0', 'in_ptr1', 'out_ptr0']
    cls = adapter.observer_class(Base, {'triton_x': contract},
                                 lambda kernel: [(n, '*tensor') for n in names])
    buffers = [torch.ones(6, dtype=torch.bfloat16), torch.empty(2),
               torch.ones(6, dtype=torch.bfloat16), torch.ones(3, dtype=torch.bfloat16),
               torch.empty(6, dtype=torch.bfloat16)]
    with cls(modules=[module]):
        kernel.run(*buffers)
        assert calls == [True]
        buffers[-1] = buffers[0]
        with pytest.raises(ValueError, match='Shared argument storage'):
            kernel.run(*buffers)
        assert calls == [True]
    assert kernel.run is original


def test_missing_symbol_substitution_forbidden():
    cls = adapter.observer_class(Base, {}, lambda kernel: [])
    with pytest.raises(ValueError): cls(modules=[], allow_missing_symbols=True)


def test_partial_base_installation_is_restored(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('placeholder')
    contract = dict(function_ast_sha256='checked', rows=2, width=3)
    monkeypatch.setattr(adapter, 'check_source', lambda *args: contract)
    original = lambda *args: None
    kernel = SimpleNamespace(run=original)
    module = SimpleNamespace(__file__=str(source), triton_x=kernel, external='original')
    class FailingBase(Base):
        def __enter__(self):
            self.modules[0].external = 'hooked'
            raise RuntimeError('installation failed')
        def __exit__(self, *args):
            self.modules[0].external = 'original'
    cls = adapter.observer_class(FailingBase, {'triton_x': contract}, lambda kernel: [])
    with pytest.raises(RuntimeError, match='installation failed'):
        with cls(modules=[module]):
            raise AssertionError('body must not execute')
    assert kernel.run is original
    assert module.external == 'original'
