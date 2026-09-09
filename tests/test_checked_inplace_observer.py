from types import SimpleNamespace
import pytest
from kernel_analyzer.checked_inplace_observer import observer_class


class Base:
    def __init__(self, modules, **kwargs): self.modules = modules
    def __enter__(self): return self
    def __exit__(self, *args): return False


def build(tmp_path, *, contract_change=None, bad_signature=False, base=Base):
    path = tmp_path/'source.py'
    path.write_text('reviewed source')
    contract = dict(body_sha256='reviewed', source_sha256='wrapper', tokens=2, vocabulary=3)
    declared = {**contract, **(contract_change or {})}
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return 7
    kernel = SimpleNamespace(run=original)
    module = SimpleNamespace(__file__=str(path), kernel=kernel)
    def source(text, name):
        assert text == 'reviewed source' and name == 'kernel'
        return contract
    def snapshot(pointers, c):
        if pointers['in_out_ptr0'] != 'valid': raise ValueError('bad operand')
    cls = observer_class(base, {'kernel': declared},
        lambda k: [('in_out_ptr0', '*fp32' if bad_signature else '*bf16')],
        check_source=source, signature={'in_out_ptr0': '*bf16'}, snapshot_inputs=snapshot)
    return cls, module, kernel, original, calls


def test_delegate_and_restore(tmp_path):
    cls, module, kernel, original, calls = build(tmp_path, contract_change={'source_sha256':'isolated-definition'})
    with cls(modules=[module]):
        assert kernel.run('valid', 2, 3, stream=19) == 7
    assert calls == [(('valid', 2, 3), {'stream': 19})]
    assert kernel.run is original


@pytest.mark.parametrize('args', [('valid', 3, 3), ('invalid', 2, 3)])
def test_reject_and_restore(tmp_path, args):
    cls, module, kernel, original, calls = build(tmp_path)
    with pytest.raises(ValueError):
        with cls(modules=[module]): kernel.run(*args)
    assert not calls and kernel.run is original


def test_changed_contract_and_missing_symbol_policy(tmp_path):
    cls, module, *_ = build(tmp_path, contract_change={'tokens':3})
    with pytest.raises(ValueError, match='contract'):
        cls(modules=[module])
    with pytest.raises(ValueError, match='substitution'):
        cls(modules=[module], allow_missing_symbols=True)


def test_signature_and_base_enter_failure_restore(tmp_path):
    cls, module, kernel, original, _ = build(tmp_path, bad_signature=True)
    with pytest.raises(ValueError, match='pointer'):
        with cls(modules=[module]): pass
    assert kernel.run is original
    class Broken(Base):
        def __enter__(self): raise RuntimeError('base failed')
    cls, module, kernel, original, _ = build(tmp_path, base=Broken)
    with pytest.raises(RuntimeError):
        with cls(modules=[module]): pass
    assert kernel.run is original
