from types import SimpleNamespace
import pytest
import torch
from kernel_analyzer import forward_recurrence_observer as observer


def test_same_call_capture_and_restore(tmp_path, monkeypatch):
    source = tmp_path/'source.py'
    source.write_text('fixture')
    contract = dict(steps=3, channels=2, state_width=2, packed_width=6,
                    state_offset=2, function_ast_sha256='fixture')
    monkeypatch.setattr(observer, 'check_source', lambda *args: contract)
    signature = {f'in_ptr{i}': '*fp32' if i == 0 else '*bf16' for i in range(5)}
    signature.update({f'out_ptr{i}': '*bf16' if i == 2 else '*fp32' for i in range(3)})
    signature['xnumel'] = 'i32'
    calls, records = [], []
    def run(*args, **kwargs):
        calls.append(1)
        for i, output in enumerate(args[5:-1]): output.fill_(i+1)
        return 'called'
    kernel = SimpleNamespace(run=run, triton_meta={'signature': signature})
    module = SimpleNamespace(__file__=str(source), recurrence=kernel)
    inputs = [torch.zeros(4)] + [torch.ones(n, dtype=torch.bfloat16) for n in (6, 2, 18, 6)]
    outputs = [torch.empty(4), torch.empty(4), torch.empty(4, dtype=torch.bfloat16)]
    with observer.Observer([module], {'recurrence': contract}, records.append) as installed:
        assert kernel.run(*inputs, *outputs, 4, stream=0) == 'called'
        assert installed.counts == {'recurrence': 1}
        with pytest.raises(ValueError): kernel.run(*inputs, *outputs, 5, stream=0)
        with pytest.raises(ValueError): kernel.run(*inputs, outputs[0], outputs[0], outputs[2], 4, stream=0)
    assert calls == [1] and kernel.run is run
    outputs[0].zero_()
    assert records[0]['post_call_outputs']['out_ptr0'].sum() == 4
    assert not records[0]['binary_identity_verified']


def test_installation_failure_restores_prior_kernel(tmp_path, monkeypatch):
    source = tmp_path/'source.py'
    source.write_text('fixture')
    contract = dict(steps=2, function_ast_sha256='fixture')
    monkeypatch.setattr(observer, 'check_source', lambda *args: contract)
    signature = {f'in_ptr{i}': '*fp32' if i == 0 else '*bf16' for i in range(5)}
    signature.update(out_ptr0='*fp32', out_ptr1='*bf16', xnumel='i32')
    run = lambda *args, **kwargs: None
    kernel = SimpleNamespace(run=run, triton_meta={'signature': signature})
    module = SimpleNamespace(__file__=str(source), recurrence=kernel)
    with pytest.raises(ValueError):
        with observer.Observer([module], {'recurrence': contract, 'missing': contract}, lambda r: None):
            pass
    assert kernel.run is run
