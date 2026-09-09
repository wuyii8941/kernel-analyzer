from types import SimpleNamespace
import torch
import pytest
from kernel_analyzer import softmax_mechanism_observer as observer


def test_observe_once_and_restore(tmp_path, monkeypatch):
    source = tmp_path/'source.py'
    source.write_text('test fixture')
    contract = dict(rows=2, width=2, scale=.5, function_ast_sha256='fixture')
    monkeypatch.setattr(observer, 'check_source', lambda *args: contract)
    signature = dict(in_out_ptr0='*bf16', in_ptr0='*i64', out_ptr0='*fp32',
                     out_ptr1='*fp32', out_ptr2='*bf16', xnumel='i32', r0_numel='i32')
    calls, records = [], []
    def run(scores, ids, maximum, denominator, probability, rows, width, stream):
        calls.append(1)
        scores.zero_(); maximum.zero_(); denominator.fill_(2); probability.fill_(.5)
    kernel = SimpleNamespace(run=run, triton_meta={'signature': signature})
    module = SimpleNamespace(__file__=str(source), softmax=kernel)
    args = [torch.ones(2, 2, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.int64),
            torch.empty(2), torch.empty(2), torch.empty(2, 2, dtype=torch.bfloat16), 2, 2]
    with observer.Observer([module], {'softmax': contract}, records.append) as installed:
        kernel.run(*args, stream=0)
        assert installed.counts == {'softmax': 1}
        with pytest.raises(ValueError): kernel.run(*args, stream=0, extra=True)
    assert calls == [1] and kernel.run is run
    assert not records[0]['binary_identity_verified']
    assert records[0]['pre_call_inputs']['in_out_ptr0'].sum() == 4
