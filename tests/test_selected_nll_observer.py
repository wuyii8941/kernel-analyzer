import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from kernel_analyzer.selected_nll_observer import observer_class
from kernel_analyzer.selected_nll_source import check_source, SIGNATURE


@pytest.mark.parametrize('valid_call', [False, True])
def test_dimensions_and_original_callable_restoration(valid_call):
    path = Path('results/coverage/runtime_releases/deepseek8b_seq128_r1/trace/model__1_backward_segment0_executed/output_code.py')
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and 'nll_loss_backward' in t.id for t in n.targets))
    symbol = node.targets[0].id
    contract = check_source(source, symbol)
    calls = []
    def original(*args, **kwargs):
        calls.append(kwargs)
        args[0].fill_(2)
        return 'original-result'
    kernel = SimpleNamespace(run=original)
    module = SimpleNamespace(__file__=str(path), **{symbol:kernel})
    class Base:
        def __init__(self, modules): self.modules=modules
        def __enter__(self): return self
        def __exit__(self, *args): return False
    cls = observer_class(Base, {symbol:contract}, lambda k:list(SIGNATURE.items()))
    if valid_call:
        tensors = [torch.ones(128, 151936, dtype=torch.bfloat16),
                   torch.zeros(129, dtype=torch.int64), torch.tensor(1.),
                   torch.tensor(128.), torch.zeros(128), torch.zeros(128)]
        with cls(modules=[module]):
            assert kernel.run(*tensors, 128, 151936, stream='declared-stream') == 'original-result'
        assert calls == [dict(stream='declared-stream')]
        assert tensors[0].min() == 2
    else:
        with pytest.raises(ValueError, match='dimensions'):
            with cls(modules=[module]):
                kernel.run(*([None]*6), 64, 151936)
        assert not calls
    assert kernel.run is original
