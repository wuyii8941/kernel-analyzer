from types import SimpleNamespace
import hashlib
import linecache

import pytest
import torch

from scripts.convolution_capture_observer import observer_class
from kernel_analyzer.depthwise_conv1d_source import check_call


CALL = ('extern_kernels.convolution(x, w, stride=(1,), padding=(3,), '
        'dilation=(1,), transposed=False, output_padding=(0,), groups=1536, bias=None)')


def setup_observer(tmp_path, contract=True):
    # Execute a real file-backed call so the production stack-frame binding,
    # not a mocked source identity, is exercised.
    source = 'def run(extern_kernels, x, w):\n    result = ' + CALL + '\n    return result\n'
    path = tmp_path / 'generated.py'
    path.write_text(source)
    digest = hashlib.sha256(source.splitlines()[1].strip().encode()).hexdigest()
    namespace = {}
    exec(compile(source, str(path), 'exec'), namespace)
    linecache.checkcache()
    contracts = {digest: {'convolution': check_call(CALL)}} if contract else {}
    cls = observer_class(contracts)
    observer = object.__new__(cls)
    calls, captured = [], []

    def original(x, w, **kwargs):
        calls.append(True)
        x.add_(1)  # Deliberately expose incorrect post-call snapshot timing.
        return x.clone()

    external = SimpleNamespace(convolution=original)
    module = SimpleNamespace(extern_kernels=external)
    observer.modules = [module]
    observer.nontriton_restores = []
    observer.nontriton_rows = {('EXTERN', digest): [{}]}
    observer._take_nontriton = lambda *args: {}
    observer._emit_nontriton = lambda row, value, endpoint, metadata: captured.append(metadata)
    observer._install_externals()
    return observer, module, external, namespace['run'], calls, captured


def test_real_callsite_snapshot_and_restore(tmp_path):
    observer, module, external, run, calls, captured = setup_observer(tmp_path)
    x, w = torch.zeros(1), torch.ones(1)
    run(module.extern_kernels, x, w)
    assert len(calls) == len(captured) == 1
    assert captured[0]['executing_line'] == 2
    assert captured[0]['runtime_args'][0].item() == 0
    assert x.item() == 1
    assert captured[0]['reference_operand_capture'] == 'PRE_INVOCATION_CLONE'
    for restore in reversed(observer.nontriton_restores):
        restore()
    assert module.extern_kernels is external


def test_selected_call_without_contract_fails_before_execution(tmp_path):
    observer, module, external, run, calls, captured = setup_observer(tmp_path, False)
    with pytest.raises(ValueError, match='lacks frozen source binding'):
        run(module.extern_kernels, torch.zeros(1), torch.ones(1))
    assert not calls and not captured
