from types import SimpleNamespace
import pytest
import torch
from scripts.indexed_accumulation_observer import IndexedAccumulationObserver, snapshot_indexed_inputs


def test_original_runs_once_and_precall_snapshot_survives_mutation():
    calls=[]
    def original(buffer,indices,values,accumulate):
        calls.append(True)
        return buffer.index_put_(tuple(indices),values,accumulate)
    namespace=SimpleNamespace(index_put_=original)
    module=SimpleNamespace(aten=namespace)
    observer=object.__new__(IndexedAccumulationObserver)
    observer.modules=[module]; observer.nontriton_restores=[]
    observer.nontriton_rows={('DIRECT_ATEN','hash'):[{}]}
    observer._source_identity=lambda:('generated.py',123,'hash')
    observer._take_nontriton=lambda *args:{}
    captured=[]
    observer._emit_nontriton=lambda row,value,endpoint,metadata:captured.append((value.clone(),metadata))
    observer._install_direct_aten()
    buffer=torch.ones(3,2); values=torch.full((2,2),2.); indices=torch.tensor([1,1])
    result=module.aten.index_put_(buffer,[indices],values,True)
    assert result is buffer and len(calls)==1
    assert torch.equal(captured[0][1]['indexed_operands']['initial'],torch.ones(3,2))
    assert torch.equal(captured[0][0][1],torch.tensor([5.,5.]))
    values.zero_(); indices.zero_()
    assert captured[0][1]['indexed_operands']['values'].sum()==8
    for restore in reversed(observer.nontriton_restores): restore()
    assert module.aten is namespace


def test_captured_operands_feed_same_reference_with_exact_identity():
    from kernel_analyzer.indexed_row_accumulation_reference import reference
    initial=torch.ones(3,2)
    operands=snapshot_indexed_inputs(initial,[torch.tensor([1,1])],torch.ones(2,2),True)
    contract=dict(executing_filename='generated.py',executing_line=12,source_line_sha256='source')
    metadata=dict(contract,indexed_operands=operands,accumulate=True,
                  reference_operand_capture='PRE_INVOCATION_CLONE',
                  implementation_kind='DIRECT_ATEN',endpoint='mutated_output_0')
    actual=reference(metadata,torch.empty_like(initial),contract)
    assert torch.equal(actual[1],torch.tensor([3.,3.]))
    for key,value in [('executing_line',13),('executing_filename','other.py'),('source_line_sha256','other')]:
        with pytest.raises(ValueError): reference(dict(metadata,**{key:value}),initial,contract)
    with pytest.raises(ValueError): reference(metadata,initial,{})


def test_other_semantics_not_silently_accepted():
    with pytest.raises(ValueError):
        snapshot_indexed_inputs(torch.zeros(3,2),[torch.tensor([1])],torch.ones(1,2),False)
    with pytest.raises(ValueError):
        snapshot_indexed_inputs(torch.zeros(3,2),[None],torch.ones(1,2),True)
