import pytest
import torch
from kernel_analyzer.segmented_recurrence_reference import first_segment


def pointers():
    p={f'in_ptr{i}':torch.ones(1,dtype=torch.bfloat16) for i in range(9)}
    p['in_ptr2']=torch.zeros(1)
    p['in_ptr3']=torch.arange(8,dtype=torch.bfloat16)
    p['in_ptr4']=torch.zeros(1,dtype=torch.bfloat16)
    return p


def test_declared_segment_reads_correct_time_rows():
    p=pointers(); state=torch.tensor(1.)
    for j,t in enumerate((6.,5.)):
        state=state*torch.exp(-torch.nn.functional.softplus(torch.tensor(t)))+1
        actual=first_segment(p,channels=1,width=1,outputs=2,output_index=j,time_start=6,sequence_length=8)
        assert torch.allclose(actual.squeeze(),state)
    assert p['in_ptr3'].numel()==8


@pytest.mark.parametrize('start',[1,8,-1,True])
def test_invalid_segment_rejected(start):
    with pytest.raises(ValueError):
        first_segment(pointers(),channels=1,width=1,outputs=2,output_index=0,time_start=start,sequence_length=8)


def test_wrong_full_sequence_layout_rejected():
    with pytest.raises(ValueError,match='layout'):
        first_segment(pointers(),channels=1,width=1,outputs=2,output_index=0,time_start=6,sequence_length=9)


def runtime_fixture():
    import hashlib
    from pathlib import Path
    from kernel_analyzer import segmented_recurrence_source,decayed_recurrence_source
    deps={str(Path(m.__file__).resolve()):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
          for m in (segmented_recurrence_source,decayed_recurrence_source)}
    contract=dict(checker_dependencies_sha256=deps,segment_input_kind='BF16_OUTER_PRODUCT',
        symbol='k',output_pointer='out_ptr1',output_pointers=['out_ptr0','out_ptr1'],
        channels=1,state_width=1,outputs=2,time_start=6,sequence_length=8)
    metadata=dict(symbol='k',formal_pointer='out_ptr1',input_output_storage_aliases=[],
                  runtime_pointers=pointers())
    return metadata,contract


def test_runtime_selected_output_matches_mathematical_reference():
    from kernel_analyzer.segmented_recurrence_reference import reference
    metadata,contract=runtime_fixture()
    actual=reference(metadata,torch.zeros(1),contract)
    expected=first_segment(pointers(),channels=1,width=1,outputs=2,output_index=1,time_start=6,sequence_length=8)
    assert torch.equal(actual,expected.reshape(1))


@pytest.mark.parametrize('field,value',[('symbol','other'),('formal_pointer','out_ptr0'),
                                      ('input_output_storage_aliases',['alias'])])
def test_runtime_boundary_changes_rejected(field,value):
    from kernel_analyzer.segmented_recurrence_reference import reference
    metadata,contract=runtime_fixture(); metadata[field]=value
    with pytest.raises(ValueError,match='boundary'): reference(metadata,torch.zeros(1),contract)


def test_dependency_change_rejected():
    from kernel_analyzer.segmented_recurrence_reference import reference
    metadata,contract=runtime_fixture(); contract['checker_dependencies_sha256']={}
    with pytest.raises(ValueError,match='dependencies'): reference(metadata,torch.zeros(1),contract)
