import pytest
import torch
from kernel_analyzer.grouped_causal_softmax_reference import snapshot_pre_call, select_output


def buffers():
    return dict(in_out_ptr0=torch.zeros(4, 2, dtype=torch.bfloat16),
                in_ptr0=torch.zeros(2, dtype=torch.int64),
                out_ptr0=torch.full((4,), float('nan')),
                out_ptr1=torch.full((4,), float('nan')),
                out_ptr2=torch.full((4, 2), float('nan'), dtype=torch.bfloat16))


def test_snapshot_precedes_write_and_ignores_uninitialized_outputs():
    p = buffers()
    saved = snapshot_pre_call(p, rows=4, width=2)
    p['in_out_ptr0'].fill_(100)
    assert saved['in_out_ptr0'].sum() == 0
    assert set(saved) == {'in_out_ptr0', 'in_ptr0'}
    for name in ('in_out_ptr0', 'out_ptr0', 'out_ptr1', 'out_ptr2'):
        output = select_output(saved, p[name], formal_pointer=name, rows=4, width=2, scale=.5)
        assert output.shape == p[name].shape
        assert output.dtype == p[name].dtype
        assert torch.isfinite(output).all()


def test_shared_storage_rejected_even_for_disjoint_views():
    p = buffers()
    storage = torch.empty(8)
    p['out_ptr0'], p['out_ptr1'] = storage[:4], storage[4:]
    with pytest.raises(ValueError, match='Shared'):
        snapshot_pre_call(p, rows=4, width=2)


def test_output_dtype_and_pointer_rejected():
    p = buffers()
    saved = snapshot_pre_call(p, rows=4, width=2)
    with pytest.raises(ValueError, match='Output storage'):
        select_output(saved, p['out_ptr0'].bfloat16(), formal_pointer='out_ptr0', rows=4, width=2, scale=.5)
    with pytest.raises(ValueError, match='Unknown'):
        select_output(saved, p['out_ptr0'], formal_pointer='missing', rows=4, width=2, scale=.5)
