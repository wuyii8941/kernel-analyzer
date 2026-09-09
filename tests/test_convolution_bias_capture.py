import pytest
import torch
from kernel_analyzer.convolution_bias_capture import PendingConvolutions


def values():
    return (torch.ones(1,1536,2,dtype=torch.bfloat16),
            torch.ones(1536,1,4,dtype=torch.bfloat16),
            torch.ones(1,1536,5,dtype=torch.bfloat16),
            torch.ones(1536,dtype=torch.bfloat16))


def test_storage_pair_and_snapshot():
    x,w,y,b = values(); pending = PendingConvolutions()
    pending.record('call', x,w,y)
    result = pending.take_before_bias('call',y.view_as(y),b)
    x.zero_(); y.zero_(); b.zero_()
    assert result['inputs'].sum() > 0 and result['convolution_output'].sum() > 0
    assert result['bias'].sum() > 0
    pending.require_empty()


def test_reject_wrong_call_copy_and_intermediate_mutation():
    x,w,y,b = values(); pending = PendingConvolutions()
    pending.record('call',x,w,y)
    with pytest.raises(ValueError): pending.take_before_bias('other',y,b)
    with pytest.raises(ValueError): pending.take_before_bias('call',y.clone(),b)
    with pytest.raises(ValueError): pending.record('call',x,w,y)
    y.add_(1)
    with pytest.raises(ValueError): pending.take_before_bias('call',y,b)
    with pytest.raises(ValueError): pending.require_empty()
