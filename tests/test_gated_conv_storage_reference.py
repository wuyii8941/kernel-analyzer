import pytest
import torch
from kernel_analyzer.gated_conv_gradient_reference import reference as original
from kernel_analyzer.gated_conv_gradient_storage_reference_v2 import reference


@pytest.mark.parametrize('output', ['in_out_ptr0', 'out_ptr0'])
def test_transposed_precall_gate_matches_pointer_arithmetic(output):
    generator = torch.Generator().manual_seed(190)
    specs = [('in_out_ptr0',(2,7),torch.bfloat16), ('in_ptr0',(2,4),torch.bfloat16),
             ('in_ptr1',(4,2),torch.float32), ('in_ptr2',(2,),torch.bfloat16),
             ('in_ptr3',(2,4),torch.float32), ('in_ptr4',(2,4),torch.bfloat16)]
    pointers = {n:torch.randn(s, generator=generator).to(d) for n,s,d in specs}
    metadata = dict(runtime_pointers=pointers, formal_pointer=output, input_output_storage_aliases=[])
    contract = dict(channels=2, steps=4, padding=3, output_pointer=output)
    candidate = torch.zeros((2,7) if output=='in_out_ptr0' else (2,), dtype=torch.bfloat16)
    expected = original(metadata, candidate, contract)
    transposed = dict(pointers, in_ptr1=pointers['in_ptr1'].T.unsqueeze(0).clone())
    actual = reference(dict(metadata, runtime_pointers=transposed), candidate, contract)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='in_ptr1'):
        original(dict(metadata, runtime_pointers=transposed), candidate, contract)


def test_gapped_snapshot_rejected():
    with pytest.raises(ValueError):
        reference(dict(runtime_pointers={'in_out_ptr0':torch.zeros(2,14)[:,::2]}),
                  torch.zeros(2), dict(channels=2,steps=4,padding=3))
