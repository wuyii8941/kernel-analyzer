import pytest
import torch
from kernel_analyzer.selected_nll_runtime import snapshot_inputs, reference_callback


def test_snapshot_survives_actual_inplace_overwrite():
    pointers = dict(in_out_ptr0=torch.ones(128, 151936, dtype=torch.bfloat16),
                    in_ptr0=torch.zeros(129, dtype=torch.int64),
                    in_ptr1=torch.tensor(1.), in_ptr2=torch.tensor(128.),
                    in_ptr3=torch.zeros(128), in_ptr4=torch.zeros(128))
    contract = dict(tokens=128, vocabulary=151936, label_offset=1)
    snapshot = snapshot_inputs(pointers, contract)
    pointers['in_out_ptr0'].zero_()
    pointers['in_ptr0'].fill_(2)
    assert snapshot['logits'].min() == 1
    assert snapshot['labels'].max() == 0
    pointers['in_ptr2'] = torch.tensor(128., dtype=torch.float64)
    with pytest.raises(ValueError, match='storage'):
        snapshot_inputs(pointers, contract)


def test_unreviewed_shape_rejected():
    with pytest.raises(ValueError, match='layout'):
        snapshot_inputs(dict.fromkeys(['in_out_ptr0', 'in_ptr0', 'in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4']),
                        dict(tokens=64, vocabulary=151936, label_offset=1))


def test_shared_callback_rejects_wrong_output_and_runtime_alias():
    callback = reference_callback({'nll': dict(tokens=128, vocabulary=151936, label_offset=1)})
    with pytest.raises(ValueError, match='Undeclared'):
        callback(dict(symbol='nll', formal_pointer='out_ptr0'), torch.zeros(1))
    with pytest.raises(ValueError, match='alias'):
        callback(dict(symbol='nll', formal_pointer='in_out_ptr0',
                      input_output_storage_aliases=['in_ptr4']), torch.zeros(1))
