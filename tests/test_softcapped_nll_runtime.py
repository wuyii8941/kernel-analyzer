import pytest
import torch
from kernel_analyzer.softcapped_nll_runtime import snapshot_inputs, reference_callback
from kernel_analyzer.softcapped_nll_source import BODY_SHA256


def contract():
    return dict(body_sha256=BODY_SHA256, tokens=128, vocabulary=262144, label_offset=1, cap=30.)


def pointers():
    return dict(in_out_ptr0=torch.zeros(128, 262144, dtype=torch.bfloat16),
                in_ptr0=torch.arange(129), in_ptr1=torch.tensor(1.), in_ptr2=torch.tensor(128.),
                in_ptr3=torch.zeros(128), in_ptr4=torch.zeros(128))


def test_snapshot_survives_inplace_write_and_shifts_labels():
    p = pointers()
    saved = snapshot_inputs(p, contract())
    p['in_out_ptr0'].fill_(1)
    p['in_ptr0'].zero_()
    assert torch.count_nonzero(saved['logits']) == 0
    torch.testing.assert_close(saved['labels'], torch.arange(1, 129))


def test_rejects_unreviewed_contract_and_storage():
    p = pointers()
    with pytest.raises(ValueError, match='contract'):
        snapshot_inputs(p, {**contract(), 'cap': 31.})
    p['in_ptr3'] = torch.zeros(127)
    with pytest.raises(ValueError, match='storage'):
        snapshot_inputs(p, contract())


def test_callback_rejects_wrong_output_before_calculation():
    callback = reference_callback({'kernel': contract()})
    with pytest.raises(ValueError, match='Undeclared'):
        callback(dict(symbol='kernel', formal_pointer='out_ptr1'), torch.zeros(1))
    with pytest.raises(ValueError, match='alias'):
        callback(dict(symbol='kernel', formal_pointer='in_out_ptr0',
                      input_output_storage_aliases=['in_ptr1']), torch.zeros(1))
