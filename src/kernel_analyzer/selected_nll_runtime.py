"""Decode the reviewed NLL backward pointer layout before its in-place write."""
import torch
from kernel_analyzer.selected_nll_reference import selected_nll_backward


def snapshot_inputs(pointers, contract):
    names = ['in_out_ptr0', 'in_ptr0', 'in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4']
    if set(pointers) != set(names):
        raise ValueError('NLL pointer signature differs')
    n, v = contract['tokens'], contract['vocabulary']
    offset = contract['label_offset']
    if (n, v, offset) != (128, 151936, 1):
        raise ValueError('Unreviewed runtime layout')
    expected = [(torch.bfloat16, n*v), (torch.int64, n+offset),
                (torch.float32, 1), (torch.float32, 1), (torch.float32, n), (torch.float32, n)]
    output = pointers['in_out_ptr0']
    for name, (dtype, size) in zip(names, expected):
        value = pointers[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.numel() != size or not value.is_contiguous()
                or value.device != output.device):
            raise ValueError('NLL storage mismatch: ' + name)
        if name != 'in_out_ptr0' and value.untyped_storage().data_ptr() == output.untyped_storage().data_ptr():
            raise ValueError('Unsupported shared input allocation')
    return dict(logits=output.detach().reshape(n, v).clone(),
                labels=pointers['in_ptr0'].detach().reshape(-1)[offset:offset+n].clone(),
                grad_output=pointers['in_ptr1'].detach().clone(),
                total_weight=pointers['in_ptr2'].detach().clone(),
                row_max=pointers['in_ptr3'].detach().reshape(n).clone(),
                log_exp_sum=pointers['in_ptr4'].detach().reshape(n).clone())


def reference_write(snapshot):
    """Declared high-precision local derivative followed by original BF16 write."""
    return selected_nll_backward(**snapshot).to(torch.bfloat16)


def reference_callback(contracts):
    """Use the shared observer's pre-call clones, preserving its sink protocol.

    Callers must also check loaded source and runtime ABI before execution.
    This callback deliberately does not claim that metadata proves those facts.
    """
    def reference(metadata, candidate, **unused):
        contract = contracts.get(metadata.get('symbol'))
        if contract is None or metadata.get('formal_pointer') != 'in_out_ptr0':
            raise ValueError('Undeclared NLL output')
        if metadata.get('input_output_storage_aliases'):
            raise ValueError('Unsupported runtime input/output alias')
        if candidate.dtype != torch.bfloat16 or candidate.numel() != contract['tokens']*contract['vocabulary']:
            raise ValueError('NLL candidate storage differs')
        snapshot = snapshot_inputs(metadata['runtime_pointers'], contract)
        return reference_write(snapshot).reshape(candidate.shape)
    return reference
