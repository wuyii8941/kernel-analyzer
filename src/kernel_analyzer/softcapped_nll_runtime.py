"""Decode the reviewed in-place softcap/NLL calculation before its write.

The caller must verify loaded source and launch dimensions separately. These
checks do not infer execution identity from a kernel name.
"""
import torch
from kernel_analyzer.softcapped_nll_reference import softcapped_nll_backward
from kernel_analyzer.softcapped_nll_source import BODY_SHA256


def snapshot_inputs(pointers, contract):
    if (contract.get('body_sha256') != BODY_SHA256 or
            (contract.get('tokens'), contract.get('vocabulary'), contract.get('label_offset'),
             contract.get('cap')) != (128, 262144, 1, 30.)):
        raise ValueError('Unreviewed softcap runtime contract')
    names = ['in_out_ptr0', 'in_ptr0', 'in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4']
    if set(pointers) != set(names):
        raise ValueError('Softcap pointer signature differs')
    output = pointers['in_out_ptr0']
    if not isinstance(output, torch.Tensor):
        raise ValueError('Tensor output required')
    expected = [(torch.bfloat16, 128*262144), (torch.int64, 129),
                (torch.float32, 1), (torch.float32, 1), (torch.float32, 128), (torch.float32, 128)]
    for name, (dtype, size) in zip(names, expected):
        value = pointers[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.numel() != size or not value.is_contiguous()
                or value.device != output.device):
            raise ValueError('Softcap storage mismatch: '+name)
        if name != 'in_out_ptr0' and value.untyped_storage().data_ptr() == output.untyped_storage().data_ptr():
            raise ValueError('Unsupported shared input allocation')
    return dict(logits=output.detach().reshape(128, 262144).clone(),
                labels=pointers['in_ptr0'].detach().reshape(-1)[1:129].clone(),
                grad_output=pointers['in_ptr1'].detach().clone(),
                total_weight=pointers['in_ptr2'].detach().clone(),
                row_max=pointers['in_ptr3'].detach().reshape(128).clone(),
                log_exp_sum=pointers['in_ptr4'].detach().reshape(128).clone(), cap=30.)


def reference_callback(contracts):
    def reference(metadata, candidate, **unused):
        contract = contracts.get(metadata.get('symbol'))
        if contract is None or metadata.get('formal_pointer') != 'in_out_ptr0':
            raise ValueError('Undeclared softcap output')
        if metadata.get('input_output_storage_aliases'):
            raise ValueError('Unsupported runtime input/output alias')
        if candidate.dtype != torch.bfloat16 or candidate.numel() != 128*262144:
            raise ValueError('Softcap candidate storage differs')
        snapshot = snapshot_inputs(metadata['runtime_pointers'], contract)
        return softcapped_nll_backward(**snapshot).to(torch.bfloat16).reshape(candidate.shape)
    return reference
