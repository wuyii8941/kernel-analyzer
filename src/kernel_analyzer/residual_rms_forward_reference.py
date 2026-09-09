"""Reference expression for an observed fused residual-add/RMSNorm ordering.

The observed kernel computes mean square before the BF16 residual write, then
normalizes values reloaded after that write. This helper is not yet a registered
source checker or a complete training adapter.
"""


def evaluate(branch, residual, weight, *, epsilon):
    import math
    import torch
    if (branch.ndim != 2 or branch.shape != residual.shape or branch.numel() == 0
            or weight.shape != (branch.shape[1],)
            or any(t.dtype != torch.bfloat16 for t in (branch, residual, weight))
            or any(t.device != branch.device for t in (residual, weight))
            or not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool)
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('Invalid residual RMS inputs')
    if any(not torch.isfinite(t).all() for t in (branch, residual, weight)):
        raise ValueError('Finite inputs required')
    summed = branch.float() + residual.float()
    written = summed.to(torch.bfloat16)
    inverse = torch.rsqrt(summed.square().mean(dim=-1) + epsilon)
    normalized = ((written.float() * inverse[:, None]) * weight.float()).to(torch.bfloat16)
    if any(not torch.isfinite(t).all() for t in (written, inverse, normalized)):
        raise ValueError('Nonfinite residual RMS output')
    return written, inverse, normalized


def from_runtime_pointers(pointers, *, rows, width, epsilon):
    """Decode pre-call buffers; caller must preserve them before in-place writes.

    This does not establish execution identity or prove original-buffer aliasing:
    those properties must be checked by the capture path before cloning buffers.
    """
    import torch
    if any(type(value) is not int or value <= 0 for value in (rows, width)):
        raise ValueError('Positive integer dimensions required')
    names = ('in_ptr0', 'in_out_ptr0', 'in_ptr1')
    values = [pointers.get(name) for name in names]
    sizes = (rows * width, rows * width, width)
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16
           or value.numel() != size or not value.is_contiguous()
           for value, size in zip(values, sizes)):
        raise ValueError('Pre-call storage contract differs')
    branch, residual, weight = values
    return evaluate(branch.reshape(rows, width), residual.reshape(rows, width),
                    weight.reshape(width), epsilon=epsilon)


def select_output(pointers, candidate, *, formal_pointer, rows, width, epsilon):
    """Select one of three outputs without conflating their storage types."""
    import torch
    names = ('in_out_ptr0', 'in_out_ptr1', 'out_ptr0')
    if formal_pointer not in names:
        raise ValueError('Unknown residual RMS output')
    outputs = from_runtime_pointers(pointers, rows=rows, width=width, epsilon=epsilon)
    expected = outputs[names.index(formal_pointer)]
    if (not isinstance(candidate, torch.Tensor) or candidate.dtype != expected.dtype
            or candidate.device != expected.device or candidate.numel() != expected.numel()
            or not candidate.is_contiguous()):
        raise ValueError('Output storage contract differs')
    return expected.reshape(candidate.shape)


def snapshot_pre_call(pointers, *, rows, width):
    """Validate original buffers before cloning; reject shared-storage layouts.

    The in-place residual is one named buffer, not an alias between arguments.
    Distinct views sharing storage are conservatively unsupported even when
    their accessed ranges do not overlap. No numerical outcome is inspected.
    """
    import torch
    if any(type(value) is not int or value <= 0 for value in (rows, width)):
        raise ValueError('Positive integer dimensions required')
    layout = dict(in_out_ptr0=(torch.bfloat16, rows * width),
                  in_out_ptr1=(torch.float32, rows),
                  in_ptr0=(torch.bfloat16, rows * width),
                  in_ptr1=(torch.bfloat16, width),
                  out_ptr0=(torch.bfloat16, rows * width))
    if set(pointers) != set(layout):
        raise ValueError('Pointer ABI differs')
    storage_ids = []
    for name, (dtype, size) in layout.items():
        value = pointers[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.numel() != size or not value.is_contiguous()):
            raise ValueError('Original buffer layout differs: ' + name)
        storage_ids.append((value.device, value.untyped_storage().data_ptr()))
    if len({value.device for value in pointers.values()}) != 1:
        raise ValueError('Mixed devices')
    if len(set(storage_ids)) != len(storage_ids):
        raise ValueError('Shared argument storage requires a separate audited contract')
    # Do not read uninitialized output-only buffers.
    return {name: pointers[name].detach().clone()
            for name in ('in_ptr0', 'in_out_ptr0', 'in_ptr1')}
