"""Independent FP32 reference for a grouped causal attention forward region.

This formula alone is not a source matcher or an execution adapter. Callers must
establish the complete source contract and capture scores BEFORE in-place writes.
The finite BF16 mask sentinel is intentional; do not replace it with -infinity.
"""
import math


def snapshot_pre_call(pointers, *, rows, width):
    """Check original storage before cloning; never read output-only buffers.

    The caller must invoke this before the kernel, not after its in-place write.
    This function cannot establish that temporal ordering by itself.
    """
    import torch
    if (type(rows) is not int or type(width) is not int or min(rows, width) <= 0
            or rows % width or width & (width - 1)):
        raise ValueError('Invalid declared dimensions')
    layout = dict(in_out_ptr0=(torch.bfloat16, rows*width),
                  in_ptr0=(torch.int64, width), out_ptr0=(torch.float32, rows),
                  out_ptr1=(torch.float32, rows), out_ptr2=(torch.bfloat16, rows*width))
    if set(pointers) != set(layout):
        raise ValueError('Pointer ABI differs')
    storage = []
    for name, (dtype, size) in layout.items():
        value = pointers[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.numel() != size or not value.is_contiguous()):
            raise ValueError('Original buffer layout differs: ' + name)
        storage.append((value.device, value.untyped_storage().data_ptr()))
    if len({v.device for v in pointers.values()}) != 1:
        raise ValueError('Mixed devices')
    if len(set(storage)) != len(storage):
        raise ValueError('Shared argument storage requires a separate audited contract')
    return {name: pointers[name].detach().clone() for name in ('in_out_ptr0', 'in_ptr0')}


def select_output(pointers, candidate, *, formal_pointer, rows, width, scale):
    """Decode saved pre-call inputs and select one output, retaining its dtype."""
    import torch
    if formal_pointer not in ('in_out_ptr0', 'out_ptr0', 'out_ptr1', 'out_ptr2'):
        raise ValueError('Unknown output')
    if (type(rows) is not int or type(width) is not int or min(rows, width) <= 0
            or rows % width or width & (width - 1)):
        raise ValueError('Invalid dimensions')
    required = {'in_out_ptr0', 'in_ptr0'}
    full_abi = required | {'out_ptr0', 'out_ptr1', 'out_ptr2'}
    # The shared observer snapshots every pointer before the call.  Output-only
    # snapshots are intentionally ignored, but accepting them is required for
    # the production metadata path.  No other pointer set is accepted.
    if set(pointers) not in (required, full_abi):
        raise ValueError('Expected preserved inputs or the complete pointer ABI')
    for name, dtype, size in [('in_out_ptr0', torch.bfloat16, rows*width),
                              ('in_ptr0', torch.int64, width)]:
        value = pointers[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.numel() != size or not value.is_contiguous()):
            raise ValueError('Preserved input layout differs: ' + name)
    outputs = evaluate(pointers['in_out_ptr0'].reshape(rows, width),
                       pointers['in_ptr0'].reshape(width), scale)
    expected = outputs[formal_pointer]
    if (not isinstance(candidate, torch.Tensor) or candidate.dtype != expected.dtype
            or candidate.device != expected.device or candidate.numel() != expected.numel()
            or not candidate.is_contiguous()):
        raise ValueError('Output storage contract differs')
    return expected.reshape(candidate.shape)


def evaluate(scores, group_ids, scale):
    """Return all four declared outputs, without mutating the input scores.

    Rows flatten attention heads and query positions; IDs are shared across
    heads. Statistics use FP32 scores prior to their BF16 output write.
    """
    import torch
    if (not isinstance(scores, torch.Tensor) or scores.dtype != torch.bfloat16
            or scores.ndim != 2 or not scores.is_contiguous()):
        raise ValueError('Expected contiguous BF16 score matrix')
    rows, width = scores.shape
    if rows <= 0 or width <= 0 or rows % width or width & (width - 1):
        raise ValueError('Expected whole heads with power-of-two sequence length')
    if (not isinstance(group_ids, torch.Tensor) or group_ids.dtype != torch.int64
            or group_ids.shape != (width,) or not group_ids.is_contiguous()
            or group_ids.device != scores.device):
        raise ValueError('Expected one shared contiguous int64 group-ID vector')
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid scale')
    factor = torch.tensor(scale, dtype=torch.float32, device=scores.device)
    if not torch.isfinite(factor) or factor <= 0:
        raise ValueError('Scale is not representable as positive finite FP32')
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite input outside declared reference domain')
    query = torch.arange(rows, device=scores.device) % width
    key = torch.arange(width, device=scores.device)
    allowed = ((key[None, :] <= query[:, None])
               & (group_ids[query, None] == group_ids[None, :]))
    mask = torch.where(allowed, 0.0, float(torch.finfo(torch.bfloat16).min))
    scaled = scores.float() * factor + mask
    if not torch.isfinite(scaled).all():
        raise ValueError('Nonfinite scaled score outside declared reference domain')
    maximum = scaled.amax(-1, keepdim=True)
    exponential = (scaled - maximum).exp()
    denominator = exponential.sum(-1, keepdim=True)
    return dict(in_out_ptr0=scaled.to(torch.bfloat16),
                out_ptr0=maximum.squeeze(-1),
                out_ptr1=denominator.squeeze(-1),
                out_ptr2=(exponential / denominator).to(torch.bfloat16))
