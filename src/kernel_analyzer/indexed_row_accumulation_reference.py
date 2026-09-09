"""Declared input-order reference for embedding's indexed row accumulation.

This is a same-dtype implementation variant, not an exact real-arithmetic oracle.
It deliberately does not call index_put_/index_add_ to reproduce the candidate.
Actual generated-call binding remains required before training measurements.
"""


def evaluate(initial, indices, values):
    """Compute initial[row] + sum(values for row), in original input order.

    Supports the observed one-index, accumulate=True, FP32 matrix operation.
    Negative indices follow ordinary row indexing; broadcasting is not supported.
    Inputs are never mutated. Repeated indices are included, never deduplicated.
    """
    import torch
    if (initial.ndim != 2 or initial.dtype != torch.float32
            or indices.ndim < 1 or indices.dtype != torch.int64
            or values.shape != (*indices.shape, initial.shape[1])
            or values.dtype != initial.dtype or initial.shape[0] <= 0
            or any(x.device != initial.device for x in (indices, values))):
        raise ValueError('Expected FP32 rows, one int64 index tensor and matching values')
    if not torch.isfinite(initial).all() or not torch.isfinite(values).all():
        raise ValueError('Finite operands required')
    rows=initial.shape[0]
    if torch.any(indices < -rows) or torch.any(indices >= rows):
        raise ValueError('Index outside declared row range')
    result=initial.clone()
    # Flatten batch/sequence in logical input order, never sort duplicate rows.
    indices=indices.reshape(-1)
    values=values.reshape(-1, initial.shape[1])
    for position in range(indices.numel()):
        row=indices[position]
        result[row]=result[row]+values[position]
    return result


def reference(metadata,candidate,contract):
    """Reference from an exactly declared generated call and pre-call snapshot."""
    import torch
    required=('source_line_sha256','executing_filename','executing_line')
    if (metadata.get('reference_operand_capture')!='PRE_INVOCATION_CLONE'
            or metadata.get('accumulate') is not True
            or metadata.get('implementation_kind')!='DIRECT_ATEN'
            or metadata.get('endpoint')!='mutated_output_0'
            or any(not contract.get(k) or metadata.get(k)!=contract[k] for k in required)):
        raise ValueError('Indexed call identity or operand capture differs')
    operands=metadata.get('indexed_operands',{})
    if any(not isinstance(operands.get(k),torch.Tensor) for k in ('initial','indices','values')):
        raise ValueError('Missing indexed operands')
    initial=operands['initial']
    if candidate.shape!=initial.shape or candidate.dtype!=initial.dtype or candidate.device!=initial.device:
        raise ValueError('Candidate output does not match declared input matrix')
    return evaluate(initial,operands['indices'],operands['values'])
