"""Direct input-order depthwise cross-correlation reference.

Independent of the candidate convolution API. This is a declared numerical
reference, not real-arithmetic truth or evidence of runtime training support.
Bias is deliberately excluded: the observed generated graph adds it later.
"""


def evaluate(inputs, weight, *, padding, accumulation_dtype=None):
    import torch
    dtype = torch.float32 if accumulation_dtype is None else accumulation_dtype
    if (inputs.ndim != 3 or weight.ndim != 3 or weight.shape[1] != 1
            or weight.shape[0] != inputs.shape[1] or min(inputs.shape) <= 0
            or weight.shape[2] <= 0 or type(padding) is not int or padding < 0
            or inputs.dtype not in (torch.bfloat16, torch.float32, torch.float64)
            or weight.dtype != inputs.dtype or weight.device != inputs.device
            or dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(inputs).all() or not torch.isfinite(weight).all()):
        raise ValueError('Expected finite depthwise inputs NCL and weights C1K')
    length = inputs.shape[2] + 2*padding-weight.shape[2]+1
    if length <= 0:
        raise ValueError('Nonpositive convolution output length')
    x, w = inputs.to(dtype), weight.to(dtype)
    padded = torch.nn.functional.pad(x, (padding, padding))
    result = torch.zeros((*inputs.shape[:2], length), device=x.device, dtype=dtype)
    for k in range(weight.shape[2]):
        result = result + padded[:, :, k:k+length]*w[None, :, 0, k, None]
    if not torch.isfinite(result).all():
        raise ValueError('Reference accumulation overflow')
    return result.to(inputs.dtype)


def reference(metadata, candidate, contract):
    """Consume the existing external-call observer's pre-call snapshots.

    Source/callsite binding remains the runner's responsibility. No alternate
    observer or convolution-specific statistical formula is introduced.
    """
    import torch
    if (metadata.get('external_symbol') != 'convolution'
            or metadata.get('reference_operand_capture') != 'PRE_INVOCATION_CLONE'
            or contract.get('groups') != 1536 or contract.get('padding') != 3
            or contract.get('required_weight_shape') != [1536, 1, 4]
            or contract.get('bias_included') is not False):
        raise ValueError('Missing declared convolution contract')
    args = metadata.get('runtime_args', ())
    options = metadata.get('runtime_kwargs', {})
    expected = dict(stride=(1,), padding=(3,), dilation=(1,), transposed=False,
                    output_padding=(0,), groups=1536, bias=None)
    if (not isinstance(args, (list, tuple)) or len(args) != 2
            or any(not isinstance(x, torch.Tensor) for x in args)
            or options != expected
            or any(type(options[k]) is not type(v) for k, v in expected.items())
            or any(type(x) is not int for k in ('stride', 'padding', 'dilation', 'output_padding')
                   for x in options[k])):
        raise ValueError('Actual convolution call options differ')
    inputs, weight = args
    if (inputs.ndim != 3 or inputs.shape[1] != 1536 or tuple(weight.shape) != (1536, 1, 4)
            or inputs.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not isinstance(candidate, torch.Tensor)
            or candidate.shape != (inputs.shape[0], 1536, inputs.shape[2]+3)
            or candidate.dtype != inputs.dtype or candidate.device != inputs.device):
        raise ValueError('Actual depthwise tensor layout differs')
    return evaluate(inputs, weight, padding=3)


def evaluate_with_separate_bias(inputs, weight, bias, *, padding,
                                accumulation_dtype=None):
    """Closed conv+bias reference preserving the intermediate storage cast.

    Not a fused high-precision convolution: candidate's convolution output is
    stored in input dtype before the subsequent FP32 bias add and dtype write.
    """
    import torch
    if (inputs.dtype != torch.bfloat16 or bias.dtype != inputs.dtype
            or bias.shape != (inputs.shape[1],) or bias.device != inputs.device
            or not torch.isfinite(bias).all()):
        raise ValueError('Expected separate BF16 channel bias')
    stored = evaluate(inputs, weight, padding=padding,
                      accumulation_dtype=accumulation_dtype)
    return (stored.float()+bias.float()[None, :, None]).to(inputs.dtype)
