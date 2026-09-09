"""Declared tanh-GELU backward product, independent of generated Triton code.

This is not erf-GELU and must not replace it. Source binding and physical input
layout must be checked by the caller before this mathematical reference runs.
"""


def decode_inputs(pointers, *, elements, width, stride, offset):
    """Decode the checked packed selection, preserving BF16 input values.

    Accept complete contiguous tensors, including nonzero storage offsets.
    Arbitrary views or undersized pointer slices must not be guessed into shape.
    Caller must invoke this before the candidate changes any aliased storage.
    """
    import torch
    if (any(type(v) is not int for v in (elements,width,stride,offset))
            or min(elements,width,stride)<=0 or offset<0 or elements%width
            or offset+width>stride
            or set(pointers)!={'in_ptr0','in_ptr1','in_ptr2'}):
        raise ValueError('Invalid declared GELU pointer layout')
    values=[pointers[f'in_ptr{i}'] for i in range(3)]
    sizes=(elements,(elements//width)*stride,elements)
    for value,size in zip(values,sizes):
        if (not isinstance(value,torch.Tensor) or value.dtype!=torch.bfloat16
                or value.numel()!=size or not value.is_contiguous()
                or value.device!=values[0].device or not torch.isfinite(value).all()):
            raise ValueError('GELU pointer does not match complete declared storage view')
    gradient=values[0].detach().reshape(-1).clone()
    multiplier=values[1].detach().reshape(elements//width,stride)[
        :,offset:offset+width].reshape(-1).clone()
    saved_input=values[2].detach().reshape(-1).clone()
    return gradient,multiplier,saved_input


def evaluate(gradient, multiplier, saved_input, *, accumulation_dtype=None,
             output_dtype=None):
    import math
    import torch
    tensors=(gradient,multiplier,saved_input)
    if (any(not isinstance(t,torch.Tensor) for t in tensors)
            or any(t.shape!=gradient.shape or t.device!=gradient.device for t in tensors)
            or any(t.dtype not in (torch.bfloat16,torch.float32,torch.float64) for t in tensors)
            or any(not torch.isfinite(t).all() for t in tensors)):
        raise ValueError('Finite shape-aligned GELU product operands required')
    dtype=torch.float32 if accumulation_dtype is None else accumulation_dtype
    output_dtype=gradient.dtype if output_dtype is None else output_dtype
    if dtype not in (torch.float32,torch.float64) or output_dtype not in (torch.bfloat16,torch.float32,torch.float64):
        raise ValueError('Unsupported reference precision')
    g,m,x=(t.to(dtype) for t in tensors)
    x2=x*x
    argument=(x+0.044715*(x2*x))*math.sqrt(2/math.pi)
    t=torch.tanh(argument)
    derivative=0.5*(1+t)+(0.5*x)*(1-t*t)*(1+0.134145*x2)*math.sqrt(2/math.pi)
    result=(g*m)*derivative
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite reference arithmetic')
    return result.to(output_dtype)
