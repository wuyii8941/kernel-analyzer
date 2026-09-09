"""View dense tensor storage in pointer order without a logical-order copy."""
import math
import torch


def dense_pointer_view(value, shape):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError('Strided tensor required')
    if any(type(n) is not int or n <= 0 for n in shape) or math.prod(shape) != value.numel():
        raise ValueError('Declared pointer extent differs')
    # Accept permutations of a dense allocation, not slices with gaps or
    # overlapping/expanded tensors. Singleton strides do not affect addressing.
    expected = 1
    for stride, size in sorted((s, n) for n, s in zip(value.shape, value.stride()) if n > 1):
        if stride != expected:
            raise ValueError('Pointer snapshot is not nonoverlapping dense storage')
        expected *= size
    available = value.untyped_storage().nbytes()//value.element_size()-value.storage_offset()
    if available < value.numel():
        raise ValueError('Pointer extent exceeds storage')
    flat = value.as_strided((value.numel(),), (1,), value.storage_offset())
    return flat.reshape(shape)
