"""PyTorch 2.10 public average-pooling forward and VJP, including ceil tail windows."""
import torch
import torch.nn.functional as F


def _forward(x):
    return F.avg_pool2d(x, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0),
                        ceil_mode=True, count_include_pad=True, divisor_override=None)


_compiled = torch.compile(_forward, fullgraph=True, backend='inductor')


def call(inputs):
    if torch.__version__ != '2.10.0+cu128':
        raise RuntimeError('this call requires PyTorch 2.10.0+cu128')
    shapes = {'x': (1, 1, 4, 4), 'upstream': (1, 1, 2, 2)}
    if set(inputs) != set(shapes):
        raise ValueError('expected exactly x and upstream; all pooling options are fixed')
    for name, shape in shapes.items():
        value = inputs[name]
        if (not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != torch.float32
                or not value.is_cuda or not value.is_contiguous()):
            raise ValueError(f'{name} must be contiguous CUDA float32 {shape}')
        if not bool(torch.isfinite(value).all()) or bool((value.abs() > 2).any()):
            raise ValueError(f'{name} must be finite in [-2,2]')
    x = inputs['x'].detach().requires_grad_(True)
    y = _compiled(x)
    dx, = torch.autograd.grad(y, x, inputs['upstream'])
    return {'y': y.detach(), 'dx': dx.detach()}
