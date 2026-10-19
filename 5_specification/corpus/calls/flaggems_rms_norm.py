"""FlagGems 5.0.2 replacing public PyTorch RMSNorm with its default epsilon."""
import importlib.metadata

import flag_gems
import torch
import torch.nn.functional as F


def call(inputs):
    if importlib.metadata.version('flag_gems') != '5.0.2' or torch.__version__ != '2.10.0+cu128':
        raise RuntimeError('this call requires FlagGems 5.0.2 and PyTorch 2.10.0+cu128')
    shapes = {'x': (2, 7), 'weight': (7,), 'upstream': (2, 7)}
    if set(inputs) != set(shapes):
        raise ValueError('expected exactly x, weight and upstream; shape and public eps=None are fixed')
    for name, shape in shapes.items():
        t = inputs[name]
        if (not torch.is_tensor(t) or tuple(t.shape) != shape or t.dtype != torch.float32
                or not t.is_cuda or not t.is_contiguous()):
            raise ValueError(f'{name} must be contiguous CUDA float32 {shape}')
        if not bool(torch.isfinite(t).all()) or bool((t.abs() > 8).any()):
            raise ValueError(f'{name} must be finite in [-8,8]')
    x = inputs['x'].detach().requires_grad_(True)
    weight = inputs['weight'].detach().requires_grad_(True)
    with flag_gems.use_gems(include=['rms_norm']):
        y = F.rms_norm(x, (7,), weight, eps=None)
        dx, dweight = torch.autograd.grad(y, (x, weight), inputs['upstream'])
    return {'y': y.detach(), 'dx': dx.detach(), 'dweight': dweight.detach()}
