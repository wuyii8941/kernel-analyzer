"""Uninstrumented versioned pooling witness, followed by a separate dispatcher trace."""
import argparse
import contextlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
import flag_gems

import run as batch
import run_pool as runner


def average(x):
    return F.avg_pool2d(x, 3, stride=2, padding=0, ceil_mode=True, count_include_pad=True)


def observe(fn, device):
    x = torch.ones(1, 1, 4, 4, device=device, requires_grad=True)
    y = fn(x)
    upstream = torch.zeros_like(y)
    upstream[..., -1, -1] = 1.
    dx, = torch.autograd.grad(y, x, upstream)
    changed = x.detach().clone()
    changed[..., -1, -1] += .5
    delta = (fn(changed)[..., -1, -1] - y.detach()[..., -1, -1]) / .5
    return {'out': batch.record(y), 'dx': batch.record(dx),
            'last_output_last_input_difference_quotient_step_half': float(delta.item())}


class DispatchTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if torch.__version__ != '2.10.0+cu128' or importlib.metadata.version('flag_gems') != '5.0.2':
        raise ValueError('version mismatch')
    report = {'torch': torch.__version__, 'flag_gems': '5.0.2', 'recorder': 'not enabled',
              'scope': 'Specific versioned task, not a new admitted mathematical profile or statistical bias claim',
              'source': Path(__file__).read_text(), 'observations': {}}
    compiled = torch.compile(average, fullgraph=True)
    for label, fn, device, context in (
        ('native_cpu', average, 'cpu', contextlib.nullcontext()),
        ('native_cuda', average, 'cuda', contextlib.nullcontext()),
        ('flag_gems', average, 'cuda', flag_gems.use_gems(include=['avg_pool2d', 'avg_pool2d_backward'])),
        ('inductor', compiled, 'cuda', contextlib.nullcontext()),
    ):
        with context:
            report['observations'][label] = observe(fn, device)
    # Trace separately: it must not be the cause of the uninstrumented observations.
    trace = DispatchTrace()
    x = torch.arange(16, dtype=torch.float32, device='cuda').reshape(1, 1, 4, 4)
    with flag_gems.use_gems(include=['max_pool2d_with_indices', 'max_pool2d_backward']), trace:
        result, _, _ = runner.execute(runner.functional, x, torch.ones(1, 1, 2, 2, device='cuda'),
                                     {'kind': 'max', 'kernel_size': 2})
    report['max_public_dispatch'] = trace.ops
    report['max_public_outputs'] = {k: batch.record(v) for k, v in result.items()}
    import torch._inductor.lowering as lowering
    report['inductor_source'] = Path(inspect.getfile(lowering)).read_text()
    with args.out.open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
