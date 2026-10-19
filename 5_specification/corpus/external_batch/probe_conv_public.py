"""Uninstrumented public-call controls for the observed FlagGems rejections."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F

import run as batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    import flag_gems
    assert torch.__version__ == '2.10.0+cu128'
    assert importlib.metadata.version('flag_gems') == '5.0.2'
    cases = [
        ('same_1d_default_stride', F.conv1d, (1, 1, 5), (1, 1, 2), {'padding': 'same'}),
        ('same_2d_default_stride', F.conv2d, (1, 1, 4, 5), (1, 1, 2, 2), {'padding': 'same'}),
        ('valid_2d_unbatched', F.conv2d, (1, 4, 5), (1, 1, 2, 2), {'padding': 'valid'}),
    ]
    report = {'source': Path(__file__).read_text(), 'torch': torch.__version__,
              'flag_gems': '5.0.2', 'instrumentation': 'none', 'cases': [],
              'schemas': {name: str(getattr(torch.ops.aten, name).padding._schema) for name in ('conv1d', 'conv2d')}}
    for name, fn, xshape, wshape, options in cases:
        row = {'case': name, 'options': options}
        report['cases'].append(row)
        for engine in ('native_cpu', 'native_cuda', 'flag_gems'):
            device = 'cpu' if engine == 'native_cpu' else 'cuda'
            x = torch.arange(1, 1 + int(torch.tensor(xshape).prod()), dtype=torch.float32, device=device).reshape(xshape).requires_grad_()
            w = torch.ones(wshape, device=device, requires_grad=True)
            b = torch.ones(wshape[0], device=device, requires_grad=True)
            try:
                if engine == 'flag_gems':
                    with flag_gems.use_gems(include=['conv1d', 'conv2d']):
                        y = fn(x, w, b, **options)
                        gradients = torch.autograd.grad(y, (x, w, b), torch.ones_like(y))
                else:
                    y = fn(x, w, b, **options)
                    gradients = torch.autograd.grad(y, (x, w, b), torch.ones_like(y))
                if device == 'cuda':
                    torch.cuda.synchronize()
                row[engine] = {'status': 'executed', 'out': batch.record(y),
                               'gradients': [batch.record(g) for g in gradients]}
            except Exception as exc:
                row[engine] = {'status': 'error', 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}
        print(name, {e: row[e]['status'] for e in ('native_cpu', 'native_cuda', 'flag_gems')}, flush=True)
    with args.out.open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
