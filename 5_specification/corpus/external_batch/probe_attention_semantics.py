"""Uninstrumented square-shape checks for the FlagGems public SDPA call."""
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import run as batch
import run_attention as attention_batch


def make_inputs(case, seed):
    generator = torch.Generator().manual_seed(seed)

    def draw(shape):
        return (torch.randint(-4, 5, shape, generator=generator).float() / 8).cuda()

    hq, hkv = case.get('heads', [1, 1])
    n, d = 128, 16
    q = draw((1, hq, n, d))
    k = draw((1, hkv, n, d))
    v = draw((1, hkv, n, d))
    upstream = draw((1, hq, n, d))
    mask = None
    if case.get('mask') == 'boolean':
        mask = torch.tensor([[(i + 2*j) % 3 != 0 for j in range(n)] for i in range(n)], device='cuda')[None, None]
    elif case.get('mask') == 'additive':
        mask = torch.tensor([[(i-j) % 4 / -4 for j in range(n)] for i in range(n)],
                            dtype=torch.float32, device='cuda')[None, None]
    options = {name: case.get(name, default) for name, default in
               (('is_causal', False), ('scale', None), ('enable_gqa', False))}
    return q, k, v, mask, upstream, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('standard version mismatch')

    import flag_gems
    distribution = importlib.metadata.distribution('flag_gems')
    if distribution.version != '5.0.2':
        raise ValueError('candidate version mismatch')
    candidate = flag_gems.scaled_dot_product_attention
    source_path = Path(distribution.locate_file('flag_gems/ops/attention.py'))
    cases = [
        {'id': 'causal_square_control', 'is_causal': True, 'scale': 0.25},
        {'id': 'noncausal_square', 'is_causal': False, 'scale': 0.25},
        {'id': 'boolean_mask_square', 'mask': 'boolean', 'scale': 0.25},
        {'id': 'additive_mask_square', 'mask': 'additive', 'scale': 0.25},
        {'id': 'gqa_causal_square', 'heads': [4, 2], 'enable_gqa': True, 'is_causal': True, 'scale': 0.25},
    ]
    report = {
        'purpose': 'Separate public-call behavior from launch recording and rectangular-shape effects.',
        'instrumentation': 'none',
        'torch': torch.__version__,
        'flag_gems': distribution.version,
        'device': torch.cuda.get_device_name(),
        'seed': 20261319,
        'cases': [],
        'candidate_source': source_path.read_text(),
        'standard_source': Path(inspect.getfile(F)).read_text(),
    }
    for case in cases:
        row = {'case': case}
        report['cases'].append(row)
        print('START', case['id'], flush=True)
        try:
            q, k, v, mask, upstream, options = make_inputs(case, report['seed'])
            expected, _ = attention_batch.execute(attention_batch.standard, q, k, v, mask, upstream, options)
            actual, _ = attention_batch.execute(candidate, q, k, v, mask, upstream, options)
            row.update(
                status='compared',
                inputs={name: attention_batch.record(value) for name, value in
                        (('q', q), ('k', k), ('v', v), ('mask', mask), ('upstream', upstream))},
                options=options,
                standard={name: attention_batch.record(value) for name, value in expected.items()},
                candidate={name: attention_batch.record(value) for name, value in actual.items()},
                comparison=batch.standard_comparison(actual, expected),
                input_values_unchanged=True,
            )
        except Exception as exc:
            row.update(status='execution_error', error=f'{type(exc).__name__}: {exc}')
        print('END', case['id'], row['status'], flush=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if any(row['status'] != 'compared' for row in report['cases']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
