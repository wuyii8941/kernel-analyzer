"""Public normalization tasks reuse the admitted NORM reference and full VJP observation."""
import argparse
import contextlib
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F

import run as batch
import run_matrix as observation
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite


def functional(operands, kind, eps, normalized_shape=None, num_groups=None):
    x, weight, bias = operands
    if kind == 'layer_norm':
        return F.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)
    if kind == 'rms_norm':
        return F.rms_norm(x, normalized_shape, weight=weight, eps=eps)
    if kind == 'group_norm':
        return F.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps)
    raise ValueError('unselected normalization task')


def inputs(case, dimensions, seed):
    n, height, width = dimensions
    generator = torch.Generator().manual_seed(seed)
    def draw(shape):
        return (torch.randint(-8, 9, shape, generator=generator).float() / 8).cuda()
    if case['op'] == 'group_norm':
        channels = case['groups'] * 2
        x = draw((n, width, channels)).transpose(1, 2) if case.get('noncontiguous') else draw((n, channels, width))
        parameter_shape = (channels,)
        options = {'num_groups': case['groups']}
    else:
        shape = (n, height, width) if case.get('multi_axis') else (n, width)
        x = draw(tuple(reversed(shape))).t() if case.get('noncontiguous') else draw(shape)
        parameter_shape = shape[1:]
        options = {'normalized_shape': list(parameter_shape)}
    if case.get('constant'):
        x.fill_(2.)
    values = [x, draw(parameter_shape) if case['weight'] else None,
              draw(parameter_shape) if case['bias'] else None]
    return values, draw(tuple(x.shape)), {'kind': case['op'], 'eps': case['eps'], **options}


def mathematical_query(operands, upstream, options):
    # The selected batch is physical finite float32. Nested NORM inputs cannot
    # preserve empty trailing dimensions, so those tasks must not be inferred.
    for v in [*operands, upstream]:
        if v is not None and (v.dtype != torch.float32 or v.numel() == 0 or not bool(torch.isfinite(v).all())):
            raise ValueError('normalization batch requires finite nonempty float32 tensors')
    x, weight, bias = operands
    op = options['kind']
    eps = options['eps']
    if eps is None:
        if op != 'rms_norm':
            raise ValueError('only public RMSNorm defines eps=None in this task')
        eps = torch.finfo(x.dtype).eps
    def values(v):
        return None if v is None else v.detach().cpu().tolist()
    q = {'op': op, 'x': values(x), 'weight': values(weight), 'eps': eps, 'upstream': values(upstream)}
    if op != 'rms_norm':
        q['bias'] = values(bias)
    elif bias is not None:
        raise ValueError('RMSNorm has no bias operand')
    if op == 'group_norm':
        q['num_groups'] = options['num_groups']
    else:
        q['normalized_shape'] = options['normalized_shape']
    return q


def complete_reference(fields, operands):
    expected_names = {k for k, v in zip(('x', 'weight', 'bias'), operands) if v is not None}
    if set(fields) != {'y', 'grad'} or set(fields['grad']) != expected_names:
        raise ValueError('reference does not return the complete public task')
    return {'out': fields['y'], **{f'g{i}': fields['grad'].get(k)
            for i, k in enumerate(('x', 'weight', 'bias'))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'acceptance'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(Path(__file__).with_name('selection_norm_20261019_v1.json').read_text())
    split = selection[args.split]
    args.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('standard version mismatch')
    report = {'selection': selection, 'split': args.split, 'torch': torch.__version__, 'samples': [],
              'sources': {'runner': Path(__file__).read_text(), 'observation': Path(observation.__file__).read_text(),
                          'comparison': Path(batch.__file__).read_text()},
              'formal_binding': 'not_established', 'bias_verdict': 'not_established',
              'capture_setup': 'Recorder enabled before warmup with no warmup launches selected; recorder disables static CUDA host launcher; no precision setting changed.'}
    if split['implementation'] == 'flag_gems':
        import flag_gems
        dist = importlib.metadata.distribution('flag_gems')
        if dist.version != '5.0.2':
            raise ValueError('candidate version mismatch')
        report['candidate_sources'] = {p: Path(dist.locate_file('flag_gems/' + p)).read_text()
                                      for p in ['__init__.py', 'ops/layernorm.py', 'ops/rms_norm.py', 'ops/groupnorm.py']}
        report['candidate_metadata'], report['candidate_license'] = dist.read_text('METADATA'), dist.read_text('LICENSE')
        candidate = functional
        # use_gems selects Python function names, not ATen dispatcher names.
        report['candidate_function_selection'] = ['layer_norm', 'layer_norm_backward',
                                                  'group_norm', 'group_norm_backward', 'rms_norm']
        context = lambda: flag_gems.use_gems(include=report['candidate_function_selection'])
    else:
        candidate, context = torch.compile(functional, fullgraph=True, dynamic=True), contextlib.nullcontext
    for case in selection['cases']:
        for seed in split['seeds']:
            row = {'case': case['id'], 'seed': seed}
            report['samples'].append(row)
            print('START', case['id'], seed, flush=True)
            try:
                operands, upstream, options = inputs(case, split['dimensions'], seed)
                row.update(options=options, inputs=[observation.record(v) for v in operands], upstream=batch.record(upstream),
                           original_input_strides=[None if v is None else list(v.stride()) for v in operands])
                q = mathematical_query(operands, upstream, options)
                ref = registry.query('NORM', 'torch_autograd', q)
                row['reference'] = {'query': q, 'outcome': ref.outcome, 'note': ref.note}
                if ref.outcome != 'value':
                    raise ValueError('complete mathematical reference unavailable')
                expected = complete_reference(ref.fields, operands)
                row['mathematical'] = {k: None if v is None else batch.map_values(v, encode_finite) for k, v in expected.items()}
                standard, _, _ = observation.execute(functional, operands, upstream, options)
                row['standard'] = {k: observation.record(v) for k, v in standard.items()}
                row['standard_math'] = {k: batch.reference_comparison(v, expected[k]) for k, v in standard.items() if v is not None}
                with context():
                    with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                        observation.execute(candidate, operands, upstream, options)
                    actual, launches, strides = observation.execute(candidate, operands, upstream, options, capture=True)
                row.update(candidate={k: observation.record(v) for k, v in actual.items()}, launches=launches,
                           observed_input_strides=strides, input_values_unchanged=True,
                           comparison=observation.compare(actual, standard, expected))
                if {l['phase'] for l in launches} != {'forward', 'backward'}:
                    raise ValueError('forward/backward Triton execution not fully captured')
                row['status'] = 'compared'
            except Exception as exc:
                row.update(status='execution_or_reference_error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('END', case['id'], seed, row['status'], row.get('error', ''), flush=True)
    failed = any(s['status'] != 'compared' or any(c['nonfinite_pairs'] or not c['allclose_rtol_1e-5_atol_1e-6']
                  for c in s['comparison']['standard'].values()) for s in report['samples'])
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
