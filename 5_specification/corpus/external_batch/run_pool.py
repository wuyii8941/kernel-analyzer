"""Versioned public pooling tasks, preserving indices and complete input VJPs."""
import argparse
import contextlib
import gzip
import importlib.metadata
import inspect
import json
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F

import run as batch
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite

AVERAGE_PROFILE = 'finite_avg2d_autograd'
AVERAGE_TASK = ('Select the real idealization of PyTorch 2.10 avg_pool2d and its complete input VJP: '
                'count explicit padding when requested, exclude ceil overhang, and let divisor_override take precedence.')
MAX_PROFILE = 'torch210_unique_max2d_autograd'
MAX_TASK = ('Select PyTorch 2.10 functional max_pool2d with return_indices=true on unique-max windows, '
            'using row-major spatial indices h*W+w and the complete input VJP.')


def functional(x, kind, **options):
    if kind == 'max':
        return F.max_pool2d(x, return_indices=True, **options)
    if kind == 'avg':
        return F.avg_pool2d(x, **options)
    raise ValueError('unselected pooling task')


def execute(fn, x, upstream, options, capture=False):
    x = x.detach().clone().requires_grad_(True)
    watched = [x, upstream]
    before = [v.detach().contiguous().view(torch.uint8).clone() for v in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder else contextlib.nullcontext():
        result = fn(x, **options)
        out, indices = result if options['kind'] == 'max' else (result, None)
        forward_count = len(recorder.launches) if recorder else 0
        dx, = torch.autograd.grad(out, x, upstream)
    if x.is_cuda:
        torch.cuda.synchronize()
    if not all(torch.equal(v.detach().contiguous().view(torch.uint8), old)
               for v, old in zip(watched, before)):
        raise ValueError('input or upstream values changed')
    observed = {'out': out.detach(), 'dx': dx.detach()}
    if indices is not None:
        observed['indices'] = indices.detach()
    launches = [] if recorder is None else [
        {'phase': 'forward' if n < forward_count else 'backward', 'kernel': l.kernel_name,
         'grid': list(l.grid), 'ttir': l.asm.get('ttir')} for n, l in enumerate(recorder.launches)]
    return observed, launches, list(x.stride())


def compare(actual, standard, kind):
    required = {'out', 'dx', 'indices'} if kind == 'max' else {'out', 'dx'}
    if kind not in ('max', 'avg') or set(actual) != required or set(standard) != required:
        raise ValueError('missing or extra public output/gradient')
    for name in required:
        dtype = torch.int64 if name == 'indices' else torch.float32
        if actual[name].dtype != dtype or standard[name].dtype != dtype or actual[name].shape != standard[name].shape:
            raise ValueError(f'{name}: task shape/dtype mismatch')
    floating = batch.standard_comparison({k: actual[k] for k in ('out', 'dx')},
                                        {k: standard[k] for k in ('out', 'dx')})
    result = {'floating': floating}
    if kind == 'max':
        result['max_values_exact_equal'] = bool(torch.equal(actual['out'], standard['out']))
        result['indices'] = {'coordinates': actual['indices'].numel(),
                             'unequal_coordinates': int((actual['indices'] != standard['indices']).sum()),
                             'exact_equal': bool(torch.equal(actual['indices'], standard['indices']))}
    result['compatible'] = all(not v['nonfinite_pairs'] and v['allclose_rtol_1e-5_atol_1e-6']
                               for v in floating.values()) and result.get('indices', {}).get('exact_equal', True)
    result['compatible'] &= result.get('max_values_exact_equal', True)
    return result


def mathematical_values(x, options):
    # This is deliberately a projection to values, never certification of the
    # complete with-indices/autograd call. Do not guess an average-pooling formula.
    if options['kind'] != 'max':
        return {'outcome': 'unsupported', 'note': 'No admitted average-pooling mathematical profile'}, None
    q = {'op': 'aten.max_pool2d', 'args': [batch.record(x)],
         'kwargs': {k: v for k, v in options.items() if k != 'kind'}}
    ref = registry.query('POOL', 'finite_max2d_values', q)
    info = {'query': q, 'outcome': ref.outcome, 'note': ref.note,
            'scope': 'forward values only; indices and dx not covered'}
    if ref.outcome != 'value':
        return info, None
    expected = ref.fields['out']
    info['out'] = batch.map_values(expected, encode_finite)
    return info, expected


def mathematical_reference(x, upstream, options):
    if options['kind'] == 'max':
        q = {'op': 'torch.nn.functional.max_pool2d', 'x': batch.record(x),
             'upstream': batch.record(upstream), 'stride': None, 'padding': 0,
             'dilation': 1, 'ceil_mode': False, 'return_indices': True,
             **{k: v for k, v in options.items() if k != 'kind'}}
        ref = registry.query('POOL', MAX_PROFILE, q, task_bindings={MAX_PROFILE: MAX_TASK})
        info = {'query': q, 'outcome': ref.outcome, 'note': ref.note, 'task_choice': MAX_TASK,
                'scope': 'complete unique-max forward values, int64 indices and input VJP',
                'labels': ref.labels, 'output_mapping': {'out': 'out', 'indices': 'indices', 'grad': 'dx'}}
        if ref.outcome != 'value':
            return info, None
        if set(ref.fields) != {'out', 'indices', 'grad'}:
            raise ValueError('max-pooling reference omitted or added task outputs')
        expected = {'out': ref.fields['out'], 'indices': ref.fields['indices'], 'dx': ref.fields['grad']}
        info['fields'] = {k: batch.map_values(v, encode_finite) for k, v in expected.items()}
        return info, expected
    if options['kind'] != 'avg':
        return {'outcome': 'unsupported', 'note': 'unselected pooling task'}, None
    q = {'op': 'avg_pool2d', 'x': batch.record(x), 'upstream': batch.record(upstream),
         'stride': None, 'padding': 0, 'ceil_mode': False, 'count_include_pad': True,
         'divisor_override': None, **{k: v for k, v in options.items() if k != 'kind'}}
    ref = registry.query('POOL', AVERAGE_PROFILE, q, task_bindings={AVERAGE_PROFILE: AVERAGE_TASK})
    info = {'query': q, 'outcome': ref.outcome, 'note': ref.note, 'task_choice': AVERAGE_TASK,
            'scope': 'complete average-pooling forward and input VJP', 'labels': ref.labels,
            'output_mapping': {'out': 'out', 'grad': 'dx'}}
    if ref.outcome != 'value':
        return info, None
    if set(ref.fields) != {'out', 'grad'}:
        raise ValueError('average-pooling reference omitted or added task outputs')
    expected = {'out': ref.fields['out'], 'dx': ref.fields['grad']}
    info['fields'] = {k: batch.map_values(v, encode_finite) for k, v in expected.items()}
    return info, expected


def mathematical_comparison(observed, expected, name):
    if name != 'indices':
        return batch.reference_comparison(observed, expected)
    wanted = torch.tensor(expected, dtype=torch.int64, device=observed.device)
    if observed.dtype != torch.int64 or observed.shape != wanted.shape:
        raise ValueError('indices: task shape/dtype mismatch')
    unequal = int((observed != wanted).sum())
    return {'coordinates': observed.numel(), 'unequal_coordinates': unequal,
            'exact_equal': unequal == 0, 'meaning': 'Exact discrete comparison; no floating tolerance.'}


def inputs(case, dimensions, seed, device='cuda'):
    generator = torch.Generator().manual_seed(seed)
    shape = tuple(dimensions[1:] if case.get('unbatched') else dimensions)
    stored = (*shape[:-2], shape[-1], shape[-2]) if case.get('noncontiguous') else shape
    if case.get('unique'):
        count = int(torch.tensor(stored).prod())
        x = ((torch.randperm(count, generator=generator).reshape(stored) - count // 2).float() / 8).to(device)
    else:
        x = (torch.randint(-16, 17, stored, generator=generator).float() / 8).to(device)
    if case.get('noncontiguous'):
        x = x.transpose(-1, -2)
    if case.get('constant'):
        x.fill_(2.)
    options = {'kind': case['kind'], **case['options']}
    # Only the explicitly chosen native reference determines the output shape.
    output = functional(x, **options)
    values = output[0] if case['kind'] == 'max' else output
    upstream = (torch.randint(-8, 9, values.shape, generator=generator).float() / 8).to(device)
    return x, upstream, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'acceptance'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path, default=Path(__file__).with_name('selection_pool_20261019_v1.json'))
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    split = selection[args.split]
    args.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('standard version mismatch')
    spec_root = Path(__file__).resolve().parents[2]
    report = {'selection': selection, 'split': args.split, 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(), 'samples': [],
              'sources': {'runner': Path(__file__).read_text(), 'comparison': Path(batch.__file__).read_text(),
                          'torch_functional': Path(inspect.getfile(F)).read_text()},
              'specification_sources': {name: (spec_root / name).read_text() for name in
                                         ('contracts/POOL.yaml', 'adapters/POOL.py', 'kernel_spec/e2_pool.py')},
              'official_documents': {name: gzip.decompress((spec_root / 'sources' / (name + '.txt.gz')).read_bytes()).decode()
                                     for name in ('T_MAXPOOL2D', 'T_F_MAX_POOL2D', 'T_AVGPOOL2D', 'T_F_AVG_POOL2D')},
              'formal_binding': 'not_established', 'bias_verdict': 'not_established',
              'reference_protocol': 'Complete unique-max PyTorch 2.10 max-pooling and selected real average-pooling references; availability is recorded per call.',
              'capture_setup': 'Recorder enabled before warmup with no launches selected; static CUDA host launcher disabled; no precision setting changed.'}
    if split['implementation'] == 'flag_gems':
        import flag_gems
        dist = importlib.metadata.distribution('flag_gems')
        if dist.version != '5.0.2':
            raise ValueError('candidate version mismatch')
        report['candidate_sources'] = {p: Path(dist.locate_file('flag_gems/' + p)).read_text()
                                      for p in ('__init__.py', 'ops/max_pool2d_with_indices.py', 'ops/avg_pool2d.py')}
        report['candidate_metadata'], report['candidate_license'] = dist.read_text('METADATA'), dist.read_text('LICENSE')
        candidate = functional
        context = lambda: flag_gems.use_gems(include=['max_pool2d_with_indices', 'max_pool2d_backward',
                                                    'avg_pool2d', 'avg_pool2d_backward'])
    else:
        candidate, context = torch.compile(functional, fullgraph=True, dynamic=True), contextlib.nullcontext
        report['compiler_setup'] = 'Reset Dynamo between task cases, so ten distinct APIs/options do not exhaust one Python frame recompile limit; numerical compiler options unchanged.'
    for case in split.get('cases', selection.get('cases', [])):
        if split['implementation'] == 'inductor':
            torch._dynamo.reset()
            candidate = torch.compile(functional, fullgraph=True, dynamic=True)
        for seed in split['seeds']:
            row = {'case': case['id'], 'seed': seed}
            report['samples'].append(row)
            print('START', case['id'], seed, flush=True)
            try:
                x, upstream, options = inputs(case, split['dimensions'], seed)
                row.update(options=options, input=batch.record(x), upstream=batch.record(upstream),
                           original_input_strides=list(x.stride()))
                row['mathematical_reference'], expected = mathematical_reference(x, upstream, options)
                standard, _, _ = execute(functional, x, upstream, options)
                row['standard'] = {k: batch.record(v) for k, v in standard.items()}
                if expected is not None:
                    row['standard_math'] = {k: mathematical_comparison(standard[k], v, k) for k, v in expected.items()}
                with context():
                    with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                        execute(candidate, x, upstream, options)
                    actual, launches, strides = execute(candidate, x, upstream, options, capture=True)
                row.update(candidate={k: batch.record(v) for k, v in actual.items()}, launches=launches,
                           observed_input_strides=strides, input_values_unchanged=True,
                           comparison=compare(actual, standard, options['kind']))
                if expected is not None:
                    row['candidate_math'] = {k: mathematical_comparison(actual[k], v, k) for k, v in expected.items()}
                captured_phases = sorted({l['phase'] for l in launches})
                missing_phases = sorted({'forward', 'backward'} - set(captured_phases))
                row['triton_phase_coverage'] = {'expected': ['backward', 'forward'],
                                                'captured': captured_phases,
                                                'missing': missing_phases}
                row['status'] = 'compared' if not missing_phases else 'compared_partial_triton'
            except Exception as exc:
                row.update(status='execution_or_reference_error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('END', case['id'], seed, row['status'], row.get('error', ''), flush=True)
    raise SystemExit(int(any(s['status'] != 'compared' or not s.get('comparison', {}).get('compatible', False)
                             for s in report['samples'])))


if __name__ == '__main__':
    main()
