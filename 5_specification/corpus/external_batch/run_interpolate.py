"""Real explicit-size nearest interpolation calls with complete forward and input VJP."""
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

PROFILE = 'torch210_nearest2d_size_autograd'
TASK = ('Select PyTorch 2.10 rank-4 functional interpolate with explicit size and legacy nearest coordinates '
        'floor(output_index*input_size/output_size), plus the complete input VJP.')


def functional(x, *, size, scale_factor=None, mode='nearest', align_corners=None,
               recompute_scale_factor=None, antialias=False):
    return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode,
                         align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
                         antialias=antialias)


def public_options(case):
    return {'size': list(case['size']), 'scale_factor': None, 'mode': 'nearest', 'align_corners': None,
            'recompute_scale_factor': None, 'antialias': False}


def inputs(case, dimensions, seed, device):
    generator = torch.Generator().manual_seed(seed)
    shape = tuple(dimensions)
    stored_shape = (*shape[:-2], shape[-1], shape[-2]) if case.get('noncontiguous') else shape
    x = (torch.randint(-24, 25, stored_shape, generator=generator).float() / 8).to(device)
    if case.get('noncontiguous'):
        x = x.transpose(-1, -2)
    options = public_options(case)
    output_shape = x.shape[:2] + tuple(options['size'])
    upstream = (torch.randint(-8, 9, output_shape, generator=generator).float() / 8).to(device)
    return x, upstream, options


def execute(fn, x, upstream, options, capture=False):
    x = x.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    upstream = upstream.detach().clone(memory_format=torch.preserve_format)
    watched = [x, upstream]
    before = [value.detach().contiguous().view(torch.uint8).clone() for value in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder else contextlib.nullcontext():
        out = fn(x, **options)
        forward_count = len(recorder.launches) if recorder else 0
        dx, = torch.autograd.grad(out, x, upstream)
    if x.is_cuda:
        torch.cuda.synchronize()
    if any(not torch.equal(value.detach().contiguous().view(torch.uint8), old)
           for value, old in zip(watched, before)):
        raise ValueError('input or upstream values changed')
    launches = [] if recorder is None else [
        {'phase': 'forward' if index < forward_count else 'backward', 'kernel': launch.kernel_name,
         'grid': list(launch.grid), 'ttir': launch.asm.get('ttir')}
        for index, launch in enumerate(recorder.launches)]
    return {'out': out.detach(), 'dx': dx.detach()}, launches, list(x.stride())


def mathematical_reference(x, upstream, options):
    query = {'op': 'torch.nn.functional.interpolate', 'x': batch.record(x),
             'upstream': batch.record(upstream), **options}
    result = registry.query('INTERP', PROFILE, query, task_bindings={PROFILE: TASK})
    info = {'query': query, 'outcome': result.outcome, 'note': result.note, 'labels': result.labels,
            'task_choice': TASK, 'scope': 'complete forward and input VJP',
            'output_mapping': {'out': 'out', 'grad': 'dx'}}
    if result.outcome != 'value':
        return info, None
    if set(result.fields) != {'out', 'grad'}:
        raise ValueError('interpolation reference omitted or added task outputs')
    expected = {'out': result.fields['out'], 'dx': result.fields['grad']}
    info['fields'] = {name: batch.map_values(value, encode_finite) for name, value in expected.items()}
    return info, expected


def exact_tensor_comparison(actual, standard):
    if set(actual) != {'out', 'dx'} or set(standard) != {'out', 'dx'}:
        raise ValueError('complete interpolation call requires out and dx')
    comparison = batch.standard_comparison(actual, standard)
    exact = {}
    for name in ('out', 'dx'):
        if actual[name].dtype != torch.float32 or standard[name].dtype != torch.float32:
            raise ValueError(f'{name}: selected task requires float32')
        exact[name] = bool(torch.equal(actual[name], standard[name]))
    return {'standard': comparison, 'exact_equal': exact, 'compatible': all(exact.values())}


def exact_math_comparison(observed, expected):
    return {name: batch.reference_comparison(observed[name], expected[name]) for name in ('out', 'dx')}


def math_exact(comparison):
    return all(item['outside_mathematical_enclosure'] == 0 and item['inside_nonpoint_unresolved'] == 0
               and item['nonfinite'] == 0 and item['exact_equal'] == item['coordinates']
               for item in comparison.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'acceptance'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path,
                        default=Path(__file__).with_name('selection_interpolate_nearest_20261019_v1.json'))
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    split = selection[args.split]
    args.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('standard version mismatch')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')

    spec_root = Path(__file__).resolve().parents[2]
    report = {
        'selection': selection, 'split': args.split, 'torch': torch.__version__, 'device': args.device,
        'samples': [], 'formal_binding': 'not_established', 'bias_verdict': 'not_established',
        'reference_protocol': ('Complete PyTorch 2.10 legacy-nearest forward and input VJP, cross-checked against '
                               'an independent exact coordinate reference.'),
        'sources': {'runner': Path(__file__).read_text(), 'comparison': Path(batch.__file__).read_text(),
                    'torch_functional': Path(inspect.getfile(F)).read_text()},
        'specification_sources': {name: (spec_root / name).read_text()
                                  for name in ('contracts/INTERP.yaml', 'adapters/INTERP.py')},
        'retained_official_sources': ['sources/TORCH210_NN_FUNCTIONAL.py.gz',
                                      'sources/TORCH210_DECOMPOSITIONS.py.gz',
                                      'sources/TS_NATIVE_FUNCTIONS.yaml.gz'],
        'capture_setup': ('For CUDA candidates the recorder is enabled before warmup with no warmup launches '
                          'selected; static CUDA host launch is disabled and numerical precision is unchanged.')
    }

    candidate = context = None
    if args.device == 'cuda' and split['implementation'] == 'flag_gems':
        import flag_gems
        distribution = importlib.metadata.distribution('flag_gems')
        if distribution.version != '5.0.2':
            raise ValueError('candidate version mismatch')
        report['candidate_sources'] = {path: Path(distribution.locate_file('flag_gems/' + path)).read_text()
                                       for path in ('__init__.py', 'ops/upsample_nearest2d.py')}
        report['candidate_metadata'] = distribution.read_text('METADATA')
        report['candidate_license'] = distribution.read_text('LICENSE')
        report['candidate_function_selection'] = ['upsample_nearest2d']
        report['candidate_scope'] = ('FlagGems replaces only the forward upsample_nearest2d operator. The public '
                                     'input VJP is computed by the native PyTorch backward and is context evidence, '
                                     'not a FlagGems Triton-backward result.')
        candidate, context = functional, lambda: flag_gems.use_gems(include=['upsample_nearest2d'])
    elif args.device == 'cuda' and split['implementation'] == 'inductor':
        report['candidate_scope'] = 'TorchInductor compiles the complete forward and input VJP.'
        candidate, context = torch.compile(functional, fullgraph=True, dynamic=True), contextlib.nullcontext

    device = torch.device(args.device)
    for case in split['cases']:
        if args.device == 'cuda' and split['implementation'] == 'inductor':
            torch._dynamo.reset()
            candidate = torch.compile(functional, fullgraph=True, dynamic=True)
        for seed in split['seeds']:
            row = {'case': case['id'], 'seed': seed}
            report['samples'].append(row)
            print('START', case['id'], seed, flush=True)
            try:
                x, upstream, options = inputs(case, split['dimensions'], seed, device)
                row.update(options=options, input=batch.record(x), upstream=batch.record(upstream),
                           original_input_strides=list(x.stride()))
                row['mathematical_reference'], expected = mathematical_reference(x, upstream, options)
                if expected is None:
                    raise ValueError('selected mathematical reference is unavailable')
                standard, _, standard_strides = execute(functional, x, upstream, options)
                row['standard'] = {name: batch.record(value) for name, value in standard.items()}
                row['standard_input_strides'] = standard_strides
                row['standard_math'] = exact_math_comparison(standard, expected)
                if not math_exact(row['standard_math']):
                    raise ValueError('versioned standard differs from the selected exact reference')

                if args.device == 'cpu':
                    row['status'] = 'reference_crosschecked'
                else:
                    with context():
                        with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                            execute(candidate, x, upstream, options)
                        actual, launches, strides = execute(candidate, x, upstream, options, capture=True)
                    row.update(candidate={name: batch.record(value) for name, value in actual.items()},
                               launches=launches, observed_input_strides=strides,
                               input_values_unchanged=True,
                               comparison=exact_tensor_comparison(actual, standard),
                               candidate_math=exact_math_comparison(actual, expected))
                    captured = sorted({launch['phase'] for launch in launches})
                    required_phases = {'forward'} if split['implementation'] == 'flag_gems' else {'forward', 'backward'}
                    missing = sorted(required_phases - set(captured))
                    row['triton_phase_coverage'] = {'expected': sorted(required_phases),
                                                    'captured': captured, 'missing': missing}
                    if not row['comparison']['compatible'] or not math_exact(row['candidate_math']):
                        row['status'] = 'compared_mismatch'
                    else:
                        row['status'] = 'compared' if not missing else 'compared_partial_triton'
            except Exception as error:
                row.update(status='execution_or_reference_error', error=f'{type(error).__name__}: {error}',
                           traceback=traceback.format_exc())
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('END', case['id'], seed, row['status'], row.get('error', ''), flush=True)

    if args.device == 'cpu':
        failed = any(sample['status'] != 'reference_crosschecked' for sample in report['samples'])
    else:
        failed = any(sample['status'] != 'compared'
                     or not sample.get('comparison', {}).get('compatible', False)
                     or not math_exact(sample.get('candidate_math', {}))
                     for sample in report['samples'])
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
