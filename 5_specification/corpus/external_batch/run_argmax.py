"""Real public argmax calls that reuse the admitted REDUCE reference."""
import argparse
import contextlib
import importlib.metadata
import json
import math
from pathlib import Path
import traceback

import torch

import run as batch
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite


def functional(x, *, dim, keepdim):
    return torch.argmax(x, dim=dim, keepdim=keepdim)


def inputs(case, dimensions, seed, device):
    generator = torch.Generator().manual_seed(seed)
    shape = tuple(dimensions)
    stored_shape = (*shape[:-2], shape[-1], shape[-2]) if case.get('noncontiguous') else shape
    count = math.prod(stored_shape)
    x = ((torch.randperm(count, generator=generator).reshape(stored_shape) - count // 2).float() / 8).to(device)
    if case.get('noncontiguous'):
        x = x.transpose(-1, -2)
    dim = case['dim']
    if case.get('tie'):
        high = float(count + 1)
        if dim is None:
            if not x.is_contiguous():
                raise ValueError('the frozen flat-tie case must be contiguous')
            x.view(-1)[2] = high
            x.view(-1)[-2] = high
        else:
            axis = dim % x.ndim
            if x.shape[axis] < 4:
                raise ValueError('tie witness needs at least four values along the reduced dimension')
            x.select(axis, 1).fill_(high)
            x.select(axis, 3).fill_(high)
    return x, {'dim': dim, 'keepdim': bool(case['keepdim'])}


def output_record(value):
    if value.dtype != torch.int64:
        raise ValueError('argmax must return int64 indices')
    return {'t': 'i64', 'shape': list(value.shape), 'v': value.detach().cpu().tolist()}


def execute(fn, x, options, capture=False):
    x = x.detach().clone(memory_format=torch.preserve_format)
    before = x.detach().contiguous().view(torch.uint8).clone()
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder else contextlib.nullcontext():
        out = fn(x, **options)
    if x.is_cuda:
        torch.cuda.synchronize()
    if not torch.equal(x.detach().contiguous().view(torch.uint8), before):
        raise ValueError('argmax modified its input')
    launches = [] if recorder is None else [
        {'phase': 'forward', 'kernel': launch.kernel_name, 'grid': list(launch.grid),
         'ttir': launch.asm.get('ttir')} for launch in recorder.launches]
    return out.detach(), launches, list(x.stride())


def mathematical_reference(x, options):
    query = {'op': 'aten.argmax', 'args': [batch.record(x), options['dim'], options['keepdim']]}
    result = registry.query('REDUCE', 'real', query)
    info = {'public_call': {'op': 'torch.argmax', **options}, 'mapped_query': query,
            'mapping': 'torch.argmax(input, dim, keepdim) -> aten.argmax(input, dim, keepdim)',
            'outcome': result.outcome, 'note': result.note, 'labels': result.labels,
            'scope': 'complete int64 index output'}
    if result.outcome != 'value':
        return info, None
    if set(result.fields) != {'out'}:
        raise ValueError('argmax reference omitted or added task outputs')
    info['out'] = batch.map_values(result.fields['out'], encode_finite)
    return info, result.fields['out']


def math_comparison(observed, expected):
    return batch.reference_comparison(observed, expected, expected_dtype=torch.int64)


def exact_math(item):
    return (item['outside_mathematical_enclosure'] == 0 and item['inside_nonpoint_unresolved'] == 0
            and item['nonfinite'] == 0 and item['exact_equal'] == item['coordinates'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'acceptance'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path,
                        default=Path(__file__).with_name('selection_argmax_20261019_v1.json'))
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    split = selection[args.split]
    args.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('standard version mismatch')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')

    spec_root = Path(__file__).resolve().parents[2]
    report = {'selection': selection, 'split': args.split, 'torch': torch.__version__, 'device': args.device,
              'samples': [], 'formal_binding': 'not_established', 'bias_verdict': 'not_established',
              'reference_protocol': 'Reuse REDUCE.real first-extreme argmax; only the public call mapping is new.',
              'sources': {'runner': Path(__file__).read_text(), 'comparison': Path(batch.__file__).read_text(),
                          'torch_public_documentation': torch.argmax.__doc__},
              'specification_sources': {name: (spec_root / name).read_text()
                                        for name in ('contracts/REDUCE.yaml', 'adapters/REDUCE.py')},
              'capture_setup': ('For CUDA candidates the recorder is enabled before warmup with no warmup launches '
                                'selected; static CUDA host launch is disabled and numerical precision is unchanged.')}

    candidate = context = None
    if args.device == 'cuda' and split['implementation'] == 'flag_gems':
        import flag_gems
        distribution = importlib.metadata.distribution('flag_gems')
        if distribution.version != '5.0.2':
            raise ValueError('candidate version mismatch')
        report['candidate_sources'] = {path: Path(distribution.locate_file('flag_gems/' + path)).read_text()
                                       for path in ('__init__.py', 'ops/argmax.py')}
        report['candidate_metadata'] = distribution.read_text('METADATA')
        report['candidate_license'] = distribution.read_text('LICENSE')
        candidate, context = functional, lambda: flag_gems.use_gems(include=['argmax'])
    elif args.device == 'cuda' and split['implementation'] == 'inductor':
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
                x, options = inputs(case, split['dimensions'], seed, device)
                row.update(options=options, input=batch.record(x), original_input_strides=list(x.stride()))
                row['mathematical_reference'], expected = mathematical_reference(x, options)
                if expected is None:
                    raise ValueError('selected mathematical reference is unavailable')
                standard, _, standard_strides = execute(functional, x, options)
                row.update(standard=output_record(standard), standard_input_strides=standard_strides,
                           standard_math=math_comparison(standard, expected))
                if not exact_math(row['standard_math']):
                    raise ValueError('versioned standard differs from REDUCE.real')
                if args.device == 'cpu':
                    row['status'] = 'reference_crosschecked'
                else:
                    with context():
                        with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                            execute(candidate, x, options)
                        actual, launches, strides = execute(candidate, x, options, capture=True)
                    row.update(candidate=output_record(actual), launches=launches,
                               observed_input_strides=strides, input_values_unchanged=True,
                               exact_standard_equal=bool(torch.equal(actual, standard)),
                               candidate_math=math_comparison(actual, expected))
                    row['triton_coverage'] = {'captured': bool(launches),
                                              'kernels': [launch['kernel'] for launch in launches]}
                    if not launches:
                        row['status'] = 'compared_no_triton_capture'
                    elif not row['exact_standard_equal'] or not exact_math(row['candidate_math']):
                        row['status'] = 'compared_mismatch'
                    else:
                        row['status'] = 'compared'
            except Exception as error:
                row.update(status='execution_or_reference_error', error=f'{type(error).__name__}: {error}',
                           traceback=traceback.format_exc())
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('END', case['id'], seed, row['status'], row.get('error', ''), flush=True)

    expected_status = 'reference_crosschecked' if args.device == 'cpu' else 'compared'
    raise SystemExit(int(any(sample['status'] != expected_status for sample in report['samples'])))


if __name__ == '__main__':
    main()
