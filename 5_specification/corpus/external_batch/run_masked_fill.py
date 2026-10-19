"""Real public scalar masked_fill calls with complete forward and input VJP."""
import argparse
import contextlib
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch

import run as batch
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite

PROFILE = 'torch210_masked_fill_scalar_autograd'


def functional(x, mask, value):
    return x.masked_fill(mask, value)


def scalar(value):
    return float(Fraction(value))


def inputs(case, seed, device):
    generator = torch.Generator().manual_seed(seed)
    logical_shape = tuple(case['input_shape'])
    stored_shape = (*logical_shape[:-2], logical_shape[-1], logical_shape[-2]) \
        if case.get('noncontiguous') else logical_shape
    x = (torch.randint(-24, 25, stored_shape, generator=generator).float() / 8).to(device)
    if case.get('noncontiguous'):
        x = x.transpose(-1, -2)
    mask_shape = tuple(case['mask_shape'])
    if case['mask'] == 'all_true':
        mask = torch.ones(mask_shape, dtype=torch.bool, device=device)
    elif case['mask'] == 'all_false':
        mask = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    else:
        mask = ((torch.arange(torch.tensor(mask_shape).prod().item()).reshape(mask_shape) + seed) % 3 == 0).to(device)
    output_shape = torch.broadcast_shapes(x.shape, mask.shape)
    upstream = (torch.randint(-8, 9, output_shape, generator=generator).float() / 8).to(device)
    return x, mask, scalar(case['value']), upstream


def execute(fn, x, mask, value, upstream, capture=False):
    x = x.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    mask = mask.detach().clone(memory_format=torch.preserve_format)
    upstream = upstream.detach().clone(memory_format=torch.preserve_format)
    watched = [x, mask, upstream]
    before = [item.detach().contiguous().view(torch.uint8).clone() for item in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder else contextlib.nullcontext():
        out = fn(x, mask, value)
        forward_count = len(recorder.launches) if recorder else 0
        dx, = torch.autograd.grad(out, x, upstream)
    if x.is_cuda:
        torch.cuda.synchronize()
    if any(not torch.equal(item.detach().contiguous().view(torch.uint8), old)
           for item, old in zip(watched, before)):
        raise ValueError('input, mask or upstream values changed')
    launches = [] if recorder is None else [
        {'phase': 'forward' if index < forward_count else 'backward',
         'kernel': launch.kernel_name, 'grid': list(launch.grid), 'ttir': launch.asm.get('ttir')}
        for index, launch in enumerate(recorder.launches)]
    return {'out': out.detach(), 'dx': dx.detach()}, launches, {
        'input': list(x.stride()), 'mask': list(mask.stride()), 'out': list(out.stride()),
        'out_contiguous': out.is_contiguous()}


def mathematical_reference(x, mask, value, upstream):
    query = {'op': 'torch.Tensor.masked_fill', 'input': batch.record(x), 'mask': batch.record(mask),
             'value': str(Fraction(value)), 'upstream': batch.record(upstream)}
    result = registry.query('ARITH', PROFILE, query)
    info = {'query': query, 'outcome': result.outcome, 'note': result.note, 'labels': result.labels,
            'scope': 'complete public forward and input VJP',
            'mapping': 'Tensor.masked_fill(mask, Python scalar) -> aten.masked_fill.Scalar',
            'output_mapping': {'out': 'out', 'grad': 'dx'}}
    if result.outcome != 'value':
        return info, None
    if set(result.fields) != {'out', 'grad'}:
        raise ValueError('masked_fill reference omitted or added task outputs')
    expected = {'out': result.fields['out'], 'dx': result.fields['grad']}
    info['fields'] = {name: batch.map_values(field, encode_finite) for name, field in expected.items()}
    return info, expected


def exact_math(observed, expected):
    return {name: batch.reference_comparison(observed[name], expected[name]) for name in ('out', 'dx')}


def math_exact(comparison):
    return all(item['outside_mathematical_enclosure'] == 0
               and item['inside_nonpoint_unresolved'] == 0
               and item['nonfinite'] == 0
               and item['exact_equal'] == item['coordinates'] for item in comparison.values())


def exact_standard(actual, standard):
    details = batch.standard_comparison(actual, standard)
    equal = {name: bool(torch.equal(actual[name], standard[name])) for name in ('out', 'dx')}
    return {'standard': details, 'exact_equal': equal, 'compatible': all(equal.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'acceptance'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path,
                        default=Path(__file__).with_name('selection_masked_fill_20261019_v1.json'))
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
        'reference_protocol': ('PyTorch 2.10 public Scalar masked_fill cross-checked against exact independent '
                               'broadcast selection and input-gradient reduction.'),
        'sources': {'runner': Path(__file__).read_text(), 'comparison': Path(batch.__file__).read_text(),
                    'torch_method_documentation': torch.Tensor.masked_fill.__doc__,
                    'aten_schema': str(torch.ops.aten.masked_fill.Scalar._schema)},
        'specification_sources': {name: (spec_root / name).read_text()
                                  for name in ('contracts/ARITH.yaml', 'adapters/ARITH.py')},
        'retained_official_sources': ['sources/T_MASKED_FILL.txt.gz', 'sources/TS_REFS.py.gz',
                                      'sources/TS_NATIVE_FUNCTIONS.yaml.gz'],
        'capture_setup': ('For CUDA candidates the recorder is enabled before warmup with no warmup launches '
                          'selected; static CUDA host launch is disabled and numerical precision is unchanged.')}

    candidate = context = None
    if args.device == 'cuda' and split['implementation'] == 'flag_gems':
        import flag_gems
        distribution = importlib.metadata.distribution('flag_gems')
        if distribution.version != '5.0.2':
            raise ValueError('candidate version mismatch')
        report['candidate_sources'] = {path: Path(distribution.locate_file('flag_gems/' + path)).read_text()
                                       for path in ('__init__.py', 'ops/masked_fill.py', 'utils/shape_utils.py')}
        report['candidate_metadata'] = distribution.read_text('METADATA')
        report['candidate_license'] = distribution.read_text('LICENSE')
        report['candidate_function_selection'] = ['masked_fill']
        report['candidate_scope'] = ('FlagGems replaces the forward Scalar masked_fill operator. The public input '
                                     'VJP is native autograd context, not a FlagGems Triton backward result.')
        candidate, context = functional, lambda: flag_gems.use_gems(include=['masked_fill'])
    elif args.device == 'cuda' and split['implementation'] == 'inductor':
        report['candidate_scope'] = 'TorchInductor compiles the public forward and input VJP.'
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
                x, mask, value, upstream = inputs(case, seed, device)
                row.update(input=batch.record(x), mask=batch.record(mask), value=str(Fraction(value)),
                           upstream=batch.record(upstream), original_input_strides=list(x.stride()))
                row['mathematical_reference'], expected = mathematical_reference(x, mask, value, upstream)
                if expected is None:
                    raise ValueError('selected mathematical reference is unavailable')
                standard, _, standard_layout = execute(functional, x, mask, value, upstream)
                row.update(standard={name: batch.record(tensor) for name, tensor in standard.items()},
                           standard_layout=standard_layout, standard_math=exact_math(standard, expected))
                if not standard_layout['out_contiguous'] or not math_exact(row['standard_math']):
                    raise ValueError('versioned standard differs from the selected reference or layout promise')
                if args.device == 'cpu':
                    row['status'] = 'reference_crosschecked'
                else:
                    with context():
                        with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                            execute(candidate, x, mask, value, upstream)
                        actual, launches, layout = execute(candidate, x, mask, value, upstream, capture=True)
                    row.update(candidate={name: batch.record(tensor) for name, tensor in actual.items()},
                               launches=launches, observed_layout=layout, input_values_unchanged=True,
                               comparison=exact_standard(actual, standard),
                               candidate_math=exact_math(actual, expected))
                    captured = sorted({launch['phase'] for launch in launches})
                    required = {'forward'} if split['implementation'] == 'flag_gems' else {'forward', 'backward'}
                    missing = sorted(required - set(captured))
                    row['triton_phase_coverage'] = {'expected': sorted(required),
                                                    'captured': captured, 'missing': missing}
                    if not layout['out_contiguous'] or not row['comparison']['compatible'] \
                            or not math_exact(row['candidate_math']):
                        row['status'] = 'compared_mismatch'
                    else:
                        row['status'] = 'compared' if not missing else 'compared_partial_triton'
            except Exception as error:
                row.update(status='execution_or_reference_error', error=f'{type(error).__name__}: {error}',
                           traceback=traceback.format_exc())
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('END', case['id'], seed, row['status'], row.get('error', ''), flush=True)

    expected_status = 'reference_crosschecked' if args.device == 'cpu' else 'compared'
    raise SystemExit(int(any(row['status'] != expected_status for row in report['samples'])))


if __name__ == '__main__':
    main()
