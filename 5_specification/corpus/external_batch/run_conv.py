"""Public convolution training calls, reusing the reviewed real CONV reference.

Public rank and padding normalization only affects reference queries. The
candidate receives the original call. Captured tensor arguments are diagnostic;
their presence does not certify a complete producer or a formal task binding.
"""
import argparse
import contextlib
import faulthandler
import gzip
import importlib.metadata
import inspect
import json
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import run as batch
import run_matrix as matrix
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite


def functional(operands, kind, **options):
    fn = {'conv1d': F.conv1d, 'conv2d': F.conv2d}[kind]
    return fn(operands[0], operands[1], operands[2], **options)


def spatial(value, rank, name, minimum):
    values = [value] * rank if type(value) is int else list(value)
    if len(values) == 1:
        values *= rank
    if len(values) != rank or any(type(v) is not int or v < minimum for v in values):
        raise ValueError(f'invalid {name}')
    return values


def mathematical_reference(operands, upstream, options):
    """Map the documented public call to batched, symmetric CONV queries.

    For same padding, P is an explicit zero insertion: y = CONV(P x, w, b).
    Its input VJP is P^T dx_padded, i.e. crop back to the original coordinates.
    Adding/removing a batch of size one preserves all coordinates and gradients.
    """
    rank = {'conv1d': 1, 'conv2d': 2}[options['kind']]
    if set(options) - {'kind', 'stride', 'padding', 'dilation', 'groups'}:
        raise ValueError('unselected convolution option')
    x, weight, bias = operands
    if weight.ndim != rank + 2 or x.ndim not in (rank + 1, rank + 2):
        raise ValueError('public convolution rank mismatch')
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all()
           for v in (x, weight, bias, upstream) if v is not None):
        raise ValueError('batch selects finite float32 only')
    unbatched = x.ndim == rank + 1
    if upstream.ndim != x.ndim:
        raise ValueError('upstream rank mismatch')
    xr = batch.record(x.unsqueeze(0) if unbatched else x)
    gr = batch.record(upstream.unsqueeze(0) if unbatched else upstream)
    wr, br = matrix.record(weight), matrix.record(bias)
    stride = spatial(options.get('stride', 1), rank, 'stride', 1)
    dilation = spatial(options.get('dilation', 1), rank, 'dilation', 1)
    padding = options.get('padding', 0)
    pads = [(0, 0)] * rank
    original_shape = xr['shape']
    if isinstance(padding, str):
        if padding not in ('valid', 'same'):
            raise ValueError('unselected padding string')
        if padding == 'same':
            if stride != [1] * rank:
                raise ValueError('same padding requires stride one')
            total = [d * (k - 1) for d, k in zip(dilation, weight.shape[2:])]
            pads = [(p // 2, p - p // 2) for p in total]
            # Copy received values into explicit zeros, without calling a
            # convolution or using a candidate output to choose the mapping.
            padded_shape = [*original_shape[:2], *[n + l + r for n, (l, r) in zip(original_shape[2:], pads)]]
            padded = np.zeros(padded_shape, dtype=np.float32)
            crop = (slice(None), slice(None), *[slice(l, l + n) for n, (l, _) in zip(original_shape[2:], pads)])
            padded[crop] = np.asarray(xr['v'], dtype=np.float32)
            xr = {'t': 'f32', 'shape': padded_shape, 'v': padded.tolist()}
        padding = [0] * rank
    else:
        padding = spatial(padding, rank, 'padding', 0)
    groups = options.get('groups', 1)
    if type(groups) is not int or groups < 1:
        raise ValueError('invalid groups')
    tail = [stride, padding, dilation, False, [0] * rank, groups]
    queries = [{'op': 'aten.convolution', 'args': [xr, wr, br, *tail]},
               {'op': 'aten.convolution_backward', 'args': [gr, xr, wr,
                None if bias is None else list(bias.shape), *tail, [True, True, bias is not None]]}]
    results = [registry.query('CONV', 'real', q) for q in queries]
    info = {'queries': queries, 'results': [{'outcome': r.outcome, 'note': r.note} for r in results],
            'mapping': {'added_batch': unbatched, 'explicit_zero_padding': pads,
                        'outputs': {'forward': 'out', 'backward': ['g0', 'g1', 'g2']},
                        'absent_bias_gradient': 'explicit null'},
            'scope': 'complete forward and input/weight/present-bias VJP; exact real task, not rounding semantics'}
    if any(r.outcome != 'value' for r in results):
        return info, None
    if any(set(r.fields) != {'out'} for r in results) or len(results[1].fields['out']) != 3:
        raise ValueError('incomplete convolution reference outputs')
    out = results[0].fields['out']
    gx, gw, gb = results[1].fields['out']
    if any(l or r for l, r in pads):
        crop = (slice(None), slice(None), *[slice(l, l + n) for n, (l, _) in zip(original_shape[2:], pads)])
        gx = np.asarray(gx, dtype=object)[crop].tolist()
    if unbatched:
        out, gx = out[0], gx[0]
    expected = {'out': out, 'g0': gx, 'g1': gw, 'g2': gb}
    info['fields'] = {k: None if v is None else batch.map_values(v, encode_finite) for k, v in expected.items()}
    return info, expected


def execute(fn, operands, upstream, options, capture=False):
    values = [None if v is None else v.detach().clone().requires_grad_(True) for v in operands]
    leaves = [v for v in values if v is not None]
    watched = [*leaves, upstream]
    before = [v.detach().contiguous().reshape(-1).view(torch.uint8).clone() for v in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False, keep_storages=True) if capture else None
    with recorder if recorder is not None else contextlib.nullcontext():
        out = fn(values, **options)
        forward_count = len(recorder.launches) if recorder else 0
        gradients = iter(torch.autograd.grad(out, leaves, upstream))
    if out.is_cuda:
        torch.cuda.synchronize()
    if not all(torch.equal(v.detach().contiguous().reshape(-1).view(torch.uint8), old)
               for v, old in zip(watched, before)):
        raise ValueError('input or upstream bytes changed')
    result = {'out': out.detach(), **{f'g{n}': None if v is None else next(gradients).detach()
                                    for n, v in enumerate(values)}}
    launches = [] if recorder is None else [
        {'phase': 'forward' if n < forward_count else 'backward', 'kernel': l.kernel_name,
         'grid': list(l.grid), 'ttir': l.asm.get('ttir')} for n, l in enumerate(recorder.launches)]
    matches = {} if recorder is None else {name: [
        {'launch': i, 'argument': a.name} for i, l in enumerate(recorder.launches) for a in l.args
        if a.kind == 'tensor' and a.storage_id == v.untyped_storage()._cdata]
        for name, v in result.items() if v is not None}
    return result, {'launches': launches, 'output_storage_arguments': matches,
                    'triton_scope': 'not_established',
                    'scope_note': 'Matching an argument storage is not proof of a complete write or producer. Native operations and hidden intermediates remain unbound.',
                    'input_strides': [None if v is None else list(v.stride()) for v in values]}


def inputs(case, dimensions, seed, device='cuda'):
    generator = torch.Generator().manual_seed(seed)
    def draw(shape):
        return (torch.randint(-8, 9, shape, generator=generator).float() / 8).to(device)
    n, h, w = dimensions
    rank = int(case['kind'][-2])
    shape = (n, case['cin'], *([w] if rank == 1 else [h, w]))
    x = draw(shape)
    if case.get('channels_last'):
        x = x.contiguous(memory_format=torch.channels_last)
    if case.get('unbatched'):
        x = x[0]
    weight = draw((case['cout'], case['cin'] // case.get('options', {}).get('groups', 1), *case['kernel']))
    bias = draw((case['cout'],)) if case.get('bias', True) else None
    operands = [x, weight, bias]
    options = {'kind': case['kind'], **case.get('options', {})}
    # The native public call determines the upstream shape, never the candidate.
    upstream = draw(functional(operands, **options).shape)
    return operands, upstream, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'acceptance'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compiler-defaults', action='store_true', help='Separate default-backend control; not a replacement for Triton-only acceptance')
    parser.add_argument('--case', action='append', help='Run only these original case IDs; keep the complete selection in the record')
    parser.add_argument('--diagnostic-after', type=int, default=0, help='Dump execution stacks after this many seconds, without changing the candidate')
    parser.add_argument('--selection', type=Path, default=Path(__file__).with_name('selection_conv_20261019_v1.json'))
    args = parser.parse_args()
    if args.compiler_defaults and args.split != 'acceptance':
        parser.error('--compiler-defaults applies only to Inductor')
    selection = json.loads(args.selection.read_text())
    if args.case and set(args.case) - {c['id'] for c in selection['cases']}:
        parser.error('unknown case ID')
    if args.diagnostic_after:
        faulthandler.dump_traceback_later(args.diagnostic_after, repeat=True)
    split = selection[args.split]
    args.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('standard version mismatch')
    root = Path(__file__).resolve().parents[2]
    torch_root = Path(torch.__file__).parent
    report = {'selection': selection, 'split': args.split, 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(), 'samples': [],
              'selected_cases': args.case,
              'formal_binding': 'not_established', 'bias_verdict': 'not_established',
              'precision': {'matmul_precision': torch.get_float32_matmul_precision(),
                            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32},
              'sources': {'runner': Path(__file__).read_text(), 'comparison': Path(batch.__file__).read_text(),
                          'complete_output_comparison': Path(matrix.__file__).read_text(),
                          'torch_functional': Path(inspect.getfile(F)).read_text(),
                          'torch_conv_module': (torch_root / 'nn/modules/conv.py').read_text()},
              'specification_sources': {name: (root / name).read_text() for name in
                                        ('contracts/CONV.yaml', 'adapters/CONV.py', 'kernel_spec/aten.py')},
              'official_documents': {name: gzip.decompress((root / 'sources' / (name + '.txt.gz')).read_bytes()).decode()
                                     for name in ('T_CONV1D', 'T_CONV2D')},
              'capture_setup': 'Recorder active before warmup, selecting none. No numerical precision change. No tracing substitution.'}
    if split['implementation'] == 'flag_gems':
        import flag_gems
        dist = importlib.metadata.distribution('flag_gems')
        if dist.version != '5.0.2':
            raise ValueError('candidate version mismatch')
        report['candidate_sources'] = {p: Path(dist.locate_file('flag_gems/' + p)).read_text()
                                      for p in ('__init__.py', 'ops/conv1d.py', 'ops/conv2d.py')}
        report['candidate_metadata'] = dist.read_text('METADATA')
        candidate, context = functional, lambda: flag_gems.use_gems(include=['conv1d', 'conv2d'])
    else:
        report['compiler_options'] = {} if args.compiler_defaults else {'max_autotune': True, 'max_autotune_conv_backends': 'TRITON'}
        report['execution_role'] = 'default-backend control after acceptance' if args.compiler_defaults else 'Triton-template acceptance'
        report['candidate_sources'] = {'conv_lowering': (torch_root / '_inductor/kernel/conv.py').read_text()}
        context = contextlib.nullcontext
    for case in selection['cases']:
        if args.case and case['id'] not in args.case:
            continue
        if split['implementation'] == 'inductor':
            torch._dynamo.reset()
            candidate = torch.compile(functional, fullgraph=True, options=report['compiler_options'])
        for seed in split['seeds']:
            row = {'case': case['id'], 'seed': seed}
            report['samples'].append(row)
            print('START', case['id'], seed, flush=True)
            try:
                operands, upstream, options = inputs(case, split['dimensions'], seed)
                row.update(options=options, inputs=[matrix.record(v) for v in operands], upstream=batch.record(upstream))
                row['mathematical_reference'], expected = mathematical_reference(operands, upstream, options)
                standard, _ = execute(functional, operands, upstream, options)
                row['standard'] = {k: matrix.record(v) for k, v in standard.items()}
                if expected is not None:
                    row['standard_math'] = {k: batch.reference_comparison(v, expected[k]) for k, v in standard.items() if v is not None}
                row['status'] = 'candidate_pending'
                (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                with context():
                    with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                        execute(candidate, operands, upstream, options)
                    observed, capture = execute(candidate, operands, upstream, options, capture=True)
                row.update(candidate={k: matrix.record(v) for k, v in observed.items()}, capture=capture,
                           input_values_unchanged=True)
                if expected is None:
                    raise ValueError('complete mathematical reference unavailable')
                row['comparison'] = matrix.compare(observed, standard, expected)
                row['status'] = 'compared' if capture['launches'] else 'no_triton_capture'
            except Exception as exc:
                row.update(status='execution_or_reference_error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('END', case['id'], seed, row['status'], row.get('error', ''), flush=True)
    failed = any(s['status'] != 'compared' or any(c['nonfinite_pairs'] or not c['allclose_rtol_1e-5_atol_1e-6']
                 for c in s['comparison']['standard'].values()) for s in report['samples'])
    raise SystemExit(int(failed))


if __name__ == '__main__':
    main()
