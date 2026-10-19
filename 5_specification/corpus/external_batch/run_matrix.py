"""Real matrix calls and complete operand VJPs; references come from the existing task."""
import argparse
import contextlib
import importlib.metadata
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import run as batch
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite


def functional(operands, kind, beta=1., alpha=1.):
    if kind == 'linear':
        return F.linear(input=operands[0], weight=operands[1], bias=operands[2])
    if kind == 'addmm':
        return torch.addmm(input=operands[0], mat1=operands[1], mat2=operands[2], beta=beta, alpha=alpha)
    if kind == 'bmm':
        return torch.bmm(input=operands[0], mat2=operands[1])
    raise ValueError('unselected operation')


def record(value):
    return None if value is None else batch.record(value)


def execute(fn, operands, upstream, options, capture=False):
    values = [None if v is None else v.detach().clone().requires_grad_(True) for v in operands]
    leaves = [v for v in values if v is not None]
    watched = [*leaves, upstream]
    before = [v.detach().contiguous().reshape(-1).view(torch.uint8).clone() for v in watched]
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder is not None else contextlib.nullcontext():
        out = fn(values, **options)
        forward_count = len(recorder.launches) if recorder else 0
        gradients = iter(torch.autograd.grad(out, leaves, upstream))
    torch.cuda.synchronize()
    if not all(torch.equal(v.detach().contiguous().reshape(-1).view(torch.uint8), old)
               for v, old in zip(watched, before)):
        raise ValueError('input values changed, including ignored addend bytes')
    result = {'out': out.detach()}
    result.update({f'g{n}': None if v is None else next(gradients).detach() for n, v in enumerate(values)})
    launches = [] if recorder is None else [
        {'phase': 'forward' if n < forward_count else 'backward', 'kernel': l.kernel_name,
         'grid': list(l.grid), 'ttir': l.asm.get('ttir')} for n, l in enumerate(recorder.launches)]
    return result, launches, [None if v is None else list(v.stride()) for v in values]


def inputs(case, dimensions, seed):
    m, k, n = dimensions
    generator = torch.Generator().manual_seed(seed)
    def draw(shape):
        return (torch.randint(-8, 9, shape, generator=generator).float() / 8).cuda()
    if case['op'] == 'linear':
        x = draw((2, m, k) if case.get('batched') else (m, k))
        bias = None if case['bias'] == 'absent' else draw(()) if case['bias'] == 'scalar' else draw((n,))
        operands, output_shape = [x, draw((n, k)), bias], (*x.shape[:-1], n)
    elif case['op'] == 'addmm':
        c = draw((n,)) if case['addend'] == 'vector' else draw((m, n))
        if case['addend'] in ('nan', 'inf'):
            c.fill_(float(case['addend']))
        b = draw((n, k)).t() if case.get('noncontiguous') else draw((k, n))
        operands, output_shape = [c, draw((m, k)), b], (m, n)
    elif case['op'] == 'bmm':
        operands = [draw((2, m, k)), draw((2, n, k)).transpose(1, 2)]
        output_shape = (2, m, n)
    else:
        raise ValueError('unselected input task')
    return operands, draw(output_shape)


def queries(case, operands, upstream):
    args = [record(v) for v in operands]
    names = {'linear': ['input', 'weight', 'bias'], 'addmm': ['self', 'mat1', 'mat2'],
             'bmm': ['self', 'mat2']}[case['op']]
    scales = {k: case[k] for k in ('alpha', 'beta') if k in case}
    forward = {'op': 'aten.' + case['op'], 'kwargs': {**dict(zip(names, args)), **scales}}
    vjp = {'op': 'vjp.aten.' + case['op'], 'args': [record(upstream), *args], 'kwargs': scales}
    return forward, vjp


def compare(candidate, standard, reference):
    if set(candidate) != set(standard) or set(candidate) != set(reference):
        raise ValueError('missing complete output or operand gradient')
    for name, expected in reference.items():
        if expected is None:
            if candidate[name] is not None or standard[name] is not None:
                raise ValueError('absent bias gradient must be explicit null')
        elif candidate[name] is None or standard[name] is None:
            raise ValueError('a required tensor gradient is missing')
    numerical = {k: v for k, v in candidate.items() if v is not None}
    native = {k: v for k, v in standard.items() if v is not None}
    return {'standard': batch.standard_comparison(numerical, native),
            'candidate_math': {k: batch.reference_comparison(v, reference[k]) for k, v in numerical.items()},
            'standard_math': {k: batch.reference_comparison(v, reference[k]) for k, v in native.items()},
            'absent_gradients': [k for k, v in reference.items() if v is None]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'acceptance'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--triton-gemm', action='store_true', help='Explicitly select Inductor Triton matrix templates')
    args = parser.parse_args()
    if args.triton_gemm and args.split != 'acceptance':
        parser.error('--triton-gemm applies only to the Inductor acceptance implementation')
    selection = json.loads(Path(__file__).with_name('selection_matrix_20261019_v1.json').read_text())
    split = selection[args.split]
    args.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('standard version mismatch')
    report = {'selection': selection, 'split': args.split, 'torch': torch.__version__,
              'runner_source': Path(__file__).read_text(), 'comparison_source': Path(batch.__file__).read_text(),
              'samples': [], 'formal_binding': 'not_established', 'bias_verdict': 'not_established',
              'capture_setup': 'Recorder active before warmup, selecting no warmup launches; static CUDA host launcher disabled by recorder; no precision setting changed.'}
    if split['implementation'] == 'flag_gems':
        import flag_gems
        dist = importlib.metadata.distribution('flag_gems')
        if dist.version != '5.0.2':
            raise ValueError('candidate version mismatch')
        report['candidate_sources'] = {p: Path(dist.locate_file('flag_gems/' + p)).read_text()
                                      for p in ['__init__.py', 'ops/mm.py', 'ops/bmm.py', 'ops/addmm.py']}
        report['candidate_metadata'], report['candidate_license'] = dist.read_text('METADATA'), dist.read_text('LICENSE')
        candidate, context = functional, lambda: flag_gems.use_gems(include=['mm', 'bmm', 'addmm'])
    else:
        options = {'max_autotune': True, 'max_autotune_gemm_backends': 'TRITON'} if args.triton_gemm else {}
        report['compiler_options'] = options
        candidate, context = torch.compile(functional, fullgraph=True, dynamic=True, options=options), contextlib.nullcontext
    for case in selection['cases']:
        for seed in split['seeds']:
            row = {'case': case['id'], 'seed': seed}
            report['samples'].append(row)
            print('START', case['id'], seed, flush=True)
            try:
                operands, upstream = inputs(case, split['dimensions'], seed)
                fquery, gquery = queries(case, operands, upstream)
                row['queries'] = {'forward': fquery, 'vjp': gquery}
                refs = [registry.query('MATMUL', 'real', q) for q in (fquery, gquery)]
                row['references'] = [{'outcome': r.outcome, 'note': r.note} for r in refs]
                if any(r.outcome != 'value' for r in refs):
                    raise ValueError('complete mathematical reference unavailable')
                expected = {'out': refs[0].fields['out'],
                            **{f'g{n}': v for n, v in enumerate(refs[1].fields['out'])}}
                row['mathematical'] = {k: None if v is None else batch.map_values(v, encode_finite)
                                       for k, v in expected.items()}
                options = {'kind': case['op'], **{k: case[k] for k in ('alpha', 'beta') if k in case}}
                standard, _, _ = execute(functional, operands, upstream, options)
                row['standard'] = {k: record(v) for k, v in standard.items()}
                with context():
                    with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                        execute(candidate, operands, upstream, options)
                    observed, launches, strides = execute(candidate, operands, upstream, options, capture=True)
                row.update(candidate={k: record(v) for k, v in observed.items()}, launches=launches,
                           observed_input_strides=strides, input_values_unchanged=True,
                           comparison=compare(observed, standard, expected))
                if {l['phase'] for l in launches} != {'forward', 'backward'}:
                    raise ValueError('forward/backward Triton execution not fully captured')
                row['status'] = 'compared'
            except Exception as exc:
                row.update(status='execution_or_reference_error', error=f'{type(exc).__name__}: {exc}')
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('END', case['id'], seed, row['status'], row.get('error', ''), flush=True)
    failed = any(s['status'] != 'compared' or any(c['nonfinite_pairs'] or not c['allclose_rtol_1e-5_atol_1e-6']
                  for c in s['comparison']['standard'].values()) for s in report['samples'])
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
