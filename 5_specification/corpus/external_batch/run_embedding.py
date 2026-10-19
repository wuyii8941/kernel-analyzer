"""Public embedding, complete VJP and read-only state against two sourced references."""
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


def functional(input, weight, padding_idx, scale_grad_by_freq, max_norm=None, norm_type=2.0):
    return F.embedding(input, weight, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type,
                       scale_grad_by_freq=scale_grad_by_freq, sparse=False)


def execute(fn, weight, indices, upstream, options, capture=False):
    weight = weight.detach().clone().requires_grad_(True)
    before = [t.detach().clone() for t in (weight, indices, upstream)]
    recorder = TritonLaunchRecorder(copy_tensors=False) if capture else None
    with recorder if recorder is not None else contextlib.nullcontext():
        output = fn(input=indices, weight=weight, **options)
        forward_count = len(recorder.launches) if recorder else 0
        gradient, = torch.autograd.grad(output, weight, upstream)
    torch.cuda.synchronize()
    mutable_weight = options.get('max_norm') is not None
    watched = (indices, upstream) if mutable_weight else (weight, indices, upstream)
    expected = before[1:] if mutable_weight else before
    if not all(torch.equal(a, b) for a, b in zip(watched, expected)):
        raise ValueError('a declared read-only input changed')
    launches = [] if recorder is None else [
        {'phase': 'forward' if n < forward_count else 'backward', 'kernel': l.kernel_name,
         'grid': list(l.grid), 'ttir': l.asm.get('ttir')} for n, l in enumerate(recorder.launches)]
    results = {'out': output.detach(), 'dweight': gradient.detach()}
    if mutable_weight:
        results['weight_after'] = weight.detach().clone()
    return results, launches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'acceptance'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    selection_path = Path(__file__).with_name('selection_embedding_20261019_v1.json')
    selection = json.loads(selection_path.read_text())
    split = selection[args.split]
    args.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != selection['versions']['torch']:
        raise ValueError('unexpected standard implementation version')
    report = {'selection': selection, 'split': args.split, 'torch': torch.__version__,
              'runner_source': Path(__file__).read_text(), 'shared_runner_source': Path(batch.__file__).read_text(),
              'capture_setup': 'Warmup executes under a recorder selecting no launches, so Inductor creates a '
                               'recordable Python launcher before caching it. Measured calls use the same launcher; '
                               'the recorder disables the static CUDA host launcher. No numerical precision switch is changed.',
              'formal_binding': 'not_established', 'bias_verdict': 'not_established', 'samples': []}
    candidate = functional
    if split['implementation'] == 'flag_gems':
        import flag_gems
        distribution = importlib.metadata.distribution('flag_gems')
        if distribution.version != selection['versions']['flag_gems']:
            raise ValueError('unexpected candidate version')
        report['candidate_sources'] = {name: Path(distribution.locate_file('flag_gems/' + name)).read_text()
            for name in ['__init__.py', 'ops/embedding.py', 'ops/embedding_dense_backward.py']}
        report['candidate_metadata'] = distribution.read_text('METADATA')
        report['candidate_license'] = distribution.read_text('LICENSE')
        candidate_context = lambda: flag_gems.use_gems(include=['embedding', 'embedding_dense_backward'])
    else:
        candidate = torch.compile(functional, fullgraph=True, dynamic=True)
        candidate_context = contextlib.nullcontext
    for case in selection['cases']:
        options = {k: case[k] for k in ('padding_idx', 'scale_grad_by_freq')}
        for seed in split['seeds']:
            row = {'case': case['id'], 'seed': seed, 'options': options}
            report['samples'].append(row)
            try:
                generator = torch.Generator().manual_seed(seed)
                rows, width = split['weight_shape']
                weight = (torch.randint(-32, 33, (rows, width), generator=generator).float() / 8).cuda()
                indices = torch.randint(rows, split['index_shape'], generator=generator)
                indices.view(-1)[:5] = torch.tensor([0, rows - 1, 1, 1, 1])
                indices = indices.cuda()
                upstream = (torch.randint(-8, 9, (*indices.shape, width), generator=generator).float() / 8).cuda()
                query = {'weight': weight.cpu().tolist(), 'indices': indices.cpu().tolist(),
                         'upstream': upstream.cpu().tolist(), 'dtype': 'f32', 'index_dtype': 'i64',
                         'max_norm': None, 'norm_type': 2, 'sparse': False, **options}
                row['query'] = query
                ref = registry.query('EMB', 'torch210_dense_real', query)
                row['reference'] = {'outcome': ref.outcome, 'note': ref.note}
                if ref.outcome != 'value':
                    raise ValueError('selected mathematical reference is unavailable: ' + ref.note)
                row['reference']['fields'] = {k: batch.map_values(v, encode_finite) for k, v in ref.fields.items()}
                standard, _ = execute(functional, weight, indices, upstream, options)
                with candidate_context():
                    # Inductor caches its host launcher on first execution. Set
                    # up the existing capture path before warmup, but exclude
                    # warmup/autotuning launches from the measured record.
                    with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                        execute(candidate, weight, indices, upstream, options)
                    result, launches = execute(candidate, weight, indices, upstream, options, capture=True)
                row.update(candidate={k: batch.record(v) for k, v in result.items()},
                           standard={k: batch.record(v) for k, v in standard.items()}, launches=launches,
                           inputs_unchanged=True, standard_comparison=batch.standard_comparison(result, standard))
                row['candidate_math'] = {k: batch.reference_comparison(v, ref.fields[k]) for k, v in result.items()}
                row['standard_math'] = {k: batch.reference_comparison(v, ref.fields[k]) for k, v in standard.items()}
                if options == {'padding_idx': None, 'scale_grad_by_freq': False}:
                    index_query = {'op': 'torch.nn.functional.embedding', 'args': [
                        {'t': 'i64', 'shape': list(indices.shape), 'v': query['indices']}, batch.record(weight)]}
                    index_ref = registry.query('INDEX', 'real', index_query)
                    row['lookup_reference'] = {'outcome': index_ref.outcome, 'note': index_ref.note}
                    if index_ref.outcome != 'value' or index_ref.fields['out'] != ref.fields['out']:
                        raise ValueError('INDEX public lookup and EMB forward are not both established and equal')
                    row['lookup_reference']['matches_emb_forward'] = True
                phases = {l['phase'] for l in launches}
                row['status'] = 'compared' if phases == {'forward', 'backward'} else 'partial_triton_capture'
                if not launches:
                    raise ValueError('no captured Triton execution')
            except Exception as exc:
                row.update(status='execution_or_reference_error', error=f'{type(exc).__name__}: {exc}')
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(case['id'], seed, row['status'], row.get('error', ''), flush=True)
    failed = any(s['status'] != 'compared' or any(
        c['nonfinite_pairs'] or not c['allclose_rtol_1e-5_atol_1e-6'] for c in s['standard_comparison'].values())
        or s['candidate_math']['out']['outside_mathematical_enclosure']
        or s['standard_math']['out']['outside_mathematical_enclosure'] for s in report['samples'])
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
