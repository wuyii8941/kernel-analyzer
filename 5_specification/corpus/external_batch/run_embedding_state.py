"""Versioned standard comparison of a stateful public task; no invented strict oracle."""
import argparse
import contextlib
import importlib.metadata
import json
from pathlib import Path

import torch

import run as batch
from run_embedding import execute, functional
import run_embedding
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry


def state_relations(results, before, indices):
    if set(results) != {'out', 'dweight', 'weight_after'}:
        raise ValueError('stateful embedding requires out, dweight and complete weight_after')
    after = results['weight_after']
    if after.shape != before.shape or after.dtype != before.dtype:
        raise ValueError('weight post-state is incomplete')
    untouched = torch.ones(before.shape[0], dtype=torch.bool, device=before.device)
    untouched[indices] = False
    return {'lookup_matches_post_state': torch.equal(results['out'], after[indices]),
            'unvisited_rows_unchanged': torch.equal(before[untouched], after[untouched]),
            'changed_rows': torch.nonzero((before != after).any(dim=1)).flatten().cpu().tolist(),
            'unvisited_rows': torch.nonzero(untouched).flatten().cpu().tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['development', 'acceptance'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(Path(__file__).with_name('selection_embedding_state_20261019_v1.json').read_text())
    split = selection[args.split]
    args.out.mkdir(parents=True, exist_ok=False)
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('standard reference version mismatch')
    report = {'selection': selection, 'split': args.split, 'runner_source': Path(__file__).read_text(),
              'embedding_runner_source': Path(run_embedding.__file__).read_text(),
              'comparison_runner_source': Path(batch.__file__).read_text(), 'samples': [],
              'formal_binding': 'not_established', 'bias_verdict': 'not_established',
              'reference_kind': 'versioned_standard_implementation',
              'strict_renormalisation_reference': 'unsupported',
              'shared_implementation': 'Candidates may call the same native ATen renormalisation as the standard. Matching state is not independent verification of its formula.',
              'capture_scope': 'Forward/backward Triton launches only; native ATen state updates are not interpreted. Warmup uses recorder setup, disabling static CUDA host launch without changing numerical precision.'}
    if split['implementation'] == 'flag_gems':
        import flag_gems
        dist = importlib.metadata.distribution('flag_gems')
        if dist.version != '5.0.2':
            raise ValueError('candidate version mismatch')
        report['candidate_sources'] = {p: Path(dist.locate_file('flag_gems/' + p)).read_text()
            for p in ['__init__.py', 'ops/embedding.py', 'ops/embedding_dense_backward.py']}
        report['candidate_metadata'] = dist.read_text('METADATA')
        report['candidate_license'] = dist.read_text('LICENSE')
        candidate = functional
        context = lambda: flag_gems.use_gems(include=['embedding', 'embedding_dense_backward'])
    else:
        candidate = torch.compile(functional, fullgraph=True, dynamic=True)
        context = contextlib.nullcontext
    for case in selection['cases']:
        options = {k: v for k, v in case.items() if k != 'id'}
        for seed in split['seeds']:
            row = {'case': case['id'], 'seed': seed, 'options': options}
            report['samples'].append(row)
            try:
                generator = torch.Generator().manual_seed(seed)
                n, d = split['weight_shape']
                w = torch.randint(-8, 9, (n, d), generator=generator).float() / 4
                w[0] = 0; w[0, :2] = torch.tensor([3., 4.])
                w[1] = 0; w[1, 0] = .25
                w[-1] = 0; w[-1, :2] = torch.tensor([6., 8.])
                indices = torch.randint(3, split['index_shape'], generator=generator)
                indices.view(-1)[:5] = torch.tensor([0, 0, 1, n - 1, n - 1])
                g = torch.randint(-8, 9, (*indices.shape, d), generator=generator).float() / 8
                query = {'weight': w.tolist(), 'indices': indices.tolist(), 'upstream': g.tolist(),
                         'dtype': 'f32', 'index_dtype': 'i64', 'sparse': False, **options}
                row['inputs'] = query
                strict = registry.query('EMB', 'torch210_dense_real', query)
                row['strict_reference'] = {'outcome': strict.outcome, 'note': strict.note}
                w, indices, g = w.cuda(), indices.cuda(), g.cuda()
                standard, _ = execute(functional, w, indices, g, options)
                with context():
                    with TritonLaunchRecorder(select=lambda name, index: False, copy_tensors=False):
                        execute(candidate, w, indices, g, options)
                    observed, launches = execute(candidate, w, indices, g, options, capture=True)
                row.update(standard={k: batch.record(v) for k, v in standard.items()},
                           candidate={k: batch.record(v) for k, v in observed.items()}, launches=launches,
                           comparison=batch.standard_comparison(observed, standard),
                           standard_relations=state_relations(standard, w, indices),
                           candidate_relations=state_relations(observed, w, indices),
                           readonly_inputs_unchanged=True)
                if {l['phase'] for l in launches} != {'forward', 'backward'}:
                    raise ValueError('forward/backward Triton execution was not fully captured')
                row['status'] = 'compared_to_versioned_standard'
            except Exception as exc:
                row.update(status='execution_or_reference_error', error=f'{type(exc).__name__}: {exc}')
            (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(case['id'], seed, row['status'], row.get('error', ''), flush=True)
    failed = any(s['status'] != 'compared_to_versioned_standard'
                 or s['strict_reference']['outcome'] != 'unsupported'
                 or any(v['nonfinite_pairs'] or not v['allclose_rtol_1e-5_atol_1e-6'] for v in s['comparison'].values())
                 or any(not s[k]['lookup_matches_post_state'] or not s[k]['unvisited_rows_unchanged']
                        for k in ('standard_relations', 'candidate_relations')) for s in report['samples'])
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
