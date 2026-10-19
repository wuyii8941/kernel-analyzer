"""Real public calls, two references, and separate producer visibility diagnostics."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import flag_gems
import torch

from kernel_analyzer.provenance import _Trace
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
from kernel_spec import registry


def record(value):
    if torch.is_tensor(value):
        dtype = {torch.float32: 'f32', torch.int64: 'i64'}[value.dtype]
        return {'t': dtype, 'shape': list(value.shape), 'v': value.detach().cpu().tolist()}
    return [record(v) for v in value]


def tensors(value):
    return [value] if torch.is_tensor(value) else list(value)


def same(a, b):
    aa, bb = tensors(a), tensors(b)
    return len(aa) == len(bb) and all(
        x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y) for x, y in zip(aa, bb))


def inputs(case, shape, seed):
    generator = torch.Generator().manual_seed(seed)
    if case['input'] == 'unique_rows':
        x = torch.stack([torch.randperm(shape[1], generator=generator) for _ in range(shape[0])]).float() / 8
    elif case['input'] == 'ties':
        x = torch.zeros(shape)
    else:
        x = torch.randint(-8, 9, shape, generator=generator).float() / 8
    x = x.cuda()
    kwargs = dict(case['kwargs'])
    if case['op'] in ('gather', 'index_select'):
        dim = kwargs['dim']
        index_shape = ([3, shape[1]] if dim == 0 else [shape[0], 7]) if case['op'] == 'gather' else [7]
        kwargs['index'] = torch.randint(shape[dim], index_shape, generator=generator).cuda()
    return x, kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'selection.json').write_bytes(args.selection.read_bytes())
    (args.out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    distribution = importlib.metadata.distribution('flag_gems')
    if distribution.version != '5.0.2' or torch.__version__ != '2.10.0+cu128':
        raise ValueError('runtime versions differ from the selected task')
    sources = {op: Path(distribution.locate_file(f'flag_gems/ops/{op}.py'))
               for op in {c['op'] for c in selection['cases']}}
    source_texts = {op: path.read_text() for op, path in sources.items()}
    report = {'selection': selection, 'torch': torch.__version__, 'candidate_version': distribution.version,
              'source_modules': source_texts, 'package_metadata': distribution.read_text('METADATA'),
              'license': distribution.read_text('LICENSE'), 'samples': [],
              'formal_binding': 'not_established', 'bias_verdict': 'not_established'}
    for case in selection['cases']:
        for shape in selection['shapes']:
            for seed in selection['seeds']:
                row = {'case': case['id'], 'shape': shape, 'seed': seed}
                report['samples'].append(row)
                try:
                    x, kwargs = inputs(case, shape, seed)
                    watched = [x] + [v for v in kwargs.values() if torch.is_tensor(v)]
                    before = [t.clone() for t in watched]
                    query = {'op': case.get('reference_op', 'aten.' + case['op']), 'args': [record(x)],
                             'kwargs': {k: record(v) if torch.is_tensor(v) else v for k, v in kwargs.items()}}
                    call_args = (x,)
                    if case.get('keyword_input'):
                        query['args'] = []
                        query['kwargs']['input'] = record(x)
                        call_args, kwargs = (), {'input': x, **kwargs}
                    row['query'] = query
                    ref = registry.query(case['contract'], case['profile'], query)
                    row['reference'] = {'outcome': ref.outcome, 'note': ref.note,
                                        'expected_outcome': case['expected_reference']}
                    fn = getattr(torch, case['op'])
                    standard = fn(*call_args, **kwargs)
                    rec = TritonLaunchRecorder(copy_tensors=False, keep_storages=True)
                    with rec, flag_gems.use_gems(include=[case['op']]):
                        candidate = fn(*call_args, **kwargs)
                    torch.cuda.synchronize()
                    row.update(standard=record(standard), candidate=record(candidate),
                               standard_equal=same(candidate, standard),
                               inputs_unchanged=all(torch.equal(a, b) for a, b in zip(watched, before)),
                               launches=[{'kernel': l.kernel_name, 'grid': list(l.grid)} for l in rec.launches])
                    if not rec.launches:
                        raise ValueError('no Triton launch: cannot count native fallback as a tested candidate')
                    if ref.outcome == 'value':
                        adapter = registry.load(case['contract']).adapter
                        row['candidate_math_equal'] = adapter.check_output(case['profile'], query, {'out': record(candidate)})
                        row['standard_math_equal'] = adapter.check_output(case['profile'], query, {'out': record(standard)})
                    traced_rec = TritonLaunchRecorder(copy_tensors=False, keep_storages=True)
                    trace = _Trace(traced_rec)
                    with traced_rec, torch.compiler.set_stance('force_eager'), trace.mode, flag_gems.use_gems(include=[case['op']]):
                        traced = fn(*call_args, **kwargs)
                    torch.cuda.synchronize()
                    row['trace'] = {'enclosed_launches': trace.enclosed_launches,
                                    'events': [{'after_launches': e[0], 'operator': str(e[2])} for e in trace.events],
                                    'output_unchanged': same(traced, candidate),
                                    'inputs_unchanged': all(torch.equal(a, b) for a, b in zip(watched, before)),
                                    'launches': [{'kernel': l.kernel_name, 'grid': list(l.grid)} for l in traced_rec.launches]}
                    row['status'] = 'compared' if ref.outcome == 'value' else 'reference_unavailable'
                except Exception as exc:
                    row.update(status='execution_or_reference_error', error=f'{type(exc).__name__}: {exc}')
                (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                print(case['id'], shape, seed, row['status'], row.get('error', ''), flush=True)
    if any(path.read_text() != source_texts[op] for op, path in sources.items()):
        raise ValueError('candidate source changed during execution')
    failed = any(s['status'] == 'execution_or_reference_error'
                 or s['reference']['outcome'] != s['reference']['expected_outcome']
                 or not s['inputs_unchanged'] or not s['trace']['inputs_unchanged']
                 or not s['trace']['output_unchanged']
                 or (s['reference']['outcome'] == 'value' and not (
                     s['candidate_math_equal'] and s['standard_math_equal'] and s['standard_equal']))
                 for s in report['samples'])
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
