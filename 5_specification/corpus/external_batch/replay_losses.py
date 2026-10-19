"""Add newly reviewed mathematical references to retained public outputs, without rerunning candidates."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as batch
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(gzip.decompress(args.source.read_bytes()))
    result = {'source': str(args.source), 'kind': 'mathematical_replay_of_development_outputs',
              'new_kernel_execution': False, 'formal_binding': 'not_established',
              'bias_verdict': 'not_established', 'cases': []}
    failed = False
    for case in source['cases']:
        if case['id'] not in ('attorch_smooth_l1', 'attorch_huber'):
            continue
        entry = {'id': case['id'], 'entry': case['entry'], 'samples': []}
        result['cases'].append(entry)
        for sample in case['samples']:
            row = {'shape': sample['shape'], 'seed': sample['seed'], 'references': []}
            entry['samples'].append(row)
            if sample['status'] != 'compared':
                row.update(outcome='not_established', reason='source candidate did not finish')
                failed = True
                continue
            kwargs = {'reduction': 'mean', 'beta' if case['op'] == 'smooth_l1' else 'delta': .5}
            for backward in (False, True):
                names = ['dx', 'dtarget'] if backward else ['out']
                inputs = ([{'t': 'f32', 'v': sample['upstream'], 'shape': []}] if backward else [])
                inputs += [sample['inputs']['x'], sample['inputs']['target']]
                query = {'op': ('vjp.' if backward else '')+'aten.'+case['op']+'_loss',
                         'args': inputs, 'kwargs': kwargs}
                r = registry.query('HUBER', 'torch210_unweighted_f32', query)
                item = {'query': query, 'outcome': r.outcome, 'note': r.note}
                row['references'].append(item)
                if r.outcome != 'value':
                    failed = True
                    continue
                values = r.fields['out'] if backward else [r.fields['out']]
                if len(values) != len(names):
                    raise ValueError('incomplete public result')
                item['values'] = {n: batch.map_values(v, encode_finite) for n, v in zip(names, values)}
                for origin in ('candidate', 'standard'):
                    comparisons = {}
                    for name, value in zip(names, values):
                        rec = sample[origin][name]
                        if rec['t'] != 'f32':
                            raise ValueError('source dtype outside selected task')
                        observed = torch.tensor(rec['v'], dtype=torch.float32).reshape(rec['shape'])
                        comparisons[name] = batch.reference_comparison(observed, value)
                    item[origin+'_differences'] = comparisons
    if len(result['cases']) != 2:
        raise ValueError('both selected external loss cases must be retained')
    with args.out.open('xb') as f:
        f.write(gzip.compress(json.dumps(result, indent=2, allow_nan=False).encode(), mtime=0))
    print('replayed', sum(len(c['samples']) for c in result['cases']), 'retained calls; failed references:', failed)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
