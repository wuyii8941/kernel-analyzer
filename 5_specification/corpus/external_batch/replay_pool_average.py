"""Apply the newly admitted exact task to frozen observations; never call this new acceptance."""
import argparse
import gzip
import json
from pathlib import Path
import torch

import run_pool as runner


def tensor(record):
    if record['t'] != 'f32':
        raise ValueError('frozen replay requires its declared float32 task')
    value = torch.tensor(record['v'], dtype=torch.float32)
    if list(value.shape) != record['shape']:
        raise ValueError('frozen tensor shape is incomplete')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = Path(__file__).resolve().parents[2]
    report = {'kind': 'replay of frozen observations, not new acceptance', 'samples': [],
              'source': Path(__file__).read_text(), 'mapping': Path(runner.__file__).read_text(),
              'reviewed_reference': {p: (root / p).read_text() for p in
                  ('contracts/POOL.yaml', 'adapters/POOL.py', 'kernel_spec/e2_pool.py', 'evidence/pool_implementation.md')}}
    for source in args.source:
        original = json.loads(gzip.decompress(source.read_bytes()))
        for sample in original['samples']:
            if sample.get('options', {}).get('kind') != 'avg':
                continue
            info, expected = runner.mathematical_reference(tensor(sample['input']), tensor(sample['upstream']), sample['options'])
            row = {'source': str(source), 'case': sample['case'], 'seed': sample['seed'],
                   'original_status': sample['status'], 'original_error': sample.get('error'), 'reference': info}
            report['samples'].append(row)
            if expected is not None:
                for who in ('standard', 'candidate'):
                    if who not in sample:
                        row[who] = {'outcome': 'not_observed', 'note': 'original execution failed; no result invented'}
                    else:
                        if set(sample[who]) != set(expected):
                            raise ValueError('incomplete frozen task outputs')
                        row[who] = {k: runner.batch.reference_comparison(tensor(sample[who][k]), v) for k, v in expected.items()}
    with args.out.open('xb') as output:
        output.write(gzip.compress(json.dumps(report, indent=2).encode(), mtime=0))
    raise SystemExit(int(any(s['reference']['outcome'] != 'value' for s in report['samples'])))


if __name__ == '__main__':
    main()
