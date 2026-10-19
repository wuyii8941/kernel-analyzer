"""Compute all selected task references even if candidate compilation times out.

This is a CPU reference replay, not a candidate execution or a new acceptance set.
"""
import argparse
import json
from pathlib import Path

import run_conv as conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    assert conv.torch.__version__ == '2.10.0+cu128'
    selection = json.loads(Path(conv.__file__).with_name('selection_conv_20261019_v1.json').read_text())
    split = selection['development']
    report = {'selection': selection, 'role': 'CPU reference replay of the development selection, not GPU candidate evidence',
              'source': Path(__file__).read_text(), 'mapping_source': Path(conv.__file__).read_text(), 'samples': []}
    for case in selection['cases']:
        for seed in split['seeds']:
            operands, upstream, options = conv.inputs(case, split['dimensions'], seed, 'cpu')
            reference, expected = conv.mathematical_reference(operands, upstream, options)
            actual, _ = conv.execute(conv.functional, operands, upstream, options)
            if expected is None:
                raise ValueError('a selected task reference is unavailable')
            comparison = conv.matrix.compare(actual, actual, expected)
            report['samples'].append({'case': case['id'], 'seed': seed, 'options': options,
                'inputs': [conv.matrix.record(v) for v in operands], 'upstream': conv.batch.record(upstream),
                'mathematical_reference': reference, 'native_cpu': {k: conv.matrix.record(v) for k, v in actual.items()},
                'standard_math': comparison['standard_math']})
    with args.out.open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
    assert all(v['exact_equal'] == v['coordinates'] for r in report['samples'] for v in r['standard_math'].values())
    print('All 20 development references available; every CPU output and gradient exact.', flush=True)


if __name__ == '__main__':
    main()
