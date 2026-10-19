"""Repeat the channels-last failure without a Triton recorder or dispatch trace."""
import argparse
import json
from pathlib import Path
import traceback

import flag_gems
import run_conv as conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    assert conv.torch.__version__ == '2.10.0+cu128'
    assert conv.importlib.metadata.version('flag_gems') == '5.0.2'
    selection = json.loads(Path(conv.__file__).with_name('selection_conv_20261019_v1.json').read_text())
    case = next(c for c in selection['cases'] if c['id'] == 'conv2d_groups_channels_last')
    operands, upstream, options = conv.inputs(case, selection['development']['dimensions'], 20261231)
    reference, expected = conv.mathematical_reference(operands, upstream, options)
    standard, _ = conv.execute(conv.functional, operands, upstream, options)
    report = {'source': Path(__file__).read_text(), 'runner_source': Path(conv.__file__).read_text(),
              'instrumentation': 'none', 'case': case, 'reference': reference,
              'input_strides': [list(v.stride()) for v in operands],
              'standard': {k: conv.matrix.record(v) for k, v in standard.items()}}
    leaves = [v.detach().clone().requires_grad_() for v in operands]
    try:
        with flag_gems.use_gems(include=['conv2d']):
            y = conv.functional(leaves, **options)
            report['forward'] = conv.batch.record(y)
            report['forward_math'] = conv.batch.reference_comparison(y, expected['out'])
            conv.torch.autograd.grad(y, leaves, upstream)
        report['status'] = 'executed'
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    with args.out.open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(report['status'], report.get('error', ''), flush=True)


if __name__ == '__main__':
    main()
