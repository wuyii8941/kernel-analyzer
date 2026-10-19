"""Reduce retained attention reports to reviewable execution facts."""
import argparse
import json
from pathlib import Path


OUTPUTS = ('out', 'dq', 'dk', 'dv')


def error_head(row):
    error = row.get('error')
    return None if error is None else error.splitlines()[0]


def comparison(row, source_atol=None):
    values = row.get('comparison', {}).get('standard', {})
    result = {
        name: {
            'coordinates': values[name]['coordinates'],
            'nonfinite_pairs': values[name]['nonfinite_pairs'],
            'max_abs_difference': values[name]['max_abs_difference'],
            'common_strict_allclose': values[name]['allclose_rtol_1e-5_atol_1e-6'],
        }
        for name in OUTPUTS if name in values
    }
    if source_atol is not None and len(result) == len(OUTPUTS):
        result['candidate_source_tolerance'] = {
            'atol': source_atol,
            'rtol': 0,
            'all_outputs': all(value['nonfinite_pairs'] == 0 and value['max_abs_difference'] <= source_atol
                               for name, value in result.items() if name in OUTPUTS),
            'meaning': 'The retained tutorial source uses this threshold; it is not exact semantic equality.',
        }
    return result


def rows(path, source_atol=None):
    report = json.loads(path.read_text())
    result = []
    for row in report['samples']:
        result.append({
            'case': row['case'],
            'seed': row['seed'],
            'status': row['status'],
            'mathematical_reference': row.get('mathematical_reference', {}).get('outcome'),
            'error': error_head(row),
            'comparison': comparison(row, source_atol),
        })
    return result


def probe_rows(path):
    report = json.loads(path.read_text())
    return [{
        'case': row['case']['id'],
        'status': row['status'],
        'error': error_head(row),
        'comparison': {
            name: {
                'coordinates': value['coordinates'],
                'nonfinite_pairs': value['nonfinite_pairs'],
                'max_abs_difference': value['max_abs_difference'],
                'common_strict_allclose': value['allclose_rtol_1e-5_atol_1e-6'],
            }
            for name, value in row.get('comparison', {}).items()
        },
    } for row in report['cases']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--tutorial-original', type=Path, required=True)
    parser.add_argument('--tutorial-f32-supported-shapes', type=Path, required=True)
    parser.add_argument('--tutorial-fp16', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    report = {
        'scope': 'Execution and standard-comparison summary; no formal kernel binding and no statistical bias verdict.',
        'development_flaggems': rows(args.development),
        'tutorial_original_float32': rows(args.tutorial_original),
        'tutorial_tested_shapes_float32': rows(args.tutorial_f32_supported_shapes),
        'tutorial_tested_float16': rows(args.tutorial_fp16, source_atol=0.01),
        'flag_gems_uninstrumented_square_probe': probe_rows(args.probe),
        'source_review': {
            'flag_gems_boolean_mask': 'The public wrapper converts True to -1e6 and False to zero, opposite to the selected PyTorch public task.',
            'flag_gems_additive_mask': 'The noncausal path converts qk to base-2 units but adds the public additive mask without the same conversion.',
            'flag_gems_backward_mask': 'The autograd wrapper invokes backward with attn_mask=None.',
            'flag_gems_backward_mode': 'The backward kernel has no causal mode argument and executes its causal masked/nonmasked block split for every call.',
            'triton_tutorial_dtype': 'The retained official test call uses float16; its float32 calls are outside that tested implementation path.',
            'triton_tutorial_mode': 'The retained source marks noncausal tests as failing and tests backward only for causal float16 shapes.',
        },
        'refusals': [
            'No result here establishes complete producer coverage or a formal task binding.',
            'Numerical proximity under a source test threshold is not exact equality to the mathematical reference.',
            'Execution failures and unsupported calls remain in their original denominators.',
            'No statistical bias conclusion is made from this development batch.',
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
