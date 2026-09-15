"""Read-only legacy interval sensitivity; never relabel historical confirmation."""
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

from kernel_analyzer.mean_inference import mean_test_p, student_quantile


def legacy_critical(df):
    z = NormalDist().inv_cdf(.975)
    return (z + (z**3 + z)/(4*df) + (5*z**5 + 16*z**3 + 3*z)/(96*df**2)
            + (3*z**7 + 19*z**5 + 17*z**3 - 15*z)/(384*df**3))


def walk(value, location=''):
    if isinstance(value, dict):
        if {'estimate', 'confidence_interval_95', 'independent_unit_count',
                'raw_studentized_signflip_p'} <= value.keys():
            yield location, value
        for key, child in value.items():
            yield from walk(child, location + '/' + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, location + '/' + str(index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path; historical records are not overwritten')
    rows, skipped = [], []
    paths = set()
    for source in args.input:
        if not source.exists():
            raise FileNotFoundError(source)
        paths.update(source.rglob('*.json') if source.is_dir() else [source])
    for path in sorted(paths):
        for location, branch in walk(json.loads(path.read_text())):
            n = branch['independent_unit_count']
            estimate = branch['estimate']
            low, high = branch['confidence_interval_95']
            if n < 2 or not all(math.isfinite(x) for x in (estimate, low, high)):
                skipped.append({'file': str(path), 'location': location, 'reason': 'invalid interval or unit count'})
                continue
            if not math.isclose((low + high)/2, estimate, rel_tol=1e-7, abs_tol=1e-12):
                skipped.append({'file': str(path), 'location': location, 'reason': 'not a symmetric mean interval'})
                continue
            se = branch.get('standard_error')
            if se is None:
                se = (high-low)/(2*legacy_critical(n-1))
            critical = student_quantile(n-1, .975)
            interval = [estimate-critical*se, estimate+critical*se]
            p = mean_test_p(estimate, se, n-1)
            rows.append({'file': str(path), 'location': location, 'units': n,
                         'standard_error_source': 'stored' if 'standard_error' in branch else 'recovered_assuming_legacy_expansion',
                         'old_interval': [low, high], 'new_interval': interval,
                         'old_signflip_p': branch['raw_studentized_signflip_p'],
                         'new_mean_p': p,
                         'raw_significance_changed': (branch['raw_studentized_signflip_p'] < .05) != (p < .05),
                         'zero_sample_variance': se == 0})
    payload = {'scope': 'RETROSPECTIVE_NUMERICAL_SENSITIVITY_NOT_NEW_CONFIRMATION',
               'limitations': 'No Holm reclassification; saved interval recovery assumes the historical expansion. No inference-unit independence or original-vector validation.',
               'file_count': len(paths), 'branch_count': len(rows),
               'raw_significance_changes': sum(row['raw_significance_changed'] for row in rows),
               'skipped': skipped, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({key: value for key, value in payload.items() if key != 'rows'}))


if __name__ == '__main__':
    main()
