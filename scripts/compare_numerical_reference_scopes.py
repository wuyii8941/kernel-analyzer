#!/usr/bin/env python3
"""Compare recorded reference choices at the same declared training location."""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha
from scripts.verify_numerical_coverage import verify


def compare(first, second):
    roots = (first, second)
    verified = [verify(root) for root in roots]
    records = [{r['task_id']: r for r in result['records'] if r['verified_measurement']} for result in verified]
    protocols = [read(root / 'protocol.json') for root in roots]
    rows = []
    for task_id in sorted(set(records[0]) & set(records[1])):
        raw = [read(Path(records[i][task_id]['raw'])) for i in (0, 1)]
        checks = {key: raw[0].get(key) == raw[1].get(key) for key in
                  ('carrier', 'state_ids', 'calibration_state_ids', 'confirmation_state_ids',
                   'optimizer', 'parameter_write_protocol', 'runtime_boundary')}
        checks.update({key: protocols[0].get(key) == protocols[1].get(key)
                       for key in ('model', 'architecture', 'input_bank', 'release', 'primary_stage',
                                   'claim_scope', 'fixed_suite_margins', 'allow_graph_breaks')})
        bank_hashes = [p['source_sha256'].get(p['input_bank']) for p in protocols]
        checks['input_bank_digest'] = bool(bank_hashes[0]) and bank_hashes[0] == bank_hashes[1]
        if not all(checks.values()):
            raise ValueError('More than the declared reference choice differs: ' + str(checks))
        rows.append({'task_id': task_id, 'declared_setup_checks': checks,
                     'same_recorded_reference_scope': raw[0].get('reference_comparison_scope') == raw[1].get('reference_comparison_scope'),
                     'original_coordinate_statistics_identical': raw[0].get('original_coordinate_statistics') == raw[1].get('original_coordinate_statistics'),
                     'measurements': [{'root': str(roots[i].resolve()),
                                       'raw_sha256': records[i][task_id]['raw_sha256'],
                                       'reference_scope': raw[i].get('reference_comparison_scope'),
                                       'bias_analysis': records[i][task_id]['result']}
                                      for i in (0, 1)]})
    if not rows:
        raise ValueError('No shared verified training locations')
    return {'schema': 'reference-scope-comparison-v1', 'rows': rows,
            'protocol_sha256': {str(root / 'protocol.json'): sha(root / 'protocol.json') for root in roots},
            'scope': 'Same declared checkpoint/location/inputs/optimizer; different reference constructions',
            'full_checkpoint_content_identity_independently_verified': False,
            'upstream_only_causal_attribution': False,
            'interpretation': 'Differences between references do not establish that upstream differences are the sole cause; local reference precision/evaluation also differs.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expect-identical-measurement', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Use new output under /data1/tzh')
    result = compare(args.first, args.second)
    if args.expect_identical_measurement:
        result['execution_reuse_check_passed'] = all(row['same_recorded_reference_scope'] and
                                                     row['original_coordinate_statistics_identical'] for row in result['rows'])
    save(args.output, result)
    print({'compared_locations': len(result['rows'])})
    if args.expect_identical_measurement and not result['execution_reuse_check_passed']:
        raise SystemExit('Shared capture changed measured statistics; inspect retained comparison')
