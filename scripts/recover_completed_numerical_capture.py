#!/usr/bin/env python3
"""Recover fully written raw captures without pretending the process exit was observed.

Only records absent analysis/status files. No capture is restarted, historical
raw data is never edited. Declared analysis hashes must match; dependencies
absent from older protocols are explicitly recorded as current reanalysis.
"""
import argparse
import hashlib
import json
from pathlib import Path

from kernel_analyzer.numerical_campaign import digest, reference_scope
from kernel_analyzer.training_numerical_analysis import analyze_artifact
from scripts.run_numerical_coverage import ROOT, read, save, sha


def recover(root):
    protocol = read(root / 'protocol.json')
    coverage = read(root / 'coverage.json')
    if digest(coverage) != protocol['coverage_sha256']:
        raise ValueError('Coverage changed')
    analysis_sources = {}
    unfrozen_dependencies = []
    for relative in ('src/kernel_analyzer/training_numerical_analysis.py',
                     'src/kernel_analyzer/training_equivalence.py',
                     'src/kernel_analyzer/training_bias_profile.py',
                     'src/kernel_analyzer/analysis_result.py'):
        path = ROOT / relative
        actual = sha(path)
        analysis_sources[str(path)] = actual
        expected = protocol['source_sha256'].get(str(path))
        if expected is None:
            unfrozen_dependencies.append(str(path))
        elif expected != actual:
            raise ValueError('Original analysis source is unavailable: ' + relative)
    records = []
    for row in coverage['rows']:
        if row['status'] != 'READY_FOR_CAPTURE':
            continue
        case = row['case']
        run = root / 'runs' / hashlib.sha256(row['task_id'].encode()).hexdigest()[:20]
        raw_path = run / 'raw' / (case['case_id'] + '.json')
        if not raw_path.exists() or (run / 'status.json').exists():
            continue
        raw = read(raw_path)
        ids = raw.get('state_ids', [])
        if (raw.get('status') != 'COMPLETE' or len(ids) != 32 or len(set(ids)) != 32
                or raw.get('case_id') != case['case_id'] or raw.get('carrier') != case['carrier']
                or raw.get('runtime_boundary', {}).get('task_id') != row['task_id']
                or raw.get('determinism', {}).get('all_exact') is not True
                or any(len(raw.get('original_coordinate_statistics', {}).get(stage, [])) != 32
                       for stage in ('LOCAL', 'PARAMETER_GRADIENT', 'PARAMETER_WRITE'))):
            raise ValueError('Raw capture incomplete or identity mismatch: ' + str(raw_path))
        report = analyze_artifact(raw, protocol)
        report['provenance']['raw_sha256'] = sha(raw_path)
        target = run / 'analysis.json'
        if target.exists():
            if read(target) != report:
                raise ValueError('Existing analysis disagrees; not overwritten')
        else:
            save(target, report)
        save(run / 'status.json', {
            'status': report['measurement_status'], 'returncode': None,
            'recovery_basis': 'COMPLETE_RAW_CAPTURE_AND_RECOMPUTATION_NOT_OBSERVED_PROCESS_EXIT',
            'raw_sha256': sha(raw_path),
            'analysis_source_sha256': analysis_sources,
            'dependencies_not_frozen_in_original_protocol': unfrozen_dependencies,
            'analysis_data_use': 'CURRENT_REANALYSIS' if unfrozen_dependencies else 'FROZEN_ANALYSIS',
            'reference_comparison_scope': reference_scope(case.get('reference_method')),
        })
        records.append({'case_id': case['case_id'], 'status': report['measurement_status'],
                        'bias_analysis': report['bias_analysis']})
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Recovery writes must remain under /data1/tzh')
    print(json.dumps(recover(args.root), indent=2))
