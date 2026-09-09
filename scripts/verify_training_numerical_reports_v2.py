#!/usr/bin/env python3
"""Recompute saved VALID reports; verify arithmetic, not execution or sampling."""
import argparse
import json
from kernel_analyzer.training_numerical_analysis import analyze_artifact
from scripts.run_training_numerical_v2 import BASE, hashes, save_new
from scripts.run_liger_single_boundary_collapse import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = json.loads((BASE / 'protocol.json').read_text())
    if hashes() != protocol['source_sha256']:
        raise RuntimeError('Frozen core source differs')
    raw_paths = set(BASE.rglob('raw.json')) | set(BASE.glob('**/raw/*.json'))
    by_digest = {file_sha256(p): p for p in raw_paths}
    report_paths = set(BASE.rglob('recomputed.json')) | set(BASE.glob('**/recomputed/*.json'))
    checked = []
    fields = ('case_id', 'contrast_id', 'measurement_status', 'claim_scope',
              'bias_analysis', 'equivalence_decision', 'mandatory_endpoints')
    for path in sorted(report_paths):
        recorded = json.loads(path.read_text())
        if recorded.get('measurement_status') != 'VALID':
            continue
        digest = recorded.get('provenance', {}).get('raw_sha256')
        if digest not in by_digest:
            raise RuntimeError(f'Raw source missing or changed: {path}')
        raw = by_digest[digest]
        recomputed = analyze_artifact(json.loads(raw.read_text()), protocol)
        for field in fields:
            if recorded[field] != recomputed[field]:
                raise RuntimeError(f'Recomputation differs: {path}: {field}')
        checked.append({'case_id': recorded['case_id'], 'report': str(path.relative_to(BASE)),
                        'report_sha256': file_sha256(path), 'raw': str(raw.relative_to(BASE)),
                        'raw_sha256': digest})
    if not checked:
        raise RuntimeError('No valid reports checked')
    from pathlib import Path
    result = {'status': 'VERIFIED_RECOMPUTATION', 'reports': checked,
              'report_count': len(checked), 'core_sources': hashes(),
              'scope': 'Recorded original-coordinate statistics to fixed-suite decisions only; execution identity and independent sampling require their separate evidence.'}
    save_new(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('reports', 'core_sources')}, indent=2))


if __name__ == '__main__':
    main()
