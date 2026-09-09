#!/usr/bin/env python3
"""Combine family scans without treating unmatched kernels as numerical negatives."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(manifests):
    if not manifests:
        raise ValueError('No family scans')
    sources = None
    rows = {}
    families = set()
    for manifest in manifests:
        family = manifest['reference_family']
        if family in families:
            raise ValueError('Duplicate family scan')
        families.add(family)
        current = {}
        current_keys = set()
        for source in manifest['sources']:
            path = source['source']
            if path in current:
                raise ValueError('Duplicate source in scan')
            current[path] = source['source_sha256']
            for item in source['rows']:
                key = (path, item['symbol'])
                if key in current_keys:
                    raise ValueError('Duplicate source/symbol')
                current_keys.add(key)
                row = rows.setdefault(key, {'source': path, 'symbol': item['symbol'],
                                            'source_sha256': source['source_sha256'],
                                            'matched_reference_families': []})
                if item['status'] == 'SOURCE_CHECKED':
                    if item['contract']['source_sha256'] != source['source_sha256']:
                        raise ValueError('Contract source digest differs')
                    row['matched_reference_families'].append(family)
                elif item['status'] != 'NOT_THIS_REFERENCE_FAMILY':
                    raise ValueError('Unrecognized scan status')
        if sources is None:
            sources, expected_keys = current, current_keys
        elif current != sources or current_keys != expected_keys:
            raise ValueError('Family scans cover different source versions or definitions')
    records = []
    for key in sorted(rows):
        row = rows[key]
        matches = row['matched_reference_families']
        row['status'] = ('REFERENCE_TEMPLATE_AVAILABLE' if len(matches) == 1 else
                         'MULTIPLE_REFERENCES_REQUIRE_EXPLICIT_SELECTION' if matches else
                         'REFERENCE_ADAPTER_REQUIRED')
        row['runtime_measurement_status'] = 'NOT_ASSESSED_BY_SOURCE_SCAN'
        records.append(row)
    return {'schema': 'source-reference-coverage-v1',
            'scope': 'Saved Triton definition records only; excludes external calls and does not establish runtime training support',
            'source_count': len(sources), 'definition_records': len(records),
            'reference_families_checked': sorted(families),
            'counts': dict(Counter(r['status'] for r in records)),
            'numerical_results_read': False, 'unmatched_means_unbiased': False,
            'all_kernel_support_established': False, 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Use a new output under /data1/tzh')
    data = [(path, path.read_bytes()) for path in args.manifest]
    report = summarize([json.loads(value) for _, value in data])
    report['input_sha256'] = {str(path.resolve()): hashlib.sha256(value).hexdigest() for path, value in data}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: report[key] for key in ('source_count', 'definition_records', 'counts')}))


if __name__ == '__main__':
    main()
