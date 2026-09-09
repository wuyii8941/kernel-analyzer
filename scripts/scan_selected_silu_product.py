#!/usr/bin/env python3
"""Check an additional reference on exactly the sources of an existing census.

Only source is read. This does not claim dynamic parameter reachability or
completed training measurements, and does not modify running capture code.
"""
import argparse
import json
from pathlib import Path

from kernel_analyzer.selected_silu_product_reference import check_source
from kernel_analyzer.scaled_masked_softmax_reference import check_source as check_scaled_softmax
from scripts.build_row_reduction_contracts import FAMILIES, discover
from scripts.finalize_numerical_family import read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=('SELECTED_SILU_PRODUCT','SCALED_MASKED_SOFTMAX'),
                        default='SELECTED_SILU_PRODUCT')
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    coverage = read(args.coverage)
    sources = {}
    # Empty source files have no definition rows, but remain in the census.
    for manifest, digest in coverage['input_sha256'].items():
        if sha(Path(manifest)) != digest:
            raise ValueError('Original source scan changed: ' + manifest)
        for row in read(Path(manifest))['sources']:
            old = sources.setdefault(row['source'], row['source_sha256'])
            if old != row['source_sha256']:
                raise ValueError('Conflicting source digests')
    for row in coverage['records']:
        old = sources.setdefault(row['source'], row['source_sha256'])
        if old != row['source_sha256']:
            raise ValueError('Conflicting source digests')
    if len(sources) != coverage['source_count']:
        raise ValueError('Source census is incomplete')
    family = args.family
    checker, filename = {
        'SELECTED_SILU_PRODUCT': (check_source, 'selected_silu_product_reference.py'),
        'SCALED_MASKED_SOFTMAX': (check_scaled_softmax, 'scaled_masked_softmax_reference.py'),
    }[family]
    FAMILIES[family] = (checker, ('FP32_NATIVE',), filename)
    rows = []
    for path, digest in sorted(sources.items()):
        if sha(Path(path)) != digest:
            raise ValueError('Source changed: ' + path)
        rows.append(discover(Path(path), family))
    report = dict(reference_family=family, sources=rows,
                  input_coverage_sha256=sha(args.coverage),
                  checker_sha256=sha(Path(__file__).resolve().parents[1] / 'src/kernel_analyzer' / filename),
                  scanner_sha256=sha(Path(__file__)),
                  discovery_engine_sha256=sha(Path(__file__).with_name('build_row_reduction_contracts.py')),
                  legal_variants=['FP32_NATIVE'],
                  runtime_measurement_status='NOT_ASSESSED_BY_SOURCE_SCAN',
                  numerical_results_read=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(sources=len(rows), matched=sum(
        r['status']=='SOURCE_CHECKED' for s in rows for r in s['rows']))))


if __name__ == '__main__':
    main()
