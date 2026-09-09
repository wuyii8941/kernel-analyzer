#!/usr/bin/env python3
"""Recheck a frozen saved-source pool against every registered reference.

This is source eligibility, not runtime coverage. Adding a family to the shared
registry automatically includes it; neither case names nor numerical outcomes
select the sources or reference families.
"""
import argparse
import hashlib
import json
from pathlib import Path

from kernel_analyzer.source_reference_registry import REFERENCES
from scripts.build_row_reduction_contracts import discover_prepared, prepare_source
from scripts.summarize_source_reference_coverage import summarize


def checked_sources(pool):
    sources = {}
    for row in pool['records']:
        path, digest = row['source'], row['source_sha256']
        if sources.setdefault(path, digest) != digest:
            raise ValueError('Conflicting source versions: ' + path)
    if not sources:
        raise ValueError('Empty source pool')
    for path, digest in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source changed: ' + path)
    return sorted(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Use a new output directory under /data1/tzh')
    raw = args.pool.read_bytes()
    sources = checked_sources(json.loads(raw))
    prepared_sources = [prepare_source(path) for path in sources]
    args.output.mkdir(parents=True)
    manifests = []
    for family, specification in REFERENCES.items():
        adapter = Path(__file__).resolve().parents[1] / 'src/kernel_analyzer' / specification.source_filename
        manifest = dict(schema='source-checked-row-reduction-v1', reference_family=family,
                        sources=[discover_prepared(source, family) for source in prepared_sources],
                        adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                        legal_variants=list(specification.variants), numerical_results_read=False)
        with (args.output / (family.lower() + '.json')).open('x') as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
        manifests.append(manifest)
        print(f'{family}: {sum(r["status"] == "SOURCE_CHECKED" for s in manifest["sources"] for r in s["rows"])} source matches', flush=True)
    report = summarize(manifests)
    report['input_pool_sha256'] = hashlib.sha256(raw).hexdigest()
    report['selection'] = 'ALL_FROZEN_SOURCES_ALL_REGISTERED_FAMILIES'
    with (args.output / 'coverage.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()
