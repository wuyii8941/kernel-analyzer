#!/usr/bin/env python3
"""Bind every checked continued segment from a source scan, without case picking."""
import argparse
from pathlib import Path
import subprocess
import sys
from scripts.run_numerical_coverage import read, save, sha


def candidates(scan, source):
    if scan.get('schema') != 'continued-recurrence-source-scan-v1':
        raise ValueError('Unsupported discovery schema')
    for path, digest in scan['source_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Discovery implementation changed: '+path)
    matches = [s for s in scan['sources'] if Path(s['path']).resolve() == source.resolve()]
    if len(matches) != 1 or matches[0]['sha256'] != sha(source):
        raise ValueError('Missing, ambiguous or changed source')
    rows = matches[0]['rows']
    checked = [r for r in rows if r['status'] == 'SOURCE_CHECKED']
    if len({r['symbol'] for r in checked}) != len(checked):
        raise ValueError('Duplicate checked definition')
    for row in checked:
        c = row['contract']
        if (c['symbol'] != row['symbol'] or c['source_sha256'] != sha(source)
                or c['segment_input_kind'] != 'FP32_PREVIOUS_STATE'):
            raise ValueError('Discovery contract identity differs')
    return checked, [r for r in rows if r['status'] != 'SOURCE_CHECKED']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scan', 'source', 'mapping', 'tasks', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--sequence-length', type=int, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose new output under /data1/tzh')
    checked, rejected = candidates(read(a.scan), a.source)
    a.output.mkdir(parents=True)
    results = []
    for i, row in enumerate(checked):
        output = a.output/f'plan_{i:03d}.json'
        subprocess.run([sys.executable, '-B', 'scripts/bind_continued_recurrence.py',
            '--source', str(a.source), '--mapping', str(a.mapping), '--tasks', str(a.tasks),
            '--symbol', row['symbol'], '--time-start', str(row['contract']['time_start']),
            '--sequence-length', str(a.sequence_length), '--output', str(output)], check=True)
        plan = read(output)
        # Rechecking must reproduce all source evidence from discovery.
        if any(plan['contract'].get(k) != v for k, v in row['contract'].items()):
            raise ValueError('Binding does not reproduce discovered contract')
        results.append(dict(symbol=row['symbol'], plan=str(output.resolve()), sha256=sha(output),
                            positions=len(plan['cases']), unresolved=len(plan['unresolved'])))
    save(a.output/'index.json', dict(schema='discovered-recurrence-binding-index-v1',
        plans=results, rejected_definitions=rejected, numerical_results_read=False,
        runtime_measurement_complete=False,
        input_sha256={str(x.resolve()):sha(x) for x in (a.scan,a.source,a.mapping,a.tasks)},
        source_sha256={str(Path(__file__).resolve()):sha(Path(__file__)),
            str(Path('scripts/bind_continued_recurrence.py').resolve()):sha(Path('scripts/bind_continued_recurrence.py'))}))
    print(dict(definitions=len(results), positions=sum(r['positions'] for r in results)))


if __name__ == '__main__':
    main()
