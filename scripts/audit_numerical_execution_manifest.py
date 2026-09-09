#!/usr/bin/env python3
"""Recheck multiple full family plans without treating queued work as measured."""
import argparse
import json
from pathlib import Path

from scripts.audit_family_plan_execution import audit
from scripts.finalize_numerical_family import ROOT, read, sha


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build(manifest_path):
    manifest = read(manifest_path)
    names = [item['name'] for item in manifest['families']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate family entry')
    reports = []
    for item in manifest['families']:
        queues = [resolve(p) for p in item.get('queues', [])]
        started = [q for q in queues if (q / 'queue_protocol.json').exists()]
        pending = [str(q) for q in queues if q not in started]
        # The original complete plan retains positions from unstarted queues.
        # Missing queue metadata is neither a negative result nor proof of launch.
        report = audit(resolve(item['plan']),
                       [resolve(p) for p in item.get('roots', [])], started)
        reports.append(dict(name=item['name'], pending_queue_metadata=pending,
                            **report))
    return dict(
        schema='numerical-execution-manifest-audit-v1',
        manifest_sha256=sha(manifest_path),
        families=reports,
        eligible_positions=sum(r['eligible_positions'] for r in reports),
        verified_positions=sum(r['counts'].get('VERIFIED', 0) for r in reports),
        selected_plans_complete=bool(reports) and all(
            r['eligible_measurement_complete'] for r in reports),
        all_kernel_support_established=False,
        whole_research_plan_complete=False,
        process_liveness='NOT_INFERRED_FROM_FILES',
        scope='Saved measurements rechecked against complete selected plans; '
              'not a census of all kernels, mechanism count, or loss evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    report = build(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({r['name']: r['counts'] for r in report['families']}))


if __name__ == '__main__':
    main()
