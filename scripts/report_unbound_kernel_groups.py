"""Prioritize missing reference bindings without inspecting numerical outcomes."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def summarize(records):
    groups = defaultdict(list)
    seen = set()
    for row in records:
        identity = (row['release'], row['task_id'])
        if identity in seen:
            raise ValueError('Repeated release-qualified position')
        seen.add(identity)
        if row['eligibility'] != 'NO_CHECKED_REFERENCE_FOR_THIS_OUTPUT':
            continue
        key = (row['implementation_kind'], row['phase'], row['symbol'])
        groups[key].append(row)
    result = []
    for (backend, phase, symbol), rows in groups.items():
        result.append(dict(implementation_kind=backend, phase=phase, symbol=symbol,
            positions=len(rows), releases=sorted({r['release'] for r in rows}),
            output_pointers=sorted({r['formal_pointer'] for r in rows}),
            position_ids=[dict(release=r['release'], task_id=r['task_id']) for r in rows]))
    return sorted(result, key=lambda r: (-r['positions'], r['implementation_kind'],
                                       r['phase'], r['symbol']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    raw = args.inventory.read_bytes()
    groups = summarize(json.loads(raw)['records'])
    report = dict(schema='unbound-kernel-groups-v1',
        inventory_sha256=hashlib.sha256(raw).hexdigest(),
        selection_uses_numerical_results=False,
        scope='EXACT_SYMBOL_GROUPS_NOT_AUDITED_SEMANTIC_FAMILIES',
        unsupported_positions=sum(r['positions'] for r in groups), groups=groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k != 'position_ids'} for r in groups[:10]], indent=2))


if __name__ == '__main__':
    main()
