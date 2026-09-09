#!/usr/bin/env python3
"""Split a frozen family plan by capacity, never by measured numerical effect."""
import argparse
import hashlib
import json
from pathlib import Path


def partitions(plan, size):
    cases = plan['cases']
    if size < 1 or not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Nonempty unique cases and positive batch size required')
    return [dict(cases=cases[i:i+size], selection='SOURCE_PLAN_ORDER_CAPACITY_ONLY',
                 numerical_results_read=False, full_plan_case_count=len(cases),
                 batch_index=i//size, unresolved_preserved_in_source_plan=True)
            for i in range(0, len(cases), size)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use new output under /data1/tzh')
    raw = a.plan.read_bytes()
    batches = partitions(json.loads(raw), a.batch_size)
    a.output.mkdir(parents=True)
    for batch in batches:
        batch['source_plan_sha256'] = hashlib.sha256(raw).hexdigest()
        batch['source_plan'] = str(a.plan.resolve())
        with (a.output / f"batch_{batch['batch_index']:03d}.json").open('x') as f:
            json.dump(batch, f, indent=2, allow_nan=False)
    print(json.dumps(dict(batches=len(batches), cases=sum(len(b['cases']) for b in batches))))


if __name__ == '__main__': main()
