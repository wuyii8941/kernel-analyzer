#!/usr/bin/env python3
"""Freeze a small source-diverse capture set without reading numerical results."""
import argparse
import json
from pathlib import Path

from scripts.run_numerical_coverage import read, save, sha


def select(plan, limit):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError('Positive selection limit required')
    cases = plan.get('cases', [])
    if len({case['task_id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate case task identity')
    ordered = sorted(cases, key=lambda case: (
        case.get('reference_contract_symbol', ''), case['task_id']))
    chosen, seen_symbols = [], set()
    for case in ordered:
        symbol = case.get('reference_contract_symbol')
        if symbol in seen_symbols:
            continue
        chosen.append(case)
        seen_symbols.add(symbol)
        if len(chosen) == limit:
            break
    if len(chosen) < limit:
        chosen_ids = {case['task_id'] for case in chosen}
        chosen.extend(case for case in ordered if case['task_id'] not in chosen_ids)
        chosen = chosen[:limit]
    return dict(
        schema='registered-reference-source-diverse-selection-v1',
        cases=chosen,
        source_case_count=len(cases),
        selected_case_count=len(chosen),
        unresolved_count=len(plan.get('unresolved', [])),
        selection_rule='DISTINCT_REFERENCE_SYMBOL_THEN_TASK_ID_NO_NUMERICAL_RESULTS',
        numerical_results_read=False,
        runtime_measurement_complete=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    result = select(read(args.plan), args.limit)
    result['source_sha256'] = {str(args.plan.resolve()): sha(args.plan),
                               str(Path(__file__).resolve()): sha(Path(__file__))}
    save(args.output, result)
    print(json.dumps({key: result[key] for key in
                      ('source_case_count', 'selected_case_count', 'unresolved_count')}))


if __name__ == '__main__':
    main()
