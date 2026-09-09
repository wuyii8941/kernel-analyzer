#!/usr/bin/env python3
"""Recompute paired quality and recorded runtime/memory from complete run sets.

Recorded runtime includes evaluation and shared-machine interference. It is
not an isolated throughput benchmark or evidence of a speed improvement.
"""
import argparse
import math
import statistics
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha


def summarize(root):
    plan = read(root / 'plan.json')
    rows = []
    provenance = {str(root / 'plan.json'): sha(root / 'plan.json')}
    for pair in range(plan['pairs']):
        paths = [root / f'pair{pair}' / condition / 'status.json'
                 for condition in ('candidate', 'reference')]
        if not all(path.exists() for path in paths):
            raise ValueError(f'Incomplete pair {pair}; no subset summary')
        c, r = [read(path) for path in paths]
        if any(state['status'] != 'COMPLETE_DEVELOPMENT_PILOT' for state in (c, r)):
            raise ValueError(f'Non-complete pair {pair}')
        if not c.get('initial_parameters_sha256') or c['initial_parameters_sha256'] != r.get('initial_parameters_sha256'):
            raise ValueError(f'Initial parameters differ in pair {pair}')
        evaluations = [[v for v in state['evaluations'] if v['step'] == plan['steps']]
                       for state in (c, r)]
        if any(len(values) != 1 for values in evaluations):
            raise ValueError(f'Primary evaluation missing or duplicated in pair {pair}')
        losses = [values[0]['shared_evaluation_loss'] for values in evaluations]
        seconds = [state['elapsed_seconds_including_evaluation'] for state in (c, r)]
        memory = [state['peak_allocated_bytes'] for state in (c, r)]
        if not all(math.isfinite(x) for x in losses + seconds + memory) or min(seconds) <= 0 or min(memory) < 0:
            raise ValueError(f'Invalid recorded metrics in pair {pair}')
        rows.append({'pair': pair, 'step': plan['steps'],
                     'candidate_evaluation_loss': losses[0], 'reference_evaluation_loss': losses[1],
                     'candidate_minus_reference_loss': losses[0] - losses[1],
                     'candidate_seconds_including_evaluation': seconds[0],
                     'reference_seconds_including_evaluation': seconds[1],
                     'candidate_over_reference_recorded_time': seconds[0] / seconds[1],
                     'candidate_peak_allocated_bytes': memory[0],
                     'reference_peak_allocated_bytes': memory[1]})
        provenance.update({str(path): sha(path) for path in paths})
    if not rows:
        raise ValueError('No declared pairs')
    return {'schema': 'paired-quality-recorded-cost-v1', 'pairs': len(rows), 'rows': rows,
            'mean_loss_gap': statistics.mean(row['candidate_minus_reference_loss'] for row in rows),
            'median_recorded_time_ratio': statistics.median(row['candidate_over_reference_recorded_time'] for row in rows),
            'source_sha256': provenance, 'scope': plan.get('scope', plan.get('population_scope')),
            'cost_role': 'DESCRIPTIVE_INCLUDES_EVALUATION_AND_SHARED_MACHINE_INTERFERENCE',
            'isolated_throughput_claim': False, 'new_quality_test_performed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Use a new output under /data1/tzh')
    result = summarize(args.root)
    save(args.output, result)
    print({key: result[key] for key in ('pairs', 'mean_loss_gap', 'median_recorded_time_ratio')})
