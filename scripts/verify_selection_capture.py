"""Recompute selection probes from retained tensors; no population certificate."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from kernel_analyzer.selection_reference import validate_selection
from scripts.run_training_bias_profile_v2_empirical import _original_coordinate_row


def same_recomputed_value(saved, checked):
    """Compare JSON records while allowing last-bit reduction variation."""
    if isinstance(saved, dict) and isinstance(checked, dict):
        return saved.keys() == checked.keys() and all(
            same_recomputed_value(saved[key], checked[key]) for key in saved)
    if isinstance(saved, list) and isinstance(checked, list):
        return len(saved) == len(checked) and all(
            same_recomputed_value(a, b) for a, b in zip(saved, checked))
    if isinstance(saved, float) and isinstance(checked, float):
        return saved == checked or abs(saved-checked) <= 1e-14 * max(abs(saved), abs(checked))
    return saved == checked


def check_outcomes(outcomes):
    if len(outcomes) != 4:
        raise ValueError('Expected original/variant and two repeat executions')
    for row in outcomes:
        for key in ('scores', 'values', 'gradient', 'write', 'loss'):
            if not torch.isfinite(row[key]).all():
                raise ValueError('Nonfinite recorded tensor: ' + key)
        validate_selection(row['scores'], row['values'], row['indices'],
                           row['values'].shape[1], dim=1)
    same_scores = all(torch.equal(outcomes[0]['scores'], r['scores']) for r in outcomes[1:])
    repeat = all(torch.equal(outcomes[j][key], outcomes[j+2][key])
                 for j in (0, 1) for key in ('values', 'indices', 'gradient', 'write', 'loss'))
    candidate, reference = outcomes[:2]
    statistics = {stage: _original_coordinate_row(candidate[key]-reference[key], reference[key])
                  for stage, key in (('LOCAL', 'values'), ('PARAMETER_GRADIENT', 'gradient'),
                                     ('PARAMETER_WRITE', 'write'))}
    sets_differ = (candidate['indices'].sort(dim=1).values !=
                   reference['indices'].sort(dim=1).values).any(dim=1)
    return dict(same_scores=same_scores, repeat_determinism=repeat,
                original_coordinate_statistics=statistics,
                changed_index_coordinates=int((candidate['indices'] != reference['indices']).sum()),
                changed_expert_sets=int(sets_differ.sum()),
                candidate_minus_reference_loss=float(candidate['loss']-reference['loss']),
                status='MEASURED_ENGINEERING_PROBE' if same_scores and repeat else 'INVALID_COMPARISON')


def verify(root):
    summary = json.loads((root/'summary.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol != summary['protocol'] or protocol['schema'] != 'granite-selection-probe-v1':
        raise ValueError('Protocol mismatch')
    records = summary['records']
    if len(records) != protocol['states'] or len({r['state_id'] for r in records}) != len(records):
        raise ValueError('Missing or duplicate states')
    checked = []
    for index, record in enumerate(records):
        raw = Path(record['raw_path'])
        if raw.resolve() != (root/f'state_{index:03d}.pt').resolve():
            raise ValueError('Unexpected raw path')
        if hashlib.sha256(raw.read_bytes()).hexdigest() != record['raw_sha256']:
            raise ValueError('Raw hash mismatch')
        if json.loads((root/f'state_{index:03d}.json').read_text()) != record:
            raise ValueError('State record mismatch')
        result = check_outcomes(torch.load(raw, map_location='cpu', weights_only=True))
        for key, value in result.items():
            if key != 'changed_expert_sets' and not same_recomputed_value(record.get(key), value):
                raise ValueError('Recomputation differs: ' + key)
        checked.append(dict(state_id=record['state_id'], **result))
    return dict(schema='selection-capture-recomputation-v1', records=checked,
                all_comparisons_valid=all(r['status']=='MEASURED_ENGINEERING_PROBE' for r in checked),
                claim_scope=protocol['state_scope'], equivalence_decision='NOT_ASSESSED',
                execution_identity_independently_proven=False, training_quality_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        raise ValueError('Output must be under /data1/tzh')
    result = verify(args.root)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
