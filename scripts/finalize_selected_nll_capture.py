"""Reuse common raw-record verification for reviewed loss-output captures."""
import argparse
import math
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha
from scripts.snapshot_frozen_sources import preserve
from scripts.finalize_explicit_output_capture import finalize
from kernel_analyzer.explicit_output_capture_audit import CONTRACTS


LOSS_CONTRACTS = {
    'selected-nll-capture-v1': (
        'INTERNAL_NLL_BACKWARD_OUTPUT_REPLACEMENT',
        'SAVED_INPUT_NLL_FP64_BF16_WRITE',
    ),
    'softcapped-nll-capture-v1': (
        'INTERNAL_SOFTCAPPED_NLL_OUTPUT_REPLACEMENT',
        'SOFTCAPPED_NLL_FP64_BF16_WRITE',
    ),
}


def descriptive_effects(raw):
    """Report exact saved-coordinate magnitudes without inventing a margin."""
    ids = list(raw.get('state_ids', []))
    confirmation = set(raw.get('confirmation_state_ids', []))
    if not ids or not confirmation <= set(ids):
        raise ValueError('Declared confirmation IDs must belong to the complete suite')

    def summarize(rows, selected):
        chosen = [rows[index] for index in selected]
        effect = math.fsum(float(row['effect_energy']) for row in chosen)
        repair = math.fsum(float(row['repair_energy']) for row in chosen)
        inner = math.fsum(float(row['effect_repair_inner_product']) for row in chosen)
        if repair <= 0.0 or effect < 0.0:
            raise ValueError('Saved coordinate energies are invalid')
        return {
            'total_rms_ratio': math.sqrt(effect / repair),
            'aligned_ratio_of_sums': inner / repair,
            'state_count': len(chosen),
            'nonzero_effect_state_count': sum(
                int(row.get('nonzero_effect_coordinates', 0)) > 0 for row in chosen
            ),
        }

    all_indices = list(range(len(ids)))
    confirmation_indices = [index for index, value in enumerate(ids) if value in confirmation]
    result = {}
    statistics = raw.get('original_coordinate_statistics', {})
    for stage in ('LOCAL', 'PARAMETER_GRADIENT', 'PARAMETER_WRITE'):
        rows = statistics.get(stage, [])
        if len(rows) != len(ids):
            raise ValueError('Original-coordinate statistics are incomplete: ' + stage)
        result[stage] = {
            'complete_suite': summarize(rows, all_indices),
            'confirmation_suite': (
                summarize(rows, confirmation_indices) if confirmation_indices else None
            ),
        }
    return {
        'scope': 'DIRECT_SAVED_COORDINATE_DESCRIPTION_NOT_EQUIVALENCE_OR_POPULATION_INFERENCE',
        'stages': result,
    }


def verify(root):
    protocol = read(root/'raw/family_execution_protocol.json')
    if protocol.get('schema') not in LOSS_CONTRACTS:
        raise ValueError('Supported loss capture protocol required')
    schema = protocol['schema']
    if schema in CONTRACTS:
        raise ValueError('Loss verifier contract already registered')
    CONTRACTS[schema] = LOSS_CONTRACTS[schema]
    try:
        result = finalize(root)
    finally:
        del CONTRACTS[schema]
    for record in result['records']:
        if record.get('status') == 'RECORDED_MEASUREMENT_CHECKED':
            record['descriptive_effects'] = descriptive_effects(read(Path(record['raw_path'])))
    result['loss_capture_verifier_sha256'] = sha(Path(__file__))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preserve-matching-sources', action='store_true')
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        raise ValueError('New output under /data1/tzh required')
    snapshot = a.root/'source_snapshot.json'
    if a.preserve_matching_sources and not snapshot.exists():
        preserve(a.root/'raw/family_execution_protocol.json', snapshot)
    result = verify(a.root)
    save(a.output, result)
    print(result['counts'])
    if not result['recorded_measurement_complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
