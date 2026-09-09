"""Report captured three-stage energy and scaling without inventing a margin."""
import argparse
import math
from pathlib import Path
from kernel_analyzer.forward_capture_audit import valid_statistics
from scripts.finalize_residual_rms_forward import finalize
from scripts.run_numerical_coverage import read, save, sha


def summarize(raw):
    ids = raw.get('state_ids', [])
    selected = raw.get('confirmation_state_ids', [])
    if not ids or len(set(ids)) != len(ids) or not selected or len(set(selected)) != len(selected) or not set(selected) <= set(ids):
        raise ValueError('Invalid confirmation state selection')
    indices = [ids.index(s) for s in selected]
    stages = {}
    for stage in ('LOCAL', 'PARAMETER_GRADIENT', 'PARAMETER_WRITE'):
        rows = raw.get('original_coordinate_statistics', {}).get(stage)
        if not valid_statistics(rows, len(ids)):
            raise ValueError('Invalid original-coordinate statistics: ' + stage)
        x = math.fsum(float(rows[i]['effect_energy']) for i in indices)
        b = math.fsum(float(rows[i]['repair_energy']) for i in indices)
        a = math.fsum(float(rows[i]['effect_repair_inner_product']) for i in indices)
        stages[stage] = dict(effect_energy_sum=x, repair_energy_sum=b,
            effect_repair_inner_product_sum=a,
            total_relative_rms=math.sqrt(x/b) if b > 0 else None,
            aligned_ratio_of_sums=a/b if b > 0 else None,
            candidate_to_repair_energy_ratio=(x+b+2*a)/b if b > 0 else None,
            energy_identity_scope='ALGEBRAIC_RECONSTRUCTION_NOT_INDEPENDENT_MECHANISM_TEST',
            status='MEASURED' if b > 0 else 'ZERO_REPAIR_ENERGY',
            mean_direction_claim='NOT_ESTABLISHED_BY_ENERGY_AND_ALIGNMENT_ALONE')
    return dict(confirmation_state_ids=selected, stages=stages,
        parameter_scope=raw.get('carrier'),
        optimizer=raw.get('optimizer'),
        parameter_write_protocol=raw.get('parameter_write_protocol'),
        reference_comparison_scope=raw.get('reference_comparison_scope'),
        scope='DESCRIPTIVE_FIXED_SUITE', population_guarantee=False,
        equivalence_decision='NOT_ASSESSED', training_consequence='NOT_MEASURED_HERE')


def report(root):
    audit = finalize(root)
    records = []
    for checked in audit['records']:
        row = dict(case_id=checked['case_id'], task_id=checked['task_id'],
                   measurement_status=checked['status'])
        if checked['status'] == 'RECORDED_MEASUREMENT_CHECKED':
            path = Path(checked['raw_path'])
            row['summary'] = summarize(read(path))
            if sha(path) != checked['raw_sha256']:
                raise ValueError('Raw record changed during report')
            row.update(raw_path=str(path), raw_sha256=checked['raw_sha256'])
        records.append(row)
    return dict(schema='forward-measurement-descriptive-summary-v1', records=records,
                protocol_sha256=audit['protocol_sha256'],
                reporting_script_sha256=sha(Path(__file__)),
                data_use='DESCRIPTIVE_REANALYSIS_NOT_NEW_CONFIRMATION')


def queue_report(queue):
    from scripts.join_forward_measurements import queue_roots
    roots, statuses = queue_roots(queue)
    captures = [dict(root=str(root), report=report(root)) for root in roots]
    return dict(schema='forward-queue-descriptive-summary-v1', captures=captures,
                jobs=statuses, queue_protocol_sha256=sha(queue/'queue_protocol.json'),
                data_use='DESCRIPTIVE_REANALYSIS_NOT_NEW_CONFIRMATION',
                whole_research_goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--root', type=Path)
    source.add_argument('--queue', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    result = queue_report(args.queue) if args.queue else report(args.root)
    save(args.output, result)
    for row in result.get('records', []):
        print(row['case_id'], row.get('summary', {}).get('stages', {}))
    if args.queue:
        print(dict(audited_captures=len(result['captures']), declared_jobs=len(result['jobs'])))
