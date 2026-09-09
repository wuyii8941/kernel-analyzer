"""Audit each capture, then merge the large coverage inventory only once."""
import argparse
from pathlib import Path
from scripts.join_forward_measurements import queue_roots
from scripts.finalize_residual_rms_forward import finalize
from scripts.join_coverage_measurements import join, validate_retained_measurements
from scripts.run_numerical_coverage import read, save, sha


def collect(release, roots):
    task_path = (release/'same_dtype_tasks.json.gz').resolve()
    task_sha = sha(task_path)
    records, audits = [], []
    for root in roots:
        path = root/'raw/family_execution_protocol.json'
        protocol = read(path)
        hashes = {str(Path(k).resolve()): v for k, v in protocol['source_sha256'].items()}
        if hashes.get(str(task_path)) != task_sha:
            raise ValueError('Capture task inventory differs')
        report = finalize(root)
        for original in report['records']:
            row = dict(original)
            if 'raw_path' in row:
                row['raw_artifact'] = row.pop('raw_path')
            records.append(row)
        audits.append(dict(root=str(root), protocol_sha256=sha(path), counts=report['counts']))
    return dict(records=records, errors=[]), audits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'release', 'queue', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose new output under /data1/tzh')
    inventory = read(a.inventory)
    checked = validate_retained_measurements(inventory)
    roots, statuses = queue_roots(a.queue)
    report, audits = collect(a.release, roots)
    result = join(inventory, a.release, report)
    result['retained_measurement_integrity'] = dict(checked=checked)
    result['forward_audit_scope'] = 'SAVED_RECORD_CHECK_NOT_INDEPENDENT_EXECUTION_VERIFICATION'
    result['capture_audits'] = audits
    result['queue_snapshot'] = dict(jobs=statuses, audited_terminal_captures=len(roots),
        protocol_sha256=sha(a.queue/'queue_protocol.json'), whole_queue_complete=False)
    result['input_inventory_sha256'] = sha(a.inventory)
    result['reporting_script_sha256'] = sha(Path(__file__))
    save(a.output, result)
    print(result['runtime_counts'])


if __name__ == '__main__':
    main()
