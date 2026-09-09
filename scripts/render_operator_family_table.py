"""Render the checked operator-family report without changing its claims."""
import argparse
import csv
import json
from pathlib import Path


def rows_from_report(report):
    if report.get('schema')!='operator-family-report-v2':
        raise ValueError('Unsupported operator-family report')
    rows=[]
    for family in report['families']:
        counts=family['support_stage_counts']
        evidence=family.get('additional_measurement_evidence',[])
        rows.append(dict(
            family_id=family['family_id'],label=family['label'],
            classified_positions=family['classified_positions'],
            valid_measurement_positions=counts.get('VALID_MEASUREMENT_COMPLETED',0),
            bound_not_measured_positions=counts.get(
                'REFERENCE_AND_TRAINING_BINDING_READY_NOT_VALIDLY_MEASURED',0),
            reference_only_positions=counts.get('REFERENCE_AVAILABLE_TRAINING_BINDING_REQUIRED',0),
            additional_fixed_suite_evidence=len(evidence),
            historical_role_records=family['historical_role_records'],
            historical_artifacts=len(family['additional_historical_artifacts'])))
    return rows


def write_csv(path,rows):
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_markdown(path,rows,source):
    columns=list(rows[0])
    lines=[f'# 算子族覆盖表\n',f'来源：`{source}`。\n',
           '| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    for row in rows:
        lines.append('| '+' | '.join(str(row[c]) for c in columns)+' |')
    lines.extend(['',
        '`classified_positions` 是 release-qualified 输出位置，不是独立算子数。',
        '`additional_fixed_suite_evidence` 用于没有编译图位置的常规实现测量，不加入位置数。',
        '有效测量不等于 bias 阳性、总体等价或训练质量结论。',''])
    path.write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--markdown',type=Path,required=True)
    args=parser.parse_args()
    for output in (args.csv,args.markdown):
        if output.exists() or not output.resolve().is_relative_to(Path('/data1/tzh')):
            parser.error('New outputs under /data1/tzh required')
        output.parent.mkdir(parents=True,exist_ok=True)
    rows=rows_from_report(json.loads(args.report.read_text()))
    if not rows: raise ValueError('No operator families')
    write_csv(args.csv,rows); write_markdown(args.markdown,rows,args.report)
    print(json.dumps(dict(families=len(rows),csv=str(args.csv),markdown=str(args.markdown))))


if __name__=='__main__': main()
