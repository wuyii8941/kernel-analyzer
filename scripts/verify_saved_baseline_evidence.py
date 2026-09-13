"""Verify existing baseline records against original statistics and tolerance logs.

No new thresholds, no implicit ground truth, no inference of absent allclose.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from kernel_analyzer.numerical_baselines import stage_baselines
from scripts.build_same_data_baselines import local_comparison_summary, full_mean_summary

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/property/numerical_coverage_v1'


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify():
    tol_paths={}
    for p in sorted(BASE.rglob('local_tolerance_comparisons.json')):
        tol_paths.setdefault(digest(p),p)
    result_rows=[];errors=[];seen=set();sources={}
    for report_path in sorted(BASE.rglob('baselines.json')):
        report=read(report_path)
        if report.get('schema')!='same-data-baselines-v1':continue
        sources[str(report_path.relative_to(ROOT))]=digest(report_path)
        captures={}
        for row in report['rows']:
            captures.setdefault((row['case_id'],row['raw_sha256']),[]).append(row)
        for key,reported_rows in captures.items():
            # Strongest enriched reports are all verified, then captures deduplicated below.
            raw=Path(reported_rows[0]['raw_path'])
            checks={};item=dict(case_id=key[0],raw_sha256=key[1],report=str(report_path.relative_to(ROOT)))
            try:
                data=raw.read_bytes();payload=json.loads(data)
                checks['raw_digest']=hashlib.sha256(data).hexdigest()==key[1]
                checks['case_identity']=payload['case_id']==key[0]
                calculated={r['stage']:r for r in stage_baselines(payload)}
                for row in reported_rows:
                    measured=calculated[row['stage']]
                    for metric in ['relative_rms','aligned_ratio_of_sums','total_effect_energy','total_reference_energy']:
                        if metric in row:
                            checks[row['stage']+'_'+metric]=metric in measured and math.isclose(row[metric],measured[metric],rel_tol=1e-12,abs_tol=1e-20)
                tol_hash=report.get('local_comparisons_sha256')
                local=next((r for r in reported_rows if r['stage']=='LOCAL'),None)
                item['allclose']='NOT_RECORDED'
                if local and isinstance(local.get('local_allclose'),bool):
                    if tol_hash not in tol_paths:raise ValueError('Tolerance log not found by frozen hash')
                    records=read(tol_paths[tol_hash])['records']
                    record=next(r for r in records if r['case_id']==key[0])
                    measured=local_comparison_summary(payload,data,record)
                    checks['tolerance_reanalysis']=all(local.get(k)==v for k,v in measured.items())
                    item.update(allclose=local['local_allclose'],rtol=local['local_rtol'],atol=local['local_atol'],
                                tolerance_log=str(tol_paths[tol_hash].relative_to(ROOT)),tolerance_sha256=tol_hash)
                for mean_path,h in report.get('full_means_sha256',{}).items():
                    mp=Path(mean_path)
                    if digest(mp)!=h:raise ValueError('Full mean log changed')
                    record=read(mp)
                    if record['case_id']==key[0]:
                        for row in reported_rows:
                            values=full_mean_summary(payload,data,record,row['stage'])
                            checks['full_mean_'+row['stage']]=all(row.get(k)==v for k,v in values.items())
                item['reference_scope']=payload.get('reference_comparison_scope')
                item['write_rms']=calculated.get('PARAMETER_WRITE',{}).get('relative_rms')
                item['local_rms']=calculated.get('LOCAL',{}).get('relative_rms')
            except Exception as exc:
                checks['exception']=False;item['error']=str(exc)
            item['checks']=checks;item['verified']=bool(checks) and all(checks.values())
            if not item['verified']:errors.append(item)
            result_rows.append(item)
    captures={}
    for row in result_rows:
        key=(row['case_id'],row['raw_sha256'])
        if row['verified'] and (key not in captures or isinstance(row['allclose'],bool)):
            captures[key]=row
    passed=[r for r in captures.values() if r['allclose'] is True and r['write_rms'] is not None]
    return dict(schema='baseline-original-evidence-verification-v1',
                status='VERIFIED_RECORDS' if not errors else 'HAS_UNVERIFIED_RECORDS',
                checked_report_capture_count=len(result_rows),unique_verified_captures=len(captures),
                local_results=dict(Counter(str(r['allclose']) for r in captures.values())),
                maximum_write_rms_among_allclose_passed=max((r['write_rms'] for r in passed),default=None),
                policies=dict(Counter(str((r.get('rtol'),r.get('atol'))) for r in captures.values() if isinstance(r['allclose'],bool))),
                errors=errors,rows=result_rows,sources=sources,
                conclusion='Recorded comparison and original statistics verified; not detection accuracy or independent GPU reproduction')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('Use a new repository output')
    result=verify();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','sources','errors']}))
    print('errors',len(result['errors']))
