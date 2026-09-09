#!/usr/bin/env python3
"""Reverify a complete family plan and summarize observed update magnitudes.

No magnitude-only promotion to long training; missing warm-state and mechanism
evidence is explicit. Pending, invalid and zero results remain in the denominator.
"""
import argparse
import json
import math
from pathlib import Path
from scripts.audit_family_plan_execution import audit
from scripts.finalize_numerical_family import read
from kernel_analyzer.numerical_baselines import stage_baselines


def summarize(report):
    values=[]
    for row in report['records']:
        if row['status']!='VERIFIED': continue
        analysis=row['analysis']
        if analysis is not None:
            if analysis['claim_scope']!='FIXED_SUITE_UPDATE':
                raise ValueError('Do not mix population and fixed-suite observations')
            bias=analysis['bias_analysis']
            rms=bias['fixed_suite_total_rms']
            aligned=bias['fixed_suite_aligned_ratio_of_sums']
            confirmation_ids=bias['confirmation_state_ids']
        else:
            # Some source-checked family protocols deliberately declared no
            # equivalence margin.  Their verified raw statistics still support
            # a descriptive fixed-suite magnitude, but never a PASS/FAIL.
            raw=read(Path(row['raw_artifact']))
            stages={entry['stage']:entry for entry in stage_baselines(raw)}
            bias=stages['PARAMETER_WRITE']
            if bias['status']!='VALID':
                raise ValueError('Verified record lacks valid parameter-write statistics')
            rms=bias['relative_rms']
            aligned=bias['aligned_ratio_of_sums']
            confirmation_ids=raw['confirmation_state_ids']
        if not math.isfinite(rms) or rms<0: raise ValueError('Invalid RMS')
        values.append(dict(case_id=row['case_id'],update_rms_relative=rms,
             aligned_ratio=aligned,
             confirmation_state_ids=confirmation_ids,
             raw_artifact=row['raw_artifact'],raw_sha256=row['raw_sha256']))
    return dict(schema='family-measurement-summary-v1',audit=report,
         verified_positions=len(values),observations=values,
         nonzero_confirmation_rms_positions=sum(v['update_rms_relative']>0 for v in values),
         maximum_confirmation_update_rms=max((v['update_rms_relative'] for v in values),default=None),
         automatic_long_training_selection='NOT_PERFORMED_REQUIRES_WARM_STATE_AND_MECHANISM_EVIDENCE',
         strong_training_result_established=False,
         scope='Descriptive fixed-suite update magnitude; not a bias prevalence or quality claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--root',type=Path,nargs='*',default=[])
    p.add_argument('--queue',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose a new output under /data1/tzh')
    result=summarize(audit(a.plan,a.root,a.queue))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('audit','observations')}))


if __name__=='__main__': main()
