#!/usr/bin/env python3
"""Audit the existing asymptotic Q test across independent-unit counts.

No classifier, margin or calibration threshold is tuned by this program.
The infinite-variance case is deliberately outside the stated CLT conditions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from kernel_analyzer.training_equivalence import population_total_energy_equivalence


def wilson(k,n):
    z=1.959963984540054; center=(k/n+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-half),min(1.,center+half)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=2000)
    a=p.parse_args()
    if a.repetitions<100 or a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use a new output under /data1/tzh and at least 100 repeats')
    rng=np.random.default_rng(2026090617); rows=[]; margin=.01
    for name in ('gamma2','t8_energy','t3_energy_outside_finite_variance'):
        for n in (16,64,256,1024):
            for scale in (.95,1.,1.05):
                decisions={'EQUIVALENT':0,'NON_EQUIVALENT':0,'INCONCLUSIVE':0}
                covered=0
                for _ in range(a.repetitions):
                    e=(rng.gamma(2,.5,n) if name=='gamma2' else
                       rng.standard_t(8,n)**2*.75 if name=='t8_energy' else rng.standard_t(3,n)**2/3)
                    # Explicit vectors -> original-coordinate energies -> the
                    # same production population function. R=1 in expectation.
                    b=rng.gamma(3,1/3,n)
                    u=np.sqrt(e)*margin*scale; r=np.sqrt(b)
                    result=population_total_energy_equivalence(u*u,r*r,rms_margin=margin)
                    decisions[result['decision']]+=1
                    lo,hi=result['one_sided_mean_bounds']
                    truth=margin**2*(scale**2-1)
                    covered+=lo<=truth<=hi
                k=decisions['EQUIVALENT']
                rows.append({'distribution':name,'independent_units':n,'true_rms_ratio':margin*scale,
                             'finite_variance_condition':name!='t3_energy_outside_finite_variance',
                             'counts':decisions,'repetitions':a.repetitions,
                             'equivalence_rate':k/a.repetitions,'equivalence_rate_mc_interval_95':wilson(k,a.repetitions),
                             'two_one_sided_bounds_coverage':covered/a.repetitions,
                             'nominal_two_one_sided_bounds_coverage':.90})
    result={'schema':'population-sample-size-audit-v1','status':'DIAGNOSTIC_NOT_A_UNIVERSAL_CALIBRATION_PASS',
            'production_function':'population_total_energy_equivalence', 'margin':margin,'alpha':.05,
            'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                (Path(__file__),Path('src/kernel_analyzer/training_equivalence.py'))},
            'rows':rows,'population_certificate_enabled_by_this_run':False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'rows':len(rows),'repetitions_per_row':a.repetitions}))


if __name__=='__main__':main()
