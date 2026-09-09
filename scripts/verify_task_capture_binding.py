#!/usr/bin/env python3
"""Verify that a separate AOT capture is the one bound by the release tasks."""
import argparse
import json
from pathlib import Path

from scripts.run_numerical_coverage import read, sha


def verify(tasks, proof, capture):
    capture=capture.get('capture',capture)
    recorded=proof['standard_aot_capture']
    checks=dict(
        task_proof_identity=bool(proof.get('result_sha256')) and
            tasks['bindings']['proof_capture_result_sha256']==proof['result_sha256'],
        capture_identity=bool(recorded.get('capture_sha256')) and
            capture.get('capture_sha256')==recorded['capture_sha256'],
        complete_graphs_equal=bool(capture.get('graphs')) and capture['graphs']==recorded['graphs'])
    return dict(status='CAPTURE_BOUND_TO_TASK_PROOF' if all(checks.values()) else 'BINDING_MISMATCH',
                checks=checks, runtime_measurement_complete=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tasks','proof','capture','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose a new output under /data1/tzh')
    result=verify(read(a.tasks),read(a.proof),read(a.capture))
    result['source_sha256']={str(v.resolve()):sha(v) for v in (a.tasks,a.proof,a.capture,Path(__file__))}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(result['status'])
    if result['status']!='CAPTURE_BOUND_TO_TASK_PROOF': raise SystemExit(1)


if __name__=='__main__': main()
