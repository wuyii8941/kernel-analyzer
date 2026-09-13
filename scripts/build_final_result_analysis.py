"""Finish a finite evidence-analysis task, never declare universal science complete."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.verify_same_path_training_attribution import verify as verify_training
from scripts.analyze_review_evidence import build as verify_measurements

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/property/result_analysis_v1'


def read(name):return json.loads((BASE/name).read_text())


def build():
    measurements=verify_measurements()
    training=verify_training(BASE/'same_path_training')
    baseline=read('baseline_verification.json')
    bounded=read('bounded_validation_actual_readback.json')
    exceedance=read('exceedance_validation_replay.json')
    tests=read('test_run.json')
    additional_tests=read('final_additional_test_run.json')
    control=read('compensation_control_scalar_followup/result.json')
    checks=dict(measurement_records_recomputed=measurements['status']=='RECORDS_RECOMPUTED',
                baseline_records_verified=baseline['status']=='VERIFIED_RECORDS' and not baseline['errors'],
                bounded_validation=bounded['status']=='PASS',
                exceedance_validation=exceedance['status']=='PASS',
                full_tests=tests['exit_code']==0,
                final_additional_tests=additional_tests['exit_code']==0,
                scalar_control=control['tensor_off_matches_eager_default_all_steps'],
                all_training_records_and_checkpoints=training['status']=='VERIFIED',
                historical_compensation_reproduction=training.get('historical_on_reproduced') is True)
    return dict(schema='finite-result-analysis-completion-v1',
        status='COMPLETED_ANALYSIS_UNDER_DECLARED_SCOPE' if all(checks.values()) else 'ANALYSIS_NOT_FINISHED',
        checks=checks,measurement_counts=measurements['counts'],
        training=training,
        conclusions=dict(
            standalone_large_update_detection_beyond_local_allclose='NOT_ESTABLISHED_IN_RECORDED_COMPARISONS',
            negative_alignment_means_shorter_step='FALSE_IN_GENERAL; USE_RECOMPUTED_TOTAL_NORM_RATIO',
            all_signature_results_are_single_kernel_bias='NOT_ESTABLISHED; GRAPH_ENDPOINT_SUBSTITUTION',
            statistical_guarantees='CONDITIONAL_BY_ESTIMAND_NOT_UNIVERSAL',
            compensation_same_path_training=training.get('primary',{}).get('decision','NOT_FINISHED'),
            universal_mainline_complete=False),
        baseline_summary={k:baseline[k] for k in ['unique_verified_captures','local_results','maximum_write_rms_among_allclose_passed','policies']},
        preserved_negative_results=['local-allclose incremental-detection claim unsupported',
            'legacy bounded-validation fixture incompatible with actual-write gate',
            'default compiled and eager arithmetic differ independently of compensation'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),BASE/'evidence.json',BASE/'baseline_verification.json',BASE/'bounded_validation_actual_readback.json',BASE/'exceedance_validation_replay.json',BASE/'test_run.json',BASE/'final_additional_test_run.json']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('Use new repository output')
    result=build()
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({'status':result['status'],'checks':result['checks'],'conclusions':result['conclusions']}))
    if result['status']!='COMPLETED_ANALYSIS_UNDER_DECLARED_SCOPE':raise SystemExit(1)
