"""Build a checked reporting record for an ordinary Top-k/sort measurement.

This does not invent a compiled-kernel position and does not issue an
equivalence decision.  It verifies the retained fixed-suite tensors and makes
their bounded evidence available to the operator-family report.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.verify_selection_capture import same_recomputed_value, verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, recomputation_path):
    saved=json.loads(recomputation_path.read_text())
    checked=verify(root)
    if not same_recomputed_value(saved, checked):
        raise ValueError('Saved selection recomputation differs from retained tensors')
    records=checked['records']
    stages=('LOCAL','PARAMETER_GRADIENT','PARAMETER_WRITE')
    exact_zero={stage: all(
        row['original_coordinate_statistics'][stage]['effect_energy']==0.0 and
        row['original_coordinate_statistics'][stage]['nonzero_effect_coordinates']==0
        for row in records) for stage in stages}
    return dict(schema='operator-family-additional-evidence-v1',records=[dict(
        family='SELECTION',
        artifact_path=str(recomputation_path),
        artifact_sha256=sha(recomputation_path),
        evidence_kind='ORDINARY_IMPLEMENTATION_FIXED_SUITE_THREE_STAGE_MEASUREMENT',
        backend='TRANSFORMERS_EAGER_ATEN',
        state_count=len(records),
        claim_scope=checked['claim_scope'],
        measurement_status='VALID_FIXED_SUITE_ENGINEERING_MEASUREMENT',
        exact_zero_effect_by_stage=exact_zero,
        equivalence_decision=checked['equivalence_decision'],
        execution_identity_independently_proven=checked['execution_identity_independently_proven'],
        training_quality_claim=checked['training_quality_claim'],
        position_inventory_count=0,
        scope='Additional family evidence only; no compiled-position, population, equivalence, or training-quality claim')])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--recomputation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('New output under /data1/tzh required')
    result=build(args.root,args.recomputation)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result['records'][0],indent=2))


if __name__=='__main__': main()
