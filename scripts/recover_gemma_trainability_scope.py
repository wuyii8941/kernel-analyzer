#!/usr/bin/env python3
"""Restore the original single-parameter backward scope before source validation."""
import argparse
import json
from pathlib import Path
import sys
from scripts import recover_gemma_square_sum_boundary as recovery
from scripts.run_training_numerical_v2 import BASE, save_new
from scripts.run_liger_single_boundary_collapse import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','capture'));args=p.parse_args()
    recovery.OUT=BASE/'recovery/gemma_original_trainability_scope'
    carrier='model.language_model.per_layer_model_projection.weight'
    if args.command=='prepare':
        sys.argv=['recover_gemma_square_sum_boundary.py','prepare'];recovery.main()
        save_new(recovery.OUT/'trainability_protocol.json',{
            'source_sha256':file_sha256(Path(__file__)),
            'trainable_parameters':[carrier],
            'historical_runner':'scripts/run_gemma4_v3_validation.py: requested_carrier with retain_full_backward=False',
            'historical_runner_sha256':file_sha256(Path('scripts/run_gemma4_v3_validation.py')),
            'reason':'The generic recapture left all parameters trainable, changing the generated backward and saved internal outputs. Restore the declared original carrier scope; retain the exact original kernel AST gate.',
            'identity_checks_relaxed':False})
        return
    protocol=json.loads((recovery.OUT/'trainability_protocol.json').read_text())
    if protocol['source_sha256']!=file_sha256(Path(__file__)):raise RuntimeError('Trainability adapter changed')
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    original=capture.load_model
    def load(*a,**kw):
        model=original(*a,**kw)
        for name,parameter in model.named_parameters():parameter.requires_grad_(name==carrier)
        if [n for n,p in model.named_parameters() if p.requires_grad]!=[carrier]:raise RuntimeError('Trainability differs from original scope')
        return model
    capture.load_model=load
    sys.argv=['recover_gemma_square_sum_boundary.py','capture','--device','cuda:0']
    try:recovery.main()
    except Exception as error:
        save_new(recovery.OUT/'failure.json',{'status':'ORIGINAL_SCOPE_RECAPTURE_FAILED','error':str(error),'replacement_used':False})
        raise


if __name__=='__main__':main()
