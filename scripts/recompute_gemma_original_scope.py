#!/usr/bin/env python3
"""Recompute the recovered original Gemma boundary with the unchanged analyzer."""
import json
from kernel_analyzer.training_numerical_analysis import analyze_artifact
from scripts.run_training_numerical_v2 import BASE, save_new
from scripts.run_liger_single_boundary_collapse import file_sha256


def main():
    root=BASE/'recovery/gemma_original_trainability_scope'
    raw_path=root/'raw/gemma4_text128_scan_0037.json'
    if not raw_path.exists():raise SystemExit('Capture not complete')
    binding=json.loads((root/'protocol.json').read_text())
    scope=json.loads((root/'trainability_protocol.json').read_text())
    if binding['aot_endpoint_claimed'] or binding['larger_boundary_substituted'] or scope['identity_checks_relaxed']:
        raise RuntimeError('Reference claim or scope differs')
    protocol=json.loads((BASE/'protocol.json').read_text())
    raw=json.loads(raw_path.read_text())
    result=analyze_artifact(raw,protocol)
    result['provenance'].update(raw_sha256=file_sha256(raw_path),
        explicit_reference_protocol=str(root/'protocol.json'),explicit_reference_sha256=file_sha256(root/'protocol.json'),
        trainability_protocol=str(root/'trainability_protocol.json'),trainability_sha256=file_sha256(root/'trainability_protocol.json'),
        reference_scope='Original internal sum-of-squares output with runtime kernel AST guard; not an invented AOT endpoint',
        data_use=binding['data_use'])
    save_new(root/'recomputed.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
